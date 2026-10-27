"""A geração do G2 (DOC-12 §3.7.4 e §3.7.5): o `llama-server` com o GGUF Q8_0 e o cliente
de amostragem, igual para a base e para o ΦGen.

- **Servidor.** `-c 40960 -np 4 --no-kv-unified`, ou seja, 10.240 fichas por slot. Mais KV
  q8_0 com flash attention, `-lm none` (ver `phifm.rag.llm`) e uma porta livre pedida ao
  sistema (a 8080 é do assistente).
- **Amostragem.** T=0,6 e top-p 0,95. Todos os outros amostradores ficam explicitamente
  DESLIGADOS (top-k, min-p, typical, penalidades, DRY, XTC, mirostat), porque os padrões do
  servidor não são neutros. A semente da amostra j é `sha256('bench:id:j')`, a mesma nos
  dois braços. Até 4.096 fichas.
- **Guarda de repetição, no cliente, por streaming.** Se uma janela de 32 fichas volta 4
  vezes em intervalos iguais (±2 fichas), ou 8 vezes de qualquer jeito, a geração é
  cortada: é laço. O corte conta como TRUNCADO: sem a guarda, ela iria até o teto.
  Interpretação fixada antes de medir.
- **Retomada.** Cada amostra vira uma linha de jsonl, com o SHA-256 do texto. Rodar de novo
  pula o que já foi gerado.

O hash da configuração reúne o commit do servidor, as flags, o SHA-256 do GGUF, a GPU e o
pedido. Ele vai em toda linha. A base só vale no veredito com o MESMO hash do ΦGen.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
EXECUTAVEL = RAIZ / "ferramentas/llama.cpp/llama-server.exe"
COMMIT_DO_SERVIDOR = "b11159 (6b790a9c2)"

FICHAS_MAX = 4096
JANELA_REPETICAO = 32
VEZES_REPETICAO = 4
FOLGA_REPETICAO = 2       # fichas de diferença aceitas entre os intervalos de um laço
TETO_REPETICAO = 8        # a mesma janela 8 vezes corta, com intervalos iguais ou não


def amostragem(temperatura: float = 0.6, top_p: float = 0.95) -> dict:
    """Os parâmetros do pedido. Tudo o que não é T e top-p fica DESLIGADO de forma
    explícita (os padrões do servidor ligam top-k 40 e min-p 0,05)."""
    return {"temperature": temperatura, "top_p": top_p, "top_k": 0, "min_p": 0.0,
            "typical_p": 1.0, "repeat_penalty": 1.0, "presence_penalty": 0.0,
            "frequency_penalty": 0.0, "dry_multiplier": 0.0, "xtc_probability": 0.0,
            "mirostat": 0}


def semente(bench: str, item_id: str, j: int) -> int:
    """A semente da amostra j do item: os 32 bits de `sha256('bench:id:j')`."""
    return int(hashlib.sha256(f"{bench}:{item_id}:{j}".encode()).hexdigest()[:8], 16)


class GuardaDeRepeticao:
    """Corta o LAÇO: uma janela de `janela` fichas que volta `vezes` vezes em intervalos
    iguais (±`folga`), ou `teto` vezes de qualquer jeito.

    Na iteração 1 do dev (2026-10-24), a guarda antiga (a janela 4 vezes em qualquer ponto)
    cortou respostas legítimas: numa derivação, a expressão longa aparece derivada,
    simplificada, retomada e posta na caixa, e o corte vinha no meio da caixa final. Isso
    pesa diferente conforme o estilo do braço. O laço de verdade se repete em intervalos
    iguais, e a guarda nova o corta, no máximo umas voltas depois da antiga."""

    def __init__(self, janela: int = JANELA_REPETICAO, vezes: int = VEZES_REPETICAO,
                 folga: int = FOLGA_REPETICAO, teto: int = TETO_REPETICAO):
        self.janela, self.vezes, self.folga, self.teto = janela, vezes, folga, teto
        self.fichas: list[int] = []
        self._posicoes: dict[tuple, list[int]] = defaultdict(list)

    def alimentar(self, novas: Iterable[int]) -> bool:
        """True se, com as fichas novas, alguma janela fechou um laço."""
        for f in novas:
            self.fichas.append(f)
            if len(self.fichas) >= self.janela:
                pos = self._posicoes[tuple(self.fichas[-self.janela:])]
                pos.append(len(self.fichas))
                if len(pos) >= self.teto:
                    return True
                if len(pos) >= self.vezes:
                    ult = pos[-self.vezes:]
                    intervalos = [b - a for a, b in zip(ult, ult[1:], strict=False)]
                    if max(intervalos) - min(intervalos) <= self.folga:
                        return True
        return False


@dataclass
class Amostra:
    bench: str
    item_id: str
    j: int
    semente: int
    texto: str
    fichas: int
    parada: str            # eos | word | limit | repeticao | erro
    segundos: float

    @property
    def truncada(self) -> bool:
        return self.parada in ("limit", "repeticao")


def ler_fluxo(linhas: Iterable[bytes], guarda: GuardaDeRepeticao) -> tuple[str, int, str]:
    """Lê o `stream` do `/completion` (uma linha `data: {…}` por pedaço). Devolve (texto,
    fichas, parada). A parada é `repeticao` se a guarda cortou; senão é o `stop_type` do
    servidor."""
    pedacos: list[str] = []
    n = 0
    for bruta in linhas:
        linha = bruta.decode("utf-8").strip()
        if not linha.startswith("data:"):
            continue
        d = json.loads(linha[len("data:"):])
        if d.get("content"):
            pedacos.append(d["content"])
        novas = d.get("tokens") or []
        n += len(novas)
        if guarda.alimentar(novas):
            return "".join(pedacos), n, "repeticao"
        if d.get("stop"):
            return "".join(pedacos), max(n, int(d.get("tokens_predicted") or 0)), str(
                d.get("stop_type") or "eos")
    return "".join(pedacos), n, "erro"


def porta_livre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServidorG2:
    """O `llama-server` da medida, numa porta livre."""

    def __init__(self, gguf: Path, contexto: int = 40960, slots: int = 4,
                 executavel: Path = EXECUTAVEL, log: Path | None = None):
        self.gguf, self.contexto, self.slots = Path(gguf), contexto, slots
        self.executavel, self.log = executavel, log
        self.porta = porta_livre()
        self.url = f"http://127.0.0.1:{self.porta}"
        self._processo: subprocess.Popen | None = None
        self._arq_log = None

    def flags(self) -> list[str]:
        return ["-ngl", "99", "-c", str(self.contexto), "-np", str(self.slots),
                "--no-kv-unified", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
                "-lm", "none", "--no-webui"]

    def no_ar(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=3) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def iniciar(self, espera_s: int = 300) -> None:
        self._arq_log = open(self.log, "a", encoding="utf-8") if self.log else None  # noqa: SIM115
        self._processo = subprocess.Popen(
            [str(self.executavel), "-m", str(self.gguf), *self.flags(),
             "--host", "127.0.0.1", "--port", str(self.porta)],
            stdout=self._arq_log or subprocess.DEVNULL, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL)
        t0 = time.time()
        while time.time() - t0 < espera_s:
            if self._processo.poll() is not None:
                raise SystemExit(f"o llama-server saiu com {self._processo.returncode}")
            if self.no_ar():
                return
            time.sleep(2)
        raise SystemExit(f"o llama-server não ficou no ar em {espera_s} s")

    def parar(self) -> None:
        if self._processo is not None and self._processo.poll() is None:
            self._processo.terminate()
            try:
                self._processo.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._processo.kill()
        self._processo = None
        if self._arq_log is not None:
            self._arq_log.close()
            self._arq_log = None

    def __enter__(self):
        self.iniciar()
        return self

    def __exit__(self, *_):
        self.parar()


def hash_da_configuracao(servidor: ServidorG2, gguf_sha256: str, gpu: str,
                         pedido: dict) -> str:
    conf = {"servidor": COMMIT_DO_SERVIDOR, "flags": servidor.flags(), "gguf": gguf_sha256,
            "gpu": gpu, "pedido": pedido, "fichas_max": FICHAS_MAX,
            "guarda": [JANELA_REPETICAO, VEZES_REPETICAO, FOLGA_REPETICAO, TETO_REPETICAO,
                       "intervalos iguais"]}
    return hashlib.sha256(json.dumps(conf, sort_keys=True).encode()).hexdigest()


def chave_da_amostra(a: dict) -> tuple[str, str, int]:
    """Identidade lógica; tentativas operacionais compartilham esta chave."""
    try:
        b, i, j = a["bench"], a["item_id"], a["j"]
    except KeyError as e:
        raise ValueError("amostra sem identidade completa") from e
    if not isinstance(b, str) or not b or not isinstance(i, str) or not i:
        raise ValueError("benchmark e id da amostra devem ser textos não vazios")
    if type(j) is not int or j < 0:
        raise ValueError("índice da amostra deve ser inteiro não negativo")
    return b, i, j


def hash_da_execucao(hash_config: str, pedidos: list[tuple[str, str, int, str]],
                     paradas: list[str], pedido: dict, fichas_max: int = FICHAS_MAX,
                     artefatos: dict | None = None) -> str:
    """Identifica também prompts, seleção, k, paradas e versões dos artefatos.

    O hash do motor sozinho não detecta trocar exemplos ou selecionar outros itens no
    desenvolvimento. A ordem de envio não faz parte da identidade; o conteúdo faz.
    """
    if not pedidos:
        raise ValueError("conjunto de pedidos vazio")
    if type(fichas_max) is not int or fichas_max <= 0:
        raise ValueError("limite de fichas deve ser inteiro positivo")
    vistos = set()
    for b, i, j, prompt in pedidos:
        chave = chave_da_amostra({"bench": b, "item_id": i, "j": j})
        if chave in vistos:
            raise ValueError("pedidos duplicados para a mesma amostra")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt vazio ou inválido")
        vistos.add(chave)
    conf = {"hash_config": hash_config, "pedidos": sorted(pedidos), "paradas": paradas,
            "pedido": pedido, "fichas_max": fichas_max, "artefatos": artefatos or {},
            "guarda": [JANELA_REPETICAO, VEZES_REPETICAO]}
    return hashlib.sha256(json.dumps(conf, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def unificar_amostras(amostras: Iterable[dict]) -> list[dict]:
    """Uma linha por amostra, sem apagar o histórico nem escolher a melhor resposta.

    Uma tentativa válida substitui erros operacionais anteriores. Duas respostas válidas
    diferentes para a mesma chave indicam mistura/corrupção e interrompem a medição.
    Repetir a mesma linha ou acumular vários ERROR não aumenta o peso do item.
    """
    unicas: dict[tuple[str, str, int], dict] = {}
    campos = ("texto", "fichas", "parada", "truncada", "semente", "sha256_texto")
    for a in amostras:
        chave = chave_da_amostra(a)
        if a.get("parada") not in ("eos", "word", "limit", "repeticao", "erro"):
            raise ValueError("motivo de parada inválido na amostra")
        anterior = unicas.get(chave)
        if anterior is not None and anterior["parada"] != "erro":
            if a["parada"] == "erro":
                continue
            if any(anterior.get(c) != a.get(c) for c in campos):
                raise ValueError("respostas válidas conflitantes para a mesma amostra")
        unicas[chave] = a
    return [unicas[c] for c in sorted(unicas)]


def exigir_completude(amostras: Iterable[dict], esperadas: set[tuple[str, str, int]]) -> None:
    if not esperadas:
        raise ValueError("conjunto esperado vazio")
    obtidas = {chave_da_amostra(a) for a in amostras}
    faltantes, extras = esperadas - obtidas, obtidas - esperadas
    if faltantes or extras:
        raise ValueError(f"medida incompleta: {len(faltantes)} amostras ausentes, "
                         f"{len(extras)} amostras inesperadas; nenhum agregado publicado")


def sha256_bytes(caminho: Path) -> str:
    """Hash dos bytes reais, inclusive de GGUF e do log de tentativas."""
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def gerar_uma(url: str, prompt: str, paradas: list[str], pedido: dict, sem: int,
              fichas_max: int = FICHAS_MAX, tempo_max_s: int = 900) -> tuple[str, int, str]:
    if set(pedido) & {"prompt", "n_predict", "stop", "seed", "cache_prompt", "stream",
                      "return_tokens"}:
        raise ValueError("amostragem não pode sobrescrever identidade ou limites do pedido")
    corpo = {"prompt": prompt, "n_predict": fichas_max, "stop": paradas, "seed": sem,
             "cache_prompt": True, "stream": True, "return_tokens": True, **pedido}
    req = urllib.request.Request(url + "/completion", data=json.dumps(corpo).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=tempo_max_s) as r:
        return ler_fluxo(r, GuardaDeRepeticao())


def carregar_amostras(saida: Path, hash_config: str, hash_execucao: str) -> list[dict]:
    """Valida todo o log antes de reutilizar qualquer tentativa. Nunca o reescreve.

    Logs antigos sem a identidade completa ou com a última linha interrompida precisam
    ser preservados e investigados, não misturados silenciosamente com a nova execução.
    """
    if not saida.exists():
        return []
    amostras = []
    for n, linha in enumerate(saida.read_text(encoding="utf-8").splitlines(), 1):
        if not linha.strip():
            continue
        try:
            d = json.loads(linha)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON inválido na linha {n}; preserve o log antes de recuperar") from e
        if not isinstance(d, dict):
            raise ValueError(f"amostra inválida na linha {n}")
        if d.get("hash_config") != hash_config:
            raise ValueError(f"outra configuração na linha {n}; use uma saída nova")
        if d.get("hash_execucao") != hash_execucao:
            raise ValueError(f"outra execução ou identidade ausente na linha {n}; use uma saída nova")
        b, i, j = chave_da_amostra(d)
        if not isinstance(d.get("texto"), str) or d.get("sha256_texto") != hashlib.sha256(
                d["texto"].encode()).hexdigest():
            raise ValueError(f"hash do texto inválido na linha {n}")
        if type(d.get("semente")) is not int or d["semente"] != semente(b, i, j):
            raise ValueError(f"semente inválida na linha {n}")
        if type(d.get("fichas")) is not int or d["fichas"] < 0:
            raise ValueError(f"contagem de fichas inválida na linha {n}")
        if (type(d.get("segundos")) not in (int, float) or not math.isfinite(d["segundos"])
                or d["segundos"] < 0):
            raise ValueError(f"tempo inválido na linha {n}")
        if d.get("truncada") is not (d.get("parada") in ("limit", "repeticao")):
            raise ValueError(f"estado de truncagem inválido na linha {n}")
        amostras.append(d)
    return unificar_amostras(amostras)


def ja_gerados(saida: Path, hash_config: str, hash_execucao: str) -> set[tuple[str, str, int]]:
    return {chave_da_amostra(a) for a in carregar_amostras(saida, hash_config, hash_execucao)
            if a["parada"] != "erro"}


def gerar_lote(url: str, pedidos: list[tuple[str, str, int, str]], paradas: list[str],
               pedido: dict, saida: Path, hash_config: str, paralelos: int = 4,
               fichas_max: int = FICHAS_MAX, *, artefatos: dict | None = None) -> Iterator[Amostra]:
    """Gera (bench, item_id, j, prompt) em `paralelos` pedidos simultâneos, um por slot.
    Grava cada amostra ao terminar e pula as que o arquivo já tem."""
    if type(paralelos) is not int or paralelos <= 0:
        raise ValueError("paralelismo deve ser inteiro positivo")
    hash_execucao = hash_da_execucao(hash_config, pedidos, paradas, pedido, fichas_max, artefatos)
    feitas = carregar_amostras(saida, hash_config, hash_execucao)
    esperadas = {(p[0], p[1], p[2]) for p in pedidos}
    if any(chave_da_amostra(a) not in esperadas for a in feitas):
        raise ValueError("amostra inesperada no log da execução")
    feitos = {chave_da_amostra(a) for a in feitas if a["parada"] != "erro"}
    fila = [p for p in pedidos if (p[0], p[1], p[2]) not in feitos]
    trava = threading.Lock()

    def uma(p: tuple[str, str, int, str]) -> Amostra:
        bench, item_id, j, prompt = p
        sem = semente(bench, item_id, j)
        t0 = time.perf_counter()
        try:
            texto, n, parada = gerar_uma(url, prompt, paradas, pedido, sem, fichas_max)
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            texto, n, parada = "", 0, "erro"
        a = Amostra(bench, item_id, j, sem, texto, n, parada, time.perf_counter() - t0)
        linha = {**a.__dict__, "truncada": a.truncada, "hash_config": hash_config,
                 "hash_execucao": hash_execucao,
                 "sha256_texto": hashlib.sha256(texto.encode()).hexdigest()}
        with trava, saida.open("a", encoding="utf-8") as f:
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return a

    saida.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(paralelos) as ex:
        for fut in as_completed([ex.submit(uma, p) for p in fila]):
            yield fut.result()
