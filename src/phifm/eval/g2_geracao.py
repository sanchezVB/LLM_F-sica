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


def gerar_uma(url: str, prompt: str, paradas: list[str], pedido: dict, sem: int,
              fichas_max: int = FICHAS_MAX, tempo_max_s: int = 900) -> tuple[str, int, str]:
    corpo = {"prompt": prompt, "n_predict": fichas_max, "stop": paradas, "seed": sem,
             "cache_prompt": True, "stream": True, "return_tokens": True, **pedido}
    req = urllib.request.Request(url + "/completion", data=json.dumps(corpo).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=tempo_max_s) as r:
        return ler_fluxo(r, GuardaDeRepeticao())


# U+0085, U+2028 e U+2029 crus numa linha de jsonl: o `json.dumps(ensure_ascii=False)` os
# deixa crus, e o `str.splitlines()` parte a linha neles. Na medida da base, uma geração com
# U+0085 travou a retomada (emenda 1, 2026-10-25). O gravador escapa; o leitor separa pelo
# "\n" de verdade, e assim lê também os arquivos antigos.
_SEPARADORES = {"\u0085": "\\u0085", " ": "\\u2028", " ": "\\u2029"}


def linha_jsonl(registro: dict) -> str:
    """O registro numa linha de jsonl que nenhum leitor parte."""
    linha = json.dumps(registro, ensure_ascii=False)
    for c, e in _SEPARADORES.items():
        linha = linha.replace(c, e)
    return linha


def ler_jsonl(arquivo: Path) -> Iterator[dict]:
    """Os registros de um jsonl, separados SÓ pelo "\\n"."""
    for linha in Path(arquivo).read_text(encoding="utf-8").split("\n"):
        linha = linha.rstrip("\r")
        if linha.strip():
            yield json.loads(linha)


def ja_gerados(saida: Path) -> set[tuple[str, str, int]]:
    if not saida.exists():
        return set()
    return {(d["bench"], d["item_id"], d["j"]) for d in ler_jsonl(saida)
            if d.get("parada") != "erro"}


def gerar_lote(url: str, pedidos: list[tuple[str, str, int, str]], paradas: list[str],
               pedido: dict, saida: Path, hash_config: str, paralelos: int = 4,
               fichas_max: int = FICHAS_MAX) -> Iterator[Amostra]:
    """Gera (bench, item_id, j, prompt) em `paralelos` pedidos simultâneos, um por slot.
    Grava cada amostra ao terminar e pula as que o arquivo já tem."""
    feitos = ja_gerados(saida)
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
                 "sha256_texto": hashlib.sha256(texto.encode()).hexdigest()}
        with trava, saida.open("a", encoding="utf-8") as f:
            f.write(linha_jsonl(linha) + "\n")
        return a

    saida.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(paralelos) as ex:
        for fut in as_completed([ex.submit(uma, p) for p in fila]):
            yield fut.result()
