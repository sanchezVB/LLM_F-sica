"""O modelo de linguagem local: `llama-server` (llama.cpp) com o Qwen3-8B na RX 7600.

Sem dependência nova: HTTP pela biblioteca padrão, e o servidor é o executável oficial
do llama.cpp com backend Vulkan (`ferramentas/llama.cpp/`), que roda na GPU AMD sem ROCm.
Medido em 2026-09-24 (build b11159): 802 tokens/s lendo o prompt e 44–48 tokens/s
gerando — uma resposta de 300 palavras em ~8 s.

## O caminho de CHAT do servidor não é usado

`/v1/chat/completions` travou o `llama-server` b11159 com o Qwen3: o pedido não virava
tarefa (nada no log), a GPU ficava parada com 6,8 GB ocupados, e depois dele o servidor
não respondia nem a `/health`. Atribuí isso ao template Jinja; a seção abaixo mostra
uma causa melhor, e a atribuição fica em aberto.

O prompt é montado AQUI de qualquer forma, no formato ChatML do Qwen3, e vai por `/completion`. Isso
também fixa o modo sem raciocínio longo: o bloco `<think>` vazio é o que o próprio
template do Qwen3 insere com `enable_thinking=False`.

## ⚠️ O antivírus congela o servidor — e parece defeito do servidor

Em 2026-09-24 o `llama-server` parava de responder ~10 s depois de carregar, em GPU e em
CPU, lançado do bash, do Python ou do `Start-Process`, com log ligado ou desligado. Não
era o servidor: as 21 threads do processo estavam em `Wait: Suspended` — suspensas de
fora — e o Avast registrava os executáveis de `ferramentas/llama.cpp/` no Auto-Sandbox.
A travada "do caminho de chat" acima tem o mesmo sintoma; não foi medida de novo.

O sintoma, para reconhecer: `/health` responde nos primeiros segundos e depois nem ele;
o log para em `all slots are idle` sem erro. O remédio é do dono da máquina (exceção no
antivírus para `ferramentas/llama.cpp/`), não deste código — ver SETUP.md.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
EXECUTAVEL = RAIZ / "ferramentas/llama.cpp/llama-server.exe"
MODELO = RAIZ / "models/llm/Qwen3-8B-Q4_K_M.gguf"
PORTA = 8080
FIM_DE_TURNO = "<|im_end|>"
DICA_TRAVADO = ("Se o processo existe e nem /health responde, confira se o antivírus o "
                "suspendeu (ver o cabeçalho de phifm.rag.llm e o SETUP.md).")


def chatml(sistema: str, usuario: str, pensar: bool = False) -> str:
    """O prompt no formato de conversa do Qwen3. Sem `pensar`, o raciocínio longo fica
    DESLIGADO pelo bloco `<think>` vazio; com ele, o modelo escreve o raciocínio antes."""
    return (f"<|im_start|>system\n{sistema}{FIM_DE_TURNO}\n"
            f"<|im_start|>user\n{usuario}{FIM_DE_TURNO}\n"
            "<|im_start|>assistant\n" + ("" if pensar else "<think>\n\n</think>\n\n"))


def sem_raciocinio(texto: str) -> str:
    """O que vem depois de `</think>`. Sem o fechamento — o limite de tokens cortou o
    raciocínio no meio —, devolve vazio: meio raciocínio não é resposta."""
    if "<think>" not in texto and "</think>" not in texto:
        return texto.strip()
    _, fechou, depois = texto.partition("</think>")
    return depois.strip() if fechou else ""


class ModeloLocal:
    """Cliente do `llama-server`. Sobe o servidor se ele não estiver no ar."""

    def __init__(self, porta: int = PORTA, modelo: Path = MODELO,
                 executavel: Path = EXECUTAVEL, contexto: int = 8192):
        self.url = f"http://127.0.0.1:{porta}"
        self.modelo, self.executavel, self.contexto = modelo, executavel, contexto
        self._processo: subprocess.Popen | None = None
        self._log = None

    def no_ar(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=3) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def iniciar(self, log: Path | None = None, espera_s: int = 180) -> None:
        """Sobe o servidor na GPU (`-ngl 99`), UMA conversa por vez (`-np 1`)."""
        if self.no_ar():
            return
        for f in (self.executavel, self.modelo):
            if not f.exists():
                raise SystemExit(f"{f} não existe. Ver SETUP.md, seção do assistente.")
        # O arquivo vive enquanto o servidor escreve nele; `parar` o fecha.
        self._log = open(log, "w", encoding="utf-8") if log else None  # noqa: SIM115
        porta = self.url.rsplit(":", 1)[1]
        self._processo = subprocess.Popen(
            [str(self.executavel), "-m", str(self.modelo), "-ngl", "99",
             "-c", str(self.contexto), "-np", "1", "--host", "127.0.0.1", "--port", porta],
            stdout=self._log or subprocess.DEVNULL, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL)
        t0 = time.time()
        while time.time() - t0 < espera_s:
            if self._processo.poll() is not None:
                raise SystemExit(f"o llama-server saiu com {self._processo.returncode}"
                                 + (f"; log em {log}" if log else ""))
            if self.no_ar():
                return
            time.sleep(2)
        raise SystemExit(f"o llama-server não ficou no ar em {espera_s} s. {DICA_TRAVADO}")

    def parar(self) -> None:
        """Encerra o servidor SÓ se foi este objeto que o subiu."""
        if self._processo is not None and self._processo.poll() is None:
            self._processo.terminate()
            try:
                self._processo.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._processo.kill()
        self._processo = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def _pedido(self, sistema: str, usuario: str, max_tokens: int, temperatura: float,
                semente: int | None, pensar: bool, fluxo: bool) -> urllib.request.Request:
        corpo = {"prompt": chatml(sistema, usuario, pensar), "n_predict": max_tokens,
                 "temperature": temperatura, "stop": [FIM_DE_TURNO], "cache_prompt": True,
                 "stream": fluxo}
        if semente is not None:
            corpo["seed"] = semente
        return urllib.request.Request(self.url + "/completion",
                                      data=json.dumps(corpo).encode("utf-8"),
                                      headers={"Content-Type": "application/json"})

    def gerar(self, sistema: str, usuario: str, *, max_tokens: int = 700,
              temperatura: float = 0.2, semente: int | None = None, pensar: bool = False,
              tempo_max_s: int = 300) -> str:
        req = self._pedido(sistema, usuario, max_tokens, temperatura, semente, pensar, False)
        try:
            with urllib.request.urlopen(req, timeout=tempo_max_s) as r:
                texto = json.loads(r.read())["content"]
                return sem_raciocinio(texto) if pensar else texto.strip()
        except TimeoutError:
            raise SystemExit(f"o llama-server não respondeu em {tempo_max_s} s. "
                             f"{DICA_TRAVADO}") from None

    def gerar_em_fluxo(self, sistema: str, usuario: str, *, max_tokens: int = 700,
                       temperatura: float = 0.2, semente: int | None = None,
                       tempo_max_s: int = 300):
        """Como `gerar`, mas devolve o texto aos pedaços, enquanto o modelo escreve (o
        `stream` do `/completion`: uma linha `data: {…}` por pedaço). Sem o modo de
        raciocínio — um bloco `<think>` sairia pela metade na tela. O `tempo_max_s` vale
        por pedaço, não para a resposta inteira."""
        req = self._pedido(sistema, usuario, max_tokens, temperatura, semente, False, True)
        try:
            with urllib.request.urlopen(req, timeout=tempo_max_s) as r:
                for linha in r:
                    linha = linha.decode("utf-8").strip()
                    if not linha.startswith("data:"):
                        continue
                    d = json.loads(linha[len("data:"):])
                    if d.get("content"):
                        yield d["content"]
                    if d.get("stop"):
                        return
        except TimeoutError:
            raise SystemExit(f"o llama-server não respondeu em {tempo_max_s} s. "
                             f"{DICA_TRAVADO}") from None


class ModeloSobDemanda:
    """O `ModeloLocal` que só ocupa a GPU enquanto é usado — o da página local.

    O servidor sobe na primeira chamada de `gerar` (~1–2 min com o HD frio) e é derrubado
    depois de `ocioso_s` segundos sem chamada. Numa página que fica aberta o dia todo, o
    modelo parado não segura os ~6 GB da placa. Um servidor que JÁ estava no ar (o
    `perguntar.py` de outro terminal) é usado e nunca derrubado — `ModeloLocal.parar` só
    encerra o que ele mesmo subiu.

    Uma chamada por vez (a trava): o servidor roda com `-np 1`, e o vigia não pode
    derrubá-lo no meio de uma resposta.
    """

    def __init__(self, modelo: ModeloLocal, log: Path | None = None, ocioso_s: float = 600,
                 intervalo_s: float = 15):
        import threading

        self.modelo, self.log, self.ocioso_s = modelo, log, ocioso_s
        self._trava = threading.Lock()
        self._ultimo = time.monotonic()
        self._fim = threading.Event()
        self._vigia = threading.Thread(target=self._vigiar, args=(intervalo_s,), daemon=True)
        self._vigia.start()

    def no_ar(self) -> bool:
        return self.modelo.no_ar()

    def gerar(self, sistema: str, usuario: str, **kw) -> str:
        with self._trava:
            if not self.modelo.no_ar():
                self.modelo.iniciar(log=self.log)
            try:
                return self.modelo.gerar(sistema, usuario, **kw)
            finally:
                self._ultimo = time.monotonic()

    def gerar_em_fluxo(self, sistema: str, usuario: str, **kw):
        """⚠️ Segura a trava até o gerador acabar: quem o consome tem de levá-lo até o
        fim ou chamar `.close()` — a página faz isso quando o navegador desconecta."""
        with self._trava:
            if not self.modelo.no_ar():
                self.modelo.iniciar(log=self.log)
            try:
                yield from self.modelo.gerar_em_fluxo(sistema, usuario, **kw)
            finally:
                self._ultimo = time.monotonic()

    def _vigiar(self, intervalo_s: float) -> None:
        while not self._fim.wait(intervalo_s):
            with self._trava:
                if time.monotonic() - self._ultimo > self.ocioso_s:
                    self.modelo.parar()

    def parar(self) -> None:
        self._fim.set()
        with self._trava:
            self.modelo.parar()
