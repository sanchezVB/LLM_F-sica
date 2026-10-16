"""O juiz independente (DOC-13 §9.6): outro modelo corrige as mesmas respostas.

O juiz da regra é o Qwen3-8B — o modelo que escreve as respostas —, conferido só pelo
Claude, que também escreveu as perguntas. Aqui um modelo maior e de outra família julga os
mesmos textos, com o MESMO prompt, a mesma entrada e o mesmo leitor de veredicto
(`assistente_bracos.julgar`): muda o modelo, e nada mais.

A regra dele foi escrita antes de o teste rodar:

- **vale** se o κ de `certo` contra o resto, dele contra os veredictos que o Claude deu às
  cegas e selou, for ≥ `KAPPA_MINIMO`; o prompt não é ajustado para ele;
- **não decide**: a decisão é a do juiz da regra. Cada eixo que depende de juiz é refeito
  com ele; mesmo desfecho → robusto; desfecho diferente → FRÁGIL, e a adoção volta ao dono.

O pacote que sobe para o Colab tem só o que o juiz lê — pergunta, gabarito e a resposta sem
as marcas [n] —, sem braço e sem artigo: lá ninguém sabe de qual sistema veio cada texto.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

from phifm.eval import assistente_bracos as b
from phifm.rag.assistente import sem_citacoes

# O modelo, fixado antes de o teste rodar (2026-10-13). Apache-2.0, sem cadastro.
MODELO_REPO = "bartowski/mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF"
MODELO_ARQUIVO = "mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf"
MODELO_REVISAO = "b3592d09be469bedab6be51f577824b3b6b76ba1"
MODELO_BYTES = 14_333_915_264
MODELO_SHA256 = "80f5bda68f156f12650ca03a0a2dbfae06a215ac41caa773b8631a479f82415e"
LLAMA_CPP = "b11159"              # o mesmo build do `llama-server` local
# Os binários oficiais desse build para Linux com CUDA 12 (o servidor e as bibliotecas do
# CUDA), com o sha256 que a release declara: no Colab nada é compilado.
LLAMA_BINARIOS = {
    "llama-b11159-bin-ubuntu-cuda-12.8-x64.tar.gz":
        "5840b0f87ec10400d7eea2f36138a45ddd9bdd6a794eb81d20a7f150d09c66bd",
    "cudart-llama-b11159-bin-ubuntu-cuda-12.8-x64.tar.gz":
        "f79ae034d707096f825be4b512366253950d3fa4895921a8c2577d4271ff8a18",
}
CONTEXTO = 8192


def chave(pergunta: str, gabarito: str, resposta: str) -> str:
    """A identidade de um texto julgado: a pergunta, o gabarito e a resposta como o juiz a
    lê — sem as marcas [n] e sem espaço nas pontas."""
    limpo = sem_citacoes(resposta).strip()
    return hashlib.sha256(f"{pergunta}\n{gabarito}\n{limpo}".encode()).hexdigest()[:24]


def empacotar(grupos: dict[str, list[tuple[str, str, str]]]) -> tuple[list[dict], dict]:
    """`(pacote, contagem)`: cada texto distinto uma vez, na ordem em que aparece. Um
    grupo é uma lista de `(pergunta, gabarito, resposta)`."""
    pacote, vistos, contagem = [], set(), {}
    for nome, textos in grupos.items():
        novos = 0
        for pergunta, gabarito, resposta in textos:
            k = chave(pergunta, gabarito, resposta)
            if k not in vistos:
                vistos.add(k)
                novos += 1
                pacote.append({"chave": k, "pergunta": pergunta, "gabarito": gabarito,
                               "resposta": sem_citacoes(resposta).strip()})
        contagem[nome] = {"textos": len(textos), "novos_no_pacote": novos}
    return pacote, contagem


class ModeloDeConversa:
    """Cliente do `llama-server` pelo caminho de conversa (`/v1/chat/completions`): quem
    monta o prompt é o servidor, com o modelo de conversa que vem dentro do GGUF. É o que
    serve a um modelo de outra família — o `ModeloLocal` escreve o ChatML do Qwen3 à mão."""

    def __init__(self, url: str = "http://127.0.0.1:8080"):
        self.url = url

    def gerar(self, sistema: str, usuario: str, *, max_tokens: int = 700,
              temperatura: float = 0.2, semente: int | None = None,
              tempo_max_s: int = 300) -> str:
        corpo = {"messages": [{"role": "system", "content": sistema},
                              {"role": "user", "content": usuario}],
                 "max_tokens": max_tokens, "temperature": temperatura, "stream": False,
                 "cache_prompt": True}
        if semente is not None:
            corpo["seed"] = semente
        req = urllib.request.Request(self.url + "/v1/chat/completions",
                                     data=json.dumps(corpo).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=tempo_max_s) as r:
            return (json.loads(r.read())["choices"][0]["message"]["content"] or "").strip()


def ler(saida: Path) -> dict[str, dict]:
    if not saida.exists():
        return {}
    feitos = {}
    for linha in saida.read_text(encoding="utf-8").splitlines():
        if linha.strip():
            d = json.loads(linha)
            feitos[d["chave"]] = d
    return feitos


def julgar_pacote(modelo, pacote: list[dict], saida: Path, limite: int | None = None,
                  ao_julgar=None) -> dict[str, dict]:
    """O juiz da regra, com outro `modelo`, em cada texto do pacote. Retomável: o que já
    está em `saida` não é julgado de novo."""
    feitos = ler(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with saida.open("a", encoding="utf-8") as f:
        for e in pacote:
            if e["chave"] in feitos:
                continue
            if limite is not None and n >= limite:
                break
            t0 = time.perf_counter()
            v, motivo = b.julgar(modelo, e["pergunta"], e["gabarito"], e["resposta"])
            d = {"chave": e["chave"], "veredicto": v, "motivo": motivo,
                 "segundos": round(time.perf_counter() - t0, 2)}
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            f.flush()
            feitos[e["chave"]] = d
            n += 1
            if ao_julgar:
                ao_julgar(e, d)
    return feitos


def entre_juizes(da_regra: list[str], independente: list[str]) -> dict:
    """A concordância dos dois juízes nos mesmos textos — relatada, não decide."""
    rotulos = (*b.VEREDICTOS, "ilegivel")
    bin_r = ["certo" if v == "certo" else "outro" for v in da_regra]
    bin_i = ["certo" if v == "certo" else "outro" for v in independente]
    pares = list(zip(da_regra, independente, strict=True))
    return {"n": len(pares), "kappa_certo": round(b.kappa(bin_r, bin_i), 4),
            "kappa_4_categorias": round(b.kappa(da_regra, independente), 4),
            "concordancia_certo": round(
                sum(x == y for x, y in zip(bin_r, bin_i, strict=True)) / len(pares), 4),
            "so_a_regra_deu_certo": sum(1 for x, y in pares if x == "certo" != y),
            "so_o_independente_deu_certo": sum(1 for x, y in pares if y == "certo" != x),
            "matriz_regra_x_independente": {
                x: {y: sum(1 for p in pares if p == (x, y)) for y in rotulos} for x in rotulos}}


def conferir(valido: bool, da_regra: dict[str, bool], independente: dict[str, bool]) -> dict:
    """A leitura da conferência. `da_regra` e `independente` são `{eixo: passa}` para os
    eixos que dependem de juiz."""
    if not valido:
        return {"leitura": "SEM VALOR (o juiz independente não passou na validade)",
                "eixos": None}
    eixos = {e: {"juiz_da_regra": da_regra[e], "juiz_independente": independente[e],
                 "mesmo_desfecho": da_regra[e] == independente[e]} for e in da_regra}
    robusto = all(x["mesmo_desfecho"] for x in eixos.values())
    return {"leitura": "robusto" if robusto else "FRÁGIL (os juízes dão desfechos diferentes)",
            "eixos": eixos}
