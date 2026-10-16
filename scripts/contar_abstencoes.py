#!/usr/bin/env python3
"""O assistente diz "não sei" quando as fontes não trazem a resposta? Uma contagem nas
respostas que já existem — sem GPU.

Na medida do corpo (DOC-13 §9.5) o assistente de hoje acertou 5 de 150: a resposta está no
corpo do artigo e ele só lê resumos. O que ele faz nas outras 145 é o que importa aqui —
recusar é o comportamento pedido; afirmar outra coisa é o defeito. O juiz não serve para
essa divisão (confundiu 7 de 9 abstenções com erro, I3c), então quem classifica é o Claude,
lendo cada resposta, com as classes escritas ANTES de ler:

- `absteve`  — diz que as fontes não trazem o que foi perguntado e NÃO dá valor, nome nem
  direção como resposta. Pode descrever o que as fontes trazem.
- `afirmou`  — dá um valor, um nome ou uma direção como resposta ao que foi perguntado,
  sem dizer que as fontes não o trazem.
- `ressalva` — diz que as fontes não trazem a resposta de forma direta e, MESMO ASSIM,
  oferece um valor, nome ou direção como resposta (tirado de outra grandeza ou inferido).

    P = .venv-treino\\Scripts\\python.exe scripts\\contar_abstencoes.py
    P --exportar                 # as respostas de A′, sem as marcas [n] e sem o veredicto
    P --resumir <classes.json>   # a contagem, cruzada com o juiz e com P entre as fontes

As respostas e as classes ficam em `data/processed/assistente/`, fora do git (DOC-11 §8.1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8  # noqa: E402

utf8()

DIR = RAIZ / "data/processed/assistente"
AVALIACAO = RAIZ / "data/processed/avaliacao"
PARA_CLASSIFICAR = DIR / "corpo_abstencao_para_classificar.json"
CLASSES = ("absteve", "afirmou", "ressalva")
BRACO = "A2"


def _ler_jsonl(arq: Path) -> list[dict]:
    return [json.loads(x) for x in arq.read_text(encoding="utf-8").splitlines() if x.strip()]


def _respostas() -> tuple[list[dict], dict[str, dict]]:
    itens = json.loads((DIR / "itens_corpo.json").read_text(encoding="utf-8"))
    arqs = sorted(DIR.glob("bracos_corpo_*.jsonl"))
    if len(arqs) != 1:
        raise SystemExit(f"esperava um cache dos braços do corpo, achei {len(arqs)}.")
    return itens, {r["arxiv_id"]: r for r in _ler_jsonl(arqs[0]) if r["braco"] == BRACO}


def exportar() -> None:
    from phifm.rag.assistente import sem_citacoes

    itens, de = _respostas()
    dados = [{"id": n, "pergunta": it["pergunta"],
              "resposta": sem_citacoes(de[it["arxiv_id"]]["texto"]).strip()}
             for n, it in enumerate(itens)]
    PARA_CLASSIFICAR.write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(dados)} respostas → {PARA_CLASSIFICAR}")


def resumir(arq: Path) -> None:
    from phifm.eval import assistente_bracos as b
    from phifm.rag.assistente import _ADMITE_FALTA

    itens, de = _respostas()
    bruto = arq.read_text(encoding="utf-8")
    lido = json.loads(bruto)
    classes = {int(k): v for k, v in lido["classes"].items()}
    if sorted(classes) != list(range(len(itens))) or set(classes.values()) - set(CLASSES):
        raise SystemExit(f"as classes não cobrem as {len(itens)} respostas, ou saem de {CLASSES}.")
    sem_o_pedido = set(lido.get("sem_o_pedido", []))
    if any(classes[n] != "afirmou" for n in sem_o_pedido):
        raise SystemExit("`sem_o_pedido` só subdivide as respostas `afirmou`.")
    julg = b.ler_julgamentos(next(iter(sorted(DIR.glob("juiz_eec4*.jsonl")))))
    linhas = []
    for n, it in enumerate(itens):
        r = de[it["arxiv_id"]]
        v = julg[b.chave_do_julgamento(it["pergunta"], it["gabarito"], r["texto"])]["veredicto"]
        linhas.append({"classe": classes[n], "juiz": v, "p_veio": r["posicao_p"] is not None,
                       "admite_falta": bool(_ADMITE_FALTA.search(r["texto"])),
                       "sem_citacao": not r["citadas"]})
    nao_certas = [x for x in linhas if x["juiz"] != "certo"]

    def fatia(xs: list[dict]) -> dict:
        c = Counter(x["classe"] for x in xs)
        return {"n": len(xs), **{k: c.get(k, 0) for k in CLASSES},
                "fracao_que_afirma": round((c["afirmou"] + c["ressalva"]) / len(xs), 4)
                if xs else None}

    saida = {
        "o_que_e": "o que o assistente de hoje faz quando as fontes não trazem a resposta — "
                   "as respostas de A′ nas 150 perguntas do corpo (DOC-13 §9.5)",
        "classes": {"absteve": "diz que as fontes não trazem e não dá resposta",
                    "afirmou": "dá uma resposta, sem ressalva",
                    "ressalva": "diz que as fontes não trazem de forma direta e dá uma resposta "
                                "mesmo assim"},
        "classificador": "claude-opus-5-5, lendo as 150 respostas sem o veredicto do juiz e "
                         "sem o gabarito; classes escritas antes de ler",
        "ressalvas": ["sem revisão humana", "uma leitura só, sem segunda passada às cegas",
                      "perguntas de DESENVOLVIMENTO: o número não é uma medida da regra"],
        "decisoes_tomadas_ao_ler": {
            "resposta_mista": "abstém-se numa parte da pergunta e dá valor na outra → `ressalva`",
            "valor_de_outra_grandeza": "cita o valor de uma grandeza vizinha como contexto e "
                                       "conclui que não dá para responder → `absteve`",
            "sem_o_pedido": "não admite a falta e também não dá o que foi pedido (repete a "
                            "pergunta, generaliza, ou responde outra coisa) → `afirmou`, contada "
                            "à parte. Subdivisão criada DEPOIS de ler: as três classes não a "
                            "previam, e quem lê uma resposta dessas não é avisado de nada",
            "afirmou_das_quais_sem_o_pedido": len(sem_o_pedido)},
        "sha256_classes": hashlib.sha256(bruto.encode("utf-8")).hexdigest(),
        "todas": fatia(linhas),
        "as_que_o_juiz_nao_deu_como_certas": fatia(nao_certas),
        "…com_o_resumo_de_P_entre_as_fontes": fatia([x for x in nao_certas if x["p_veio"]]),
        "…sem_P_entre_as_fontes": fatia([x for x in nao_certas if not x["p_veio"]]),
        "classe_x_juiz": {c: dict(Counter(x["juiz"] for x in linhas if x["classe"] == c))
                          for c in CLASSES},
        "o_portao_ve_a_abstencao": {
            "o_que_e": "a expressão `_ADMITE_FALTA` do portão de citações casa com a resposta",
            **{c: {"casa": sum(1 for x in linhas if x["classe"] == c and x["admite_falta"]),
                   "n": sum(1 for x in linhas if x["classe"] == c)} for c in CLASSES}},
        "respostas_sem_citacao_nenhuma": {
            c: sum(1 for x in linhas if x["classe"] == c and x["sem_citacao"]) for c in CLASSES}}
    destino = AVALIACAO / "assistente_corpo_abstencao.json"
    destino.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(saida, ensure_ascii=False, indent=2))
    print(f"→ {destino.relative_to(RAIZ)}")


def numeros_sem_fonte(resposta: str, pergunta: str, fontes: str) -> list[str]:
    """Os números da resposta que não estão nem nas fontes nem na pergunta. É o que um
    portão mecânico veria: um valor afirmado que nenhuma fonte do prompt traz."""
    from perguntas_corpo import numeros

    return sorted(numeros(resposta) - numeros(pergunta) - numeros(fontes))


def portao_de_numeros(arq: Path, spine: Path) -> None:
    """Um portão que marcasse número sem fonte pegaria as respostas afirmadas? Só conta —
    não muda o assistente. O que ele NÃO vê está na saída: resposta afirmada sem número."""
    import polars as pl

    from perguntas_corpo import numeros
    from phifm.rag.assistente import Fonte, bloco_de_fontes, sem_citacoes

    itens, de = _respostas()
    lido = json.loads(arq.read_text(encoding="utf-8"))
    classes = {int(k): v for k, v in lido["classes"].items()}
    ids = sorted({a for r in de.values() for a in r["fontes"]})
    meta = {r["arxiv_id"]: r for r in pl.scan_parquet(spine)
            .filter(pl.col("arxiv_id").is_in(ids))
            .select("arxiv_id", "title", "abstract", "year").collect().iter_rows(named=True)}
    linhas = []
    for n, it in enumerate(itens):
        r = de[it["arxiv_id"]]
        fontes = [Fonte(i, a, " ".join((meta[a]["title"] or "").split()), meta[a]["year"],
                        " ".join((meta[a]["abstract"] or "").split()), 0.0)
                  for i, a in enumerate(r["fontes"], start=1)]
        resposta = sem_citacoes(r["texto"])
        soltos = numeros_sem_fonte(resposta, it["pergunta"], bloco_de_fontes(fontes))
        linhas.append({"classe": classes[n], "soltos": soltos,
                       "tem_numero_novo": bool(numeros(resposta) - numeros(it["pergunta"]))})
    saida = {"o_que_e": "números da resposta que não estão em nenhuma fonte do prompt nem na "
                        "pergunta — o que um portão mecânico marcaria",
             "por_classe": {
                 c: {"n": sum(1 for x in linhas if x["classe"] == c),
                     "com_numero_sem_fonte": sum(1 for x in linhas
                                                 if x["classe"] == c and x["soltos"]),
                     "com_numero_que_nao_veio_da_pergunta": sum(
                         1 for x in linhas if x["classe"] == c and x["tem_numero_novo"])}
                 for c in CLASSES}}
    destino = AVALIACAO / "assistente_corpo_abstencao.json"
    agregado = json.loads(destino.read_text(encoding="utf-8"))
    agregado["portao_de_numeros"] = saida
    destino.write_text(json.dumps(agregado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(saida, ensure_ascii=False, indent=2))
    for n, x in enumerate(linhas):
        if x["soltos"]:
            print(f"   {n:3d} {x['classe']:8s} {x['soltos']}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exportar", action="store_true")
    p.add_argument("--resumir", type=Path)
    p.add_argument("--numeros", type=Path, help="o portão de números, sobre as mesmas classes")
    p.add_argument("--spine", type=Path, default=RAIZ / "data/processed/spine.parquet")
    a = p.parse_args()
    if a.exportar:
        exportar()
    elif a.resumir:
        resumir(a.resumir)
    elif a.numeros:
        sys.path.insert(0, str(RAIZ / "scripts"))
        portao_de_numeros(a.numeros, a.spine)
    else:
        p.error("--exportar, --resumir ou --numeros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
