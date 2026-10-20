#!/usr/bin/env python3
"""A precisão de citação, medida (DOC-13 §6.3; o critério G2.4): a fonte citada sustenta a
frase que a cita?

    P = .venv-treino\\Scripts\\python.exe scripts\\medir_citacoes.py
    P --pares                          # os pares (frase, fonte citada) das respostas de A5
    P --exportar-validacao --rodada 1  # 150 pares, sem nada além do texto, para o Claude
    P --selar VEREDICTOS --rodada 1    # o hash dos veredictos do Claude — ao git ANTES do juiz
    P --julgar --rodada 1              # o juiz: primeiro a amostra (guarda), depois todos
    P --apurar --rodada 1              # a guarda, a precisão com IC e a decisão

As respostas são as de A5, o sistema de hoje (§9.9), nos 650 itens do teste — já geradas:
nada é gerado de novo. Decide o primário (500); o pós-corte é relatado.

`--julgar` só roda com a regra aceita (`REGRA_6_3_ACEITA` em `phifm.eval.citacoes`, aceita
em 2026-10-20) e com os veredictos do Claude da rodada selados.
Pares, amostra e veredictos ficam em `data/processed/assistente/citacoes/`, fora do git
(são do conjunto de teste); vão ao git o manifesto dos pares, os selos e o agregado.

⚠️ `--julgar` segura a trava de rodada pesada e sobe o llama-server.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.core.console import utf8  # noqa: E402

utf8()

import medir_passagens as mp  # noqa: E402
from phifm.eval import citacoes as ct  # noqa: E402

PASTA = mp.DIR / "citacoes"
PARES = PASTA / "pares.jsonl"


def _ler(arq: Path) -> list[dict]:
    if not arq.exists():
        return []
    return [json.loads(x) for x in arq.read_text(encoding="utf-8").splitlines() if x.strip()]


def _assinatura() -> str:
    from phifm.rag.llm import MODELO

    return mp._sha(f"{MODELO.name}\n{ct.SISTEMA_SUSTENTA}")


def _cache_do_juiz(pasta: Path, assinatura: str) -> Path:
    return pasta / f"juiz_{assinatura[:12]}.jsonl"


# ── os pares ──────────────────────────────────────────────────────────────────
def montar_pares(itens: list[dict], respostas: list, fontes_de) -> list[dict]:
    """Os pares de cada resposta, com a pergunta e a fonte como o modelo a leu."""
    por_id = {(i["estrato"], i["arxiv_id"]): i for i in itens}
    saida = []
    for r in respostas:
        it = por_id[(r.estrato, r.arxiv_id)]
        fontes = fontes_de(r.fontes) if r.fontes else []
        texto = {f.arxiv_id: ct.bloco_da_fonte(f.titulo, f.ano, f.arxiv_id, f.resumo)
                 for f in fontes}
        for p in ct.pares(r.texto, list(r.fontes)):
            fonte = texto[p["arxiv_id"]]
            saida.append({"estrato": r.estrato, "item": r.arxiv_id, **p,
                          "pergunta": it["pergunta"], "fonte": fonte,
                          "e_o_artigo_da_pergunta": p["arxiv_id"] == r.arxiv_id,
                          "chave": ct.chave(it["pergunta"], p["frase"], fonte)})
    return saida


def pares_fase(a) -> None:
    import medir_busca_lexical as mbl
    from phifm.eval import assistente_bracos as b

    itens, _, _, cache = mbl._carregar()
    r5 = [r for r in b.ler_respostas(cache) if r.braco == b.BRACO_BUSCA_LEXICA]
    if len(r5) < len(itens):
        raise SystemExit(f"A5 tem {len(r5)} de {len(itens)} respostas (medir_busca_lexical.py).")
    fontes_de = mp._refazer_fontes({(r.estrato, r.arxiv_id): SimpleNamespace(fontes=r.fontes)
                                    for r in r5}, a.indice, a.spine)
    ps = montar_pares(itens, r5, fontes_de)
    PASTA.mkdir(parents=True, exist_ok=True)
    bruto = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in ps)
    PARES.write_text(bruto, encoding="utf-8")
    man = {"regra": "DOC-13 §6.3", "o_que_e": "os pares (frase, fonte citada) das respostas "
           "de A5 no teste; o conteúdo fica fora do git", "sha256_pares": mp._sha(bruto),
           "respostas": len(r5),
           "respostas_com_citacao": len({(p["estrato"], p["item"]) for p in ps}),
           "pares": len(ps),
           "por_estrato": {e: sum(p["estrato"] == e for p in ps) for e in ("primario", "pos_corte")},
           "citam_o_artigo_da_pergunta": sum(p["e_o_artigo_da_pergunta"] for p in ps)}
    (mp.AVALIACAO / "citacoes_pares.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(man, ensure_ascii=False, indent=1))


def _pares_conferidos(pasta: Path, avaliacao: Path) -> list[dict]:
    arq = pasta / "pares.jsonl"
    man = json.loads((avaliacao / "citacoes_pares.json").read_text(encoding="utf-8"))
    if mp._sha(arq.read_text(encoding="utf-8")) != man["sha256_pares"]:
        raise SystemExit("os pares no disco não são os do manifesto versionado.")
    return _ler(arq)


def _primario(ps: list[dict]) -> list[dict]:
    return [p for p in ps if p["estrato"] == "primario"]


# ── a amostra do Claude e o selo ──────────────────────────────────────────────
def exportar_validacao(rodada: int, pasta: Path, avaliacao: Path) -> Path:
    """Os 150 pares da rodada, só com o que o juiz lê, numerados — sem item, sem posição."""
    prim = _primario(_pares_conferidos(pasta, avaliacao))
    amostra = [{"indice": k, "pergunta": prim[j]["pergunta"], "frase": prim[j]["frase"],
                "fonte": prim[j]["fonte"]}
               for k, j in enumerate(ct.amostra_validacao(len(prim), rodada))]
    destino = pasta / f"validacao_r{rodada}.json"
    destino.write_text(json.dumps(amostra, ensure_ascii=False, indent=1), encoding="utf-8")
    return destino


def selar(veredictos: Path, rodada: int, pasta: Path, avaliacao: Path) -> dict:
    amostra = pasta / f"validacao_r{rodada}.json"
    lidos = json.loads(veredictos.read_text(encoding="utf-8"))
    por_indice = {int(v["indice"]): v["veredicto"] for v in lidos}
    if sorted(por_indice) != list(range(ct.N_VALIDACAO)):
        raise SystemExit(f"os veredictos não cobrem exatamente os {ct.N_VALIDACAO} pares.")
    if any(v not in ct.VEREDICTOS for v in por_indice.values()):
        raise SystemExit(f"veredicto fora de {ct.VEREDICTOS}.")
    selo_arq = avaliacao / f"citacoes_selo_r{rodada}.json"
    if selo_arq.exists():
        raise SystemExit(f"a rodada {rodada} já foi selada: veredicto não muda depois de selado.")
    copia = pasta / f"veredictos_claude_r{rodada}.json"
    copia.write_text(veredictos.read_text(encoding="utf-8"), encoding="utf-8")
    selo = {"regra": "DOC-13 §6.3", "rodada": rodada,
            "sha256_amostra": mp._sha(amostra.read_text(encoding="utf-8")),
            "sha256_veredictos": mp._sha(copia.read_text(encoding="utf-8")),
            "n": ct.N_VALIDACAO, "julgador": "Claude, às cegas, antes de o juiz rodar",
            "selado_em": time.strftime("%Y-%m-%dT%H:%M:%S")}
    selo_arq.write_text(json.dumps(selo, ensure_ascii=False, indent=2), encoding="utf-8")
    return selo


def _selados(rodada: int, pasta: Path, avaliacao: Path) -> dict[int, str]:
    selo_arq = avaliacao / f"citacoes_selo_r{rodada}.json"
    if not selo_arq.exists():
        raise SystemExit(f"a rodada {rodada} não foi selada (--selar): o juiz não roda na "
                         "amostra antes dos veredictos do Claude estarem no git.")
    selo = json.loads(selo_arq.read_text(encoding="utf-8"))
    amostra, copia = pasta / f"validacao_r{rodada}.json", pasta / f"veredictos_claude_r{rodada}.json"
    if (mp._sha(amostra.read_text(encoding="utf-8")), mp._sha(copia.read_text(encoding="utf-8"))) \
            != (selo["sha256_amostra"], selo["sha256_veredictos"]):
        raise SystemExit("a amostra ou os veredictos não são os selados.")
    return {int(v["indice"]): v["veredicto"] for v in json.loads(copia.read_text(encoding="utf-8"))}


# ── o juiz ────────────────────────────────────────────────────────────────────
def julgar_pares(modelo, ps: list[dict], cache: Path, passo=None) -> dict[str, dict]:
    """Cada par distinto uma vez (pela chave), retomável."""
    feitos = {d["chave"]: d for d in _ler(cache)}
    with cache.open("a", encoding="utf-8") as f:
        for p in ps:
            if p["chave"] in feitos:
                continue
            t0 = time.perf_counter()
            v, motivo = ct.julgar(modelo, p["pergunta"], p["frase"], p["fonte"])
            d = {"chave": p["chave"], "veredicto": v, "motivo": motivo,
                 "segundos": round(time.perf_counter() - t0, 2)}
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            f.flush()
            feitos[p["chave"]] = d
            if passo:
                passo()
    return feitos


def julgar_fase(rodada: int) -> None:
    import medir_busca_lexical as mbl

    if not ct.REGRA_6_3_ACEITA:
        raise SystemExit("a §6.3 do DOC-13 ainda é PROPOSTA: o juiz não roda antes do aceite "
                         "do dono (REGRA_6_3_ACEITA em phifm.eval.citacoes).")
    claude = _selados(rodada, PASTA, mp.AVALIACAO)
    ps = _pares_conferidos(PASTA, mp.AVALIACAO)
    prim = _primario(ps)
    amostra = [prim[j] for j in ct.amostra_validacao(len(prim), rodada)]
    cache = _cache_do_juiz(PASTA, _assinatura())
    with mp.trava("medir_passagens"):
        modelo = mbl._modelo()
        try:
            julg = julgar_pares(modelo, amostra, cache)
            v = ct.validade([claude[k] for k in range(len(amostra))],
                            [julg[p["chave"]]["veredicto"] for p in amostra])
            print(json.dumps(v, ensure_ascii=False, indent=1), flush=True)
            if not v["passa"]:
                print("⚠️ o juiz não passou a guarda: não julgo o resto. Conserto do prompt e "
                      f"rodada {rodada + 1}, com a amostra seguinte (§6.3).")
                return
            passo = mbl._progresso("citações", len(ps), len(julg))
            julgar_pares(modelo, ps, cache, passo)
        finally:
            modelo.parar()
    print(f"juiz → {cache}")


# ── a apuração ────────────────────────────────────────────────────────────────
def apurar(rodada: int, pasta: Path, avaliacao: Path, assinatura: str) -> dict:
    claude = _selados(rodada, pasta, avaliacao)
    ps = _pares_conferidos(pasta, avaliacao)
    julg = {d["chave"]: d["veredicto"] for d in _ler(_cache_do_juiz(pasta, assinatura))}
    faltam = [p for p in ps if p["chave"] not in julg]
    if faltam:
        raise SystemExit(f"{len(faltam)} de {len(ps)} pares sem veredicto do juiz (--julgar).")
    prim = _primario(ps)
    amostra = [prim[j] for j in ct.amostra_validacao(len(prim), rodada)]
    guarda = ct.validade([claude[k] for k in range(len(amostra))],
                         [julg[p["chave"]] for p in amostra])

    def por_resposta(lista: list[dict]) -> list[list[str]]:
        g: dict[tuple, list[str]] = {}
        for p in lista:
            g.setdefault((p["estrato"], p["item"]), []).append(julg[p["chave"]])
        return list(g.values())

    def leitura(lista: list[dict]) -> dict:
        ponto, ic = ct.precisao(por_resposta(lista))
        vs = [julg[p["chave"]] for p in lista]
        return {"pares": len(lista), "respostas": len(por_resposta(lista)), "precisao": ponto,
                "ic95": ic, "parcial": vs.count("parcial"),
                "nao_sustenta": vs.count("nao_sustenta"), "ilegivel": vs.count("ilegivel")}

    def frase_sustentada(lista: list[dict]) -> float:
        g: dict[tuple, bool] = {}
        for p in lista:
            k = (p["estrato"], p["item"], p["frase_n"])
            g[k] = g.get(k, False) or julg[p["chave"]] == "sustenta"
        return round(sum(g.values()) / len(g), 4) if g else 0.0

    primario = leitura(prim)
    saida = {
        "regra": "DOC-13 §6.3", "regra_aceita_pelo_dono": ct.REGRA_6_3_ACEITA, "rodada": rodada,
        "assinatura_do_juiz": assinatura[:12], "meta": ct.META,
        "guarda (o juiz contra o Claude)": guarda,
        "primario (decide)": primario,
        "decisao": ct.decidir(primario["ic95"], guarda["passa"]),
        "relatado": {
            "pos_corte": leitura([p for p in ps if p["estrato"] == "pos_corte"]),
            "citando_o_artigo_da_pergunta": leitura([p for p in prim
                                                     if p["e_o_artigo_da_pergunta"]]),
            "citando_outro_artigo": leitura([p for p in prim
                                             if not p["e_o_artigo_da_pergunta"]]),
            "frases_com_ao_menos_uma_citacao_que_sustenta": frase_sustentada(prim),
            "precisao_contando_parcial": round(sum(
                julg[p["chave"]] in ("sustenta", "parcial") for p in prim) / len(prim), 4)}}
    (avaliacao / "citacoes.json").write_text(json.dumps(saida, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    return saida


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pares", action="store_true")
    g.add_argument("--exportar-validacao", action="store_true")
    g.add_argument("--selar", type=Path, metavar="VEREDICTOS")
    g.add_argument("--julgar", action="store_true")
    g.add_argument("--apurar", action="store_true")
    p.add_argument("--rodada", type=int, default=1)
    p.add_argument("--indice", type=Path, default=RAIZ / "data/processed/indice_busca")
    p.add_argument("--spine", type=Path, default=RAIZ / "data/processed/spine.parquet")
    a = p.parse_args()
    if a.pares:
        pares_fase(a)
    elif a.exportar_validacao:
        print(f"→ {exportar_validacao(a.rodada, PASTA, mp.AVALIACAO)}")
    elif a.selar:
        print(json.dumps(selar(a.selar, a.rodada, PASTA, mp.AVALIACAO), indent=1))
        print("→ commit do selo ANTES de --julgar")
    elif a.julgar:
        julgar_fase(a.rodada)
    else:
        r = apurar(a.rodada, PASTA, mp.AVALIACAO, _assinatura())
        print(json.dumps({k: r[k] for k in ("guarda (o juiz contra o Claude)",
                                            "primario (decide)", "decisao")},
                         ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
