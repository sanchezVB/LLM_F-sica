#!/usr/bin/env python3
"""O que muda na RESPOSTA do assistente, no desenvolvimento: com ou sem as passagens do
texto completo, com a instrução de hoje ou com uma que cobra o "não sei".

Uma CÉLULA é conjunto_fontes_instrução:

- conjunto: `corpo` — as 150 perguntas cuja resposta está no corpo do artigo (DOC-13 §9.5);
  `primario` — as 150 de desenvolvimento da busca, com a resposta no RESUMO do artigo.
- fontes: `res` — as 6 fontes só com o resumo, o sistema de hoje; `pas` — as mesmas, com
  as passagens sob demanda (`phifm.rag.passagens`).
- instrução: `v1` — a de hoje; `v2` — a mesma, cobrando que a fonte diga a coisa
  ESPECÍFICA pedida (`SISTEMA_RESPOSTA_V2`).

A consulta hipotética e as 6 fontes de cada pergunta são fixas, do cache: entre duas
células do mesmo conjunto só muda o que o nome diz.

    P = .venv-treino\\Scripts\\python.exe scripts\\explorar_resposta.py
    P --cortar-primario                    # os trechos das fontes do primário (CPU, ~3 min)
    P --validar-classificador              # o classificador de abstenção contra o Claude (GPU)
    P --rodar primario_res_v1 corpo_pas_v2 # responde, julga e classifica (GPU, retomável)
    P --resumir

⚠️ Desenvolvimento. As perguntas foram olhadas, a regra das passagens e a instrução v2
foram escolhidas nelas, e o classificador de abstenção foi escrito olhando as 150 classes
contra as quais é conferido. Nada daqui é a medida: o que vale sai da regra do DOC-13 §9.6.

Tudo fica em `data/processed/assistente/passagens_dev/`, fora do git (DOC-11 §8.1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.core.console import utf8  # noqa: E402
from phifm.rag.assistente import SISTEMA_RESPOSTA, SISTEMA_RESPOSTA_V2  # noqa: E402

utf8()

DIR = RAIZ / "data/processed/assistente"
SAIDA = DIR / "passagens_dev"
TEXTOS = RAIZ / "data/processed/redpajama_fisica"
INDICE = RAIZ / "data/processed/indice_busca"
K = 9

# A v2 (em `phifm.rag.assistente`) nomeia o erro que a contagem da §9.5 achou: entregar um
# valor que está nas fontes, mas é de outra grandeza ou de outro artigo.
INSTRUCOES = {"v1": SISTEMA_RESPOSTA, "v2": SISTEMA_RESPOSTA_V2}

# O que a resposta FAZ, sem gabarito: as três classes de `contar_abstencoes.py`.
SISTEMA_ABSTENCAO = (
    "You read a question and the answer given by a research assistant that works from a set "
    "of sources. Classify what the answer DOES — not whether it is right. The question and "
    "the answer may be in Portuguese.\n"
    "- absteve: it says the sources do not contain what was asked, and gives no value, name "
    "or direction as the answer. It may still describe what the sources do contain, or "
    "mention a related quantity as context before concluding that it cannot answer.\n"
    "- afirmou: it gives a value, a name, a direction or an explanation as the answer, "
    "without saying that the sources lack it. An answer that only restates the question or "
    "speaks in generalities, without admitting that the information is missing, is also "
    "afirmou.\n"
    "- ressalva: it says the sources do not give the answer directly, or do not give part "
    "of it, and STILL offers a value, a name or a direction as the answer — for the whole "
    "question or for one part of it.\n"
    'Output only one JSON object: {"motivo": "<one short sentence>", '
    '"classe": "absteve" | "afirmou" | "ressalva"}')
CLASSES = ("absteve", "afirmou", "ressalva")


def _ler_jsonl(arq: Path) -> list[dict]:
    if not arq.exists():
        return []
    return [json.loads(x) for x in arq.read_text(encoding="utf-8").splitlines() if x.strip()]


# ── os conjuntos ─────────────────────────────────────────────────────────────────────
def conjunto(nome: str) -> tuple[list[dict], dict[str, dict]]:
    """`(itens, base)`: `base[arxiv_id]` tem a consulta hipotética e as 6 fontes de A′."""
    import explorar_passagens as e

    if nome == "corpo":
        return json.loads(e.ITENS.read_text(encoding="utf-8")), e._bracos()
    if nome == "primario":
        base = {d["arxiv_id"]: d for d in _ler_jsonl(e.FONTES_PRIMARIO)}
        itens = e._itens_primario()
        if len(base) < len(itens):
            raise SystemExit("faltam fontes do primário: explorar_passagens.py --fontes-primario")
        return itens, base
    raise SystemExit(f"conjunto desconhecido: {nome}")


def _arq_trechos(nome: str) -> Path:
    return SAIDA / ("trechos.parquet" if nome == "corpo" else f"trechos_{nome}.parquet")


def trechos(nome: str) -> dict[str, list]:
    import polars as pl

    from phifm.retrieval.trechos import Trecho

    arq = _arq_trechos(nome)
    if not arq.exists():
        raise SystemExit(f"falta {arq.name}: rode --cortar-{nome}")
    saida: dict[str, list] = {}
    for a, secao, tipo, n, excede, texto in pl.read_parquet(arq).sort("arxiv_id", "i").select(
            "arxiv_id", "secao", "tipo", "n_tokens", "excede", "texto").iter_rows():
        saida.setdefault(a, []).append(Trecho(texto, secao, tipo, n, excede=excede))
    return saida


def cortar(nome: str) -> None:
    """Os trechos das fontes de um conjunto, pelo corte das passagens (160, sem repetição)."""
    import polars as pl
    from transformers import AutoTokenizer

    from phifm.rag.passagens import LIMITE_TRECHO, contador
    from phifm.retrieval.trechos import cortar as cortar_texto

    _, base = conjunto(nome)
    alvo = sorted({a for d in base.values() for a in d["fontes"]})
    modelo = json.loads((INDICE / "manifesto.json").read_text(encoding="utf-8"))["modelo"]
    contar = contador(AutoTokenizer.from_pretrained(modelo))
    linhas, vistos, t0 = [], set(), time.perf_counter()
    for p in sorted(TEXTOS.glob("part-*.parquet")):
        df = pl.scan_parquet(p).filter(pl.col("arxiv_id").is_in(alvo)).collect()
        for a, texto in df.iter_rows():
            if a in vistos:
                continue
            vistos.add(a)
            for i, t in enumerate(cortar_texto(texto, contar, limite=LIMITE_TRECHO,
                                               sobreposicao=0)):
                linhas.append((a, i, t.secao, t.tipo, t.n_tokens, t.excede, t.texto))
    pl.DataFrame(linhas, orient="row", schema={
        "arxiv_id": pl.String, "i": pl.Int32, "secao": pl.String, "tipo": pl.String,
        "n_tokens": pl.Int32, "excede": pl.Boolean, "texto": pl.String}
                 ).write_parquet(_arq_trechos(nome))
    print(f"{nome}: {len(alvo)} fontes distintas · {len(vistos)} com texto completo · "
          f"{len(linhas):,} trechos · {time.perf_counter() - t0:.0f} s")


def _ingles(nome: str, itens: list[dict], passo) -> dict[str, str]:
    """A pergunta em inglês de cada item, do cache; o que faltar, o modelo traduz agora."""
    arq = SAIDA / f"ingles_{nome}.json"
    if arq.exists():
        feitas = json.loads(arq.read_text(encoding="utf-8"))
    elif nome == "corpo":      # as de `explorar_passagens.py --consultas`, na ordem dos itens
        lista = json.loads((SAIDA / "consultas_ingles.json").read_text(encoding="utf-8"))
        feitas = {it["arxiv_id"]: t for it, t in zip(itens, lista, strict=True)}
    else:
        feitas = {}
    for it in itens:
        if it["arxiv_id"] not in feitas:
            feitas[it["arxiv_id"]] = passo.consulta(it["pergunta"])
            arq.write_text(json.dumps(feitas, ensure_ascii=False), encoding="utf-8")
    arq.write_text(json.dumps(feitas, ensure_ascii=False), encoding="utf-8")
    return feitas


# ── as células ───────────────────────────────────────────────────────────────────────
def partes(celula: str) -> tuple[str, bool, str]:
    m = re.fullmatch(r"(corpo|primario)_(res|pas)_(v\d)", celula)
    if not m or m.group(3) not in INSTRUCOES:
        raise SystemExit(f"célula inválida: {celula} (ex.: corpo_pas_v2)")
    return m.group(1), m.group(2) == "pas", m.group(3)


def respostas(celula: str) -> list:
    """As respostas de uma célula. Duas já existiam antes deste script, com outro nome:
    `corpo_res_v1` é o braço A′ da medida do corpo, e `corpo_pas_v1` é o A3 de
    `explorar_passagens.py --responder`."""
    import explorar_passagens as e
    from phifm.eval import assistente_bracos as b

    if celula == "corpo_res_v1":
        arq = next(iter(sorted(DIR.glob("bracos_corpo_*.jsonl"))))
        return [r for r in b.ler_respostas(arq) if r.braco == "A2"]
    if celula == "corpo_pas_v1":
        return b.ler_respostas(e._cache_dev(K))
    return b.ler_respostas(SAIDA / f"celula_{celula}.jsonl")


def responder(celula: str, modelo, limite: int | None) -> None:
    from dataclasses import asdict

    import explorar_passagens as e
    from phifm.eval import assistente_bracos as b
    from phifm.eval.assistente import K_FONTES, SEMENTE
    from phifm.rag import passagens as ps
    from phifm.rag.assistente import Assistente

    if celula in ("corpo_res_v1", "corpo_pas_v1"):
        print(f"{celula}: já existe ({len(respostas(celula))} respostas).")
        return
    nome, com_passagens, instrucao = partes(celula)
    itens, base = conjunto(nome)
    fontes = e._fontes_de_a2(base)
    feitos = {r.arxiv_id for r in respostas(celula)}
    assistente = Assistente(None, modelo, RAIZ / "data/processed/spine.parquet", k=K_FONTES,
                            semente=SEMENTE, sistema=INSTRUCOES[instrucao])
    passo = ingles = None
    if com_passagens:
        passo = ps.Passagens(modelo, trechos(nome).get, k=K, semente=SEMENTE)
        ingles = _ingles(nome, itens, passo)
    t_ini, n = time.perf_counter(), 0
    with (SAIDA / f"celula_{celula}.jsonl").open("a", encoding="utf-8") as f:
        for it in itens:
            p = it["arxiv_id"]
            if p in feitos:
                continue
            if limite is not None and n >= limite:
                break
            fs = passo.anexar(it["pergunta"], fontes[p], consulta=ingles[p]) if passo else fontes[p]
            t0 = time.perf_counter()
            r = assistente.responder_com(it["pergunta"], fs, base[p]["consulta"])
            reg = b._registro(it, celula, r, time.perf_counter() - t0)
            f.write(json.dumps(asdict(reg), ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 10 == 0:
                s = (time.perf_counter() - t_ini) / n
                print(f"   {celula}: {len(feitos) + n}/{len(itens)} · {s:.1f} s cada · faltam "
                      f"~{(len(itens) - len(feitos) - n) * s / 60:.0f} min", flush=True)
    print(f"{celula}: {len(respostas(celula))} respostas", flush=True)


# ── o classificador de abstenção ─────────────────────────────────────────────────────
def _cache_abstencao() -> Path:
    from phifm.rag.llm import MODELO

    sig = hashlib.sha256(f"{SISTEMA_ABSTENCAO}\n{MODELO.name}".encode()).hexdigest()[:12]
    return SAIDA / f"abstencao_{sig}.jsonl"


def _chave(pergunta: str, texto: str) -> str:
    from phifm.rag.assistente import sem_citacoes

    return hashlib.sha256(f"{pergunta}\n{sem_citacoes(texto)}".encode()).hexdigest()[:24]


def _ler_classe(bruto: str) -> str:
    m = re.search(r"\{.*\}", bruto, re.DOTALL)
    if m:
        try:
            c = str(json.loads(m.group(0)).get("classe", "")).strip().lower()
            if c in CLASSES:
                return c
        except json.JSONDecodeError:
            pass
    achados = re.findall(r"\b(" + "|".join(CLASSES) + r")\b", bruto.lower())
    return achados[-1] if achados else "ilegivel"


def classificar(modelo, itens: list[dict], resps: list) -> dict[str, str]:
    """A classe de cada resposta, do cache (pelo texto); o que faltar, o modelo classifica."""
    from phifm.eval.assistente import SEMENTE
    from phifm.rag.assistente import sem_citacoes

    arq = _cache_abstencao()
    feitos = {d["chave"]: d["classe"] for d in _ler_jsonl(arq)}
    por_id = {i["arxiv_id"]: i for i in itens}
    with arq.open("a", encoding="utf-8") as f:
        for r in resps:
            pergunta = por_id[r.arxiv_id]["pergunta"]
            chave = _chave(pergunta, r.texto)
            if chave in feitos:
                continue
            if modelo is None:
                raise SystemExit("há respostas sem classe: rode --rodar (GPU).")
            limpo = sem_citacoes(r.texto).strip()
            classe = "absteve" if not limpo else _ler_classe(modelo.gerar(
                SISTEMA_ABSTENCAO, f"Question: {pergunta}\nAnswer:\n{limpo}",
                max_tokens=160, temperatura=0.0, semente=SEMENTE))
            f.write(json.dumps({"chave": chave, "classe": classe}) + "\n")
            f.flush()
            feitos[chave] = classe
    return {r.arxiv_id: feitos[_chave(por_id[r.arxiv_id]["pergunta"], r.texto)] for r in resps}


def validar_classificador(modelo) -> None:
    """O classificador contra as 150 classes do Claude (`contar_abstencoes.py`)."""
    from phifm.eval import assistente_bracos as b

    itens, _ = conjunto("corpo")
    meu = json.loads((DIR / "corpo_abstencao_classes_claude.json").read_text(encoding="utf-8"))
    meu = {itens[int(k)]["arxiv_id"]: v for k, v in meu["classes"].items()}
    dele = classificar(modelo, itens, respostas("corpo_res_v1"))
    ids = [i["arxiv_id"] for i in itens]
    x, y = [meu[a] for a in ids], [dele[a] for a in ids]
    bin_x = ["absteve" if v == "absteve" else "afirma" for v in x]
    bin_y = ["absteve" if v == "absteve" else "afirma" for v in y]
    matriz = {h: dict(Counter(j for a, j in zip(x, y, strict=True) if a == h)) for h in CLASSES}
    print(json.dumps({
        "n": len(ids), "kappa_absteve_contra_o_resto": round(b.kappa(bin_x, bin_y), 4),
        "concordancia_absteve_contra_o_resto": round(
            sum(a == c for a, c in zip(bin_x, bin_y, strict=True)) / len(ids), 4),
        "kappa_3_classes": round(b.kappa(x, y), 4),
        "claude_x_classificador": matriz}, ensure_ascii=False, indent=1))


# ── rodar e resumir ──────────────────────────────────────────────────────────────────
def _subir():
    from phifm.rag.llm import ModeloLocal

    modelo = ModeloLocal()
    if not modelo.no_ar():
        print("subindo o llama-server (Qwen3-8B na GPU)...", flush=True)
        modelo.iniciar(log=RAIZ / "data/processed/llama_server.log")
    return modelo


def rodar(celulas: list[str], limite: int | None) -> None:
    import explorar_passagens as e
    from phifm.eval import assistente_bracos as b

    for c in celulas:
        partes(c)
    modelo = _subir()
    try:
        for c in celulas:
            responder(c, modelo, limite)
            itens, _ = conjunto(partes(c)[0])
            resps = respostas(c)
            b.julgar_todas(modelo, itens, resps, e._cache_do_juiz())
            classificar(modelo, itens, resps)
            print(f"{c}: julgada e classificada", flush=True)
    finally:
        modelo.parar()


def resumir() -> None:
    """Uma linha por célula: acerto pelo juiz e o que a resposta faz quando NÃO acerta."""
    import explorar_passagens as e
    from phifm.eval import assistente_bracos as b

    julg = b.ler_julgamentos(e._cache_do_juiz())
    todas = [f"{c}_{f}_{v}" for c in ("corpo", "primario") for f in ("res", "pas")
             for v in INSTRUCOES]
    print(f"{'célula':18s}   n  acerto  IC 95%            não certas: absteve · afirmou · "
          "ressalva   afirma errado/n   P nas fontes: acerto   P fora: afirma")
    linhas = {}
    for c in todas:
        resps = respostas(c)
        if not resps:
            continue
        itens, base = conjunto(partes(c)[0])
        if len(resps) < len(itens):
            continue
        por_id = {i["arxiv_id"]: i for i in itens}
        try:
            classe = classificar(None, itens, resps)
            certo = {r.arxiv_id: julg[b.chave_do_julgamento(
                por_id[r.arxiv_id]["pergunta"], por_id[r.arxiv_id]["gabarito"], r.texto)]
                ["veredicto"] == "certo" for r in resps}
        except (SystemExit, KeyError):
            print(f"{c:18s} respondida, falta julgar ou classificar")
            continue
        ids = [i["arxiv_id"] for i in itens]
        acc, ic = b.proporcao(np.array([certo[a] for a in ids], dtype=float))
        nc = Counter(classe[a] for a in ids if not certo[a])
        afirma_errado = nc["afirmou"] + nc["ressalva"]
        veio = [a for a in ids if a in base[a]["fontes"]]
        fora = [a for a in ids if a not in base[a]["fontes"]]
        afirma_fora = sum(1 for a in fora if classe[a] != "absteve" and not certo[a])
        linhas[c] = {"ids": ids, "certo": certo, "classe": classe}
        print(f"{c:18s} {len(ids):3d}  {acc:.3f}  [{ic[0]:.3f}; {ic[1]:.3f}]   "
              f"{nc['absteve']:3d} · {nc['afirmou']:3d} · {nc['ressalva']:3d}"
              f"{'':22s}{afirma_errado / len(ids):.3f}            "
              f"{np.mean([certo[a] for a in veio]):.3f} (n {len(veio)})       "
              f"{afirma_fora}/{len(fora)}")
    print("\ndiferenças pareadas de acerto (a segunda menos a primeira):")
    for a, c in (("primario_res_v1", "primario_pas_v1"), ("corpo_res_v1", "corpo_pas_v1"),
                 ("primario_pas_v1", "primario_pas_v2"), ("corpo_pas_v1", "corpo_pas_v2"),
                 ("primario_res_v1", "primario_res_v2"), ("corpo_res_v1", "corpo_res_v2"),
                 ("primario_res_v1", "primario_pas_v2"), ("corpo_res_v1", "corpo_pas_v2")):
        if a in linhas and c in linhas:
            ids = linhas[a]["ids"]
            x = np.array([linhas[c]["certo"][i] for i in ids], dtype=float)
            y = np.array([linhas[a]["certo"][i] for i in ids], dtype=float)
            d, ic = b.diferenca_pareada(x, y)
            print(f"   {a:16s} → {c:16s} {d:+.3f} [{ic[0]:+.3f}; {ic[1]:+.3f}] · ganha "
                  f"{int((x > y).sum())}, perde {int((x < y).sum())}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cortar-primario", action="store_true")
    p.add_argument("--validar-classificador", action="store_true")
    p.add_argument("--rodar", nargs="+", metavar="CÉLULA")
    p.add_argument("--resumir", action="store_true")
    p.add_argument("--limite", type=int)
    a = p.parse_args()
    if a.cortar_primario:
        cortar("primario")
    elif a.validar_classificador:
        modelo = _subir()
        try:
            validar_classificador(modelo)
        finally:
            modelo.parar()
    elif a.rodar:
        rodar(a.rodar, a.limite)
    elif a.resumir:
        resumir()
    else:
        p.error("--cortar-primario, --validar-classificador, --rodar ou --resumir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
