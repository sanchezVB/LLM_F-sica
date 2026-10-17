#!/usr/bin/env python3
"""Um teste de fora (DOC-13 §9.12): no AstroBench — perguntas que não escrevemos, sobre um
campo que o índice cobre —, a busca do sistema ajuda o modelo a acertar?

    P = .venv-treino\\Scripts\\python.exe scripts\\medir_astrobench.py
    P --baixar                  # o conjunto público e o manifesto (uma vez; o manifesto vai ao git)
    P --registrar ARQUIVO       # ...ou um arquivo baixado à mão (parquet, csv ou jsonl)
    P --consultas [--dev]       # fase 1, só o modelo: a consulta hipotética e o braço C
    P --buscar [--dev]          # fase 2, sem o modelo: a busca fundida + ΦRank, na CPU
    P --responder [--dev]       # fase 3, só o modelo: o braço A
    P --resumir-dev             # o desenvolvimento: leitura da letra, mudança, tempo, fontes
    P --exportar                # os dois arquivos de envio, o rascunho do e-mail e o agregado
    P --apurar --correcao-C ARQ --correcao-A ARQ   # quando a correção voltar, item a item
    P --apurar --gabarito ARQ                      # ...ou com as letras certas
    P --apurar --agregado-C K --agregado-A K       # ...ou só o acerto de cada arquivo

O gabarito do conjunto público não é público: quem corrige é a equipe do AstroMLab, pelo
e-mail. As fases do TESTE só rodam com a regra aceita (`REGRA_9_12_ACEITA` em
`phifm.eval.astrobench`, aceita em 2026-10-16); as do desenvolvimento (`--dev`, 50 itens
disjuntos) rodam sempre.

Em três fases, como a §9.9: o modelo, a busca na CPU e o índice lexical juntos paginam na
máquina de 16 GB. Respostas e fontes ficam em `data/processed/assistente/astrobench/`,
fora do git — e a correção item a item também, porque o compromisso com o AstroMLab é não
publicá-la. Vão ao git o manifesto e o agregado (`avaliacao/astrobench*.json`).

⚠️ Uma rodada pesada por vez (16 GB): segura a trava de `medir_passagens.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.core.console import utf8  # noqa: E402

utf8()

import medir_busca_lexical as mbl  # noqa: E402
import medir_passagens as mp  # noqa: E402
from phifm.eval import astrobench as ab  # noqa: E402

PASTA = mp.DIR / "astrobench"
ENVIO = PASTA / "envio"
MANIFESTO = mp.AVALIACAO / "astrobench_manifesto.json"
AGREGADO = mp.AVALIACAO / "astrobench.json"
AGREGADO_DEV = mp.AVALIACAO / "astrobench_dev.json"
LISTAGEM_HF = "https://datasets-server.huggingface.co/parquet?dataset="


# ── o conjunto, congelado ─────────────────────────────────────────────────────
def registrar(arquivo: Path, origem: str) -> None:
    """Lê com as guardas, copia para a pasta da medida e congela o manifesto. Uma vez:
    um segundo arquivo diferente do registrado é recusado."""
    itens = ab.carregar(arquivo)
    sha, registrado_em = ab.sha256_arquivo(arquivo), time.strftime("%Y-%m-%d %H:%M")
    if MANIFESTO.exists():                       # confere ANTES de copiar por cima
        antigo = json.loads(MANIFESTO.read_text(encoding="utf-8"))
        if antigo["sha256_arquivo"] != sha:
            raise SystemExit(f"já existe um conjunto registrado ({antigo['sha256_arquivo'][:16]}"
                             f"…) e este é outro ({sha[:16]}…): a medida não troca de "
                             "conjunto em silêncio.")
        registrado_em = antigo.get("registrado_em") or registrado_em
    PASTA.mkdir(parents=True, exist_ok=True)
    destino = PASTA / f"astrobench_publico{arquivo.suffix.lower()}"
    if arquivo.resolve() != destino.resolve():
        shutil.copyfile(arquivo, destino)
    man = {**ab.manifesto(destino, itens, origem), "registrado_em": registrado_em}
    MANIFESTO.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(man, ensure_ascii=False, indent=1))
    print(f"→ {MANIFESTO.relative_to(RAIZ)} (vai ao git)")


def baixar() -> None:
    """Os parquet do split `train` pela listagem do Hugging Face. `HF_TOKEN` no ambiente,
    se o conjunto pedir aceite de termos."""
    import polars as pl
    import requests

    cab = {"Authorization": f"Bearer {t}"} if (t := os.environ.get("HF_TOKEN")) else {}
    url = LISTAGEM_HF + ab.CONJUNTO
    try:
        r = requests.get(url, headers=cab, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(f"a listagem do Hugging Face falhou ({e}). Baixe à mão em "
                         f"https://huggingface.co/datasets/{ab.CONJUNTO} e rode "
                         "--registrar ARQUIVO.") from None
    arquivos = [d for d in r.json().get("parquet_files", []) if d.get("split") == "train"]
    configs = sorted({d["config"] for d in arquivos})
    if len(configs) != 1:
        raise SystemExit(f"esperava uma configuração no split train; vieram {configs}. "
                         "Baixe à mão e rode --registrar.")
    partes = PASTA / "_partes"
    partes.mkdir(parents=True, exist_ok=True)
    caminhos = []
    for d in sorted(arquivos, key=lambda d: d["filename"]):
        destino = partes / d["filename"]
        with requests.get(d["url"], headers=cab, timeout=300, stream=True) as resp:
            resp.raise_for_status()
            with destino.open("wb") as f:
                for bloco in resp.iter_content(1 << 20):
                    f.write(bloco)
        caminhos.append(destino)
    # Um arquivo só: os bytes como vieram, e o sha256 é o do Hugging Face. Mais de um:
    # juntos, na ordem dos nomes.
    unico = caminhos[0] if len(caminhos) == 1 else partes / "juntos.parquet"
    if len(caminhos) > 1:
        pl.concat([pl.read_parquet(c) for c in caminhos]).write_parquet(unico)
    registrar(unico, f"{url} · config {configs[0]} · {len(caminhos)} parquet")
    shutil.rmtree(partes)


def _conjunto(dev: bool) -> tuple[str, list[ab.Item]]:
    """`(rótulo, itens na ordem da permutação)`, conferidos contra o manifesto."""
    if not MANIFESTO.exists():
        raise SystemExit("nenhum conjunto registrado (--baixar ou --registrar ARQUIVO).")
    man = json.loads(MANIFESTO.read_text(encoding="utf-8"))
    arquivo = PASTA / man["arquivo"]
    itens = ab.carregar(arquivo)
    try:
        ab.conferir_manifesto(arquivo, itens, man)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    teste, desenvolvimento = ab.amostras(len(itens))
    return ("dev", [itens[i] for i in desenvolvimento]) if dev else \
        ("teste", [itens[i] for i in teste])


def _recusa_sem_aceite(dev: bool) -> None:
    if not dev and not ab.REGRA_9_12_ACEITA:
        raise SystemExit("a §9.12 do DOC-13 ainda é PROPOSTA: os 1.000 do teste não são "
                         "tocados antes do aceite do dono (REGRA_9_12_ACEITA em "
                         "phifm.eval.astrobench). O desenvolvimento roda com --dev.")


_ASSINATURA: list[str] = []


def _assinatura() -> str:
    """O que, se mudar, faz outra medida: modelo, instruções, semente e o índice lexical."""
    if not _ASSINATURA:
        from phifm.eval.assistente import SEMENTE
        from phifm.rag.assistente import SISTEMA_HYDE
        from phifm.rag.llm import MODELO

        lexico = json.loads((mbl.LEXICO / "manifesto.json").read_text(encoding="utf-8"))
        _ASSINATURA.append(mp._sha("\n".join(map(str, (
            MODELO.name, ab.SISTEMA_SEM_FONTES, ab.SISTEMA_COM_FONTES, SISTEMA_HYDE,
            ab.MAX_TOKENS_RESPOSTA, SEMENTE, lexico["hash_da_ordem"], lexico["celulas"],
            lexico["k1"], lexico["b"])))))
    return _ASSINATURA[0]


def _arq(rotulo: str, fase: str) -> Path:
    return PASTA / f"{rotulo}_{fase}_{_assinatura()[:12]}.jsonl"


def _feitos(arq: Path) -> dict[int, dict]:
    if not arq.exists():
        return {}
    return {d["indice"]: d for d in (json.loads(x) for x in
                                     arq.read_text(encoding="utf-8").splitlines() if x.strip())}


def _gravar(f, registro: dict) -> None:
    f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    f.flush()


# ── as três fases ─────────────────────────────────────────────────────────────
def consultas_fase(a, dev: bool) -> None:
    """Fase 1, só o modelo: a consulta hipotética do produto, com o enunciado inteiro, e o
    braço C. As duas chamadas no mesmo processo; só o texto cru é gravado — a letra sai
    na leitura, e um conserto da leitura no desenvolvimento não pede rodar de novo."""
    from phifm.eval.assistente import K_FONTES, SEMENTE
    from phifm.rag.assistente import Assistente

    _recusa_sem_aceite(dev)
    rotulo, itens = _conjunto(dev)
    arq = _arq(rotulo, "consulta_C")
    feitos = _feitos(arq)
    PASTA.mkdir(parents=True, exist_ok=True)
    with mp.trava("medir_passagens"):
        modelo = mbl._modelo()
        assistente = Assistente(None, modelo, a.spine, k=K_FONTES, semente=SEMENTE)
        passo = mbl._progresso(f"{rotulo} · consulta e C", len(itens), len(feitos))
        try:
            with arq.open("a", encoding="utf-8") as f:
                for it in itens:
                    if it.indice in feitos:
                        continue
                    t0 = time.perf_counter()
                    consulta = assistente.consulta_hipotetica(ab.enunciado(it))
                    t1 = time.perf_counter()
                    bruto = ab.responder(modelo, it)
                    _gravar(f, {"indice": it.indice, "id": it.id, "consulta": consulta,
                                "segundos_consulta": round(t1 - t0, 2), "bruto_C": bruto,
                                "segundos_C": round(time.perf_counter() - t1, 2)})
                    passo()
        finally:
            modelo.parar()
    print(f"consultas e C → {arq}")


def buscar_fase(a, dev: bool) -> None:
    """Fase 2, sem o modelo: `Assistente.recuperar` como no produto — a densa com a
    consulta hipotética fundida ao BM25 do enunciado (já em inglês: é o que a tradução
    daria), e o ΦRank nos 50."""
    _recusa_sem_aceite(dev)
    rotulo, itens = _conjunto(dev)
    consultas = _feitos(_arq(rotulo, "consulta_C"))
    if len(consultas) < len(itens):
        raise SystemExit(f"faltam consultas: {len(consultas)} de {len(itens)} (--consultas).")
    arq = _arq(rotulo, "fontes")
    feitos = _feitos(arq)
    with mp.trava("medir_passagens"):
        assistente = mbl._buscador(a)
        passo = mbl._progresso(f"{rotulo} · busca", len(itens), len(feitos))
        with arq.open("a", encoding="utf-8") as f:
            for it in itens:
                if it.indice in feitos:
                    continue
                t0 = time.perf_counter()
                fontes = assistente.recuperar(consultas[it.indice]["consulta"], ab.enunciado(it))
                _gravar(f, {"indice": it.indice, "id": it.id,
                            "fontes": [x.arxiv_id for x in fontes],
                            "segundos": round(time.perf_counter() - t0, 2)})
                passo()
    print(f"fontes → {arq}")


def responder_fase(a, dev: bool) -> None:
    """Fase 3, só o modelo: o braço A, com as 6 fontes refeitas como `recuperar` as monta
    (título e ano do índice, resumo do spine) — o mesmo caminho da §9.9."""
    _recusa_sem_aceite(dev)
    rotulo, itens = _conjunto(dev)
    buscas = _feitos(_arq(rotulo, "fontes"))
    if len(buscas) < len(itens):
        raise SystemExit(f"faltam buscas: {len(buscas)} de {len(itens)} (--buscar).")
    arq = _arq(rotulo, "A")
    feitos = _feitos(arq)
    fontes_de = mp._refazer_fontes({k: SimpleNamespace(fontes=d["fontes"])
                                    for k, d in buscas.items()}, a.indice, a.spine)
    with mp.trava("medir_passagens"):
        modelo = mbl._modelo()
        passo = mbl._progresso(f"{rotulo} · A", len(itens), len(feitos))
        try:
            with arq.open("a", encoding="utf-8") as f:
                for it in itens:
                    if it.indice in feitos:
                        continue
                    t0 = time.perf_counter()
                    bruto = ab.responder(modelo, it, fontes_de(buscas[it.indice]["fontes"]))
                    _gravar(f, {"indice": it.indice, "id": it.id, "bruto_A": bruto,
                                "segundos": round(time.perf_counter() - t0, 2)})
                    passo()
        finally:
            modelo.parar()
    print(f"A → {arq}")


# ── leitura, envio e apuração ─────────────────────────────────────────────────
def _ler(rotulo: str, itens: list[ab.Item]) -> SimpleNamespace:
    c, fo, a = (_feitos(_arq(rotulo, x)) for x in ("consulta_C", "fontes", "A"))
    for nome, d in (("consultas e C", c), ("fontes", fo), ("A", a)):
        if len(d) < len(itens):
            raise SystemExit(f"{rotulo}: {nome} em {len(d)} de {len(itens)} itens.")
        trocado = [it.indice for it in itens if d[it.indice]["id"] != it.id]
        if trocado:
            raise SystemExit(f"{rotulo}: {nome} gravado para outro texto no item {trocado[0]}.")
    return SimpleNamespace(
        brutos_c=[c[i.indice]["bruto_C"] for i in itens],
        brutos_a=[a[i.indice]["bruto_A"] for i in itens],
        letras_c=[ab.extrair_letra(c[i.indice]["bruto_C"]) for i in itens],
        letras_a=[ab.extrair_letra(a[i.indice]["bruto_A"]) for i in itens],
        fontes=[fo[i.indice]["fontes"] for i in itens],
        tempo={"consulta": [c[i.indice]["segundos_consulta"] for i in itens],
               "C": [c[i.indice]["segundos_C"] for i in itens],
               "busca": [fo[i.indice]["segundos"] for i in itens],
               "A": [a[i.indice]["segundos"] for i in itens]})


def _metadados(indice: Path, fontes: list[list[str]]) -> dict[str, tuple]:
    import polars as pl

    ids = sorted({x for fs in fontes for x in fs})
    d = (pl.scan_parquet(indice / "documentos.parquet").filter(pl.col("arxiv_id").is_in(ids))
         .select("arxiv_id", "primary_category", "year").collect())
    return {r[0]: (r[1], r[2]) for r in d.iter_rows()}


def _tempos(lidos: SimpleNamespace) -> dict:
    import numpy as np

    return {f"mediana_{k}_s": round(float(np.median(v)), 2) for k, v in lidos.tempo.items()}


def resumir_dev(a) -> None:
    """Exploratório, não decide: o que se confere no desenvolvimento antes do teste."""
    rotulo, itens = _conjunto(dev=True)
    lidos = _ler(rotulo, itens)
    saida = {"regra": "DOC-13 §9.12 — desenvolvimento (exploratório, não decide)",
             "assinatura": _assinatura()[:12],
             **ab.secundarias(lidos.letras_c, lidos.letras_a, lidos.fontes,
                              _metadados(a.indice, lidos.fontes)),
             "tempo": _tempos(lidos)}
    if all(i.gabarito for i in itens):
        saida["acerto (gabarito do próprio arquivo)"] = ab.apurar_por_item(
            ab.corrigir([i.indice for i in itens], lidos.letras_c,
                        {i.indice: i.gabarito for i in itens}),
            ab.corrigir([i.indice for i in itens], lidos.letras_a,
                        {i.indice: i.gabarito for i in itens}))
    AGREGADO_DEV.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(saida, ensure_ascii=False, indent=1))
    sem = [(b, x) for b, xs, ls in (("C", lidos.brutos_c, lidos.letras_c),
                                    ("A", lidos.brutos_a, lidos.letras_a))
           for x, letra in zip(xs, ls, strict=True) if letra is None]
    for braco, bruto in sem[:10]:
        print(f"[sem letra · {braco}] {bruto!r}")
    print(f"→ {AGREGADO_DEV.relative_to(RAIZ)}")


def exportar(a) -> None:
    """Os dois arquivos que vão por e-mail e o agregado ANTES da correção: o que foi
    enviado fica registrado (sha256) e as secundárias, que não pedem gabarito, também."""
    rotulo, itens = _conjunto(dev=False)
    lidos = _ler(rotulo, itens)
    man = json.loads(MANIFESTO.read_text(encoding="utf-8"))
    antes = json.loads(AGREGADO.read_text(encoding="utf-8")) if AGREGADO.exists() else {}
    if antes.get("estado") == "apurado":
        raise SystemExit("a correção já foi apurada: exportar de novo apagaria o resultado.")
    # Num rascunho primeiro: o que já foi enviado não pode ser sobrescrito por um envio
    # diferente nem enquanto se descobre que ele é diferente.
    rascunho = PASTA / "_envio_novo"
    envio = {}
    try:
        for braco, letras in (("C", lidos.letras_c), ("A", lidos.letras_a)):
            envio[braco] = ab.exportar(itens, letras, rascunho / f"astrobench_{braco}.csv")
    except ValueError as e:
        shutil.rmtree(rascunho, ignore_errors=True)
        raise SystemExit(str(e)) from None
    if antes.get("envio") and antes["envio"] != envio:
        shutil.rmtree(rascunho)
        raise SystemExit(f"o agregado já registra um envio diferente ({antes['envio']}): o "
                         "que foi enviado não muda depois de enviado.")
    ENVIO.mkdir(parents=True, exist_ok=True)
    for arq in rascunho.iterdir():
        arq.replace(ENVIO / arq.name)
    rascunho.rmdir()
    email =ab.rascunho_de_email(len(itens), man["sha256_arquivo"],
                                 [f"astrobench_{b}.csv" for b in envio])
    (ENVIO / "email.txt").write_text(email, encoding="utf-8")
    saida = {"regra": "DOC-13 §9.12", "regra_aceita_pelo_dono": ab.REGRA_9_12_ACEITA,
             "estado": "enviado; aguardando a correção do AstroMLab",
             "conjunto": {"sha256_arquivo": man["sha256_arquivo"], "linhas": man["linhas"],
                          "n_teste": len(itens)},
             "assinatura": _assinatura()[:12], "envio": envio,
             "secundarias (relatadas, não decidem)": ab.secundarias(
                 lidos.letras_c, lidos.letras_a, lidos.fontes,
                 _metadados(a.indice, lidos.fontes)),
             "tempo": _tempos(lidos),
             "comparadores_publicados (outro protocolo)": ab.COMPARADORES_PUBLICADOS,
             "primario": None}
    AGREGADO.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(saida, ensure_ascii=False, indent=1))
    print(f"\nanexos e rascunho em {ENVIO}\n→ mande para {ab.ENDERECO_DE_CORRECAO} e "
          f"guarde a resposta em {PASTA} (fora do git).\n→ {AGREGADO.relative_to(RAIZ)}")


def _letras_enviadas(braco: str, registrado: str) -> dict[int, str | None]:
    import csv

    arq = ENVIO / f"astrobench_{braco}.csv"
    if ab.sha256_arquivo(arq) != registrado:
        raise SystemExit(f"{arq.name} não é o arquivo enviado (sha256 diferente do agregado).")
    with arq.open(encoding="utf-8", newline="") as f:
        return {int(d["row_index"]): d["answer"] or None for d in csv.DictReader(f)}


def apurar(args) -> None:
    """A regra, com o que voltar: acerto por item (pareada), as letras certas, o gabarito
    do próprio arquivo ou — degradada — o acerto de cada arquivo."""
    if not AGREGADO.exists() or not json.loads(AGREGADO.read_text(encoding="utf-8")).get("envio"):
        raise SystemExit("nada foi enviado ainda (--exportar).")
    saida = json.loads(AGREGADO.read_text(encoding="utf-8"))
    _, itens = _conjunto(dev=False)
    indices = [i.indice for i in itens]
    enviadas = {b: _letras_enviadas(b, saida["envio"][b]) for b in ("C", "A")}
    letras = {b: [enviadas[b][i] for i in indices] for b in ("C", "A")}
    try:
        if args.correcao_C and args.correcao_A:
            r = ab.apurar_por_item(ab.acertos_de(indices, ab.ler_correcao(args.correcao_C)),
                                   ab.acertos_de(indices, ab.ler_correcao(args.correcao_A)))
            origem = "acerto por item, corrigido pelo AstroMLab"
        elif args.gabarito or all(i.gabarito for i in itens):
            gab = (ab.ler_gabarito(args.gabarito) if args.gabarito
                   else {i.indice: i.gabarito for i in itens})
            r = ab.apurar_por_item(ab.corrigir(indices, letras["C"], gab),
                                   ab.corrigir(indices, letras["A"], gab))
            origem = "gabarito por item" + (" devolvido" if args.gabarito else " do arquivo")
        elif args.agregado_C is not None and args.agregado_A is not None:
            r = ab.apurar_agregado(args.agregado_C, args.agregado_A, len(indices))
            origem = "só o acerto de cada arquivo (leitura degradada, sem pareamento)"
        else:
            raise SystemExit("diga o que voltou: --correcao-C e --correcao-A, --gabarito, ou "
                             "--agregado-C e --agregado-A.")
    except ValueError as e:
        raise SystemExit(str(e)) from None
    saida.update({"estado": "apurado", "origem_da_correcao": origem, "primario": r,
                  "decisao": r["leitura"]})
    AGREGADO.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(r, ensure_ascii=False, indent=1))
    print(f"→ {AGREGADO.relative_to(RAIZ)}")


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--baixar", action="store_true")
    g.add_argument("--registrar", type=Path, metavar="ARQUIVO")
    g.add_argument("--consultas", action="store_true")
    g.add_argument("--buscar", action="store_true")
    g.add_argument("--responder", action="store_true")
    g.add_argument("--resumir-dev", action="store_true")
    g.add_argument("--exportar", action="store_true")
    g.add_argument("--apurar", action="store_true")
    p.add_argument("--dev", action="store_true", help="os 50 do desenvolvimento, não o teste")
    p.add_argument("--correcao-C", type=Path)
    p.add_argument("--correcao-A", type=Path)
    p.add_argument("--gabarito", type=Path)
    p.add_argument("--agregado-C", type=int, help="acertos de C nos 1.000 (contagem)")
    p.add_argument("--agregado-A", type=int, help="acertos de A nos 1.000 (contagem)")
    p.add_argument("--indice", type=Path, default=RAIZ / "data/processed/indice_busca")
    p.add_argument("--spine", type=Path, default=RAIZ / "data/processed/spine.parquet")
    args = p.parse_args()
    a = SimpleNamespace(indice=args.indice, spine=args.spine)
    if args.baixar:
        baixar()
    elif args.registrar:
        registrar(args.registrar, f"arquivo local {args.registrar.name}")
    elif args.consultas:
        consultas_fase(a, args.dev)
    elif args.buscar:
        buscar_fase(a, args.dev)
    elif args.responder:
        responder_fase(a, args.dev)
    elif args.resumir_dev:
        resumir_dev(a)
    elif args.exportar:
        exportar(a)
    else:
        apurar(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
