#!/usr/bin/env python3
"""G1.5 — constrói e verifica o manifesto raiz do corpus (DOC-00 §5).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/manifesto_corpus.py --construir
    PYTHONPATH=src .venv/Scripts/python.exe scripts/manifesto_corpus.py --verificar
    PYTHONPATH=src .venv/Scripts/python.exe scripts/manifesto_corpus.py --verificar --profundo

O critério do portão pede o corpus "reprodutível ponta a ponta a partir de um
único hash de manifesto". Ver a docstring de
`phifm.core.schema.reprodutibilidade` para a cadeia de Merkle e — mais
importante — para o que este hash **não** prova.

## As etapas derivadas são declaradas aqui, não descobertas

Um construtor que varre o disco e manifesta o que encontra atesta o que *está*
lá, não o que *deveria*. Se a espinha faltar, ele produz um manifesto válido de
um corpus incompleto, e o hash confere. Por isso `ETAPAS` é uma lista explícita:
uma etapa declarada e ausente é **erro**, não é silêncio.

⚠️ Os parâmetros das etapas já executadas são **reconstruídos** do código, não
capturados na execução — as etapas rodaram antes deste manifesto existir. Cada
manifesto carrega `parametros_reconstruidos=True` para dizer isso. As etapas
futuras devem gravar o seu manifesto ao terminar, e aí a marca cai.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.core.schema.reprodutibilidade import (  # noqa: E402
    Entrada,
    EtapaRef,
    ManifestoEtapa,
    ManifestoRaiz,
    git_sha_curto,
    hash_arquivo,
    indexar,
    tamanho_de,
    verificar,
)

log = logging.getLogger("manifesto")

RAIZ_JSON = Path("data/processed/MANIFESTO-RAIZ.json")
NOME_ETAPA = "_manifesto_etapa.json"

# Fontes que uma refeitura NÃO reproduz byte a byte. Nomear é parte da garantia.
# ⚠️ Esta lista já teve uma entrada ERRADA, escrita por suposição. Eu havia posto
# o RedPajama aqui como "alvo móvel" antes de olhar: as URLs dos shards são
# `data.together.xyz/redpajama-data-1T/v1.0.0/arxiv/…`, versionadas na origem e
# fora do HuggingFace. O único ponto móvel era o arquivo de índice, agora fixado
# na revisão 398f9257. Listar uma fonte como irreprodutível sem verificar é o
# mesmo erro de sempre na direção oposta: pessimismo por inércia também é
# afirmação sem medida.
MUTAVEIS = [
    "arxiv OAI-PMH — `from`/`until` filtram por datestamp, e o datestamp muda "
    "quando o autor publica versão nova; uma coleta refeita traz registros que a "
    "de hoje não tinha. É a semântica da fonte, não defeito do coletor.",
]

# Fontes externas reprodutíveis, com o mecanismo de cada uma. Registrado porque
# "reprodutível" sem dizer POR QUE é alegação, e o G1.5 é sobre não ter alegações.
IMUTAVEIS = [
    "open-web-math/open-web-math — revisão fde8ef8d (2023-10-17), anterior à nossa "
    "coleta; o coletor agora fixa o sha em vez de usar `main`",
    "togethercomputer/RedPajama-Data-1T — shards versionados na origem "
    "(data.together.xyz/.../v1.0.0/...); índice fixado na revisão 398f9257. "
    "Risco residual é DISPONIBILIDADE, não mutabilidade",
]

# ─── As etapas derivadas, com as entradas REAIS lidas dos scripts ───────────
#
# `entradas` aponta para os diretórios que cada script recebe por padrão. Foram
# conferidos um a um em `scripts/build_spine.py`, `build_pairs.py`,
# `train_classifier.py`, `coletar_redpajama.py` e `filtrar_hf.py` — não inferidos
# pelo nome. A espinha sai de `openalex_works` (a API), não do snapshot; os pares
# saem do snapshot. Trocar os dois seria proveniência errada com aparência certa.
ETAPAS: list[dict] = [
    {
        "etapa": "spine",
        "descricao": "Espinha de metadados: arXiv juntado ao OpenAlex por DOI e título",
        "raiz": "data/processed/spine.parquet",
        "entradas": ["data/raw/arxiv_metadata", "data/raw/openalex_works"],
        "parametros": {"script": "scripts/build_spine.py"},
    },
    {
        "etapa": "isphysics_clf",
        "descricao": "Classificador binário Física/não-Física, negativos estratificados",
        "raiz": "models/isphysics-clf",
        "entradas": ["data/processed/spine.parquet", "data/raw/arxiv_negativos"],
        "parametros": {"script": "scripts/train_classifier.py", "task": "isphysics",
                       "max_por_classe": 400_000, "precision": 0.95},
    },
    {
        "etapa": "pares_citacao",
        "descricao": "Pares âncora/positivo de citação para o treino contrastivo",
        "raiz": "data/processed/pares",
        "entradas": ["data/raw/openalex_snapshot", "data/processed/spine.parquet"],
        "parametros": {"script": "scripts/build_pairs.py"},
    },
    {
        "etapa": "redpajama_fisica",
        "descricao": "Fatia de Física do RedPajama-arXiv, filtrada por casamento exato com a espinha",
        "raiz": "data/processed/redpajama_fisica",
        "entradas": ["data/processed/spine.parquet"],
        "externas": [("togethercomputer/RedPajama-Data-1T", "hf_dataset")],
        "parametros": {"script": "scripts/coletar_redpajama.py", "filtro": "spine (exato)"},
    },
    {
        "etapa": "openwebmath_fisica",
        "descricao": "Fatia de Física do OpenWebMath, filtrada pelo classificador",
        "raiz": "data/processed/openwebmath_fisica",
        "entradas": ["models/isphysics-clf"],
        "externas": [("open-web-math/open-web-math", "hf_dataset")],
        "parametros": {"script": "scripts/filtrar_hf.py", "limiar": 0.9},
    },
]


def _registros(raiz: Path) -> int | None:
    """Linhas, quando a etapa é parquet. `None` quando não faz sentido contar."""
    import polars as pl

    try:
        if raiz.is_file() and raiz.suffix == ".parquet":
            return int(pl.scan_parquet(raiz).select(pl.len()).collect().item())
        fs = sorted(raiz.glob("**/*.parquet"))
        if not fs:
            return None
        return sum(int(pl.scan_parquet(f).select(pl.len()).collect().item()) for f in fs)
    except Exception as exc:
        log.warning("contagem de registros indisponível em %s (%s)", raiz, type(exc).__name__)
        return None


def _revisao_hf(ds: str, quando_coletamos: str | None) -> dict:
    """Revisão da fonte no HuggingFace, com a evidência de valer para a NOSSA coleta.

    ⚠️ A coleta baixou de `resolve/main/`. O `sha` de hoje só é o `sha` usado se o
    dataset não mudou desde então — e isso é verificável: `lastModified` anterior
    ao início da coleta prova. Registrar o sha sem essa checagem seria inventar
    proveniência, que é pior que não ter nenhuma.
    """
    import requests

    try:
        r = requests.get(f"https://huggingface.co/api/datasets/{ds}", timeout=60)
        r.raise_for_status()
        d = r.json()
        sha, modificado = d.get("sha"), d.get("lastModified")
        vale = bool(quando_coletamos and modificado and modificado < quando_coletamos)
        return {"dataset": ds, "revisao_atual": sha, "last_modified": modificado,
                "baixado_de": "resolve/main (alvo móvel — ver MUTAVEIS)",
                "revisao_vale_para_nossa_coleta": vale,
                "evidencia": (f"last_modified {modificado} é anterior ao início da nossa "
                              f"coleta ({quando_coletamos}), então main não mudou desde"
                              if vale else
                              "não verificável: sem data de coleta registrada, ou a fonte "
                              "mudou depois de coletarmos")}
    except Exception as exc:
        return {"dataset": ds, "erro": f"{type(exc).__name__}", "revisao_atual": None,
                "revisao_vale_para_nossa_coleta": False}


def _quando(raiz: Path) -> str | None:
    """Início da coleta de uma fatia, do `_filtragem.json` quando houver."""
    for nome in ("_filtragem.json", "_manifest.json"):
        p = raiz / nome
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                q = d.get("started_at") or d.get("iniciado_em")
                if q:
                    return str(q)
            except Exception:
                pass
    # Recuo: o mtime do parquet mais antigo. É pior que um registro explícito, e o
    # manifesto diz que é recuo — não passa por dado coletado.
    fs = sorted(raiz.glob("**/*.parquet"))
    if fs:
        import datetime as dt
        t = min(f.stat().st_mtime for f in fs)
        return dt.datetime.fromtimestamp(t, dt.UTC).isoformat() + "  (recuo: mtime)"
    return None


def construir(base: Path, *, rede: bool = True) -> ManifestoRaiz:
    sha = git_sha_curto()
    refs: list[EtapaRef] = []
    bytes_tot = arq_tot = 0
    faltando: list[str] = []

    # 1. Aquisições. O manifesto do coletor entra como PROVENIÊNCIA, mas o índice
    #    de hashes é computado AQUI, e a razão é um defeito medido.
    #
    # ⚠️ O `checksum_index` dos coletores NÃO é checksum de arquivo. É
    # `canonical_hash({"rows": n, "cols": [...]})` — o hash da FORMA. Dois arquivos
    # de conteúdo completamente diferente, com as mesmas linhas e colunas, hasheiam
    # igual. O DOC-02 §8.1 especifica "mapa doc_id → BLAKE3, endereçado por
    # conteúdo"; a implementação divergiu da especificação sob um nome que promete
    # o contrário.
    #
    # Descoberto porque a verificação profunda deste script acusou 878 "alterado"
    # nos parquets de aquisição. A resposta certa a um alarme não é assumir
    # adulteração nem silenciar o alarme: é descobrir o que ele está comparando.
    #
    # Então as etapas de aquisição recebem o MESMO tratamento das derivadas: índice
    # BLAKE3 real, computado sobre os bytes que estão no disco. O manifesto do
    # coletor fica intocado e é referenciado como entrada — ele guarda o que a
    # coleta viu (cursor, licença, falhas, taxa), que nada mais guarda.
    for m in sorted(base.glob("data/raw/**/_manifest.json")):
        d = json.loads(m.read_text(encoding="utf-8"))
        dir_bruto = m.parent
        nome = d.get("source_name", dir_bruto.name)
        idx = indexar(dir_bruto)
        b, n = tamanho_de(dir_bruto)
        me = ManifestoEtapa(
            etapa=f"bruto_{nome}_{dir_bruto.name}" if nome != dir_bruto.name else f"bruto_{nome}",
            descricao=f"Coleta bruta: {nome} ({dir_bruto.relative_to(base).as_posix()})",
            raiz=dir_bruto.relative_to(base).as_posix(),
            entradas=[Entrada(caminho=m.relative_to(base).as_posix(),
                              manifesto_id=d.get("manifest_id", ""),
                              nota="manifesto de aquisição do coletor — cursor, licença, "
                                   "falhas e taxa; o `checksum_index` dele é hash de FORMA, "
                                   "não de conteúdo")],
            parametros={"endpoint": d.get("endpoint"), "metodo": d.get("harvest_method"),
                        "query_spec": d.get("query_spec"),
                        "actual_count": d.get("actual_count"),
                        "completed_at": d.get("completed_at"),
                        "hash_index_computado_em": "construção do manifesto raiz, "
                                                   "não na coleta"},
            parametros_reconstruidos=True,
            registros=d.get("actual_count"), checksum_index=idx, bytes_saida=b,
            git_sha=sha).selar()
        destino = dir_bruto / NOME_ETAPA
        destino.write_text(me.model_dump_json(indent=2), encoding="utf-8")
        refs.append(EtapaRef(tipo="aquisicao",
                             caminho=destino.relative_to(base).as_posix(),
                             etapa=me.etapa, manifesto_id=me.manifesto_id,
                             hash_manifesto=hash_arquivo(destino)))
        bytes_tot += b
        arq_tot += n

    # 2. Derivadas: manifesto construído agora.
    for spec in ETAPAS:
        raiz = base / spec["raiz"]
        if not raiz.exists():
            faltando.append(spec["etapa"])
            continue

        entradas = []
        for e in spec["entradas"]:
            p = base / e
            mid = None
            for cand in (p / "_manifest.json", p / NOME_ETAPA,
                         p.parent / f"{p.name}{NOME_ETAPA}"):
                if cand.exists():
                    mid = json.loads(cand.read_text(encoding="utf-8")).get(
                        "manifest_id") or json.loads(
                        cand.read_text(encoding="utf-8")).get("manifesto_id")
                    break
            entradas.append(Entrada(caminho=e, manifesto_id=mid,
                                    nota=None if mid else "sem manifesto a montante"))

        params = dict(spec["parametros"])
        for ds, _ in spec.get("externas", []):
            params.setdefault("fontes_externas", []).append(
                _revisao_hf(ds, _quando(raiz)) if rede
                else {"dataset": ds, "nota": "rede desativada nesta construção"})

        idx = indexar(raiz) if raiz.is_dir() else {raiz.name: hash_arquivo(raiz)}
        b, n = tamanho_de(raiz)
        me = ManifestoEtapa(
            etapa=spec["etapa"], descricao=spec["descricao"],
            raiz=spec["raiz"], entradas=entradas, parametros=params,
            parametros_reconstruidos=True,
            registros=_registros(raiz), checksum_index=idx, bytes_saida=b,
            git_sha=sha).selar()

        destino = (raiz / NOME_ETAPA if raiz.is_dir()
                   else raiz.parent / f"{raiz.name}{NOME_ETAPA}")
        destino.write_text(me.model_dump_json(indent=2), encoding="utf-8")
        refs.append(EtapaRef(tipo="processamento",
                             caminho=destino.relative_to(base).as_posix(),
                             etapa=me.etapa, manifesto_id=me.manifesto_id,
                             hash_manifesto=hash_arquivo(destino)))
        bytes_tot += b
        arq_tot += n
        log.info("etapa %-20s %6.2f GB · %s arquivos · %s registros · %s",
                 me.etapa, b / 1e9, len(idx),
                 f"{me.registros:,}" if me.registros else "—",
                 me.manifesto_id[:12])

    if faltando:
        # Etapa declarada e ausente é erro. Um manifesto de corpus incompleto que
        # confere é exatamente o modo de falha que este script existe para não ter.
        raise SystemExit(
            f"ETAPAS declaradas e ausentes no disco: {', '.join(faltando)}.\n"
            "Um manifesto raiz de corpus incompleto conferiria e não valeria nada. "
            "Rode a etapa, ou remova-a de ETAPAS declarando por quê.")

    return ManifestoRaiz(git_sha=sha, etapas=refs, bytes_totais=bytes_tot,
                         arquivos_totais=arq_tot, fontes_mutaveis=MUTAVEIS,
                         fontes_imutaveis=IMUTAVEIS).selar()


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--construir", action="store_true")
    g.add_argument("--verificar", action="store_true")
    p.add_argument("--profundo", action="store_true",
                   help="relê e re-hasheia os 22 GB; é a única verificação que pega "
                        "parquet adulterado")
    p.add_argument("--sem-rede", action="store_true",
                   help="não consulta a revisão das fontes no HuggingFace")
    p.add_argument("--base", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, default=RAIZ_JSON)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    if a.construir:
        raiz = construir(a.base, rede=not a.sem_rede)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(raiz.model_dump_json(indent=2), encoding="utf-8")
        print()
        print("=" * 72)
        print(f"  HASH RAIZ DO CORPUS: {raiz.hash_raiz}")
        print("=" * 72)
        print(f"  {len(raiz.etapas)} etapas · {raiz.arquivos_totais} arquivos · "
              f"{raiz.bytes_totais/1e9:.2f} GB · git {raiz.git_sha}")
        print(f"  -> {a.out}")
        print()
        print("  Fontes que uma refeitura NAO reproduz byte a byte:")
        for f in raiz.fontes_mutaveis:
            print(f"    - {f}")
        return 0

    rel = verificar(a.out, profundo=a.profundo, base=a.base)
    print(rel.resumo())
    return 0 if rel.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
