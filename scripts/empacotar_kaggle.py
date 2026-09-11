#!/usr/bin/env python3
"""Empacota o mínimo para um experimento de GPU rodar no Kaggle.

    .venv/Scripts/python.exe scripts/empacotar_kaggle.py --experimento t1a
    .venv/Scripts/python.exe scripts/empacotar_kaggle.py --experimento t1c

Produz `data/processed/kaggle_<exp>/`, para subir como Kaggle Dataset. Os nomes,
slugs e a lista de arquivos vêm de `phifm.core.kaggle`, para que este script e o
`publicar_kaggle.py` não possam discordar.

## T1a — ΦEmb, 211 MB e não 2,68 GB

O `pares_treino.parquet` inteiro tem 6,56 M de arestas e 2,68 GB. O T1a usa
**400 mil pares** — o volume do campeão do G1.1, escolhido por medição: o treino de
1,5 M empatou estatisticamente com ele (p=0,950) e o de 511 negativos também
(p=0,636). Subir o conjunto inteiro seria pagar 13× de banda por dados que a
medição diz não comprar nada.

## T1c — ΦRank de base diferente

Vão os negativos minerados do recuperador de verdade (o RRF top-50, já sem os
co-citados), a validação, e **dois modelos**: o ΦEmb campeão e o ΦRank de MiniLM
que serve de controle.

⚠️ Os negativos vão INTEIROS, 247 MB, com os 43,77 negativos por grupo. Cortar para
os 7 que o treino usa economizaria banda e **quebraria a comparação**: o controle
foi treinado sorteando 7 de 43,77, e sortear 7 de uma lista podada daria outros 7.
O experimento é de uma variável — a base —, então tudo o mais fica byte a byte igual.

⚠️ E os modelos vão porque não são públicos: o `phiemb-minilm-melhor` é o campeão do
G1.1 treinado aqui, e o resultado de referência do T1b (nDCG 0,1584) saiu dele. Usar
o `-t4-melhor` no lugar trocaria o recuperador junto com o reranqueador.

## O que sempre vai, e por que cada coisa

| arquivo | para quê |
|---|---|
| `MANIFESTO.json` | hashes BLAKE3 e proveniência, para o que roda lá ser o que está aqui |
| `phifm_src.zip.bin` | o pacote + os scripts — **só quando o experimento não declara `repo`** |

⚠️ O código nunca é copiado dentro do notebook. Um notebook que reimplementa o laço
de treino é um segundo laço para manter em sincronia, e a divergência entre os dois
seria invisível: os dois rodariam, com resultados diferentes, e nada apontaria qual
está certo.

⚠️ **Mas ele também deixou de viajar no dataset**, para os experimentos que declaram
`repo`. Medido em 2026-09-03: o Kaggle FIXA a versão do dataset no anexo ao kernel e
`kernels push` não re-resolve, então um conserto subiu numa versão nova, o
`datasets status` disse `ready`, e o notebook rodou 15 min sobre o código ANTIGO.
Agora o código vem de `codeload.github.com/<repo>/tar.gz/<sha>`, com o SHA injetado
pelo publicador — o notebook é reempurrado a cada publicação, então não há versão a
fixar. Os dados ficam no dataset justamente porque não mudam.

## O que NÃO vai

Pesos públicos. `all-MiniLM-L6-v2`, `gte-base` e `physbert_cased` são baixados do
HuggingFace no próprio Kaggle, que tem rede — subir 90 MB de cada seria desperdício.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.core.kaggle import (  # noqa: E402
    EXPERIMENTOS,
    VARIANTES_DE_VOLUME,
    Experimento,
    obter,
)
from phifm.core.schema.reprodutibilidade import (  # noqa: E402
    git_sha_curto,
    hash_arquivo,
)
from phifm.training.amostragem import sortear_para_parquet  # noqa: E402
from phifm.training.pretrain.dados import (  # noqa: E402
    NOME_MANIFESTO,
    NOME_MARCAS,
    NOME_TOKENS,
)

log = logging.getLogger("empacotar")

# Tudo do pacote, menos o que não roda lá nem faz sentido carregar.
EXCLUIR_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

# ⚠️ `.zip.bin`, nao `.zip`. O Kaggle DESCOMPACTA arquivos .zip no upload: medido em
# 2026-08-24, `phifm_src.zip` chegou no dataset como o diretorio `phifm_src/`, o
# notebook morreu em FileNotFoundError aos 26 s e — pior — o hash do FONTE deixou de
# ser conferivel, porque o arquivo que o manifesto descreve nao existia mais.
#
# Com uma extensao que o Kaggle nao reconhece como arquivo, ele guarda os bytes como
# estao e a conferencia por blake3 volta a valer. O `zipfile` abre pelo conteudo e
# nao pela extensao, entao nada mais muda.
SUFIXO_ZIP = ".zip.bin"


def _zipar_fonte(raiz: Path, destino: Path, scripts: tuple[str, ...]) -> int:
    n = 0
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((raiz / "src" / "phifm").rglob("*.py")):
            if any(p in EXCLUIR_DIRS for p in f.parts):
                continue
            z.write(f, f.relative_to(raiz / "src").as_posix())
            n += 1
        # Os scripts também, porque são o ponto de entrada de verdade — e assim o
        # notebook chama exatamente o que roda aqui.
        for nome in scripts:
            origem = raiz / "scripts" / nome
            if not origem.exists():
                raise SystemExit(
                    f"{origem} não existe, e o notebook do experimento a chama. "
                    "Zipar sem ela daria ModuleNotFoundError na GPU, depois do "
                    "upload inteiro.")
            z.write(origem, f"scripts/{nome}")
            n += 1
    return n


def _zipar_modelos(raiz: Path, destino: Path, modelos: tuple[str, ...]) -> int:
    """Zipa diretórios de modelo preservando só o nome final na raiz do ZIP.

    `models/phiemb-minilm-melhor/...` entra como `phiemb-minilm-melhor/...`, que é
    o que o notebook espera em `MODELOS / "phiemb-minilm-melhor"`.

    ⚠️ `estado_rank.pt` e `estado_treino.pt` ficam FORA. São o estado do otimizador
    para retomar treino, chegam a centenas de MB, e nada no notebook os lê — os
    modelos vão para inferência. Um `-melhor/` não os tem, mas o diretório de
    trabalho tem, e um dia alguém vai passar o errado.
    """
    n = 0
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in modelos:
            d = raiz / rel
            if not d.is_dir():
                raise SystemExit(
                    f"{d} não é um diretório. O experimento declara este modelo e "
                    "sem ele o notebook para no assert de pesos.")
            for f in sorted(d.rglob("*")):
                if not f.is_file() or f.suffix == ".pt":
                    continue
                z.write(f, (Path(d.name) / f.relative_to(d)).as_posix())
                n += 1
    return n


def _montar_t1a(exp: Experimento, raiz: Path, out: Path, a) -> dict:
    # ⚠️ SORTEIO, não `head`. Era `head(a.max_pares)`, e o run da T1a treinou com
    # 17.844 documentos citados distintos onde o sorteio do mesmo tamanho dá
    # 191.300 — 10,7×, pelo mesmo custo de GPU. Ver `amostrar_do_plano`.
    # ⚠️ Escreve em FLUXO. A versão anterior coletava antes de gravar, e a 6 M de
    # pares isso são 4 a 5 GB em memória com 7,1 GB livres — ver
    # `sortear_para_parquet`.
    n_linhas, total, n_doc = sortear_para_parquet(
        pl.scan_parquet(a.pares / "pares_treino.parquet"),
        a.max_pares, out / "pares_treino.parquet", a.semente)
    shutil.copy2(a.pares / "pares_validacao.parquet", out / "pares_validacao.parquet")
    # ⚠️ Sem zip de fonte quando `exp.repo` está setado — mesma razão da T1c: o
    # Kaggle fixa a versão do dataset no anexo e não re-resolve, então código no
    # dataset é código que pode ficar velho sem avisar.
    n_py = (0 if exp.repo
            else _zipar_fonte(raiz, out / f"phifm_src{SUFIXO_ZIP}", exp.scripts))
    return {"max_pares": a.max_pares, "linhas_treino": n_linhas,
            "linhas_disponiveis": total, "documentos_distintos": n_doc,
            "semente_do_sorteio": a.semente, "modulos_python": n_py,
            "codigo_de": exp.repo or "dataset"}


def _montar_t1b2(exp: Experimento, raiz: Path, out: Path, a) -> dict:
    """Só a validação e os modelos: este experimento MEDE, não treina."""
    shutil.copy2(a.pares / "pares_validacao.parquet", out / "pares_validacao.parquet")
    n_mod = _zipar_modelos(raiz, out / f"modelos{SUFIXO_ZIP}", exp.modelos)
    return {"arquivos_de_modelo": n_mod, "modelos": list(exp.modelos),
            "codigo_de": exp.repo or "dataset",
            "nota_do_protocolo": (
                "Os dois recuperadores vao juntos porque a celula mede os dois na "
                "MESMA sessao. Comparar contra o numero de agosto seria invalido: "
                "mudaram o protocolo do G1, o pool de candidatos e quatro versoes "
                "do codigo.")}


def _montar_rerank(exp: Experimento, raiz: Path, out: Path, a) -> dict:
    """T1c e T1d: negativos + validação + modelos. A diferença é QUAIS negativos,
    e eles vêm da identidade do experimento — ver `Experimento.negativos`."""
    if a.negativos is None:
        raise SystemExit(
            f"{exp.nome} não declara `negativos` e --negativos não foi dado. "
            "Adivinhar aqui empacotaria um arquivo qualquer sob o nome deste "
            "experimento.")
    origem = a.negativos
    if not origem.exists():
        raise SystemExit(
            f"{origem} não existe. Rode scripts/minerar_do_recuperador.py e depois "
            "scripts/filtrar_cocitacao.py — treinar sobre os negativos NÃO filtrados "
            "ensina o reranqueador a rebaixar co-citados, que são relevantes.")
    shutil.copy2(origem, out / origem.name)
    shutil.copy2(a.pares / "pares_validacao.parquet", out / "pares_validacao.parquet")
    n_mod = _zipar_modelos(raiz, out / f"modelos{SUFIXO_ZIP}", exp.modelos)
    # ⚠️ Sem zip de fonte quando `exp.repo` está setado. O código viaja pelo GitHub
    # num SHA porque o Kaggle fixa a versão do dataset no anexo e não re-resolve —
    # ver a nota em `phifm.core.kaggle`. Deixar o zip aqui só reintroduziria a
    # possibilidade de o notebook rodar código velho.
    n_py = (0 if exp.repo
            else _zipar_fonte(raiz, out / f"phifm_src{SUFIXO_ZIP}", exp.scripts))
    grupos = pl.scan_parquet(origem).select(pl.len()).collect().item()
    return {"grupos": grupos, "modulos_python": n_py, "arquivos_de_modelo": n_mod,
            "modelos": list(exp.modelos), "codigo_de": exp.repo or "dataset",
            "negativos": str(origem).replace("\\", "/")}


def _ligar(origem: Path, destino: Path) -> None:
    """Liga em vez de copiar quando dá. 2,7 GB por braço não precisam existir duas
    vezes no HD só para subir — e o `unlink` da limpeza tira o link, não a fatia."""
    if destino.exists():
        destino.unlink()
    try:
        os.link(origem, destino)
    except OSError:
        shutil.copy2(origem, destino)


def _montar_t2a(exp: Experimento, raiz: Path, out: Path, a) -> dict:
    """T2a — as DUAS fatias no mesmo dataset, achatadas por variante.

    As fatias vêm de `preparar_dados_phienc.py` com nomes canônicos iguais
    (`tokens.u16.bin`, `marcas.u8.bin`, `MANIFESTO_DADOS.json`) em diretórios
    diferentes. O Kaggle achata a estrutura do upload, então aqui o nome passa a
    carregar a variante e a célula refaz os links canônicos do outro lado.

    ⚠️ As duas fatias no MESMO dataset, e não uma em cada, porque os dois braços
    treinam em sessões separadas e nada mais garantiria que eles vieram do mesmo
    preparo. Um dataset, uma assinatura: se um braço rodou sobre outro bundle, a
    `assinatura_do_manifesto` acusa antes do treino.

    ⚠️ As `marcas` sobem mesmo com `p_equacao=0,0` nos dois braços, onde elas não
    são lidas — 1,8 GB de arquivo inútil para ESTE experimento. Vão por dois
    motivos: o laço memmapa e confere o tamanho delas sem olhar o `p_equacao`
    (`pretrain/dados.py`), e mexer no laço para poupar banda seria mexer no
    treinador na véspera do treino. E a ablação do §2.3, que é o experimento
    seguinte na fila, quer exatamente estas fatias — com as marcas aqui ela
    `reusa_dados_de` e custa zero de upload.
    """
    fatias = {v: raiz / f"data/processed/t2a_{v}" for v in ("A", "E")}
    mans = {}
    for v, d in fatias.items():
        if not (d / NOME_MANIFESTO).exists():
            raise SystemExit(
                f"{d / NOME_MANIFESTO} não existe. Rode "
                f"scripts/preparar_dados_phienc.py --tokenizer "
                f"data/processed/tokenizer/variante_{v}.json --out {d}")
        mans[v] = json.loads((d / NOME_MANIFESTO).read_text(encoding="utf-8"))

    # ⚠️ As guardas abaixo existem contra UM modo de falha: empacotar a mesma
    # fatia duas vezes sob dois nomes. O experimento sairia empate perfeito, a
    # leitura seria "a §8 não vale nada", e seria um NULO FABRICADO — a mesma
    # família de erro que a régua quebrada do T1b2 produziu duas vezes no artigo.
    # Um nulo convida a fechar a linha, então ele precisa de mais guarda que um
    # positivo, não menos.
    for v in ("A", "E"):
        esperado = f"variante_{v}.json"
        if not mans[v]["tokenizer"].endswith(esperado):
            raise SystemExit(
                f"a fatia {v} foi preparada com {mans[v]['tokenizer']}, não com "
                f"{esperado}. Os dois braços treinariam no mesmo dado e o "
                "experimento reportaria empate por construção.")
    if mans["A"]["tokenizer_sha"] == mans["E"]["tokenizer_sha"]:
        raise SystemExit(
            f"as duas fatias declaram o mesmo tokenizer_sha "
            f"({mans['A']['tokenizer_sha']}). A única variável do experimento não "
            "varia.")
    # O corpus, a semente e o `max_tokens` são o que TEM de ser igual: a variável
    # é o tokenizer, e qualquer outra diferença entraria no resultado sem nome.
    for campo in ("corpus", "semente_do_sorteio", "max_tokens", "em_ordem"):
        if mans["A"].get(campo) != mans["E"].get(campo):
            raise SystemExit(
                f"as fatias divergem em {campo!r}: {mans['A'].get(campo)!r} contra "
                f"{mans['E'].get(campo)!r}. O experimento é de uma variável.")
    # ⚠️ A lista de partes de E tem de ser PREFIXO da de A, e não igual a ela.
    #
    # As duas fatias sorteiam as partes com `random.Random(17).shuffle`, então a
    # ordem é a mesma; o que muda é onde cada uma para. E gasta mais tokens por
    # documento, então enche os 0,9 B antes — o conjunto de documentos de E é um
    # PREFIXO do de A, na mesma ordem. É a forma mais limpa que este experimento
    # pode ter: nenhum documento entra num braço e falta no outro por sorteio.
    #
    # Exigir igualdade aqui seria errado (E pode parar numa parte antes), e não
    # exigir nada deixaria passar duas fatias de corpora embaralhados diferentes —
    # que é o que aconteceria se o diretório do corpus ganhasse ou perdesse uma
    # parte entre um preparo e o outro.
    pa, pe = mans["A"]["partes_usadas"], mans["E"]["partes_usadas"]
    curto, longo = (pe, pa) if len(pe) <= len(pa) else (pa, pe)
    if longo[:len(curto)] != curto:
        raise SystemExit(
            f"as partes não são prefixo uma da outra:\n  A: {pa}\n  E: {pe}\n"
            "As duas sorteiam com a mesma semente, então a ordem devia ser a "
            "mesma. Divergir aqui significa que os corpora não eram o mesmo "
            "conjunto de arquivos, e os braços veriam textos diferentes.")

    # Orçamento igual em TOKENS, que é o protocolo — não em texto. E é justamente
    # porque o texto difere que o experimento tem o que medir.
    ta, te = mans["A"]["tokens"], mans["E"]["tokens"]
    if abs(ta - te) / max(ta, te) > 0.01:
        raise SystemExit(
            f"as fatias têm {ta:,} e {te:,} tokens, {abs(ta-te)/max(ta,te):.1%} de "
            "diferença. O protocolo iguala TOKENS; com orçamentos diferentes o "
            "resultado mede volume, não tokenizer.")

    for v, d in fatias.items():
        _ligar(d / NOME_TOKENS, out / f"tokens_{v}.u16.bin")
        _ligar(d / NOME_MARCAS, out / f"marcas_{v}.u8.bin")
        shutil.copy2(d / NOME_MANIFESTO, out / f"MANIFESTO_{v}.json")
        # O tokenizer vai junto: `exportar_phienc.py` precisa dele para gravar o
        # checkpoint no formato do `transformers`, e o caminho que o manifesto do
        # run nomeia é um caminho DESTA máquina, que não existe no Kaggle.
        shutil.copy2(raiz / f"data/processed/tokenizer/variante_{v}.json",
                     out / f"variante_{v}.json")

    docs = {v: mans[v]["documentos_nesta_execucao"] for v in ("A", "E")}
    por_doc = {v: mans[v]["tokens"] / docs[v] for v in ("A", "E")}
    return {
        "codigo_de": exp.repo or "dataset",
        "fatias": {v: {k: mans[v].get(k) for k in
                       ("tokens", "documentos_nesta_execucao", "tokenizer",
                        "tokenizer_sha", "partes_usadas", "corpus",
                        "semente_do_sorteio", "max_tokens")}
                   for v in ("A", "E")},
        # O efeito sob teste, medido ANTES de qualquer GPU ser gasta. Fica no
        # manifesto para o resultado poder ser lido contra a expectativa em vez de
        # contra uma lembrança.
        "efeito_disponivel": {
            "tokens_por_doc_A": round(por_doc["A"], 1),
            "tokens_por_doc_E": round(por_doc["E"], 1),
            "E_gasta_a_mais_por_doc": round(por_doc["E"] / por_doc["A"] - 1, 4),
            "A_ve_mais_texto": round(1 - por_doc["A"] / por_doc["E"], 4),
            "nota": ("A métrica intrínseca de fertilidade dava +37,7% e o corpus de "
                     "treino dá +12,6% — ela foi medida em resumos do arXiv e "
                     "superestima por 3x. O número a usar como expectativa é este."),
        },
        "nota_do_protocolo": (
            "As duas fatias no mesmo dataset porque os braços treinam em sessões "
            "separadas: uma assinatura só, conferida antes do treino nos dois. "
            "Mascaramento PADRÃO (p_equacao=0,0) em ambos — o tratamento do §2.3 "
            "depende de o tokenizer marcar equações, e ligá-lo mediria a interação "
            "em vez do tokenizer. A comparação roda LOCAL, com os dois checkpoints "
            "no mesmo processo."),
    }


# O t1a15 usa o MESMO montador: o volume é a única diferença, e ele vem
# do `max_pares` do experimento.
# O t1d usa o MESMO montador do t1c: a diferença é o arquivo de negativos, e
# ele vem da identidade do experimento — não de uma bandeira que se esquece.
MONTADORES = {"t1a": _montar_t1a, "t1a15": _montar_t1a,
              "t1a3m": _montar_t1a, "t1a6m": _montar_t1a,
              "t1b2": _montar_t1b2, "t1c": _montar_rerank,
              "t1d": _montar_rerank, "t2a_a": _montar_t2a}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--experimento", default="t1a", choices=sorted(EXPERIMENTOS))
    p.add_argument("--pares", type=Path, default=Path("data/processed/pares"))
    p.add_argument("--out", type=Path, default=None,
                   help="por omissão, o `pacote` declarado pelo experimento")
    p.add_argument("--semente", type=int, default=17,
                   help="sorteio dos pares de treino. Ver `amostrar_do_plano`: "
                        "`head` cobria 10,7x menos documentos")
    # ⚠️ Sem `default`: o volume vem do EXPERIMENTO. Ver `max_pares` em
    # `phifm.core.kaggle` — com um default fixo aqui, montar o `t1a15` sem a
    # bandeira produziria 400 mil pares sob o nome do de 1,5 M.
    p.add_argument("--max-pares", type=int, default=None,
                   help="sobrepõe o volume declarado pelo experimento. O do t1a é "
                        "o volume do campeão do G1.1; o do t1a15 testa se mais "
                        "documentos fecham os 0,033 que faltam para o G1.2")
    # ⚠️ Sem `default`: o arquivo vem do EXPERIMENTO. Ver `negativos` em
    # `phifm.core.kaggle` — com um default fixo aqui, montar o `t1d` sem a
    # bandeira empacotaria os negativos da FUSÃO sob o nome do experimento que
    # existe justamente para testar os do DENSO.
    p.add_argument("--negativos", type=Path, default=None,
                   help="sobrepõe o arquivo declarado pelo experimento; já tem de "
                        "estar sem co-citados")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s",
                        stream=sys.stdout)

    exp = obter(a.experimento)
    if a.max_pares is None:
        a.max_pares = exp.max_pares
    if a.negativos is None and exp.negativos:
        a.negativos = Path(exp.negativos)
    # ⚠️ O volume só é informação para quem SORTEIA pares. O `t1b2` não amostra
    # nada — ele mede a cadeia com modelos prontos —, e imprimir "400.000 pares"
    # ali anunciava um número que não descreve o pacote.
    if exp.nome in VARIANTES_DE_VOLUME:
        logging.info("%s · %s pares · semente %d", exp.nome, f"{a.max_pares:,}",
                     a.semente)
    else:
        logging.info("%s · semente %d", exp.nome, a.semente)
    raiz = Path(__file__).resolve().parents[1]
    out = a.out or (raiz / exp.pacote)
    out.mkdir(parents=True, exist_ok=True)

    extra = MONTADORES[exp.nome](exp, raiz, out, a)

    # ⚠️ O conteudo do pacote e uma lista DECLARADA, e o que nao esta nela sai.
    #
    # Antes o manifesto era montado de `iterdir()`, entao qualquer arquivo obsoleto
    # no diretorio entrava na atestacao e subia para o Kaggle. Aconteceu em
    # 2026-08-24: ao renomear o fonte para `.zip.bin`, o `phifm_src.zip` antigo
    # ficou, e o manifesto passou a descrever OS DOIS — 175 KB de codigo velho
    # atestados como se fizessem parte do pacote.
    #
    # Um manifesto que descreve o que sobrou no disco em vez do que a etapa produziu
    # nao atesta nada.
    poupados = {"MANIFESTO.json", "dataset-metadata.json"}
    for f in sorted(out.iterdir()):
        if f.is_file() and f.name not in exp.arquivos and f.name not in poupados:
            log.warning("removendo arquivo que nao pertence ao pacote: %s", f.name)
            f.unlink()
    arquivos = [out / n for n in exp.arquivos]
    faltando = [f.name for f in arquivos if not f.exists()]
    if faltando:
        raise SystemExit(f"o pacote ficou sem {faltando} — nao vou gravar um "
                         "manifesto que descreve menos do que o notebook exige")
    manifesto = {
        "experimento": exp.nome,
        "git_sha": git_sha_curto(),
        **extra,
        "hash_algo": "blake3",
        "arquivos": {f.name: {"blake3": hash_arquivo(f), "bytes": f.stat().st_size}
                     for f in arquivos},
        "nota": ("Subir como Kaggle Dataset. O notebook confere estes hashes antes "
                 "de treinar: rodar sobre dados que não são estes produziria um "
                 "número incomparável com os medidos aqui."),
    }
    (out / "MANIFESTO.json").write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")

    total = sum(v["bytes"] for v in manifesto["arquivos"].values())
    print()
    print("=" * 68)
    for nome, v in manifesto["arquivos"].items():
        print(f"  {nome:34s} {v['bytes']/1e6:8.1f} MB  {v['blake3'][:12]}…")
    print(f"  {'TOTAL':34s} {total/1e6:8.1f} MB")
    print("=" * 68)
    print(f"  -> {out}")
    # ⚠️ `.get`, e não indexação. O `modulos_python` só existe quando o pacote
    # leva um zip de fonte, e o t1b2 não leva nenhum — o código dele vem do
    # GitHub. Com indexação direta o `KeyError` estourava DEPOIS de gravar o
    # pacote e o manifesto, dando código de saída 1 para um trabalho que deu
    # certo: a pior das combinações, e a mesma que o `reconfigure(encoding)`
    # existe para evitar no fim destes scripts.
    modulos = manifesto.get("modulos_python")
    codigo = (f"{modulos} módulos" if modulos
              else f"código de {manifesto.get('codigo_de', 'dataset')}")
    print(f"  {exp.nome} · git {manifesto['git_sha']} · {codigo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
