#!/usr/bin/env python3
"""Tokeniza o corpus do ΦEnc uma vez, para o laço de treino só ler offsets.

    .venv-treino\\Scripts\\python.exe scripts\\preparar_dados_phienc.py \\
        --max-tokens 2_000_000_000

Produz, em `data/processed/phienc_dados/`:

| arquivo | o quê |
|---|---|
| `tokens.u16.bin` | os ids, 2 bytes cada. V=40.960 cabe em uint16 |
| `marcas.u8.bin` | os 3 bits de equação (ver `phifm.training.pretrain.dados`) |
| `MANIFESTO_DADOS.json` | contagens, proveniência e o hash do tokenizer |

## Por que offline, e não no laço

Tokenizar dentro do treino gastaria CPU competindo com o carregamento, tornaria a
vazão dependente do número de *workers*, e — o que é pior — faria o `(semente,
passo)` do DOC-08 §7.2 deixar de determinar o conteúdo: trocar o tokenizer mudaria
silenciosamente o que o passo 60.000 contém.

## O corpus, e por que este e não o outro

`data/processed/redpajama_fisica/` — 835.379 documentos, 42,15 G caracteres,
**10,54 B tokens**, construído do **fonte LaTeX** do arXiv.

⚠️ **Não o peS2o.** O texto pleno dele tem **0,0%** de ambiente de equação contra
**84,9%** desta fatia: a extração de PDF removeu a matemática (ADR-0002). Treinar o
mascaramento consciente de equações sobre peS2o seria treinar o tratamento sobre um
corpus sem tratamento possível.

## ⚠️ As partes são SORTEADAS, não as primeiras

Medido depois de uma primeira preparação que usou as 8 primeiras de 44 partes:

    partes 0-7  (usadas)      math 32,4%   display 18,8%
    partes 8-43 (NÃO usadas)  math 42,7%   display 29,5%

As partes **não são intercambiáveis**: as que ficaram de fora têm 57% mais equação
em display. Pegar as primeiras é `head()` no nível de arquivo — o mesmo erro que
este repositório já pagou três vezes (o peS2o amostrado no começo e concluído sem
LaTeX, o `val.head(500)` que eram 35 documentos, o `pares_treino` cortado por
posição com 49,6% de vazamento).

O efeito aqui vai **contra** a hipótese — menos equação enfraquece o tratamento —,
mas um viés que ajuda não deixa de ser viés, e a ablação passaria a comparar dois
braços num corpus que não representa o corpus.

O sorteio usa `--semente` e a lista escolhida vai para o manifesto, então continua
determinístico e reproduzível.

## Retomada

O progresso é gravado a cada parte. Uma execução interrompida continua da parte
seguinte, e os binários são abertos em modo de **acréscimo** — mas a retomada só é
válida se o tokenizer for o mesmo, e por isso o manifesto guarda o hash dele e a
retomada **levanta** se ele mudou. Concatenar tokens de dois tokenizers diferentes
produziria um corpus que nenhum modelo pode ler, com a contagem certa.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.core.schema.reprodutibilidade import git_sha_curto  # noqa: E402
from phifm.training.pretrain.dados import (  # noqa: E402
    NOME_MANIFESTO,
    NOME_MARCAS,
    NOME_TOKENS,
    marcas_de,
)
from phifm.training.pretrain.mascaramento import marcar_equacoes  # noqa: E402

log = logging.getLogger("preparar")

CORPUS = Path("data/processed/redpajama_fisica")
TOKENIZER = Path("data/processed/tokenizer/variante_A.json")
SAIDA = Path("data/processed/phienc_dados")
NOME_PROGRESSO = "_progresso.json"

# Ids do tokenizer treinado. Ver `phifm.models.encoder.config.ESPECIAIS`.
#
# ⚠️ Estes são os das NOSSAS variantes, e o default continua sendo eles. O
# `--especiais-do-tokenizer` deriva-os do tokenizer em uso, o que só faz sentido
# para preparar uma fatia destinada a um modelo de fora — ver a função abaixo.
ID_CLS, ID_SEP = 2, 3

# Nomes que os tokenizers costumam usar para o mesmo papel, em ordem de
# preferência. Um tokenizer estilo RoBERTa/ModernBERT chama de `<s>`/`</s>`; um
# estilo BERT, de `[CLS]`/`[SEP]`.
NOMES_INICIO = ("[CLS]", "<s>", "<|endoftext|>")
NOMES_FIM = ("[SEP]", "</s>", "<|endoftext|>")


def especiais_de(tok, derivar: bool) -> tuple[int, int]:
    """`(id_inicio, id_fim)` — os nossos, ou os do tokenizer.

    ⚠️ Com `derivar=False` (o padrão) devolve exatamente `ID_CLS, ID_SEP`. Esse é
    o caminho que produziu os 2,00 B tokens de treino, e ele não muda.

    Com `derivar=True`, procura os papéis no vocabulário do tokenizer. Falha alto
    se não achar: envolver o documento com o id errado poria um token qualquer nas
    bordas de cada sequência, o que a avaliação de MLM leria como texto.
    """
    if not derivar:
        return ID_CLS, ID_SEP
    achados = []
    for nomes, papel in ((NOMES_INICIO, "início"), (NOMES_FIM, "fim")):
        ident = next((tok.token_to_id(n) for n in nomes
                      if tok.token_to_id(n) is not None), None)
        if ident is None:
            raise SystemExit(
                f"--especiais-do-tokenizer: não achei o marcador de {papel} no "
                f"tokenizer (tentei {list(nomes)}). Envolver o documento com o id "
                "errado poria um token qualquer nas bordas de cada sequência.")
        achados.append(ident)
    return achados[0], achados[1]


def _hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()[:16]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=CORPUS)
    p.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    p.add_argument("--out", type=Path, default=SAIDA)
    p.add_argument("--max-tokens", type=int, default=2_000_000_000,
                   help="teto de tokens. O corpus inteiro tem 10,54 B; 2 B já "
                        "sobra para o proxy do bake-off e cabe em ~6 GB de disco")
    p.add_argument("--por-lote", type=int, default=256,
                   help="documentos por `encode_batch`; o Rust do tokenizers "
                        "paraleliza dentro do lote")
    p.add_argument("--recomecar", action="store_true",
                   help="apaga o que existe em --out em vez de retomar")
    p.add_argument("--semente", type=int, default=17,
                   help="sorteio das partes. Com o teto de tokens, só uma fração "
                        "das partes é lida, e pegar as primeiras enviesa — ver a "
                        "docstring")
    # ⚠️ A fatia de AVALIACAO sai daqui, e ela tem de EXCLUIR o que o treino viu.
    #
    # Medir MLM sobre o mesmo fluxo de tokens em que o modelo treinou mede
    # memorizacao. Entre seis variantes do §11.2 a comparacao continuaria
    # "justa" -- todas medidas no proprio texto de treino --, mas ordenaria
    # capacidade de memorizar, e a hipotese do DOC-07 §2.3 e sobre capacidade.
    #
    # A exclusao vem do MANIFESTO do run de treino, nao de uma lista digitada:
    # as partes sao sorteadas, e reproduzir a mao quais foram e onde se erra.
    p.add_argument("--excluir-de", type=Path, default=None, metavar="DIR",
                   help="pula as partes que o run de treino em DIR ja usou; a "
                        "lista sai do MANIFESTO_DADOS.json dele")
    # ⚠️ Só para preparar fatia destinada a um modelo DE FORA. Ver `especiais_de`.
    p.add_argument("--especiais-do-tokenizer", action="store_true",
                   help="deriva os ids de início/fim do tokenizer em uso, em vez "
                        "dos do projeto. Necessário para medir um modelo de "
                        "prateleira; NÃO usar para as variantes do §11.2")
    p.add_argument("--em-ordem", action="store_true",
                   help="lê as partes na ordem do disco em vez de sortear. Só para "
                        "reproduzir uma preparação antiga; enviesa")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8", errors="replace")

    partes = sorted(glob.glob(str(a.corpus / "part-*.parquet")))
    if not partes:
        raise SystemExit(f"nenhum part-*.parquet em {a.corpus}")
    # ⚠️ Sorteia. Com `--max-tokens` abaixo do corpus inteiro, só uma fração das
    # partes é lida, e as partes não são intercambiáveis — ver a docstring.
    if not a.em_ordem:
        random.Random(a.semente).shuffle(partes)
    if not a.tokenizer.exists():
        raise SystemExit(
            f"{a.tokenizer} não existe. Ele sai de `scripts/bakeoff_tokenizer.py`; "
            "a variante A é a do DOC-05 §7.3 (BPE, V=40.960, pré-tokenização §8).")

    excluidas: set[str] = set()
    if a.excluir_de is not None:
        man_treino = a.excluir_de / NOME_MANIFESTO
        if not man_treino.exists():
            raise SystemExit(
                f"--excluir-de {a.excluir_de} não tem {NOME_MANIFESTO}.\n"
                "Sem a lista do run de treino não há como garantir que esta "
                "fatia é disjunta dele, e uma fatia que se sobrepõe ao treino "
                "mede memorização com a cara de generalização.")
        usadas = json.loads(man_treino.read_text(encoding="utf-8")).get(
            "partes_usadas") or []
        if not usadas:
            raise SystemExit(
                f"{man_treino} não lista `partes_usadas`. Prosseguir prepararia "
                "uma fatia que pode se sobrepor inteira ao treino.")
        excluidas = {str(x) for x in usadas}
        antes = len(partes)
        partes = [x for x in partes if Path(x).name not in excluidas]
        log.info("excluindo %d partes do treino em %s: sobram %d de %d",
                 len(excluidas), a.excluir_de, len(partes), antes)
        if not partes:
            raise SystemExit(
                "todas as partes do corpus já foram usadas no treino — não há "
                "fatia disjunta possível neste corpus")

    a.out.mkdir(parents=True, exist_ok=True)
    h_tok = _hash(a.tokenizer)
    prog_p = a.out / NOME_PROGRESSO
    feitas: list[str] = []
    if a.recomecar:
        for n in (NOME_TOKENS, NOME_MARCAS, NOME_MANIFESTO, NOME_PROGRESSO):
            (a.out / n).unlink(missing_ok=True)
    elif prog_p.exists():
        prog = json.loads(prog_p.read_text(encoding="utf-8"))
        # ⚠️ Retomar com outro tokenizer concatenaria dois vocabulários no mesmo
        # arquivo. A contagem continuaria certa e o corpus seria ilegível.
        if prog.get("tokenizer_sha") != h_tok:
            raise SystemExit(
                f"o progresso em {prog_p} foi gravado com o tokenizer "
                f"{prog.get('tokenizer_sha')} e o atual é {h_tok}. Retomar "
                "concatenaria dois vocabulários no mesmo binário — use "
                "--recomecar, ou volte o tokenizer.")
        if (prog.get("semente"), prog.get("em_ordem")) != (a.semente, a.em_ordem):
            raise SystemExit(
                f"o progresso foi gravado com semente={prog.get('semente')} e "
                f"em_ordem={prog.get('em_ordem')}, e agora são {a.semente} e "
                f"{a.em_ordem}. Retomar leria partes DIFERENTES achando que "
                "continua a mesma preparação — use --recomecar, ou volte os "
                "parâmetros.")
        feitas = prog.get("partes_feitas", [])
        log.info("retomando: %d de %d partes já feitas", len(feitas), len(partes))

    tok = Tokenizer.from_file(str(a.tokenizer))
    id_inicio, id_fim = especiais_de(tok, a.especiais_do_tokenizer)
    if a.especiais_do_tokenizer:
        log.info("especiais DO TOKENIZER: início=%d (%s) · fim=%d (%s)",
                 id_inicio, tok.id_to_token(id_inicio),
                 id_fim, tok.id_to_token(id_fim))
    n_tokens = int((a.out / NOME_TOKENS).stat().st_size // 2
                   if (a.out / NOME_TOKENS).exists() else 0)
    n_docs = 0
    t0 = time.perf_counter()

    with open(a.out / NOME_TOKENS, "ab") as ft, open(a.out / NOME_MARCAS, "ab") as fm:
        for parte in partes:
            if Path(parte).name in feitas:
                continue
            if n_tokens >= a.max_tokens:
                log.info("teto de %s tokens atingido", f"{a.max_tokens:,}")
                break
            d = pl.read_parquet(parte, columns=["texto"])
            textos = [t for t in d["texto"].to_list() if t]
            for i in range(0, len(textos), a.por_lote):
                bloco = textos[i:i + a.por_lote]
                for texto, enc in zip(bloco, tok.encode_batch(bloco), strict=True):
                    ids = np.array([id_inicio, *enc.ids, id_fim], dtype=np.uint16)
                    ide, disp = marcar_equacoes(enc.offsets, texto)
                    # Os especiais que envolvem o documento não são matemática.
                    marcas = np.concatenate([
                        np.zeros(1, dtype=np.uint8),
                        marcas_de(ide, disp) if ide.size else np.zeros(0, np.uint8),
                        np.zeros(1, dtype=np.uint8)])
                    ft.write(ids.tobytes())
                    fm.write(marcas.tobytes())
                    n_tokens += ids.size
                    n_docs += 1
                if n_tokens >= a.max_tokens:
                    break
            feitas.append(Path(parte).name)
            dt = time.perf_counter() - t0
            log.info("%s · %s tokens · %s docs · %.0f mil tok/s",
                     Path(parte).name, f"{n_tokens:,}", f"{n_docs:,}",
                     n_tokens / max(dt, 1e-9) / 1e3)
            prog_p.write_text(json.dumps(
                {"partes_feitas": feitas, "tokenizer_sha": h_tok,
                 "tokens": n_tokens, "semente": a.semente,
                 "em_ordem": a.em_ordem}, indent=2), encoding="utf-8")

    manifesto = {
        "etapa": "phienc_dados",
        "git_sha": git_sha_curto(),
        "corpus": str(a.corpus).replace("\\", "/"),
        "tokenizer": str(a.tokenizer).replace("\\", "/"),
        "tokenizer_sha": h_tok,
        "tokens": n_tokens,
        "documentos_nesta_execucao": n_docs,
        "partes_feitas": len(feitas),
        "partes_totais": len(partes),
        # ⚠️ A LISTA, não só a contagem: com sorteio, saber quantas partes entraram
        # não permite reconstruir quais.
        "partes_usadas": list(feitas),
        "semente_do_sorteio": a.semente,
        "em_ordem": a.em_ordem,
        # ⚠️ O que foi EXCLUIDO vai no manifesto. Sem isto, um artefato de
        # avaliação não prova ser disjunto do treino, e "avaliei em dado
        # separado" passa a ser afirmação em vez de fato verificável.
        "excluido_de": (str(a.excluir_de).replace("\\", "/")
                        if a.excluir_de else None),
        "partes_excluidas": sorted(excluidas),
        "disjunto_do_treino": bool(excluidas),
        "max_tokens": a.max_tokens,
        "id_cls": id_inicio, "id_sep": id_fim,
        # ⚠️ TODOS os ids especiais do tokenizer, para o avaliador de MLM saber
        # quais posições não pode mascarar. Ele assumia `id < 5`, que é a
        # convenção das nossas variantes e falsa para qualquer outro tokenizer.
        "ids_especiais": sorted({tok.token_to_id(t)
                                 for t in tok.get_vocab()
                                 if tok.token_to_id(t) is not None
                                 and t in set(NOMES_INICIO) | set(NOMES_FIM)
                                 | {"[PAD]", "[UNK]", "[MASK]", "<pad>", "<unk>",
                                    "<mask>"}}),
        "especiais_do_tokenizer": bool(a.especiais_do_tokenizer),
        "nota": ("Corpus do RedPajama-arXiv, construído do fonte LaTeX (ADR-0002). "
                 "NÃO peS2o: 0,0% de ambiente de equação contra 84,9% desta fatia. "
                 "As partes são SORTEADAS: medido que as 8 primeiras têm 32,4% de "
                 "matemática contra 42,7% das demais, então pegar as primeiras "
                 "enviesaria o corpus contra a hipótese do DOC-07 §2.3."),
    }
    (a.out / NOME_MANIFESTO).write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")

    gb = (a.out / NOME_TOKENS).stat().st_size / 1e9
    print()
    print("=" * 70)
    print(f"  {n_tokens:,} tokens · {len(feitas)}/{len(partes)} partes · "
          f"{gb + gb / 2:.1f} GB")
    print(f"  tokenizer {a.tokenizer.name} · sha {h_tok}")
    print(f"  -> {a.out}")
    print("=" * 70)
    if len(feitas) < len(partes) and n_tokens < a.max_tokens:
        print("  ⚠️ ficou incompleto e o teto não foi atingido — rode de novo "
              "para retomar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
