#!/usr/bin/env python3
"""DOC-05 §11.1 — as métricas intrínsecas do bake-off de tokenizers do ΦEnc.

    .venv-treino/Scripts/python.exe scripts/bakeoff_tokenizer.py --corpus-docs 20000
    .venv-treino/Scripts/python.exe scripts/bakeoff_tokenizer.py

## O que este script é, e o que ele NÃO é

O DOC-05 §11.2 diz que a única medida que decide é treinar um encoder de ~50 M em
5 B tokens por variante e comparar a jusante — 6 variantes, ~US$ 15. Isso exige o
código de pré-treino, que não existe, e GPU.

Este script faz o §11.1: as métricas **intrínsecas**, que o próprio documento chama
de *proxies*. Elas custam zero, rodam na CPU, e respondem antes de qualquer GPU:

  · a variante E (sem as regras de pré-tokenização do §8) perde de fato?
  · Unigram bate BPE em LaTeX, como Bostrom & Durrett (2020) sugerem para
    linguagem natural?
  · V=40.960 é melhor que 32k e 64k, ou a escolha é indiferente?

⚠️ Proxy não é veredito. Nenhum número daqui autoriza congelar o tokenizer sem o
bake-off do §11.2.

## ⚠️ O corpus: resumos do arXiv, NÃO o peS2o

Medido em 2026-08-27 (`scripts/medir_equacoes_mutiladas.py`): 50,2% dos documentos
de texto pleno do peS2o têm operador matemático órfão — as equações foram removidas
na extração de PDF. Treinar um tokenizer cuja razão de existir é LaTeX sobre texto
sem LaTeX produziria um tokenizer ruim E um bake-off enganoso.

Os resumos do arXiv têm matemática íntegra (23,2% com sequência de controle, 3,0%
de órfãos) e já estão em mão. São curtos, mas tokenizer precisa de muito menos dado
que modelo.

## As variantes (DOC-05 §11.2)

    A  BPE      V=40.960  dígito único  regras do §8
    B  Unigram  V=40.960  dígito único  regras do §8
    C  BPE      V=32.768  dígito único  regras do §8
    D  BPE      V=65.536  dígito único  regras do §8
    E  BPE      V=40.960  dígito único  SEM as regras do §8   <- a mais importante
    F  Qwen3 sem modificação                                  <- controle

E é a mais importante porque isola o regex de pré-tokenização, que a §8 afirma ser
a decisão mais consequente e menos visível. **Se E empatar com A, a §8 está errada
e o documento precisa ser revisado.**
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import re
import sys
from pathlib import Path

import polars as pl

log = logging.getLogger("bakeoff")

# ── O regex do §8, com precedência máxima ───────────────────────────────────
# A ordem importa: `\begin{...}` antes de `\[a-zA-Z]+` para que a forma composta
# ganhe, senão `\begin` casaria primeiro e `{equation}` ficaria solto.
PRE_TOK_LATEX = (
    r"\\begin\{[a-zA-Z*]+\}"      # abertura de ambiente
    r"|\\end\{[a-zA-Z*]+\}"       # fechamento
    r"|\\[a-zA-Z]+\*?"            # sequência de controle atômica
    r"|\\\\"                      # quebra de linha em matriz/align
    r"|\\[,;!]"                   # espaçamento matemático
    r"|\^\{|_\{"                  # marcadores estruturais de índice
)

# §5.3: dígito único. Fica FORA das regras do §8 de propósito — é decisão da §5, e
# a variante E precisa isolar só o §8.
DIGITO = r"\d"

ESPECIAIS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

VARIANTES = {
    "A": {"algo": "bpe", "vocab": 40_960, "pre_tok_8": True},
    "B": {"algo": "unigram", "vocab": 40_960, "pre_tok_8": True},
    "C": {"algo": "bpe", "vocab": 32_768, "pre_tok_8": True},
    "D": {"algo": "bpe", "vocab": 65_536, "pre_tok_8": True},
    "E": {"algo": "bpe", "vocab": 40_960, "pre_tok_8": False},
}

# Vãos matemáticos, para a fertilidade em equações do §11.1
MATH_INLINE = re.compile(r"\$([^$]{2,400})\$")
MATH_AMB = re.compile(r"\\begin\{(equation|align|eqnarray)\*?\}(.{2,2000}?)"
                      r"\\end\{\1\*?\}", re.S)
PALAVRA = re.compile(r"\S+")


# ═══════════════════════════════════════════════════════════════════════════
# corpus
# ═══════════════════════════════════════════════════════════════════════════

def carregar_corpus(raiz: Path, n_treino: int, n_aval: int,
                    semente: int) -> tuple[list[str], list[str], list[str]]:
    """(textos de treino, textos de avaliação, subáreas da avaliação).

    ⚠️ Divisão por DOCUMENTO e reservada ANTES do treino do tokenizer. Medir
    fertilidade no texto em que o BPE aprendeu os merges mediria memorização.
    """
    spine = raiz / "data/processed/spine.parquet"
    if not spine.exists():
        raise SystemExit(f"{spine} não existe — é a fonte dos resumos do arXiv")

    d = (pl.scan_parquet(spine)
         .select(["arxiv_id", "title", "abstract", "primary_category"])
         .filter(pl.col("abstract").is_not_null()
                 & (pl.col("abstract").str.len_chars() > 200))
         .collect(engine="streaming"))
    log.info("spine: %s resumos com mais de 200 caracteres", f"{len(d):,}")

    d = d.sample(n=min(n_treino + n_aval, len(d)), seed=semente)
    textos = (d["title"].fill_null("") + ". " + d["abstract"]).to_list()
    subareas = d["primary_category"].fill_null("?").to_list()

    corte = min(n_treino, len(textos) - 1)
    return textos[:corte], textos[corte:], subareas[corte:]


# ═══════════════════════════════════════════════════════════════════════════
# treino das variantes
# ═══════════════════════════════════════════════════════════════════════════

def construir(algo: str, vocab: int, pre_tok_8: bool):
    """Monta um tokenizer não treinado com a pré-tokenização da variante."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

    modelo = models.BPE(unk_token=None) if algo == "bpe" else models.Unigram()
    tok = Tokenizer(modelo)

    passos = []
    if pre_tok_8:
        # `isolated`: o trecho casado vira pré-token PRÓPRIO, e o BPE não pode
        # atravessar a fronteira. É exatamente o que a §8 pede para `\frac`.
        passos.append(pre_tokenizers.Split(Regex(PRE_TOK_LATEX), "isolated"))
    passos.append(pre_tokenizers.Split(Regex(DIGITO), "isolated"))
    # ⚠️ `use_regex=False`. Com `True`, o ByteLevel aplica o regex do GPT-2 SOBRE os
    # pré-tokens que os passos acima isolaram, e parte `\frac` em `\` + `frac` —
    # desfazendo a atomicidade que é o ponto inteiro da §8. Com `False` ele só faz
    # o mapeamento de bytes, que é o byte-fallback obrigatório da §3.3.
    passos.append(pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False))
    if not pre_tok_8:
        # A variante E não tem as regras do §8, mas precisa de ALGUMA segmentação
        # geral, senão o BPE recebe o documento inteiro como um pré-token. O regex
        # do GPT-2 é a escolha honesta: é o que um tokenizer genérico faria.
        passos = [pre_tokenizers.Split(Regex(DIGITO), "isolated"),
                  pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)]
    tok.pre_tokenizer = pre_tokenizers.Sequence(passos)
    tok.decoder = decoders.ByteLevel()
    return tok, tok.pre_tokenizer


def treinar(tok, textos: list[str], algo: str, vocab: int):
    from tokenizers import pre_tokenizers, trainers

    alfabeto = pre_tokenizers.ByteLevel.alphabet()
    if algo == "bpe":
        tr = trainers.BpeTrainer(vocab_size=vocab, special_tokens=ESPECIAIS,
                                 initial_alphabet=alfabeto, show_progress=False)
    else:
        tr = trainers.UnigramTrainer(vocab_size=vocab, special_tokens=ESPECIAIS,
                                     initial_alphabet=alfabeto, unk_token="[UNK]",
                                     show_progress=False)
    tok.train_from_iterator(textos, trainer=tr, length=len(textos))
    return tok


CASOS_LATEX = {
    "\\frac": r"\frac{d^2x}{dt^2}",
    "\\partial": r"\partial_\mu F^{\mu\nu} = 0",
    "\\begin{equation}": r"\begin{equation} E = mc^2 \end{equation}",
}


def conferir_fronteira(pre_tok, nome: str, deve_isolar: bool) -> dict:
    """O PRÉ-TOKENIZADOR isola a sequência de controle? É a propriedade da §8.

    ⚠️ Isto mede a FRONTEIRA, não o número de tokens. A primeira versão desta
    conferência exigia que `\\frac` fosse UM token depois do BPE, e derrubou o
    bake-off num teste de fumaça de 4.000 documentos — corretamente, mas medindo a
    coisa errada.

    A §8 afirma que a pré-tokenização é uma **barreira dura**: o BPE não pode fundir
    `\\frac` com o texto vizinho, nem partir a sequência ao meio para juntá-la a
    outra coisa. Isso é diferente de `\\frac` sair como token único — o BPE ainda
    segmenta DENTRO de um pré-token quando não aprendeu o merge, e com corpus pequeno
    `\\begin{equation}` é raro demais para ser aprendido.

    Confundir as duas fez a verificação falhar por uma razão que não era defeito.
    Quantos merges o treino de fato aprendeu é métrica de qualidade, não invariante —
    e vai em `cobertura_1_token`.
    """
    achados = {}
    for alvo, texto in CASOS_LATEX.items():
        pecas = [p for p, _ in pre_tok.pre_tokenize_str(texto)]
        achados[alvo] = alvo in pecas
    if deve_isolar and not all(achados.values()):
        faltando = [k for k, v in achados.items() if not v]
        exemplo = [p for p, _ in pre_tok.pre_tokenize_str(CASOS_LATEX[faltando[0]])]
        raise SystemExit(
            f"variante {nome} deveria ter as regras do §8, mas o PRÉ-TOKENIZADOR "
            f"não isolou {faltando}. Sem a fronteira a variante é indistinguível "
            f"da E, e o bake-off compararia A com A.\n  pré-tokens: {exemplo[:10]}")
    return achados


def cobertura_1_token(tok) -> dict:
    """Quantas das sequências do §4.2 o treino aprendeu como token ÚNICO.

    Métrica de qualidade, não invariante: depende do tamanho do corpus. É o que
    diz se o orçamento de ~2.000 tokens LaTeX do §7 está sendo de fato ocupado.
    """
    fora = {}
    for alvo, texto in CASOS_LATEX.items():
        limpos = [p.replace("Ġ", "").replace("Ċ", "")
                  for p in tok.encode(texto).tokens]
        fora[alvo] = alvo in limpos
    return fora


# ═══════════════════════════════════════════════════════════════════════════
# métricas do §11.1
# ═══════════════════════════════════════════════════════════════════════════

def medir(tok, textos: list[str], subareas: list[str], rotulo: str) -> dict:
    n_tokens = n_palavras = n_bytes = 0
    falhas_roundtrip = 0
    tokens_por_subarea: dict[str, list[int]] = {}

    for texto, sub in zip(textos, subareas, strict=True):
        cod = tok.encode(texto)
        t = len(cod.tokens)
        n_tokens += t
        n_palavras += len(PALAVRA.findall(texto))
        n_bytes += len(texto.encode("utf-8"))
        p = len(PALAVRA.findall(texto))
        if p:
            tokens_por_subarea.setdefault(sub, []).append(round(t / p, 4))
        # §11.1: fidelidade de round-trip = 100%, regressão é bug crítico.
        if tok.decode(cod.ids) != texto:
            falhas_roundtrip += 1

    # fertilidade em EQUAÇÕES: só os vãos matemáticos
    eq_tokens = eq_n = 0
    for texto in textos:
        vaos = [m.group(1) for m in MATH_INLINE.finditer(texto)]
        vaos += [m.group(2) for m in MATH_AMB.finditer(texto)]
        for v in vaos:
            eq_tokens += len(tok.encode(v).tokens)
            eq_n += 1

    # consistência numérica (§5.3): `1.5` igual em todo contexto
    contextos = ["1.5", "a 1.5 b", "(1.5)", "T=1.5K", "x 1.5, y", "$1.5$"]
    segs = set()
    for c in contextos:
        pecas = [p.replace("Ġ", "") for p in tok.encode(c).tokens]
        try:
            i = pecas.index("1")
            segs.add(tuple(pecas[i:i + 3]))
        except ValueError:
            segs.add(("AUSENTE",))

    medianas = {s: sorted(v)[len(v) // 2] for s, v in tokens_por_subarea.items()
                if len(v) >= 30}
    med_global = sorted(medianas.values())[len(medianas) // 2] if medianas else 0.0
    pior_sub, pior_val = ("—", 0.0)
    if medianas and med_global:
        pior_sub = max(medianas, key=lambda s: medianas[s])
        pior_val = medianas[pior_sub] / med_global

    return {
        "variante": rotulo,
        "vocab_real": tok.get_vocab_size(),
        "fertilidade": round(n_tokens / max(n_palavras, 1), 4),
        "fertilidade_equacoes": round(eq_tokens / max(eq_n, 1), 2),
        "equacoes_medidas": eq_n,
        "bytes_por_token": round(n_bytes / max(n_tokens, 1), 3),
        "roundtrip_perfeito": falhas_roundtrip == 0,
        "falhas_roundtrip": falhas_roundtrip,
        "numero_1_5_consistente": len(segs) == 1,
        "segmentacoes_de_1_5": sorted("|".join(s) for s in segs),
        "subareas_medidas": len(medianas),
        "pior_subarea": pior_sub,
        "pior_subarea_razao": round(pior_val, 3),
    }


def carregar_qwen3():
    """Controle F. Sem ele a comparação é só entre nossas variantes."""
    from transformers import AutoTokenizer

    for nome in ("Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B"):
        try:
            hf = AutoTokenizer.from_pretrained(nome)
            log.info("controle F: %s (%s tokens)", nome, f"{hf.vocab_size:,}")
            return hf, nome
        except Exception as exc:
            log.warning("%s indisponível (%s)", nome, type(exc).__name__)
    return None, None


class EnvelopeHF:
    """Dá ao tokenizer do HF a mesma interface do `tokenizers` puro."""

    def __init__(self, hf):
        self.hf = hf

    def encode(self, texto):
        ids = self.hf.encode(texto, add_special_tokens=False)
        return type("C", (), {"ids": ids,
                              "tokens": self.hf.convert_ids_to_tokens(ids)})()

    def decode(self, ids):
        return self.hf.decode(ids, skip_special_tokens=True)

    def get_vocab_size(self):
        return self.hf.vocab_size


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raiz", type=Path, default=Path("."))
    p.add_argument("--corpus-docs", type=int, default=200_000)
    p.add_argument("--aval-docs", type=int, default=5_000)
    p.add_argument("--variantes", default="A,B,C,D,E")
    p.add_argument("--semente", type=int, default=17)
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/tokenizer/bakeoff.json"))
    p.add_argument("--guardar-tokenizers", action="store_true",
                   help="grava cada variante em data/processed/tokenizer/<X>.json")
    a = p.parse_args()

    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    treino, aval, subs = carregar_corpus(a.raiz, a.corpus_docs, a.aval_docs,
                                         a.semente)
    log.info("corpus: %s documentos de treino · %s reservados para medir",
             f"{len(treino):,}", f"{len(aval):,}")
    chars_eq = sum(1 for t in aval if MATH_INLINE.search(t) or MATH_AMB.search(t))
    log.info("dos reservados, %s (%.1f%%) têm vão matemático",
             f"{chars_eq:,}", 100 * chars_eq / max(len(aval), 1))

    resultados = []
    atomicidade = {}
    for nome in a.variantes.split(","):
        nome = nome.strip().upper()
        if nome not in VARIANTES:
            log.warning("variante %r desconhecida — ignorada", nome)
            continue
        cfg = VARIANTES[nome]
        log.info("treinando %s: %s V=%s §8=%s", nome, cfg["algo"],
                 f"{cfg['vocab']:,}", cfg["pre_tok_8"])
        tok, pre = construir(cfg["algo"], cfg["vocab"], cfg["pre_tok_8"])
        # A fronteira e conferida ANTES do treino: ela e propriedade da
        # pre-tokenizacao, e treinar nao a cria nem a destroi.
        atomicidade[nome] = conferir_fronteira(pre, nome, cfg["pre_tok_8"])
        tok = treinar(tok, treino, cfg["algo"], cfg["vocab"])
        r = medir(tok, aval, subs, nome)
        r.update({k: cfg[k] for k in ("algo", "vocab", "pre_tok_8")})
        r["latex_como_1_token"] = cobertura_1_token(tok)
        resultados.append(r)
        log.info("  fertilidade %.4f · equações %.2f tok/eq · %.3f bytes/token",
                 r["fertilidade"], r["fertilidade_equacoes"], r["bytes_por_token"])
        if a.guardar_tokenizers:
            destino = a.out.parent / f"variante_{nome}.json"
            destino.parent.mkdir(parents=True, exist_ok=True)
            tok.save(str(destino))

    hf, nome_hf = carregar_qwen3()
    if hf is not None:
        r = medir(EnvelopeHF(hf), aval, subs, "F (Qwen3, controle)")
        r.update({"algo": "bpe", "vocab": hf.vocab_size, "pre_tok_8": False,
                  "modelo": nome_hf})
        r["latex_como_1_token"] = cobertura_1_token(EnvelopeHF(hf))
        resultados.append(r)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "escopo": ("DOC-05 §11.1 — métricas INTRÍNSECAS. O §11.2 (treinar 50M em "
                   "5B tokens por variante) é o que decide, e exige GPU e o código "
                   "de pré-treino que não existe."),
        "corpus": ("resumos do arXiv (spine). NÃO peS2o: 50,2% do texto pleno dele "
                   "tem operador matemático órfão — as equações foram removidas na "
                   "extração de PDF (scripts/medir_equacoes_mutiladas.py)"),
        "documentos_treino": len(treino),
        "documentos_avaliacao": len(aval),
        "atomicidade_latex": atomicidade,
        "resultados": resultados,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 92)
    print(f"  BAKE-OFF §11.1 · {len(treino):,} documentos de treino · "
          f"{len(aval):,} reservados")
    print("=" * 92)
    print(f"  {'var':<22} {'algo':<8} {'vocab':>7} {'§8':>4} {'fert':>7} "
          f"{'tok/eq':>7} {'B/tok':>7} {'RT':>4} {'1.5':>4}")
    for r in resultados:
        print(f"  {r['variante']:<22} {r['algo']:<8} {r['vocab']:>7,} "
              f"{'sim' if r['pre_tok_8'] else 'não':>4} {r['fertilidade']:>7.4f} "
              f"{r['fertilidade_equacoes']:>7.2f} {r['bytes_por_token']:>7.3f} "
              f"{'ok' if r['roundtrip_perfeito'] else 'FALHA':>4} "
              f"{'ok' if r['numero_1_5_consistente'] else 'NÃO':>4}")
    print("=" * 92)
    print("  fert = tokens/palavra (menor é melhor) · tok/eq = tokens por vão")
    print("  matemático · B/tok = bytes por token (maior é melhor) · RT = round-trip")
    print()
    a_r = next((r for r in resultados if r["variante"] == "A"), None)
    e_r = next((r for r in resultados if r["variante"] == "E"), None)
    if a_r and e_r:
        d_eq = e_r["fertilidade_equacoes"] - a_r["fertilidade_equacoes"]
        print("  ⚠️ A CONTRA E — o teste que a §8 precisa passar:")
        print(f"     tokens por equação: A={a_r['fertilidade_equacoes']:.2f} "
              f"E={e_r['fertilidade_equacoes']:.2f} (diferença {d_eq:+.2f})")
        if abs(d_eq) < 0.5:
            print("     EMPATE. A §8 afirma ser a decisão mais consequente do")
            print("     tokenizer; se as regras dela não mudam a fertilidade em")
            print("     equações, o documento precisa ser revisado.")
        else:
            print(f"     As regras do §8 economizam {d_eq:.2f} tokens por equação.")
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
