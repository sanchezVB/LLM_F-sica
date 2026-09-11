#!/usr/bin/env python3
"""Bits por byte em dois ou mais ΦEnc exportados. A medida PRIMÁRIA do T2a.

    .venv-treino\\Scripts\\python.exe scripts\\avaliar_bits_por_byte.py \\
        --modelo "variante A=models/phienc-A" \\
        --modelo "variante E=models/phienc-E" \\
        --excluir-de data/processed/t2a_A \\
        --out data/processed/avaliacao/t2a_bits_por_byte.json

A conta e a leitura estão em `phifm.eval.bits_por_byte`, testadas na suíte rápida.
Aqui só a passagem pelo modelo — e a parte que esta camada tem de acertar sozinha,
que é garantir que os braços leram o MESMO TEXTO.

## ⚠️ Por que não dá para reusar as fatias binárias do treino

As fatias `tokens.u16.bin` já são partições diferentes do corpus: a de A cobre
12% mais texto que a de E no mesmo número de tokens — que é exatamente o efeito
sob teste. Avaliar cada braço na própria fatia mediria os dois em textos
diferentes, e a diferença resultante não teria dono.

Este script parte do TEXTO e tokeniza na hora, uma vez por braço.

## ⚠️ A janela comum, que é a única parte delicada

Truncar por tokens daria a cada braço uma quantidade diferente de texto. Truncar
por caracteres num número fixo pode estourar o contexto do tokenizer pior.

O que se faz aqui: tokenizar o documento em cada braço **com offsets**, ver em que
caractere cada um atinge o limite de contexto, e cortar o texto no MENOR desses
pontos. Todos os braços recebem então o mesmo prefixo de texto, e todos cabem.

O braço de tokens mais longos sobra contexto — vê o mesmo texto em menos posições.
Isso não é artefato: é a vantagem dele, medida.

## ⚠️ Disjunção do treino é conferida, não suposta

`--excluir-de` aponta para a fatia de TREINO e lê `partes_usadas` do manifesto
dela. As partes de avaliação são as que sobraram. Medir sobre texto de treino
mediria memorização, e entre dois braços a comparação pareceria justa — os dois
memorizando — enquanto ordenaria capacidade de memorizar.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

from phifm.core.console import utf8 as console_utf8  # noqa: E402
from phifm.eval.bits_por_byte import Acumulador, confronto  # noqa: E402
from phifm.eval.mlm_regiao import posicoes_mascaradas  # noqa: E402
from phifm.training.embedding import escolher_dispositivo  # noqa: E402
from phifm.training.pretrain.dados import NOME_MANIFESTO  # noqa: E402

# ⚠️ No IMPORT: o argparse imprime `--help` antes de qualquer código nosso, e `Φ`
# não existe em cp1252. Ver `phifm.core.console`.
console_utf8()

log = logging.getLogger("bits-por-byte")


def especiais_do_modelo(tok) -> tuple[int, list[int]]:
    """`(id_mascara, ids_especiais)` do tokenizer DO MODELO, não de constantes.

    As variantes têm `[MASK]` em 4; o ModernBERT em 50.284. Supor a convenção da
    casa mascararia com um token qualquer, e a perda seria de outra tarefa.
    """
    if tok.mask_token_id is None:
        raise SystemExit(
            f"o tokenizer de {tok.name_or_path} não declara `mask_token`. Sem ele "
            "não há como mascarar, e adivinhar o id mascararia com outro token.")
    especiais = [i for i in tok.all_special_ids if i is not None]
    return int(tok.mask_token_id), sorted({int(i) for i in especiais})


def partes_reservadas(corpus: Path, treino: Path | None) -> list[Path]:
    """As partes que o treino NÃO usou. Ver o § sobre disjunção."""
    todas = sorted(corpus.glob("part-*.parquet"))
    if not todas:
        raise SystemExit(f"nenhum part-*.parquet em {corpus}")
    if treino is None:
        log.warning("sem --excluir-de: NÃO posso afirmar disjunção do treino")
        return todas
    man_p = treino / NOME_MANIFESTO
    if not man_p.exists():
        raise SystemExit(
            f"{man_p} não existe. --excluir-de tem de apontar para a fatia de "
            "TREINO, que é quem sabe quais partes foram consumidas.")
    usadas = set(json.loads(man_p.read_text(encoding="utf-8"))["partes_usadas"])
    sobra = [p for p in todas if p.name not in usadas]
    if not sobra:
        raise SystemExit(
            f"o treino consumiu todas as {len(todas)} partes; não há dado "
            "disjunto para avaliar.")
    log.info("%d partes reservadas de %d (%d no treino)",
             len(sobra), len(todas), len(usadas))
    return sobra


def sortear_textos(partes: list[Path], quantos: int, semente: int,
                   min_caracteres: int) -> list[str]:
    """Documentos das partes reservadas, determinístico em `semente`.

    ⚠️ Descarta documentos curtos ANTES de sortear e não depois: um documento de
    200 caracteres esconde poucos bytes e entra no denominador com ruído alto, e
    um filtro aplicado depois do sorteio mudaria o tamanho da amostra sem avisar.
    """
    rng = np.random.default_rng(semente)
    ordem = rng.permutation(len(partes))
    escolhidos: list[str] = []
    for i in ordem:
        d = pl.read_parquet(partes[int(i)], columns=["texto"])
        textos = [t for t in d["texto"].to_list()
                  if t and len(t) >= min_caracteres]
        if not textos:
            continue
        # Sorteio DENTRO da parte também, senão os documentos viriam todos do
        # começo do arquivo — e as partes têm ordem de ingestão, não aleatória.
        idx = rng.permutation(len(textos))[:quantos - len(escolhidos)]
        escolhidos.extend(textos[int(j)] for j in idx)
        if len(escolhidos) >= quantos:
            break
    if len(escolhidos) < quantos:
        log.warning("só achei %d documentos com ≥%d caracteres (pedi %d)",
                    len(escolhidos), min_caracteres, quantos)
    return escolhidos[:quantos]


def janela_comum(texto: str, tokenizers: list, contexto: int,
                 teto_caracteres: int) -> str:
    """O maior prefixo de `texto` que cabe em `contexto` tokens em TODOS.

    Devolve o texto cortado. Ver o § sobre a janela comum: é isto que garante que
    os braços leem o mesmo conteúdo, em vez de a mesma contagem de tokens.
    """
    # Teto de caracteres primeiro, só para não tokenizar um documento de 2 MB
    # inteiro quando 1.024 tokens cabem nos primeiros milhares de caracteres.
    bruto = texto[:teto_caracteres]
    # `contexto - 2` deixa lugar para os dois especiais que o tokenizer insere.
    util = contexto - 2
    corte = len(bruto)
    for tok in tokenizers:
        enc = tok(bruto, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        if len(offs) > util:
            # O fim do último token que ainda cabe.
            corte = min(corte, int(offs[util - 1][1]))
    return bruto[:corte]


def medir(rotulo: str, caminho: str, textos: list[str], dev, contexto: int,
          fracao: float, semente: int, tokenizers_todos: list,
          teto_caracteres: int) -> tuple[Acumulador, dict]:
    tok = AutoTokenizer.from_pretrained(caminho)
    mod = AutoModelForMaskedLM.from_pretrained(
        caminho, attn_implementation="eager").to(dev).eval()
    id_mask, especiais = especiais_do_modelo(tok)
    acum = Acumulador()
    t0 = time.perf_counter()
    n_truncados = 0
    with torch.no_grad():
        for i, texto_bruto in enumerate(textos):
            texto = janela_comum(texto_bruto, tokenizers_todos, contexto,
                                 teto_caracteres)
            enc = tok(texto, return_offsets_mapping=True,
                      truncation=True, max_length=contexto)
            ids = np.asarray(enc["input_ids"], dtype=np.int64)
            offs = enc["offset_mapping"]
            if len(ids) >= contexto:
                n_truncados += 1
            proibidas = np.flatnonzero(np.isin(ids, especiais))
            pos = posicoes_mascaradas(len(ids), semente, i, fracao, proibidas)
            if pos.size == 0:
                continue
            entrada = ids.copy()
            entrada[pos] = id_mask
            b = {"input_ids": torch.tensor(entrada, device=dev).unsqueeze(0),
                 "attention_mask": torch.ones(1, len(ids), dtype=torch.long,
                                              device=dev)}
            logits = mod(**b).logits[0]
            lp = F.log_softmax(logits[torch.tensor(pos, device=dev)].float(),
                               dim=-1)
            alvos = torch.tensor(ids[pos], device=dev)
            log_probs = lp.gather(1, alvos.unsqueeze(1)).squeeze(1).cpu().numpy()
            certo = (lp.argmax(dim=-1) == alvos).cpu().numpy()
            # ⚠️ Os bytes que cada token mascarado cobria, do OFFSET e não de uma
            # média: a média já é a quantidade que se quer comparar, e usá-la no
            # denominador esconderia a diferença dentro do próprio denominador.
            bytes_tok = np.array(
                [len(texto[a:b_].encode("utf-8")) for a, b_ in
                 (offs[int(p)] for p in pos)], dtype=np.int64)
            acum.somar(log_probs, bytes_tok, certo)
            if (i + 1) % 50 == 0:
                log.info("  %s: %d/%d documentos", rotulo, i + 1, len(textos))
    del mod
    d = acum.como_dict()
    d["caminho"] = caminho
    d["segundos"] = round(time.perf_counter() - t0, 1)
    d["documentos_no_teto_de_contexto"] = n_truncados
    d["id_mascara"] = id_mask
    d["n_ids_especiais"] = len(especiais)
    return acum, d


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", action="append", default=[], required=True,
                   metavar="ROTULO=CAMINHO", help="repetível; mínimo dois")
    p.add_argument("--corpus", type=Path,
                   default=Path("data/processed/redpajama_fisica"))
    # ⚠️ `str` e não `Path`: os rótulos podem apontar para ids do Hub, e
    # `str(Path("org/modelo"))` vira `org\\modelo` no Windows, que o
    # `from_pretrained` recusa. Aconteceu duas vezes neste repositório.
    p.add_argument("--excluir-de", type=Path, default=None,
                   help="a fatia de TREINO; as partes dela saem da avaliação")
    p.add_argument("--n-documentos", type=int, default=500)
    p.add_argument("--contexto", type=int, default=1024)
    p.add_argument("--fracao-mascara", type=float, default=0.15)
    p.add_argument("--min-caracteres", type=int, default=2000)
    p.add_argument("--teto-caracteres", type=int, default=20_000,
                   help="corte bruto antes de tokenizar; só para não gastar "
                        "tempo num documento de 2 MB")
    p.add_argument("--semente", type=int, default=17)
    p.add_argument("--dispositivo", default="auto")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    if len(a.modelo) < 2:
        raise SystemExit(
            "bits por byte é uma COMPARAÇÃO: com um modelo só o número não tem "
            "escala. Passe --modelo pelo menos duas vezes.")
    specs = []
    for s in a.modelo:
        rotulo, _, caminho = s.partition("=")
        if not caminho:
            raise SystemExit(f"--modelo espera ROTULO=CAMINHO, recebi {s!r}")
        specs.append((rotulo, caminho))

    partes = partes_reservadas(a.corpus, a.excluir_de)
    textos = sortear_textos(partes, a.n_documentos, a.semente, a.min_caracteres)
    if not textos:
        raise SystemExit("nenhum documento sorteado")
    log.info("%d documentos, contexto %d, máscara %.0f%%",
             len(textos), a.contexto, 100 * a.fracao_mascara)

    # ⚠️ TODOS os tokenizers carregados antes de medir QUALQUER braço: a janela
    # comum depende de todos eles, e calculá-la por braço daria janelas
    # diferentes — que é exatamente o defeito que ela existe para evitar.
    tokenizers = [AutoTokenizer.from_pretrained(c) for _, c in specs]
    dev = escolher_dispositivo(a.dispositivo)

    acums, resultados = {}, {}
    for rotulo, caminho in specs:
        log.info("medindo %s (%s)", rotulo, caminho)
        acums[rotulo], resultados[rotulo] = medir(
            rotulo, caminho, textos, dev, a.contexto, a.fracao_mascara,
            a.semente, tokenizers, a.teto_caracteres)
        d = resultados[rotulo]
        log.info("%-24s %.5f bits/byte · %d bytes escondidos · %.1f s",
                 rotulo, d["bits_por_byte"], d["bytes_escondidos"],
                 d["segundos"])

    confrontos = []
    rotulos = [r for r, _ in specs]
    for i in range(len(rotulos)):
        for j in range(i + 1, len(rotulos)):
            confrontos.append(confronto(rotulos[i], acums[rotulos[i]],
                                        rotulos[j], acums[rotulos[j]],
                                        semente=a.semente))

    artefato = {
        "protocolo": {
            "documentos": len(textos), "contexto": a.contexto,
            "fracao_mascara": a.fracao_mascara, "semente": a.semente,
            "min_caracteres": a.min_caracteres,
            "corpus": str(a.corpus).replace("\\", "/"),
            "excluido_de": (str(a.excluir_de).replace("\\", "/")
                            if a.excluir_de else None),
            "disjunto_do_treino": a.excluir_de is not None,
            "partes_reservadas": len(partes),
            "dispositivo": str(dev),
        },
        "modelos": resultados,
        "confrontos": confrontos,
        "como_ler": (
            "MENOR é melhor: bits por byte é custo de reconstruir o texto. O "
            "denominador é TEXTO, então a conta vale entre vocabulários "
            "diferentes — que acurácia de MLM não vale. ⚠️ O teste continua "
            "CONSERVADOR para o tokenizer de tokens longos: ele esconde pedaços "
            "maiores por máscara. Vitória dele é robusta; empate e derrota são "
            "ambíguos e pedem a pseudo-verossimilhança canônica antes de virar "
            "conclusão."),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(artefato, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    print()
    print("=" * 78)
    print(f"  BITS POR BYTE · {len(textos)} documentos · contexto {a.contexto}")
    print(f"  {'(menor é melhor)':<26} {'bits/byte':>11} {'bytes':>12} "
          f"{'acurácia*':>10}")
    for rotulo in rotulos:
        d = resultados[rotulo]
        print(f"  {rotulo:<26} {d['bits_por_byte']:>11.5f} "
              f"{d['bytes_escondidos']:>12,} {d['acuracia_diagnostica']:>10.4f}")
    print("  * acurácia é DIAGNÓSTICO: ela não compara tokenizers.")
    print()
    for c in confrontos:
        veredicto = (f"{c['vencedor']} vence" if c["vencedor"]
                     else "EMPATE (o IC cruza zero)")
        print(f"  {c['braços']['a']} × {c['braços']['b']}: "
              f"Δ {c['diferenca_media']:+.5f} "
              f"[{c['ic95'][0]:+.5f}, {c['ic95'][1]:+.5f}] → {veredicto}")
    print("=" * 78)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
