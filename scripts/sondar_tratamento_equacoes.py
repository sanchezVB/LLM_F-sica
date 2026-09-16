"""O tratamento do DOC-07 §2.3 ACONTECE numa fatia, a um contexto dado?

    PYTHONPATH=src python scripts/sondar_tratamento_equacoes.py \\
        --fatia E=data/processed/t2a_E --fatia A=data/processed/t2a_A --contexto 1024

Roda a seleção do treino — `mascarar` com `p_equacao=1` — sobre janelas sorteadas da
fatia, sem modelo e sem GPU, e responde o que tem de estar respondido ANTES de gastar
cota no braço tratado:

- **fração tratada**: dos exemplos sorteados, quantos de fato mascararam uma equação
  inteira. `train_phienc.py` avisa abaixo de 0,5, porque aí a ablação compara
  aleatório com aleatório;
- **por que recaiu**: sem display na janela, só equações curtas, só grandes demais;
- **o tamanho da equação escolhida** em tokens E em caracteres — em caracteres porque
  é o único jeito de comparar tokenizers diferentes;
- ⚠️ **quantas escolhidas estão cortadas no FIM da janela.** `desempacotar` descarta a
  equação que a janela corta no começo ("truncada, não pode ser tratada como
  inteira"), mas a cortada no fim mantém o id e entra como se estivesse inteira.

## Por que existe (2026-09-15)

A célula do T2a avisava que, com o tokenizer E, `\\frac` fica estilhaçado e o
mascaramento por span "se comportaria de outro jeito". Com E vencendo o bake-off, o
braço tratado passa a usar E, e a pergunta tinha de ser respondida antes das 8,5 h de
T4. A calibração do módulo de mascaramento foi feita a 8.192 de contexto; o proxy
treina a 1.024, e isso muda a fração tratada mais do que o tokenizer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8  # noqa: E402

utf8()

from tokenizers import Tokenizer  # noqa: E402

from phifm.training.pretrain.dados import ConfigDados, Fluxo  # noqa: E402
from phifm.training.pretrain.mascaramento import (  # noqa: E402
    MIN_TOKENS_TRATAMENTO,
    ConfigMascara,
    Contadores,
    _escolher_equacao,
    mascarar,
)


def sondar(raiz: Path, contexto: int, n: int, taxa: float, semente: int) -> dict:
    import json

    man = json.loads((raiz / "MANIFESTO_DADOS.json").read_text(encoding="utf-8"))
    tok = Tokenizer.from_file(str(RAIZ / man["tokenizer"]))
    especiais = tuple(man.get("ids_especiais") or (0, 1, 2, 3, 4))
    id_mask = int(man.get("id_mascara", 4))
    fl = Fluxo(ConfigDados(raiz=raiz, contexto=contexto))
    idx = np.random.default_rng(semente).choice(fl.n_seq, size=min(n, fl.n_seq),
                                                replace=False)
    cont = Contadores()
    cfg = ConfigMascara(taxa=taxa, p_equacao=1.0)
    tam_tok, tam_chr, corte_fim = [], [], 0
    for i in idx.tolist():
        ids, ide, disp = fl.sequencia(i)
        mascarar(ids, ide, disp, cfg=cfg, rng=np.random.default_rng((semente, i)),
                 id_mask=id_mask, n_vocab=tok.get_vocab_size(),
                 ids_especiais=frozenset(especiais), contadores=cont)
        # A mesma escolha, com o mesmo gerador, para ver O QUE foi escolhido.
        mascaravel = ~np.isin(ids, np.array(especiais))
        n_alvo = int(round(taxa * int(mascaravel.sum())))
        esc = _escolher_equacao(ide, disp, mascaravel, n_alvo,
                                np.random.default_rng((semente, i)), None)
        if esc.size:
            tam_tok.append(int(esc.size))
            if len(tam_chr) < 3000:
                tam_chr.append(len(tok.decode(ids[esc].tolist())))
            if ide[-1] == ide[esc[0]]:
                corte_fim += 1
    d = cont.como_dict()
    t, c = np.array(tam_tok), np.array(tam_chr)

    def pct(v):
        return [int(np.percentile(v, q)) for q in (10, 50, 90)] if v.size else None

    d.update({
        "janelas": int(idx.size), "contexto": contexto,
        "fracao_de_equacao_nos_mascarados": round(
            d["tokens_de_equacao_mascarados"] / max(d["tokens_mascarados"], 1), 4),
        "escolhida_tokens_p10_p50_p90": pct(t),
        "escolhida_caracteres_p10_p50_p90": pct(c),
        "escolhidas_cortadas_no_fim": corte_fim,
        "fracao_cortadas_no_fim": round(corte_fim / max(t.size, 1), 4),
    })
    return d


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fatia", action="append", required=True, metavar="ROTULO=CAMINHO")
    p.add_argument("--contexto", type=int, default=1024)
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--taxa", type=float, default=0.30)
    p.add_argument("--semente", type=int, default=17)
    a = p.parse_args()
    for spec in a.fatia:
        rotulo, _, caminho = spec.partition("=")
        d = sondar(Path(caminho), a.contexto, a.n, a.taxa, a.semente)
        print(f"\n=== {rotulo} · {d['janelas']:,} janelas de {a.contexto} ===")
        print(f"  fração tratada        {d['fracao_tratada']:.4f}   "
              f"({d['tratados']:,} de {d['sorteados_para_tratamento']:,})")
        print(f"  recaída sem display   {d['recaida_sem_equacao']:,}")
        print(f"  recaída só curtas     {d['recaida_equacao_curta']:,}   "
              f"(< {MIN_TOKENS_TRATAMENTO} tokens)")
        print(f"  recaída só grandes    {d['recaida_equacao_grande']:,}")
        print(f"  equação nos mascarados  {d['fracao_de_equacao_nos_mascarados']:.3f}")
        print(f"  escolhida: tokens {d['escolhida_tokens_p10_p50_p90']} · "
              f"caracteres {d['escolhida_caracteres_p10_p50_p90']} (p10/p50/p90)")
        print(f"  ⚠️ cortadas no FIM da janela: {d['escolhidas_cortadas_no_fim']:,} "
              f"({100 * d['fracao_cortadas_no_fim']:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
