#!/usr/bin/env python3
"""Aplica a regra da ablação do DOC-07 §2.3 aos quatro artefatos de MLM.

    python scripts/comparar_ablacao_phienc.py \\
        --controle-uniforme data/processed/avaliacao/t2eq_mlm_controle_uniforme.json \\
        --tratado-uniforme  data/processed/avaliacao/t2eq_mlm_tratado_uniforme.json \\
        --controle-equacao  data/processed/avaliacao/t2eq_mlm_controle_equacao.json \\
        --tratado-equacao   data/processed/avaliacao/t2eq_mlm_tratado_equacao.json \\
        --treino-tratado    data/processed/t2eq_run_E_tratado/t2eq_tratado_E.json \\
        --out data/processed/avaliacao/t2eq_ablacao.json

A regra está em `kaggle/t2eq_tratado.py` (`REGRA_ABLACAO`), escrita antes de o braço
tratado existir e corrigida antes de qualquer número dele. Este script não
acrescenta leitura nenhuma: as contas estão em `phifm.eval.mlm_regiao` e a leitura em
`ler_pela_regra`.

## ⚠️ Recusa, em vez de subtrair

Os quatro artefatos têm de vir do MESMO protocolo — fatia, tokenizer, contexto,
fração, semente, número de sequências — e cada par, das MESMAS sequências. Dois
números de protocolos diferentes são duas tarefas, e a diferença entre eles teria a
cara de um resultado.

Sem torch: roda na venv leve.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8  # noqa: E402

utf8()

from phifm.eval.mlm_regiao import (  # noqa: E402
    diferenca_das_diferencas,
    diferenca_de_acuracia,
    ler_pela_regra,
)

# O que tem de ser idêntico entre os quatro, fora a prova.
PROTOCOLO_COMUM = ("amostragem", "contexto", "fracao_mascara", "semente",
                   "n_sequencias_pedidas")


def _ressalva_de_spikes(tratado: dict, caminho_controle: Path | None) -> str:
    """O que cada braço sofreu de spike, lido dos manifestos.

    Um braço com spike e outro sem é uma assimetria REAL e entra na leitura como
    ressalva nomeada — ver `kaggle/t2eq_tratado.py`. Inventá-la, ou herdá-la de
    outro experimento, é o oposto disso.
    """
    n_t = (tratado.get("spike") or {}).get("n_spikes") or 0
    passos_t = (tratado.get("spike") or {}).get("spikes") or []
    if caminho_controle is None:
        return (f"O tratado teve {n_t} spike(s){f' (passos {passos_t})' if passos_t else ''};"
                " o controle não foi informado a este comparador.")
    controle = json.loads(Path(caminho_controle).read_text(encoding="utf-8"))
    n_c = (controle.get("spike") or {}).get("n_spikes") or 0
    passos_c = (controle.get("spike") or {}).get("spikes") or []
    if not n_t and not n_c:
        return "Nenhum dos dois braços teve spike."
    detalhe = (f"tratado {n_t}{f' {passos_t}' if passos_t else ''}, "
               f"controle {n_c}{f' {passos_c}' if passos_c else ''}")
    if n_t == n_c:
        return f"Os dois braços tiveram spike ({detalhe})."
    contra = "CONTRA o tratado" if n_t > n_c else "a favor do tratado"
    return f"Spikes assimétricos ({detalhe}) — assimetria {contra}."


def _ler(caminho: Path, prova: str) -> dict:
    d = json.loads(Path(caminho).read_text(encoding="utf-8"))
    obtida = (d.get("protocolo") or {}).get("prova")
    if obtida != prova:
        raise SystemExit(f"{caminho}: prova {obtida!r}, esperada {prova!r}")
    if "por_sequencia" not in d:
        raise SystemExit(
            f"{caminho} não tem `por_sequencia` — foi gravado por uma versão do "
            "avaliador anterior ao bootstrap por sequência. Meça de novo.")
    return d


def conferir_protocolo(artefatos: dict[str, dict]) -> dict:
    ref_nome, ref = next(iter(artefatos.items()))
    for nome, d in artefatos.items():
        for chave in PROTOCOLO_COMUM:
            if d["protocolo"].get(chave) != ref["protocolo"].get(chave):
                raise SystemExit(
                    f"{nome} e {ref_nome} divergem em `{chave}`: "
                    f"{d['protocolo'].get(chave)!r} contra "
                    f"{ref['protocolo'].get(chave)!r}. São tarefas diferentes.")
        for chave in ("dados", "tokenizer_sha"):
            if d.get(chave) != ref.get(chave):
                raise SystemExit(f"{nome} e {ref_nome} divergem em `{chave}`.")
        if d.get("indices_sorteados") != ref.get("indices_sorteados"):
            raise SystemExit(f"{nome} e {ref_nome} não sortearam as MESMAS sequências.")
    return {c: ref["protocolo"].get(c) for c in PROTOCOLO_COMUM} | {
        "dados": ref.get("dados"), "tokenizer_sha": ref.get("tokenizer_sha")}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--controle-uniforme", type=Path, required=True)
    p.add_argument("--tratado-uniforme", type=Path, required=True)
    p.add_argument("--controle-equacao", type=Path, required=True)
    p.add_argument("--tratado-equacao", type=Path, required=True)
    p.add_argument("--treino-controle", type=Path, default=None,
                   help="o phienc.json do braço CONTROLE; sem ele a ressalva de "
                        "spikes fala só do tratado")
    p.add_argument("--regra", default="kaggle/t2eq_tratado.py · REGRA_ABLACAO",
                   help="onde a regra deste run está escrita; o caminho B do "
                        "ADR-0003 usa kaggle/t2eq_cpt.py · REGRA")
    p.add_argument("--treino-tratado", type=Path, required=True,
                   help="o JSON do treino do braço tratado, com `mascaramento`")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--semente", type=int, default=17)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    art = {"controle_uniforme": _ler(a.controle_uniforme, "uniforme"),
           "tratado_uniforme": _ler(a.tratado_uniforme, "uniforme"),
           "controle_equacao": _ler(a.controle_equacao, "equacao"),
           "tratado_equacao": _ler(a.tratado_equacao, "equacao")}
    protocolo = conferir_protocolo(art)

    treino = json.loads(a.treino_tratado.read_text(encoding="utf-8"))
    fracao = (treino.get("mascaramento") or {}).get("fracao_tratada")
    if fracao is None:
        raise SystemExit(f"{a.treino_tratado} não tem `mascaramento.fracao_tratada`")

    primaria = diferenca_das_diferencas(
        art["controle_uniforme"]["por_sequencia"],
        art["tratado_uniforme"]["por_sequencia"],
        semente=a.semente, n_boot=a.n_boot)
    checagem_2 = diferenca_de_acuracia(
        [[r[0], r[1], r[2]] for r in art["controle_equacao"]["por_sequencia"]],
        [[r[0], r[1], r[2]] for r in art["tratado_equacao"]["por_sequencia"]],
        semente=a.semente, n_boot=a.n_boot)
    # ⚠️ Escala, regra e ressalvas saem dos MANIFESTOS, e não de constantes deste
    # arquivo. Até 2026-09-22 os três eram fixos no §2.3 a 48 M, e o artefato do
    # caminho B saiu declarando "0,6 B" (rodou 0,4 B), a regra do outro experimento
    # e um spike que nunca houve — "O tratado teve 1 spike (passo 8.075…)" é do run
    # de 2026-09-16. Uma ressalva falsa é pior que nenhuma: ela tem a forma da
    # honestidade e o conteúdo errado.
    tokens = ((treino.get("metricas") or {}).get("tokens")
              or (treino.get("dados") or {}).get("tokens") or 0)
    escala = f"{tokens / 1e9:.1f} B".replace(".", ",")
    leitura = ler_pela_regra(primaria, checagem_2, float(fracao), escala=escala)
    ressalvas = [f"{escala} tokens; uma semente por braço.",
                 _ressalva_de_spikes(treino, a.treino_controle),
                 "A prova uniforme é o objetivo do controle: o viés dela aponta "
                 "para ele.",
                 "Secundárias (recuperação, sonda tensorial) não entram aqui."]

    resultado = {
        "experimento": ("DOC-07 §2.3 — mascaramento de equações inteiras, a "
                        f"{escala}"),
        "regra": a.regra,
        "protocolo": protocolo,
        "modelos": {k: v.get("modelo") for k, v in art.items()},
        **leitura,
        "primaria": primaria,
        "checagem_2_detalhe": checagem_2,
        "ressalvas": ressalvas,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    c = leitura["checagens"]
    print()
    print("=" * 74)
    print(f"  DOC-07 §2.3 · ablação do mascaramento de equações · a {escala}")
    print("=" * 74)
    print(f"  checagem 1 · fração tratada {c['1_fracao_tratada']['valor']:.4f} "
          f"(mín. 0,50) · {'✅' if c['1_fracao_tratada']['aprovada'] else '❌'}")
    print(f"  checagem 2 · equação inteira: tratado − controle "
          f"{checagem_2['diferenca']:+.4f} {checagem_2['ic95']} · "
          f"{'✅' if c['2_prova_com_equacao_inteira']['aprovada'] else '❌'}")
    print(f"      controle {checagem_2['acuracia_controle']:.4f} · "
          f"tratado {checagem_2['acuracia_tratado']:.4f} · "
          f"{checagem_2['tokens']:,} tokens")
    print()
    print("  PRIMÁRIA · diferença das diferenças (prova uniforme)")
    for braco in ("controle", "tratado"):
        b = primaria[braco]
        print(f"    {braco:<9} equação {b['acuracia_equacao']:.4f} · prosa "
              f"{b['acuracia_prosa']:.4f} · vantagem {b['vantagem_em_equacao']:+.4f}")
    print(f"    tratado − controle: {primaria['diferenca_das_diferencas']:+.5f} "
          f"{primaria['ic95']} · {primaria['sequencias']:,} sequências")
    print()
    print(f"  DESFECHO: {leitura['desfecho']}")
    print(f"  {leitura['leitura']}")
    print("=" * 74)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
