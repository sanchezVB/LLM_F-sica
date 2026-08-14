"""O domínio omitido não pode vazar para o treino — senão a medição não mede nada.

`avaliar_transferencia` existe para responder «quanto o classificador perde num
negativo de tipo que nunca viu». Se o domínio omitido aparecer no treino, o
resultado sai bom e **não significa nada** — e nada no número denuncia isso. É
falha silenciosa da pior espécie: produz um relatório tranquilizador.

Medido em 2026-08-13 omitindo `q-bio`: falso positivo de 1,9% dentro do domínio
para 32,9% fora. Se houvesse vazamento, os dois seriam ~2% e eu teria concluído
que a transferência era boa — e liberado o passo 4d em cima disso.

O outro cuidado: a comparação «dentro do domínio» tem de usar negativos que
existem no treino mas **não foram usados nele**. Medir contra os próprios exemplos
de treino mediria memorização, e daria um 0% que também tranquiliza à toa.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.corpus.filter.classifier import avaliar_transferencia  # noqa: E402


def montar(tmp: Path, por_dominio: int = 60) -> tuple[Path, Path]:
    """Spine e três domínios de negativos, com vocabulário separável."""
    spine = tmp / "spine.parquet"
    pl.DataFrame({
        "arxiv_id": [f"fis/{i}" for i in range(por_dominio * 3)],
        "title": ["Quantum entanglement in condensed matter"] * (por_dominio * 3),
        "abstract": ["We compute the hamiltonian spectrum of the lattice model. " * 3]
                    * (por_dominio * 3),
    }).write_parquet(spine)

    negs = tmp / "negativos"
    vocab = {
        "cs": "Neural network training converges on the benchmark dataset. ",
        "econ": "The market equilibrium under rational expectations holds. ",
        "q_bio": "Protein folding kinetics in the ribosome complex vary. ",
    }
    for dom, texto in vocab.items():
        d = negs / dom
        d.mkdir(parents=True)
        pl.DataFrame({
            "arxiv_id": [f"{dom}/{i}" for i in range(por_dominio)],
            "title": [texto.strip()] * por_dominio,
            "abstract": [texto * 3] * por_dominio,
        }).write_parquet(d / "part-000.parquet")
    return spine, negs


def test_o_dominio_omitido_nao_entra_no_treino(tmp_path, caplog):
    """O log declara de quais domínios o treino saiu; o omitido não pode estar."""
    spine, negs = montar(tmp_path)
    import logging
    with caplog.at_level(logging.INFO):
        r = avaliar_transferencia(spine, negs, "q_bio", n_por_classe=40)
    treino = [m for m in caplog.messages if "omitindo" in m]
    assert treino, "o log não declarou a composição do treino"
    assert "q_bio" not in treino[0].split("de ")[-1], (
        f"o domínio omitido entrou no treino: {treino[0]}")
    assert "cs" in treino[0] and "econ" in treino[0]
    assert r.dominio_omitido == "q_bio"


def test_dominio_inexistente_falha_alto(tmp_path):
    """Errar o nome do domínio tem de estourar, não medir silenciosamente outro."""
    spine, negs = montar(tmp_path)
    with pytest.raises(ValueError, match="q-bio"):
        avaliar_transferencia(spine, negs, "q-bio", n_por_classe=40)  # underscore!


def test_conta_negativos_de_teste_so_do_dominio_omitido(tmp_path):
    spine, negs = montar(tmp_path, por_dominio=50)
    r = avaliar_transferencia(spine, negs, "q_bio", n_por_classe=40)
    assert r.n_teste_neg <= 50, "teste puxou negativos de fora do domínio omitido"
    assert r.n_treino_neg > 0


def test_degradacao_e_razao_e_trata_fp_zero(tmp_path):
    """`degradacao` divide por `fp_dentro`; zero não pode virar ZeroDivisionError."""
    from phifm.corpus.filter.classifier import Transferencia
    t = Transferencia("x", 1, 1, 1, 0.9, 0.5, fp_dentro=0.0, fp_fora=0.3,
                      revocacao=0.9)
    assert t.degradacao == float("inf")
    t2 = Transferencia("x", 1, 1, 1, 0.9, 0.5, fp_dentro=0.02, fp_fora=0.33,
                       revocacao=0.9)
    assert t2.degradacao == pytest.approx(16.5)


def test_a_curva_cobre_limiares_crescentes(tmp_path):
    """A curva é o que mostra se o limiar resolve. Medido: não resolve — o falso
    positivo estanca porque `modified_huber` satura as probabilidades."""
    spine, negs = montar(tmp_path)
    r = avaliar_transferencia(spine, negs, "cs", n_por_classe=40)
    limiares = [t for t, _, _, _ in r.curva]
    assert limiares == sorted(limiares) and len(limiares) >= 5
    fps = [fp for _, _, _, fp in r.curva]
    assert all(a >= b for a, b in zip(fps, fps[1:])), (
        "falso positivo tem de ser monótono não-crescente no limiar")
