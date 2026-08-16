"""Soma de caracteres em polars dá a volta em UInt32, e o erro é plausível.

Regressão de 2026-08-16. `str.len_chars()` devolve **UInt32** e `.sum()` acumula no
mesmo tipo. Um corpus de 10,5 G caracteres passa do máximo de 4.294.967.295 e a
soma **dá a volta** — sem exceção, sem aviso, com um resultado plausível.

Medido na fatia de Física do OpenWebMath:

    soma sem cast   1,88 G caracteres
    soma com cast  10,47 G caracteres
    diferenca      exatamente 2,00 voltas de 2^32

Eu quase reportei «0,47 B tokens» onde são 2,62 B. O que salvou foi outra medição
discordar: a coleta registrou 12.162 caracteres por documento e o disco parecia ter
2.185 — seis vezes de diferença que não podia ser real.

**A lição:** um número que não estoura nem avisa, e que é plausível, só é pego por
outra medição do mesmo fato. É por isso que este projeto registra a coleta E o
disco em vez de confiar num dos dois.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

U32 = 2 ** 32


def test_len_chars_e_uint32(tmp_path):
    """Fixa a causa: o tipo devolvido é UInt32, e é isso que permite a volta."""
    df = pl.DataFrame({"texto": ["abc"]})
    tipo = df.select(pl.col("texto").str.len_chars())["texto"].dtype
    assert tipo == pl.UInt32, (
        f"len_chars mudou de tipo para {tipo} — se virou Int64, o cast defensivo "
        "deixou de ser necessário e este teste deve ser revisto, não removido")


def test_a_soma_sem_cast_da_a_volta(tmp_path):
    """Reproduz o estouro com o mínimo de dados possível.

    Não dá para materializar 10 G caracteres num teste, então usamos poucas linhas
    MUITO longas: 5 linhas de 1 GiB são impraticáveis também. A alternativa honesta
    é somar um valor grande diretamente e mostrar que UInt32 satura.
    """
    grande = U32 - 10          # cabe em UInt32
    s = pl.Series("n", [grande, 100], dtype=pl.UInt32)
    # Sem cast, a soma de 4.294.967.286 + 100 não cabe em UInt32.
    sem_cast = pl.DataFrame({"n": s}).select(pl.col("n").sum()).item()
    com_cast = pl.DataFrame({"n": s}).select(
        pl.col("n").cast(pl.Int64).sum()).item()
    assert com_cast == grande + 100
    assert sem_cast != com_cast, (
        "a soma em UInt32 não deu a volta neste polars — o comportamento mudou, e "
        "o cast pode ter deixado de ser necessário; verifique antes de remover")
    assert com_cast - sem_cast == U32, "a diferença tem de ser uma volta exata"


def test_o_relatorio_usa_o_cast():
    """Lê o próprio fonte: se o cast sair, o relatório volta a subnotificar.

    O modo de falha não é uma exceção — é um número menor e plausível num PDF que
    alguém vai ler como verdade.
    """
    fonte = (Path(__file__).resolve().parents[2] / "scripts" / "relatorio_pdf.py"
             ).read_text(encoding="utf-8")
    assert "cast(pl.Int64).sum()" in fonte, (
        "a contagem de caracteres do relatório perdeu o cast para Int64")


def test_contagem_real_bate_com_o_esperado(tmp_path):
    """Ponta a ponta em parquet, no caminho que o relatório usa."""
    from importlib import util

    spec = util.spec_from_file_location(
        "relatorio_pdf",
        Path(__file__).resolve().parents[2] / "scripts" / "relatorio_pdf.py")
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    texto = "x" * 1000
    p = tmp_path / "part-00000.parquet"
    pl.DataFrame({"texto": [texto] * 500}).write_parquet(p)
    assert mod._caracteres([str(p)], "texto") == 500_000


def test_coluna_ausente_devolve_zero_em_vez_de_estourar(tmp_path):
    from importlib import util

    spec = util.spec_from_file_location(
        "relatorio_pdf",
        Path(__file__).resolve().parents[2] / "scripts" / "relatorio_pdf.py")
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    p = tmp_path / "part-00000.parquet"
    pl.DataFrame({"outra": ["a"]}).write_parquet(p)
    assert mod._caracteres([str(p)], "texto") == 0
