"""T1i: as duas vencedoras do T1h a 1 M de pares, nos MESMOS pares do T1g.

O que este arquivo tranca:

1. **A receita é a de todos os ajustes**, e o volume é o do T1g (1 M), lidos pela AST.
2. **As bases são as duas que o T1h mandou subir**, e nenhuma outra.
3. **Os pares são os do T1g** (pacote do T1a 1,5 M), e a validação é byte a byte a do
   pacote de 400 mil — é por isso que só o treino sobe para o Drive.
4. **Retoma pulando o que concluiu**, e a regra está escrita antes, com Bonferroni.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "kaggle"))
sys.path.insert(0, str(RAIZ / "colab"))

import t1f_base_gte  # noqa: E402
import t1g_gte_1m  # noqa: E402
import t1h_bases_pequenas as t1h  # noqa: E402
import t1i_pequenas_1m as t1i  # noqa: E402

NOMES = ("LOTE", "MAX_TOKENS", "SEMENTE", "N_CANDIDATOS", "PASSOS_AVAL")


def _sem_marcadores(texto: str) -> str:
    for marcador, valor in (("__SHA__", "0" * 40), ("__REPO__", "dono/repo"),
                            ("__PASTA__", "/content/drive/MyDrive/phifm"),
                            ("__HASHES__", "{}"), ("__REGRA__", "regra"),
                            ("__BASES__", json.dumps(t1i.BASES))):
        texto = texto.replace(marcador, valor)
    return texto


CELULA = _sem_marcadores("\n".join(t1i.CELULAS))


def _constantes(celula: str, nomes: tuple[str, ...]) -> dict:
    out = {}
    for n in ast.walk(ast.parse(celula)):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in nomes:
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out


def test_as_celulas_sao_python_valido():
    for fonte in t1i.CELULAS:
        ast.parse(_sem_marcadores(fonte))


def test_a_receita_e_a_de_todos_e_o_volume_o_do_T1g():
    assert _constantes(CELULA, NOMES) == _constantes(t1f_base_gte.CELULA, NOMES)
    assert _constantes(CELULA, ("MAX_PARES",)) == _constantes(
        t1g_gte_1m.CELULA, ("MAX_PARES",)) == {"MAX_PARES": 1_000_000}
    assert '"--max-pares", str(MAX_PARES)' in CELULA


def test_as_bases_sao_as_duas_que_o_T1h_mandou_subir():
    assert t1i.BASES == {n: t1h.BASES[n] for n in ("gte-small", "bge-small")}


def test_os_pares_sao_os_do_T1g_e_a_validacao_e_a_de_400_mil():
    man = json.loads((RAIZ / "data/processed/kaggle_t1a15/MANIFESTO.json")
                     .read_text(encoding="utf-8")) \
        if (RAIZ / "data/processed/kaggle_t1a15/MANIFESTO.json").exists() else None
    if man is not None:
        assert {k: v["blake3"] for k, v in man["arquivos"].items()} == t1i.HASHES
    assert t1i.HASHES["pares_validacao.parquet"] == t1h.HASHES["pares_validacao.parquet"]
    assert '"pares_15m" / "pares_treino.parquet"' in t1i.SETUP
    assert '"pares" / "pares_validacao.parquet"' in t1i.SETUP
    assert "HASHES = __HASHES__" in t1i.SETUP


def test_ABORTA_sem_GPU_e_RETOMA_pulando_o_concluido():
    assert "assert torch.cuda.is_available()" in CELULA
    assert '.get("completed_at")' in t1i.TREINO and "continue" in t1i.TREINO


def test_a_regra_esta_escrita_ANTES_com_Bonferroni_e_o_alvo():
    r = " ".join(t1i.REGRA.split())
    for exigido in ("A REGRA, escrita ANTES", "Bonferroni", "97,5%", "0,6223", "ACIMA",
                    "EMPATE", "ABAIXO", "0,6211", "bootstrap pareado por ITEM"):
        assert exigido in r, exigido


def test_o_notebook_gerado_diz_o_que_subir_e_fixa_o_commit():
    nb = RAIZ / "colab" / "t1i_pequenas_1m.ipynb"
    assert nb.exists(), "o notebook não foi gerado"
    texto = json.dumps(json.loads(nb.read_text(encoding="utf-8")))
    for marcador in ("__SHA__", "__BASES__", "__HASHES__", "__REGRA__", "__PASTA__"):
        assert marcador not in texto, marcador
    assert "pares_15m" in texto and "691 MB" in texto
