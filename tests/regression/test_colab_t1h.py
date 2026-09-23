"""T1h: cinco bases, UMA variável — e a referência treinada na mesma receita.

O que este arquivo tranca:

1. **A receita é a de todos os ajustes de 200 mil pares**, lida pela AST. Se o lote ou a
   semente divergissem, a comparação mediria receita junto com base.
2. **A referência (MiniLM-L6) é treinada AQUI, e primeiro.** A 200 mil pares ela nunca
   foi medida; comparar com o número de 400 mil mudaria o volume junto com a base.
3. **As bases são ids do Hub**, e os pares são os mesmos bytes dos outros ajustes.
4. **Rodar a célula de novo retoma**, pulando o que concluiu — a sessão do Colab cai.
5. **A regra está escrita antes**, com a correção de Bonferroni e o critério de custo.
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
import t1h_bases_pequenas as t1h  # noqa: E402
import t2eq_emb_modernbert as passo1  # noqa: E402

NOMES = ("LOTE", "MAX_TOKENS", "SEMENTE", "N_CANDIDATOS", "PASSOS_AVAL")


def _sem_marcadores(texto: str) -> str:
    for marcador, valor in (("__SHA__", "0" * 40), ("__REPO__", "dono/repo"),
                            ("__PASTA__", "/content/drive/MyDrive/phifm"),
                            ("__HASHES__", "{}"), ("__REGRA__", "regra"),
                            ("__BASES__", json.dumps(t1h.BASES))):
        texto = texto.replace(marcador, valor)
    return texto


CELULA = _sem_marcadores("\n".join(t1h.CELULAS))


def _constantes(celula: str, nomes: tuple[str, ...]) -> dict:
    out = {}
    for n in ast.walk(ast.parse(celula)):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in nomes:
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out


def test_as_celulas_sao_python_valido():
    for fonte in t1h.CELULAS:
        ast.parse(_sem_marcadores(fonte))


def test_a_receita_e_a_do_T1f_e_o_volume_o_dos_ajustes_de_200_mil():
    assert _constantes(CELULA, NOMES) == _constantes(t1f_base_gte.CELULA, NOMES)
    assert len(_constantes(CELULA, NOMES)) == len(NOMES)
    assert _constantes(CELULA, ("MAX_PARES",)) == {"MAX_PARES": 200_000}
    assert '"--max-pares", str(MAX_PARES)' in CELULA


def test_a_referencia_e_o_MiniLM_L6_e_vem_PRIMEIRO():
    nomes = list(t1h.BASES)
    assert nomes[0] == t1h.REFERENCIA == "minilm-l6"
    assert t1h.BASES["minilm-l6"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_as_bases_sao_ids_do_Hub_e_distintas():
    ids = list(t1h.BASES.values())
    assert len(set(ids)) == len(ids) == 5
    assert all("/" in b for b in ids)


def test_os_pares_sao_os_mesmos_bytes_dos_outros_ajustes():
    assert t1h.HASHES == passo1.HASHES
    assert "HASHES = __HASHES__" in t1h.SETUP


def test_ABORTA_sem_GPU():
    assert "assert torch.cuda.is_available()" in CELULA


def test_rodar_de_novo_PULA_o_que_concluiu_e_retoma_o_resto():
    """A sessão do Colab cai; recomeçar do zero custaria as bases já prontas."""
    assert '.get("completed_at")' in t1h.TREINO
    assert "if concluido(saida)" in t1h.TREINO and "continue" in t1h.TREINO
    assert 'SAIDA_BASE = PASTA / "runs" / "t1h"' in t1h.SETUP


def test_a_regra_esta_escrita_ANTES_com_Bonferroni_e_o_criterio_de_custo():
    r = " ".join(t1h.REGRA.split())
    for exigido in ("A REGRA, escrita ANTES", "Bonferroni", "98,75%", "ACIMA",
                    "EMPATE", "ABAIXO", "2,5x", "prefixo", "bootstrap pareado por ITEM"):
        assert exigido in r, exigido


def test_o_notebook_gerado_fixa_o_commit_e_as_bases():
    nb = RAIZ / "colab" / "t1h_bases_pequenas.ipynb"
    assert nb.exists(), "o notebook não foi gerado"
    texto = json.dumps(json.loads(nb.read_text(encoding="utf-8")))
    for marcador in ("__SHA__", "__BASES__", "__HASHES__", "__REGRA__"):
        assert marcador not in texto, marcador
    for base in t1h.BASES.values():
        assert base in texto, base
