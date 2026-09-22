"""ADR-0003, opção C: os dois braços ajustados como ΦEmb têm de diferir SÓ na base.

⚠️ O teste central é `test_os_hiperparametros_sao_os_do_T1f`: ele lê as constantes
das duas células pela AST. A única diferença declarada é `MAX_PARES`.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "kaggle"))

import t1f_base_gte  # noqa: E402
import t2eq_emb  # noqa: E402

from phifm.core.kaggle import obter  # noqa: E402

CELULA = t2eq_emb.CELULA


def _constantes(celula: str, nomes: tuple[str, ...]) -> dict:
    out = {}
    for n in ast.walk(ast.parse(celula.replace("__VARIANTE__", "controle"))):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in nomes:
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out


def test_a_celula_e_python_valido():
    ast.parse(CELULA.replace("__VARIANTE__", "tratado"))


def test_os_hiperparametros_sao_os_do_T1f():
    """⚠️ Qualquer um diferente mudaria a tarefa junto com a base."""
    nomes = ("LOTE", "MAX_TOKENS", "SEMENTE", "N_CANDIDATOS", "PASSOS_AVAL")
    assert _constantes(CELULA, nomes) == _constantes(t1f_base_gte.CELULA, nomes)
    assert len(_constantes(CELULA, nomes)) == 5


def test_o_volume_e_200_mil_e_vai_para_o_treino():
    assert _constantes(CELULA, ("MAX_PARES",)) == {"MAX_PARES": 200_000}
    assert '"--max-pares", MAX_PARES' in CELULA
    for nome in ("t2eq_emb_controle", "t2eq_emb_tratado"):
        assert obter(nome).max_pares == 200_000


def test_a_variante_escolhe_a_base_e_nada_mais():
    arvore = ast.parse(CELULA.replace("__VARIANTE__", "controle"))
    bases = [ast.literal_eval(n.value) for n in ast.walk(arvore)
             if isinstance(n, ast.Assign) and len(n.targets) == 1
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "BASES"]
    assert bases == [{"controle": "phienc-t2a-E", "tratado": "phienc-t2eq-E-tratado",
                      "modernbert": "answerdotai/ModernBERT-base",
                      "gte": "thenlper/gte-base"}]
    assert "assert VARIANTE in BASES" in CELULA


def test_os_dois_bracos_compartilham_DADOS_e_CODIGO():
    c, t = obter("t2eq_emb_controle"), obter("t2eq_emb_tratado")
    for campo in ("titulo_dados", "slug_dados", "pacote", "fonte_celula", "arquivos",
                  "scripts", "modelos", "max_pares", "repo"):
        assert getattr(c, campo) == getattr(t, campo), campo
    assert t.reusa_dados_de == "t2eq_emb_controle" and c.reusa_dados_de is None
    assert (c.variante, t.variante) == ("controle", "tratado")
    c.conferir()
    t.conferir()


def test_os_modelos_do_pacote_sao_os_dois_bracos_do_2_3():
    assert obter("t2eq_emb_controle").modelos == (
        "models/phienc-t2a-E", "models/phienc-t2eq-E-tratado")


def test_o_montador_COPIA_os_pares_do_T1a_e_confere_o_hash():
    fonte = (RAIZ / "scripts" / "empacotar_kaggle.py").read_text(encoding="utf-8")
    corpo = fonte.split("def _montar_t2eq_emb")[1].split("\ndef ")[0]
    assert "kaggle_t1a" in corpo and "shutil.copy2" in corpo
    assert "hash_arquivo" in corpo and 'man_t1a["arquivos"]' in corpo
    assert "sortear_para_parquet" not in corpo, "os pares não podem ser re-sorteados"


def test_a_regra_esta_escrita_ANTES_e_liga_ao_ADR():
    for exigido in ("A REGRA, escrita ANTES", "ADR-0003", "nDCG@10",
                    "bootstrap pareado por ITEM", "TRATADO À FRENTE", "EMPATE",
                    "CONTROLE À FRENTE", "200 mil", "NÃO se compara"):
        assert exigido in CELULA, exigido


# ── o terceiro braço: a barra do caminho B (2026-09-17) ─────────────────────

def test_as_bases_do_HUB_vem_do_HUB_e_as_nossas_do_zip():
    """A BARRA no nome decide, e não o nome do braço.

    Com `VARIANTE == "modernbert"`, cada base nova do Hub precisaria de mais um
    `or VARIANTE == ...`; esquecer isso faria a célula procurar `thenlper/gte-base`
    dentro do zip de modelos — e o erro apareceria com a GPU já alocada.
    """
    assert 'if "/" in str(BASES[VARIANTE]):\n    BASE = BASES[VARIANTE]' in CELULA
    assert "BASE = MODELOS / BASES[VARIANTE]" in CELULA
    bases = {"controle": "phienc-t2a-E", "modernbert": "answerdotai/ModernBERT-base",
             "gte": "thenlper/gte-base"}
    assert [n for n, b in bases.items() if "/" in b] == ["modernbert", "gte"]


def test_o_braco_modernbert_compartilha_dados_codigo_e_hiperparametros():
    m, c = obter("t2eq_emb_modernbert"), obter("t2eq_emb_controle")
    for campo in ("titulo_dados", "slug_dados", "pacote", "fonte_celula", "arquivos",
                  "scripts", "max_pares", "repo"):
        assert getattr(m, campo) == getattr(c, campo), campo
    assert m.reusa_dados_de == "t2eq_emb_controle"
    assert m.variante == "modernbert"
    m.conferir()


def test_a_regra_do_modernbert_esta_escrita_ANTES_e_diz_que_nao_e_ablacao():
    r = " ".join(t2eq_emb.CELULA.split())  # a regra quebra linhas no meio das frases
    for exigido in ("REGRA do braço modernbert, escrita ANTES", "MODERNBERT À FRENTE",
                    "É a barra do produto, não uma ablação", "0,4712",
                    "NÃO entra sem nova decisão"):
        assert exigido in r, exigido
    assert ('REGRA = {"modernbert": REGRA_MODERNBERT, "gte": REGRA_GTE}'
            '.get(VARIANTE, REGRA_BRACOS)') in t2eq_emb.CELULA


# ── o quarto braço: a base é a certa? (2026-09-22) ──────────────────────────


def test_o_braco_gte_compartilha_dados_codigo_e_hiperparametros():
    """Uma variável: a BASE. Se qualquer outra coisa divergir, o número mede duas."""
    g, c = obter("t2eq_emb_gte"), obter("t2eq_emb_controle")
    for campo in ("titulo_dados", "slug_dados", "pacote", "fonte_celula", "arquivos",
                  "scripts", "max_pares", "repo"):
        assert getattr(g, campo) == getattr(c, campo), campo
    assert g.reusa_dados_de == "t2eq_emb_controle"
    assert g.variante == "gte"
    assert g.slug_notebook != c.slug_notebook
    g.conferir()


def test_a_regra_do_gte_esta_escrita_ANTES_e_declara_a_variavel():
    r = " ".join(t2eq_emb.CELULA.split())
    for exigido in ("REGRA do braço gte, escrita ANTES", "GTE À FRENTE",
                    "Uma variável: a BASE", "0,5270", "110 M contra 150 M"):
        assert exigido in r, exigido
