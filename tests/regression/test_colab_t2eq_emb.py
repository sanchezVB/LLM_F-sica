"""O braço do Colab é o MESMO experimento do Kaggle, ou não serve para comparar.

O passo 1 do caminho B existe para ser comparado com o controle@200k (0,3872) e o
tratado@200k (0,4712), que treinaram no Kaggle. Se um hiperparâmetro divergir, o
número que sair mede a diferença de receita junto com a diferença de base — e a
comparação, que é a única razão de o braço existir, morre.

Por isso este arquivo lê as constantes das DUAS células pela AST e exige igualdade.
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
sys.path.insert(0, str(RAIZ / "scripts"))

import t2eq_emb  # noqa: E402  (a célula do Kaggle)
import t2eq_emb_modernbert as colab  # noqa: E402

CELULA_COLAB = "\n".join(colab.CELULAS)
NOMES = ("LOTE", "MAX_TOKENS", "SEMENTE", "N_CANDIDATOS", "PASSOS_AVAL", "MAX_PARES")


def _constantes(celula: str, nomes: tuple[str, ...]) -> dict:
    out = {}
    for n in ast.walk(ast.parse(celula.replace("__VARIANTE__", "modernbert"))):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in nomes:
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out


def _sem_marcadores(texto: str) -> str:
    for marcador, valor in (("__SHA__", "0" * 40), ("__REPO__", "dono/repo"),
                            ("__PASTA__", "/content/drive/MyDrive/phifm"),
                            ("__HASHES__", "{}"), ("__REGRA__", "regra")):
        texto = texto.replace(marcador, valor)
    return texto


def test_as_celulas_sao_python_valido():
    for fonte in colab.CELULAS:
        ast.parse(_sem_marcadores(fonte))


def test_os_hiperparametros_sao_OS_MESMOS_da_celula_do_kaggle():
    do_colab = _constantes(_sem_marcadores(CELULA_COLAB), NOMES)
    do_kaggle = _constantes(t2eq_emb.CELULA, NOMES)
    assert len(do_colab) == len(NOMES), f"faltou constante na célula: {do_colab}"
    assert do_colab == do_kaggle


def test_a_base_e_o_ModernBERT_e_o_treino_a_recebe():
    assert 'BASE = "answerdotai/ModernBERT-base"' in CELULA_COLAB
    assert '"--base", BASE' in CELULA_COLAB


def test_ABORTA_sem_GPU():
    """Uma sessão de CPU levaria dias, e a descoberta viria depois de horas."""
    assert "assert torch.cuda.is_available()" in CELULA_COLAB


def test_os_pares_sao_conferidos_contra_o_pacote_do_T1a():
    """Sem isto, "os mesmos 200 mil pares" seria afirmação, e não fato."""
    assert "blake3" in CELULA_COLAB and "HASHES" in CELULA_COLAB
    manifesto = json.loads(
        (RAIZ / "data/processed/kaggle_t2eq_emb/MANIFESTO.json").read_text(
            encoding="utf-8")) if (
        RAIZ / "data/processed/kaggle_t2eq_emb/MANIFESTO.json").exists() else None
    if manifesto is None:
        return  # o pacote é dado local, e não entra no repositório
    for nome, esperado in colab.HASHES.items():
        assert manifesto["arquivos"][nome]["blake3"] == esperado, (
            f"o hash de {nome} na célula do Colab não é o do pacote montado")


def test_a_saida_vai_para_o_DRIVE_para_a_sessao_caida_retomar():
    assert "SAIDA_BASE = PASTA / \"runs\"" in CELULA_COLAB
    assert "SAIDA = SAIDA_BASE /" in CELULA_COLAB


def test_a_regra_esta_escrita_com_os_tres_desfechos():
    for desfecho in ("MODERNBERT À FRENTE", "EMPATE", "TRATADO À FRENTE"):
        assert desfecho in colab.REGRA
    assert "0,4712" in colab.REGRA and "0,3872" in colab.REGRA, (
        "os números que este braço tem de superar vão na regra, escritos antes")


def test_o_notebook_gerado_fixa_o_COMMIT_e_nao_main():
    import gerar_notebook_colab as gerador

    nb = gerador.notebook("a" * 40, "/content/drive/MyDrive/phifm", colab,
                          "t2eq_emb_modernbert", None)
    fonte = "".join("".join(c["source"]) for c in nb["cells"])
    assert "a" * 40 in fonte and "__SHA__" not in fonte
    assert nb["metadata"]["accelerator"] == "GPU"
    assert all(c["cell_type"] in ("code", "markdown") for c in nb["cells"])
