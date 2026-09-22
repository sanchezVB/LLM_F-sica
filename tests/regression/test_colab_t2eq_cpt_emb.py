"""A secundária do caminho B só vale se os três ajustes forem a MESMA receita.

Os dois braços do CPT são comparados entre si e contra a barra de 0,5270 — o
ModernBERT-base cru ajustado nos mesmos 200 mil pares. Se um hiperparâmetro divergir,
o número mede a diferença de receita junto com a diferença de base, e a comparação
morre. Este arquivo lê as constantes pela AST e exige igualdade com as células já
rodadas.

E tranca o que separa "o braço tratado" de "um arquivo de 598 MB qualquer": o blake3
dos pesos, conferido antes de treinar.
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

import t2eq_cpt_emb as colab  # noqa: E402
import t2eq_emb  # noqa: E402  (a célula do Kaggle, dos braços de 48 M)
import t2eq_emb_modernbert as passo1  # noqa: E402

NOMES = ("LOTE", "MAX_TOKENS", "SEMENTE", "N_CANDIDATOS", "PASSOS_AVAL", "MAX_PARES")
BRACOS = ("controle", "tratado")


def _sem_marcadores(texto: str, variante: str = "tratado") -> str:
    for marcador, valor in (("__SHA__", "0" * 40), ("__REPO__", "dono/repo"),
                            ("__PASTA__", "/content/drive/MyDrive/phifm"),
                            ("__HASHES__", "{}"), ("__HASHES_ENCODER__", "{}"),
                            ("__REGRA__", "regra"), ("__VARIANTE__", variante)):
        texto = texto.replace(marcador, valor)
    return texto


def _constantes(celula: str, nomes: tuple[str, ...]) -> dict:
    out = {}
    for n in ast.walk(ast.parse(celula)):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in nomes:
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out


CELULA = _sem_marcadores("\n".join(colab.CELULAS))


def test_as_celulas_sao_python_valido():
    for variante in BRACOS:
        for fonte in colab.CELULAS:
            ast.parse(_sem_marcadores(fonte, variante))


def test_os_hiperparametros_sao_OS_MESMOS_dos_outros_ajustes():
    do_cpt = _constantes(CELULA, NOMES)
    do_kaggle = _constantes(t2eq_emb.CELULA.replace("__VARIANTE__", "tratado"), NOMES)
    do_passo1 = _constantes(
        _sem_marcadores("\n".join(passo1.CELULAS)).replace("__VARIANTE__", "x"), NOMES)
    assert len(do_cpt) == len(NOMES), f"faltou constante na célula: {do_cpt}"
    assert do_cpt == do_kaggle == do_passo1


def test_os_dois_bracos_diferem_SO_na_variante():
    """⚠️ Com sentinelas, e não com os nomes dos braços.

    A própria célula cita os DOIS num comentário (`controle (p_equacao 0,0) ou
    tratado (0,6)`), então trocar um nome pelo outro no texto inteiro compara outra
    coisa — foi assim que a primeira versão deste teste reprovou sozinha.
    """
    a, b = (_sem_marcadores("\n".join(colab.CELULAS), v)
            for v in ("BRACO_A_SENTINELA", "BRACO_B_SENTINELA"))
    assert a != b
    assert a.replace("BRACO_A_SENTINELA", "BRACO_B_SENTINELA") == b, (
        "as duas células diferem em algo além da variante injetada")


def test_ABORTA_sem_GPU():
    assert "assert torch.cuda.is_available()" in CELULA, (
        "sem esta guarda, uma sessão de CPU treinaria por dias antes de alguém ver")


def test_os_PESOS_sao_conferidos_por_blake3_antes_de_treinar():
    """Os dois braços diferem SÓ nos pesos; medir o arquivo errado daria o número
    do outro braço com o nome deste, e nada acusaria."""
    assert set(colab.HASHES_ENCODER) == set(BRACOS)
    for h in colab.HASHES_ENCODER.values():
        assert len(h) == 64 and int(h, 16) >= 0
    assert len(set(colab.HASHES_ENCODER.values())) == 2, "os dois hashes são iguais"
    assert "HASHES_ENCODER[VARIANTE]" in CELULA
    # A conferência mora na célula 1 e o treino na 2 — a ORDEM é a das células.
    assert "_blake3(ENCODER" in colab.SETUP
    assert "train_embedding.py" in colab.TREINO
    assert colab.CELULAS.index(colab.SETUP) < colab.CELULAS.index(colab.TREINO)


def test_os_pares_sao_os_MESMOS_dos_outros_bracos():
    assert colab.HASHES == passo1.HASHES, (
        "os hashes dos pares divergiram do passo 1 — os números deixariam de ser "
        "comparáveis")
    assert "HASHES = __HASHES__" in colab.SETUP, (
        "os pares têm de ser conferidos na célula 1, antes do treino da 2")


def test_a_saida_vai_para_o_DRIVE_para_a_sessao_caida_retomar():
    assert "SAIDA_BASE = PASTA / \"runs\"" in CELULA
    assert "--out\", str(SAIDA)" in CELULA


def test_a_regra_esta_escrita_com_os_desfechos_e_a_BARRA():
    for trecho in ("TRATADO À FRENTE", "EMPATE", "CONTROLE À FRENTE", "0,5270",
                   "bootstrap pareado por ITEM"):
        assert trecho in colab.REGRA, trecho
    # A primária de MLM já medida entra como contexto, e não como desfecho.
    assert "NÃO DECIDIDO" in colab.REGRA and "−0,00039" in colab.REGRA


def test_os_notebooks_gerados_fixam_o_COMMIT_e_o_BRACO():
    for variante in BRACOS:
        nb = RAIZ / "colab" / f"t2eq_cpt_emb-{variante}.ipynb"
        assert nb.exists(), f"{nb} não foi gerado"
        texto = json.dumps(json.loads(nb.read_text(encoding="utf-8")))
        assert "__SHA__" not in texto and "__VARIANTE__" not in texto
        assert f'phienc-cpt-{variante}' in texto
        assert f'VARIANTE = \\"{variante}\\"' in texto
