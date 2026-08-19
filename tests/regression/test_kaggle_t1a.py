"""O notebook do Kaggle não pode reimplementar o treino.

A tentação, num notebook, é colar o laço de treino direto na célula — é mais rápido
de escrever e roda. O custo aparece depois: dois laços de treino no projeto,
divergindo em silêncio. Os dois rodariam, dariam números diferentes, e nada
apontaria qual está certo.

Então o notebook chama `scripts/train_embedding.py` a partir de um ZIP do pacote, e
estes testes fixam isso — junto com as três coisas que, se faltarem, transformam um
resultado do Kaggle em número sem valor:

  1. a conferência de hash dos dados de entrada;
  2. a exigência de GPU (rodar em CPU levaria dias e "funcionaria");
  3. o aviso de que 1.000 candidatos não é o protocolo do veredito.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

# ⚠️ A CÉLULA e a DOCSTRING são conferidas separadamente.
#
# A primeira versão deste arquivo lia o `.py` inteiro numa string e procurava
# "InfoNCE" nela — e encontrou, na docstring que explica POR QUE não usar
# `DataParallel`. Um teste que confunde o código com a explicação do código reprova
# a explicação por ser boa.
sys.path.insert(0, str(RAIZ / "kaggle"))
import t1a_phiemb  # noqa: E402

CELULA = t1a_phiemb.CELULA
DOC = t1a_phiemb.__doc__ or ""


def test_notebook_chama_o_treino_em_vez_de_reimplementar():
    """A asserção central deste arquivo.

    Se alguém colar o laço aqui, o projeto passa a ter dois — e a divergência entre
    eles é invisível, porque os dois rodam.
    """
    assert "scripts/train_embedding.py" in CELULA
    for sinal in ("for passo", "backward()", "cross_entropy", "InfoNCE"):
        assert sinal not in CELULA, (
            f"a célula parece reimplementar o treino ({sinal!r}). Ela deve CHAMAR "
            "`train_embedding.py` a partir do ZIP do pacote.")


def test_confere_hash_antes_de_treinar():
    """Upload truncado produz um número que parece comparável e não é.

    O Kaggle não garante nada sobre o conteúdo do input — só que existe.
    """
    assert "MANIFESTO.json" in CELULA
    assert "hash difere" in CELULA
    i_hash = CELULA.index("hash difere")
    i_treino = CELULA.index("train_embedding.py")
    assert i_hash < i_treino, "a conferência de hash tem de vir ANTES do treino"


def test_exige_gpu_em_vez_de_cair_para_cpu():
    """Um treino que "funciona" em CPU por 40 h é pior que um que não começa."""
    assert "torch.cuda.is_available()" in CELULA
    assert "assert torch.cuda.is_available()" in CELULA


def test_avisa_que_mil_candidatos_nao_e_o_veredito():
    """O protocolo do G1 é 2.000 candidatos.

    Este projeto já elegeu um campeão errado por comparar métricas de protocolos
    diferentes. O aviso existe para o número do log do Kaggle não virar alegação.
    """
    assert "2.000 candidatos" in CELULA
    assert "NÃO é comparável" in CELULA


def test_usa_uma_gpu_e_diz_por_que():
    """`DataParallel` num lote contrastivo corta os negativos pela metade.

    Cada réplica calcularia o InfoNCE só sobre a sua fatia: 127 negativos por âncora
    viram 63, sem nada avisar. É o mesmo tipo de erro silencioso que o GradCache
    existe para não cometer, e o comentário tem de estar lá para ninguém "otimizar"
    ligando as duas placas.
    """
    # Na DOCSTRING, que é onde a decisão é explicada; a célula só usa uma placa.
    assert "DataParallel" in DOC
    assert "63" in DOC
    assert "DataParallel" not in CELULA, "a célula não deve usar DataParallel"


def test_lote_e_o_do_campeao():
    """Mudar lote junto com dispositivo faria a comparação medir duas coisas.

    É o erro que este projeto cometeu ao mudar base e lote no mesmo treino, e que
    custou um experimento inteiro.
    """
    assert '"--lote", "128"' in CELULA


# ─── o empacotador ───────────────────────────────────────────────────────────


def test_empacotador_existe_e_declara_o_volume():
    doc = (RAIZ / "scripts" / "empacotar_kaggle.py").read_text(encoding="utf-8")
    assert "400_000" in doc
    # O volume tem justificativa medida, não é número redondo por gosto.
    assert "p=0,950" in doc, (
        "o volume de 400 mil pares precisa citar a medição que o justifica")


def test_pacote_gerado_tem_o_que_o_notebook_espera(tmp_path):
    """Contrato entre o empacotador e a célula, testado nos dois lados.

    Se o empacotador parar de gerar `phifm_src.zip`, ou o manifesto perder a chave
    `arquivos`, a célula falha no Kaggle — a 211 MB de upload de distância.
    """
    import polars as pl

    pares = tmp_path / "pares"
    pares.mkdir()
    d = pl.DataFrame({"arxiv_id": ["a"], "arxiv_citado": ["b"],
                      "ancora": ["x"], "positivo": ["y"]})
    d.write_parquet(pares / "pares_treino.parquet")
    d.write_parquet(pares / "pares_validacao.parquet")

    import subprocess

    saida = tmp_path / "saida"
    r = subprocess.run(
        [sys.executable, str(RAIZ / "scripts" / "empacotar_kaggle.py"),
         "--pares", str(pares), "--out", str(saida), "--max-pares", "1"],
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(RAIZ / "src"),
             "PYTHONUTF8": "1"})
    assert r.returncode == 0, r.stderr[-800:]

    man = json.loads((saida / "MANIFESTO.json").read_text(encoding="utf-8"))
    assert set(man["arquivos"]) == {"pares_treino.parquet",
                                    "pares_validacao.parquet", "phifm_src.zip"}
    for v in man["arquivos"].values():
        assert len(v["blake3"]) == 64 and v["bytes"] > 0
    assert man["hash_algo"] == "blake3"


def test_zip_traz_o_ponto_de_entrada_e_o_pacote(tmp_path):
    """O ZIP tem de conter `scripts/train_embedding.py` E o pacote `phifm`.

    Só o pacote não basta: a célula invoca o script pelo caminho, e um ZIP sem ele
    falharia com "arquivo não encontrado" depois do upload.
    """
    import subprocess

    import polars as pl

    pares = tmp_path / "pares"
    pares.mkdir()
    d = pl.DataFrame({"arxiv_id": ["a"], "arxiv_citado": ["b"],
                      "ancora": ["x"], "positivo": ["y"]})
    d.write_parquet(pares / "pares_treino.parquet")
    d.write_parquet(pares / "pares_validacao.parquet")
    saida = tmp_path / "s"
    subprocess.run(
        [sys.executable, str(RAIZ / "scripts" / "empacotar_kaggle.py"),
         "--pares", str(pares), "--out", str(saida), "--max-pares", "1"],
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(RAIZ / "src"),
             "PYTHONUTF8": "1"}, check=True)

    with zipfile.ZipFile(saida / "phifm_src.zip") as z:
        nomes = z.namelist()
    assert "scripts/train_embedding.py" in nomes
    assert "phifm/training/embedding.py" in nomes
    assert not any("__pycache__" in n for n in nomes)
