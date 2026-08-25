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

    Se o empacotador parar de gerar `phifm_src.zip.bin`, ou o manifesto perder a chave
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
                                    "pares_validacao.parquet", "phifm_src.zip.bin"}
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

    with zipfile.ZipFile(saida / "phifm_src.zip.bin") as z:
        nomes = z.namelist()
    assert "scripts/train_embedding.py" in nomes
    assert "phifm/training/embedding.py" in nomes
    assert not any("__pycache__" in n for n in nomes)


# ─── publicação: os dois metadados e o notebook gerado ───────────────────────


def _publicar():
    """Importa `scripts/publicar_kaggle.py` sem executá-lo."""
    import importlib.util

    raiz = Path(__file__).resolve().parents[2]
    caminho = raiz / "scripts/publicar_kaggle.py"
    spec = importlib.util.spec_from_file_location("publicar_kaggle", caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_o_slug_do_dataset_sai_de_um_lugar_so():
    """Os dois metadados carregam o slug, e divergentes dão erro só no Kaggle.

    O `id` do dataset e o `dataset_sources` do notebook têm de casar. Se não
    casarem, o notebook sobe, roda, e morre no `assert` de que não achou o
    `MANIFESTO.json` — depois de consumir cota de GPU.
    """
    m = _publicar()
    assert m.SLUG_DADOS and m.SLUG_NB
    assert m.SLUG_DADOS != m.SLUG_NB, "dataset e notebook não podem ter o mesmo slug"


def test_a_celula_extraida_e_python_valido():
    """O extrator é um regex sobre um arquivo nosso; se o formato mudar, quebra."""
    import ast

    ast.parse(_publicar()._celula())


def test_o_notebook_gerado_nao_reimplementa_o_treino():
    """A mesma garantia do arquivo-fonte, agora depois de passar pelo gerador.

    Um gerador que perdesse a chamada ao `train_embedding.py` produziria um
    notebook que não treina nada, e o teste do arquivo-fonte não veria.
    """
    m = _publicar()
    codigo = "".join(m._ipynb(m._celula())["cells"][0]["source"])
    assert "scripts/train_embedding.py" in codigo
    assert "InfoNCE" not in codigo, "o laço de treino vazou para o notebook"
    assert "backward()" not in codigo, "o laço de treino vazou para o notebook"


def test_o_ipynb_tem_uma_celula_de_codigo_e_nbformat_4():
    m = _publicar()
    nb = m._ipynb("print(1)\n")
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) == 1
    assert nb["cells"][0]["cell_type"] == "code"
    assert nb["cells"][0]["outputs"] == [], "notebook gerado não deve trazer saída"


def test_extrator_quebra_alto_se_o_formato_mudar(tmp_path, monkeypatch):
    """Gerar um notebook vazio em silêncio custaria uma sessão de GPU para descobrir."""
    import pytest

    m = _publicar()
    falso = tmp_path / "sem_celula.py"
    falso.write_text('"""nada aqui."""\n', encoding="utf-8")
    monkeypatch.setattr(m, "FONTE_CELULA", falso)
    with pytest.raises(SystemExit, match="CELULA"):
        m._celula()


def test_o_fonte_nao_sai_com_extensao_zip():
    """⚠️ O Kaggle DESCOMPACTA `.zip` no upload, e isso quebrou o T1a duas vezes.

    Medido em 2026-08-24: `phifm_src.zip` chegou no dataset como o diretório
    `phifm_src/`. O notebook morreu em `FileNotFoundError` aos 26 s — e pior, o hash
    do CÓDIGO deixou de ser conferível, porque o arquivo que o manifesto descreve não
    existia mais no dataset.

    Uma extensão que o Kaggle não reconhece como arquivo preserva os bytes, e o
    `zipfile` abre pelo conteúdo, não pela extensão.
    """
    fonte = (Path(__file__).resolve().parents[2] / "scripts/empacotar_kaggle.py"
             ).read_text(encoding="utf-8")
    assert '"phifm_src.zip.bin"' in fonte
    assert '_zipar_fonte(raiz, a.out / "phifm_src.zip")' not in fonte, (
        "voltou a gravar .zip — o Kaggle vai descompactar e o hash do código morre")


def test_a_celula_aceita_o_fonte_extraido_pelo_kaggle():
    """Um dataset publicado ANTES do conserto tem `phifm_src/` em vez do arquivo.

    Aceitar as duas formas evita exigir um reupload de 211 MB só para rodar — mas o
    caminho extraído tem de AVISAR que a integridade do código não foi conferida.
    """
    celula = CELULA
    assert 'phifm_src.zip.bin' in celula
    assert '(DADOS / "phifm_src").is_dir()' in celula
    assert "não foi conferida" in celula, (
        "o caminho extraído precisa dizer alto que o código não foi verificado")


def test_parquets_ausentes_derrubam_antes_do_treino():
    """Hash não conferido é aviso; dado faltando é parada.

    Depois que a conferência passou a TOLERAR arquivo ausente (para aceitar o fonte
    extraído), nada impediria seguir sem os parquets — e a falha apareceria no meio
    do laço, longe da causa.
    """
    celula = CELULA
    assert 'for obrigatorio in ("pares_treino.parquet", "pares_validacao.parquet")' \
        in celula


def test_manifesto_descreve_o_que_a_etapa_produziu_nao_o_que_sobrou_no_disco(tmp_path):
    """Arquivo obsoleto no diretório de saída não pode entrar na atestação.

    Medido em 2026-08-24: ao renomear o fonte para `.zip.bin`, o `phifm_src.zip`
    antigo ficou no diretório e o manifesto passou a descrever OS DOIS — 175 KB de
    código velho atestados como parte do pacote, e enviados ao Kaggle.

    Um manifesto montado de `iterdir()` descreve o que sobrou no disco, não o que a
    etapa produziu. Isso não atesta nada.
    """
    fonte = (RAIZ / "scripts/empacotar_kaggle.py").read_text(encoding="utf-8")
    assert "ESPERADOS" in fonte, "a lista declarada de conteúdo desapareceu"
    assert "f.name not in ESPERADOS" in fonte, (
        "voltou a aceitar qualquer arquivo do diretório no manifesto")


def test_o_notebook_pede_acelerador_pelo_campo_que_vale():
    """`enable_gpu` sozinho não liga a GPU, e o Kaggle não reclama.

    Medido em 2026-08-24: o push subiu com `enable_gpu: True`, o Kaggle guardou e
    devolveu esse valor no `kernels pull -m`, e a sessão rodou SEM acelerador —
    `torch.cuda.is_available()` False, e o assert do notebook derrubou aos 18 s.

    Quem decide é `accelerator`. Conferir o metadado de volta não pega isto, porque
    o Kaggle ecoa `enable_gpu` de qualquer jeito; só a execução revela.
    """
    fonte = (RAIZ / "scripts/publicar_kaggle.py").read_text(encoding="utf-8")
    assert '"machine_shape": "NvidiaTeslaT4"' in fonte, (
        "sem `machine_shape` o notebook roda em CPU — e em CPU este treino levaria "
        "dias e 'funcionaria'. `enable_gpu` e `accelerator` NAO servem: o primeiro e "
        "aceito e ignorado, o segundo nem e lido")


def test_a_imagem_docker_nao_fica_presa_a_execucao_antiga():
    """O Kaggle FIXA a imagem docker do kernel, e o pin sobrevive a novos pushes.

    Medido em 2026-08-24/25: quatro execuções rodaram com `torch 2.10.0+cpu` mesmo
    com `machine_shape: NvidiaTeslaT4` confirmado pelo `kernels pull -m`. O kernel
    seguia preso à imagem da PRIMEIRA execução, que foi sem acelerador — e GPU
    alocada não muda o PyTorch instalado dentro de uma imagem CPU-only.

    `latest` manda o Kaggle escolher a imagem atual apropriada ao acelerador.
    `original` é o comportamento que criou o problema.
    """
    fonte = (RAIZ / "scripts/publicar_kaggle.py").read_text(encoding="utf-8")
    assert '"docker_image_pinning_type": "latest"' in fonte, (
        "sem isto o kernel arrasta a imagem de CPU da primeira execução e o treino "
        "roda sem GPU mesmo com o acelerador pedido")
