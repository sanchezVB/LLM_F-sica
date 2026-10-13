"""O juiz independente da §9.6, no Colab.

O que este arquivo tranca:

1. **É o juiz da regra com outro modelo**: as células julgam por `ji.julgar_pacote`, que
   chama `assistente_bracos.julgar` — o prompt, a entrada e o leitor não são reescritos lá.
2. **O modelo é o fixado antes de o teste rodar**, conferido por tamanho e sha256, e o
   llama.cpp é o do build local.
3. **O notebook não imprime acerto**: ele não sabe o braço de texto nenhum.
4. **Retoma** de onde parou, e a regra está escrita antes.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "colab"))

import juiz_independente as cel  # noqa: E402

from phifm.eval import juiz_independente as ji  # noqa: E402


def _sem_marcadores(texto: str) -> str:
    for marcador, valor in (("__SHA__", "0" * 40), ("__REPO__", "dono/repo"),
                            ("__PASTA__", "/content/drive/MyDrive/phifm"),
                            ("__HASHES__", "{}"), ("__REGRA__", "regra")):
        texto = texto.replace(marcador, valor)
    return texto


CELULA = _sem_marcadores("\n".join(cel.CELULAS))


def test_as_celulas_sao_python_valido():
    for fonte in cel.CELULAS:
        ast.parse(_sem_marcadores(fonte))


def test_e_o_juiz_da_regra_com_outro_modelo():
    assert "ji.julgar_pacote(modelo, pacote, SAIDA" in cel.JULGAR
    assert "modelo = ji.ModeloDeConversa(URL)" in cel.MODELO
    for reescrito in ("You grade one answer", "def julgar", "def ler_veredicto"):
        assert reescrito not in CELULA, reescrito
    # o ensaio usa o prompt e os parâmetros do juiz da regra
    assert "b.SISTEMA_JUIZ" in cel.MODELO
    assert "max_tokens=260, temperatura=0.0, semente=SEMENTE" in cel.MODELO
    # uma conversa por vez, como o servidor local
    assert '"-np", "1"' in cel.MODELO and '"-c", str(ji.CONTEXTO)' in cel.MODELO


def test_o_modelo_e_o_fixado_antes_do_teste_e_e_conferido():
    assert ji.MODELO_REPO == "bartowski/mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF"
    assert ji.MODELO_ARQUIVO.endswith("Q4_K_M.gguf") and len(ji.MODELO_SHA256) == 64
    assert len(ji.MODELO_REVISAO) == 40 and ji.LLAMA_CPP == "b11159"
    assert "revision=ji.MODELO_REVISAO" in cel.MODELO
    assert "GGUF.stat().st_size == ji.MODELO_BYTES" in cel.MODELO
    assert "h.hexdigest() == ji.MODELO_SHA256" in cel.MODELO
    assert '"--branch", LLAMA' in cel.SETUP and "LLAMA = ji.LLAMA_CPP" in cel.SETUP
    assert "Qwen" not in ji.MODELO_REPO and "Qwen" not in ji.MODELO_ARQUIVO


def test_o_pacote_e_conferido_e_o_notebook_nao_imprime_acerto():
    assert 'sha_pacote != HASHES["pacote_juiz2.json"]' in cel.SETUP
    assert "HASHES = __HASHES__" in cel.SETUP
    for proibido in (".braco", "arxiv_id", "acerto", "nao_piora", "apurar_i3", "kappa"):
        assert proibido not in CELULA, proibido


def test_ABORTA_sem_L4_RETOMA_e_o_ensaio_vem_antes_de_julgar():
    assert cel.GPU == "L4" and "int(MEM_GPU) >= 20000" in cel.SETUP
    assert "len(ji.ler(SAIDA))" in cel.JULGAR and "subir_servidor()" in cel.JULGAR
    assert cel.CELULAS.index(cel.MODELO) < cel.CELULAS.index(cel.JULGAR)
    assert "veredicto == esperado" in cel.MODELO and "Nada foi julgado" in cel.MODELO


def test_a_regra_esta_escrita_ANTES_e_diz_como_se_le():
    r = " ".join(cel.REGRA.split())
    for exigido in ("escrita ANTES de o teste rodar", "180 veredictos", ">= 0,6",
                    "NÃO é ajustado", "robusto", "FRÁGIL", "a decisão é a do juiz da regra"):
        assert exigido in r, exigido


def test_o_notebook_gerado_fixa_o_commit_e_o_pacote():
    if not cel.MANIFESTO.exists():
        pytest.skip("o pacote ainda não foi empacotado (o teste não terminou)")
    nb = RAIZ / "colab" / "juiz_independente.ipynb"
    assert nb.exists(), "o notebook não foi gerado"
    dados = json.loads(nb.read_text(encoding="utf-8"))
    texto = json.dumps(dados)
    for marcador in ("__SHA__", "__HASHES__", "__REGRA__", "__PASTA__", "__REPO__"):
        assert marcador not in texto, marcador
    assert cel.HASHES["pacote_juiz2.json"] in texto and "L4 GPU" in texto
    assert dados["metadata"]["colab"]["gpuType"] == "L4"
