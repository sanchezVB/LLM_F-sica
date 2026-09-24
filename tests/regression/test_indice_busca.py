"""O índice de busca, num modelo minúsculo, em CPU.

Tranca as três coisas que um índice pode errar sem dar erro (ver `phifm.retrieval.indice`)
e a retomada, que é o que torna seguro deixar a indexação de ~1 h rodando:

1. cada documento, buscado pelo próprio texto, volta em primeiro;
2. um índice interrompido e retomado é BIT A BIT o de uma execução contínua;
3. consultar com outro modelo é recusado — e retomar com outro modelo também;
4. um índice sem manifesto (não terminado) não abre.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

torch = pytest.importorskip("torch", reason="requer a venv de treino (.venv-treino)")
pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402

from phifm.retrieval.indice import (  # noqa: E402
    NOME_VETORES,
    Busca,
    construir,
    documentos,
)

PALAVRAS = [f"w{i}" for i in range(200)]


def _modelo(destino: Path, semente: int) -> Path:
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, ".": 5}
    vocab.update({p: i + 6 for i, p in enumerate(PALAVRAS)})
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]").save_pretrained(destino)
    torch.manual_seed(semente)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                         num_attention_heads=2, intermediate_size=64,
                         max_position_embeddings=256)).save_pretrained(destino)
    return destino


def _spine(destino: Path, n: int = 40) -> Path:
    rng = np.random.default_rng(7)
    linhas = []
    for i in range(n):
        titulo = " ".join(rng.choice(PALAVRAS, 4))
        resumo = " ".join(rng.choice(PALAVRAS, 40))
        linhas.append({"arxiv_id": f"2601.{i:05d}", "title": titulo, "abstract": resumo,
                       "year": 2026, "primary_category": "hep-th",
                       "authors": [f"Autora {i}", "Outro"]})
    pl.DataFrame(linhas).write_parquet(destino)
    return destino


@pytest.fixture
def cenario(tmp_path):
    return (_spine(tmp_path / "spine.parquet"), _modelo(tmp_path / "modelo_a", 1),
            _modelo(tmp_path / "modelo_b", 2))


def test_cada_documento_se_acha_em_PRIMEIRO(tmp_path, cenario):
    spine, modelo, _ = cenario
    construir(spine, modelo, tmp_path / "indice", dispositivo="cpu", lote=8, bloco=16)
    busca = Busca(tmp_path / "indice", dispositivo="cpu")
    docs = documentos(spine)
    for i in (0, 7, 23, 39):
        res = busca.buscar(docs["texto"][i], k=3)
        assert res[0].arxiv_id == docs["arxiv_id"][i]
        assert res[0].escore == pytest.approx(1.0, abs=2e-3)  # float16 no índice
        assert res[0].link == f"https://arxiv.org/abs/{docs['arxiv_id'][i]}"
        assert res[0].autores == [f"Autora {i}", "Outro"]


def test_a_RETOMADA_da_o_mesmo_indice_de_uma_execucao_continua(tmp_path, cenario):
    spine, modelo, _ = cenario
    continuo = tmp_path / "continuo"
    construir(spine, modelo, continuo, dispositivo="cpu", lote=8, bloco=16)
    retomado = tmp_path / "retomado"
    r = construir(spine, modelo, retomado, dispositivo="cpu", lote=8, bloco=16,
                  max_blocos=1)
    assert r["interrompido"] and r["feitos"] == 16
    construir(spine, modelo, retomado, dispositivo="cpu", lote=8, bloco=16)
    assert np.array_equal(np.load(continuo / NOME_VETORES), np.load(retomado / NOME_VETORES))


def test_consultar_com_OUTRO_modelo_e_recusado(tmp_path, cenario):
    spine, modelo_a, modelo_b = cenario
    construir(spine, modelo_a, tmp_path / "indice", dispositivo="cpu", lote=8, bloco=16)
    with pytest.raises(SystemExit, match="não é o que gerou este índice"):
        Busca(tmp_path / "indice", modelo=modelo_b, dispositivo="cpu")


def test_retomar_com_OUTRO_modelo_e_recusado(tmp_path, cenario):
    spine, modelo_a, modelo_b = cenario
    saida = tmp_path / "indice"
    construir(spine, modelo_a, saida, dispositivo="cpu", lote=8, bloco=16, max_blocos=1)
    with pytest.raises(SystemExit, match="OUTRO modelo"):
        construir(spine, modelo_b, saida, dispositivo="cpu", lote=8, bloco=16)


def test_indice_NAO_TERMINADO_nao_abre(tmp_path, cenario):
    spine, modelo, _ = cenario
    saida = tmp_path / "indice"
    construir(spine, modelo, saida, dispositivo="cpu", lote=8, bloco=16, max_blocos=1)
    with pytest.raises(SystemExit, match="não terminou"):
        Busca(saida, dispositivo="cpu")


# ── a página no navegador ────────────────────────────────────────────────────


def test_o_SERVIDOR_responde_a_pagina_e_a_busca(tmp_path, cenario):
    """A rota JSON devolve o documento em primeiro, e as recusas têm status certo."""
    import json
    import threading
    import urllib.error
    import urllib.parse
    import urllib.request

    from phifm.serving.web import criar_servidor

    spine, modelo, _ = cenario
    construir(spine, modelo, tmp_path / "indice", dispositivo="cpu", lote=8, bloco=16)
    busca = Busca(tmp_path / "indice", dispositivo="cpu")
    servidor = criar_servidor(busca, porta=0)
    assert servidor.server_address[0] == "127.0.0.1", "só esta máquina enxerga a busca"
    fio = threading.Thread(target=servidor.serve_forever, daemon=True)
    fio.start()
    base = f"http://127.0.0.1:{servidor.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/") as r:
            assert r.status == 200 and "Busca em artigos de Física" in r.read().decode()
        docs = documentos(spine)
        q = urllib.parse.quote(docs["texto"][5])
        with urllib.request.urlopen(f"{base}/api/buscar?k=3&q={q}") as r:
            dados = json.loads(r.read())
        assert dados["resultados"][0]["arxiv_id"] == docs["arxiv_id"][5]
        assert len(dados["resultados"]) == 3
        for caminho, status in (("/api/buscar?q=", 400), ("/nada", 404),
                                ("/api/buscar?q=w1&k=abc", 400)):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(base + caminho)
            assert e.value.code == status, caminho
    finally:
        servidor.shutdown()
        servidor.server_close()
