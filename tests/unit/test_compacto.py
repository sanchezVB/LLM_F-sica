"""Os vetores do índice em int8 (DOC-13 §8.1): a conta, a conversão, a guarda contra o
índice errado e a regra de equivalência — com vetores sintéticos, sem modelo."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.retrieval import compacto as cp  # noqa: E402


def _vetores(n=20_000, d=384, semente=0) -> np.ndarray:
    v = np.random.default_rng(semente).normal(size=(n, d)).astype(np.float32)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float16)


def test_pontuar_e_o_produto_com_o_vetor_reconstruido():
    v = _vetores(2000)
    e = cp.escala(v, n_amostra=2000)
    m = cp.quantizar(v, e, bloco=700)
    q = v[5].astype(np.float32)
    esperado = (m.astype(np.float32) * e) @ q
    np.testing.assert_allclose(cp.pontuar(m, e, q, bloco=300), esperado, rtol=1e-5, atol=1e-6)
    assert m.dtype == np.int8 and m.nbytes * 2 == v.nbytes


def test_a_ordem_quase_nao_muda_nos_50_primeiros():
    v = _vetores()
    e = cp.escala(v)
    m = cp.quantizar(v, e)
    rng = np.random.default_rng(1)
    sobre, alvo_no_topo = [], 0
    for j in rng.choice(v.shape[0], 30, replace=False):
        q = v[j].astype(np.float32) + rng.normal(scale=0.03, size=v.shape[1]).astype(np.float32)
        q /= np.linalg.norm(q)
        exato = v.astype(np.float32) @ q
        aprox = cp.pontuar(m, e, q)
        assert np.corrcoef(exato, aprox)[0, 1] > 0.999
        sobre.append(cp.sobreposicao(cp.topo(exato, 50), cp.topo(aprox, 50)))
        alvo_no_topo += cp.topo(aprox, 1)[0] == j
    assert np.mean(sobre) > 0.9 and alvo_no_topo == 30


def test_corta_em_127_e_recusa_dimensao_constante():
    v = np.zeros((10, 4), dtype=np.float16)
    with pytest.raises(ValueError, match="escala zero"):
        cp.escala(v)
    e = np.full(4, 0.01, dtype=np.float32)
    m = cp.quantizar(np.array([[5.0, -5.0, 0.004, -0.006]], dtype=np.float16), e)
    assert m.tolist() == [[127, -127, 0, -1]]


def test_topo_desempata_como_a_busca():
    s = np.array([0.5, 0.9, 0.9, 0.1, 0.7])
    assert cp.topo(s, 3) == [1, 2, 4] and cp.topo(s, 99) == [1, 2, 4, 0, 3]


def _indice(tmp_path: Path, modelo="abc") -> Path:
    indice = tmp_path / "indice"
    indice.mkdir()
    np.save(indice / "vetores.npy", _vetores(3000))
    (indice / "manifesto.json").write_text(json.dumps({"modelo_blake3": modelo}),
                                           encoding="utf-8")
    return indice


def test_converter_e_carregar_amarrados_ao_indice(tmp_path):
    indice = _indice(tmp_path)
    man = cp.converter(indice, tmp_path / "int8")
    assert man["documentos"] == 3000 and man["bytes_int8"] * 2 == man["bytes_float16"]
    m, e = cp.carregar(tmp_path / "int8", indice)
    assert m.shape == (3000, 384) and e.shape == (384,)
    (indice / "manifesto.json").write_text(json.dumps({"modelo_blake3": "outro"}),
                                           encoding="utf-8")
    with pytest.raises(ValueError, match="não é deste índice"):
        cp.carregar(tmp_path / "int8", indice)


def test_a_regra_de_equivalencia():
    bom = [{"sobreposicao_50_fundida": 0.99, "p_exato": True, "p_int8": True}] * 100
    assert cp.decidir(bom)["equivalente"]
    raso = [{"sobreposicao_50_fundida": 0.95, "p_exato": True, "p_int8": True}] * 100
    assert not cp.decidir(raso)["equivalente"]
    perde = bom[:97] + [{"sobreposicao_50_fundida": 1.0, "p_exato": True, "p_int8": False}] * 2
    r = cp.decidir(perde + [{"sobreposicao_50_fundida": 1.0, "p_exato": False, "p_int8": True}])
    assert (r["perde_o_artigo"], r["ganha_o_artigo"], r["equivalente"]) == (2, 1, True)
    assert not cp.decidir(perde)["equivalente"]
    with pytest.raises(ValueError):
        cp.decidir([])


def test_medir_item_compara_as_duas_buscas_com_o_mesmo_bm25():
    import medir_vetores_compactos as script

    v = _vetores(500)
    e = cp.escala(v, n_amostra=500)
    m = cp.quantizar(v, e)

    class Busca:
        def embutir(self, texto):
            return v[7].astype(np.float32)

        def pontuar(self, q):
            return v.astype(np.float32) @ q

    class Lexica:
        def pontuar(self, texto):
            s = np.zeros(500)
            s[[3, 7, 9]] = [5.0, 4.0, 3.0]
            return s

    r = script.medir_item(Busca(), Lexica(), m, e, {"x7": 7}, "corpo_1",
                          {"arxiv_id": "x7", "hyde": "h", "ingles": "i"})
    assert r["p_exato"] and r["p_int8"] and r["sobreposicao_50_fundida"] > 0.9


def test_a_regra_e_proposta_e_a_medida_recusa(monkeypatch):
    import medir_vetores_compactos as script

    assert cp.REGRA_8_1_ACEITA is False
    with pytest.raises(SystemExit, match="PROPOSTA"):
        script.medir()
