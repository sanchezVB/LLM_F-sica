"""Objetivo causal, retomada e identidade com modelo local minúsculo."""
import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel

from phifm.training.causal import ConfigCausal, perda_de_dev, treinar_causal


def modelo():
    torch.manual_seed(7)
    return GPT2LMHeadModel(GPT2Config(vocab_size=24, n_positions=16, n_embd=16,
                                    n_layer=1, n_head=2, resid_pdrop=0.1))


def dados():
    return torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]] * 4)


def test_retomada_repete_pesos_e_historico(tmp_path):
    cfg = ConfigCausal(passos=4, lr=0.01, salvar_a_cada=2)
    direto, parcial = modelo(), modelo()
    r1 = treinar_causal(direto, dados(), dados(), tmp_path / "direto", cfg, {"base": "mini"})
    r2 = treinar_causal(parcial, dados(), dados(), tmp_path / "resume", cfg,
                        {"base": "mini"}, passos_nesta_sessao=2)
    assert not r2["concluido"]
    novo = modelo()
    r3 = treinar_causal(novo, dados(), dados(), tmp_path / "resume", cfg, {"base": "mini"})
    assert r3["concluido"] and r3["historico"] == r1["historico"]
    assert all(torch.equal(p, novo.state_dict()[k]) for k, p in direto.state_dict().items())
    assert r1["perda_dev_final"] < r1["perda_dev_inicial"]


def test_objetivo_preve_proximo_token_e_avaliacao_preserva_modo():
    m = modelo()
    m.eval()
    x = dados()
    with torch.no_grad():
        logits = m(input_ids=x).logits
        esperado = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 24),
                                                      x[:, 1:].reshape(-1)).item()
    assert perda_de_dev(m, x, 2) == pytest.approx(esperado)
    assert not m.training


def test_troca_dos_dados_recusa_checkpoint(tmp_path):
    cfg = ConfigCausal(passos=2)
    treinar_causal(modelo(), dados(), dados(), tmp_path / "run", cfg, {"base": "mini"})
    alterado = dados().clone()
    alterado[0, 0] = 6
    with pytest.raises(ValueError, match="outra"):
        treinar_causal(modelo(), alterado, dados(), tmp_path / "run", cfg, {"base": "mini"})


def test_falha_de_gravacao_preserva_checkpoint(tmp_path, monkeypatch):
    from phifm.training import causal
    arquivo = tmp_path / "estado.pt"
    causal._gravar_atomico(arquivo, {"passo": 1})
    original = copy.copy(arquivo.read_bytes())
    def falhar(*args, **kwargs):
        raise OSError("disco")
    monkeypatch.setattr(causal.torch, "save", falhar)
    with pytest.raises(OSError):
        causal._gravar_atomico(arquivo, {"passo": 2})
    assert arquivo.read_bytes() == original
