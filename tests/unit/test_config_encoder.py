"""A configuração do ΦEnc, e a conferência que impede um modelo do tamanho errado.

"ΦEnc-150M" é um nome, e um nome não impõe nada. O orçamento de computação do
DOC-07 §2.4 é `C = 6 × N × D` — **linear em N** —, então um modelo 27% maior que o
declarado custa 27% mais e ninguém percebe até a fatura.

Estes testes rodam na suíte RÁPIDA: a contagem é analítica e não constrói pesos. O
teste que compara com o modelo de verdade está separado e pula sem torch — a mesma
divisão de `test_phirank_do_sistema.py`, e pela mesma razão.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.models.encoder.config import (  # noqa: E402
    CONFIGS,
    PHIENC_150M,
    PHIENC_400M,
    PROXY_BAKEOFF,
    VOCAB_PHIENC,
    ConfigEnc,
    flops_de_treino,
    obter,
)


def test_as_configs_do_documento_fecham_com_o_tamanho_declarado():
    """A asserção central. Se falhar, ou a config mudou ou o documento mentia."""
    for cfg in (PHIENC_150M, PHIENC_400M):
        cfg.conferir_declarado()  # levanta se fugir
        real = cfg.parametros()["total"] / 1e6
        assert abs(real - cfg.declarado_M) / cfg.declarado_M <= 0.08, real


def test_o_150M_da_exatamente_o_que_foi_medido():
    """142.420.480 — conferido contra `ModernBertForMaskedLM` construído.

    Fixar o número exato aqui é o que transforma uma mudança de estrutura do
    `transformers` em falha de teste, em vez de num modelo silenciosamente de outro
    tamanho com o orçamento do §2.4 desatualizado.
    """
    p = PHIENC_150M.parametros()
    assert p["total"] == 142_420_480
    assert p["embedding"] == VOCAB_PHIENC * 768 + 768


def test_a_conferencia_reprova_uma_config_do_tamanho_errado():
    """Era com `ffn=1536` que o 400M dava 292,7 M, e foi isto que pegou."""
    errada = ConfigEnc(nome="ΦEnc-400M-errado", camadas=28, d_model=1024,
                       cabecas=16, ffn=1536, declarado_M=400)
    with pytest.raises(ValueError, match="declara 400"):
        errada.conferir_declarado()
    # E `obter` confere, para que ninguém pegue uma config torta pelo registro.
    for nome in CONFIGS:
        obter(nome)


def test_a_fracao_de_embedding_nao_e_os_16_por_cento_do_documento():
    """⚠️ O DOC-07 §2.2 afirma 16% e o real é 22,1%.

    De onde vem o 16%: a tabela do §7.2 calcula `V × d` e divide pelos 150 M
    NOMINAIS, não pelo total, e usa V=32.000. A consequência não é acadêmica — eu
    disse ao usuário que V=65.536 levaria a embedding "de 16% para ~24%", e o real é
    22,1% → 31,2%. A troca é mais dura que o documento diz, e o argumento da §7.3
    para ficar em 40.960 fica mais forte.
    """
    assert PHIENC_150M.fracao_de_embedding() == pytest.approx(0.221, abs=0.002)
    maior = ConfigEnc(nome="v65k", camadas=22, d_model=768, cabecas=12, ffn=1152,
                      vocab=65_536)
    assert maior.fracao_de_embedding() == pytest.approx(0.312, abs=0.002)
    # E o documento continua com o número errado até alguém corrigi-lo; o teste
    # afirma o MEDIDO, não o documentado.
    assert PHIENC_150M.fracao_de_embedding() > 0.16


def test_o_flops_inclui_a_projecao_de_saida():
    """⚠️ Eu errei isto primeiro, e quase registrei como correção do DOC-07.

    "Consulta de tabela não é matmul" está certo para a ENTRADA e errado para o
    resto: com `tie_word_embeddings`, a mesma matriz volta como projeção de saída, e
    aí é `d × V` por token — 31,5 M de MACs contra 110 M do corpo, 29% a mais.

    O §2.4 usa N=1,5e8 e dá 2,7e19 para 30 B tokens. A conta correta dá 2,56e19,
    dentro de 5%: **o documento estava certo.**
    """
    c = flops_de_treino(PHIENC_150M, 30e9)
    assert c == pytest.approx(2.56e19, rel=0.02), f"{c:.3e}"
    assert abs(c - 2.7e19) / 2.7e19 < 0.06, "deixou de bater com o §2.4"

    # Excluir a projeção subestimaria em ~26%, que é o erro que eu quase gravei.
    p = PHIENC_150M.parametros()
    so_corpo = 6.0 * (p["camadas"] + p["cabeca"] - PHIENC_150M.vocab) * 30e9
    assert so_corpo < 0.8 * c


def test_o_flops_e_linear_nos_tokens():
    a = flops_de_treino(PHIENC_150M, 1e9)
    assert flops_de_treino(PHIENC_150M, 3e9) == pytest.approx(3 * a)


def test_o_proxy_do_bakeoff_expoe_o_problema_do_paragrafo_11_2():
    """O DOC-05 §11.2 pede "~50 M por variante", e a 50 M a embedding domina.

    Com V=40.960 o proxy é 43,7% embedding; com V=65.536 a embedding sozinha
    passaria de 50 M, antes de qualquer camada. Nesse regime a comparação entre
    vocabulários mede quantos parâmetros de embedding cada variante tem, não qual
    tokenizer produz o melhor modelo.

    O teste fixa o fato, para que o §11.2 seja revisado antes de rodar.
    """
    assert PROXY_BAKEOFF.fracao_de_embedding() > 0.40
    so_embedding = 65_536 * PROXY_BAKEOFF.d_model
    assert so_embedding > 30e6, (
        "a embedding de V=65.536 neste d_model deixou de dominar; a ressalva do "
        "§11.2 pode ter sido resolvida — reconfira antes de apagar este teste")
    # E o proxy não declara tamanho, porque o documento não declara nenhum.
    assert PROXY_BAKEOFF.declarado_M is None


def test_dimensoes_incoerentes_levantam_na_construcao_da_config():
    with pytest.raises(ValueError, match="não divide por"):
        ConfigEnc(nome="x", camadas=2, d_model=768, cabecas=7, ffn=1152)
    with pytest.raises(ValueError, match="janela local"):
        ConfigEnc(nome="x", camadas=2, d_model=768, cabecas=12, ffn=1152,
                  contexto=1000)


def test_o_contexto_e_8192_e_nao_512():
    """É a vantagem estrutural do DOC-07 §2.1 sobre SciBERT e PhysBERT, que operam
    em 512. Um paper de Física não cabe em 512 tokens — nem a introdução cabe."""
    assert PHIENC_150M.contexto == 8192
    assert PHIENC_150M.para_transformers()["max_position_embeddings"] == 8192


def test_sem_bias_em_nenhum_lugar():
    """DOC-07 §2.1: pre-norm sem termos de bias, por estabilidade de treino."""
    hf = PHIENC_150M.para_transformers()
    for campo in ("norm_bias", "mlp_bias", "attention_bias"):
        assert hf[campo] is False, campo


def test_os_ids_especiais_batem_com_o_tokenizer_treinado():
    """⚠️ Um `pad_token_id` diferente do do tokenizer faz o unpadding do ModernBERT
    descartar as posições erradas — sem erro, só com perda que não desce.

    Os ids vêm de `data/processed/tokenizer/variante_A.json`, que é gitignored; por
    isso estão fixos aqui e o teste compara com os literais em vez de ler o arquivo.
    """
    hf = PHIENC_150M.para_transformers()
    assert hf["pad_token_id"] == 0
    assert hf["cls_token_id"] == 2
    assert hf["sep_token_id"] == 3
    assert hf["vocab_size"] == 40_960


def test_obter_com_nome_desconhecido_lista_os_conhecidos():
    with pytest.raises(SystemExit, match="Conhecidas"):
        obter("ΦEnc-1B")


def test_como_dict_carrega_a_contagem_para_o_manifesto():
    d = PHIENC_150M.como_dict()
    assert d["parametros"]["total"] == 142_420_480
    assert d["fracao_de_embedding"] == pytest.approx(0.221, abs=0.001)


# ─── a construção de verdade, que pula sem torch ─────────────────────────────


def test_o_modelo_construido_tem_a_contagem_analitica():
    """O que liga a fórmula ao `transformers`. Pula sem torch, e é por isso que os
    testes de cima fixam o número exato: eles rodam no CI, este não."""
    pytest.importorskip("torch", reason="requer a venv de treino (.venv-treino)")
    pytest.importorskip("transformers")
    from phifm.models.encoder.modelo import construir

    modelo = construir(PROXY_BAKEOFF)
    real = sum(p.numel() for p in modelo.parameters())
    assert real == PROXY_BAKEOFF.parametros()["total"]


def test_nunca_pede_flash_attention_2():
    """FA-2 exige SM 8.0+; a T4 é 7.5 e o DirectML não a tem.

    Pedi-la levanta na construção do modelo — descobrir isso depois de montar o
    dataloader é perder a sessão de GPU.
    """
    # ⚠️ `so_codigo_de` remove comentarios E docstrings. A docstring de
    # `modelo.py` explica que a T4 nao tem FA-2, e a primeira versao deste teste
    # reprovou justamente essa explicacao — quinta ocorrencia da armadilha, e a
    # primeira que aconteceu DEPOIS de `tests/conftest.py` existir para evita-la.
    from conftest import so_codigo_de

    codigo = so_codigo_de(Path(__file__).resolve().parents[2]
                          / "src/phifm/models/encoder/modelo.py")
    assert "flash_attention_2" not in codigo, (
        "o código voltou a pedir FA-2, que nenhuma das duas GPUs tem")
    assert "reference_compile=False" in codigo, (
        "o `torch.compile` do ModernBERT voltou; ele falha no DirectML e custa "
        "minutos sem CUDA")
