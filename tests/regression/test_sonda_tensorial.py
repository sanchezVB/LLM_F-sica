"""A sonda de estrutura tensorial, e as formas de ela mentir.

A sonda não treina nada: ela compara quanto uma edição **significativa** move a
representação contra quanto uma edição **no-op** move. Isso remove a confusão de
capacidade da sondagem supervisionada, e põe todo o peso no inventário de pares —
que passa a ser onde os defeitos moram.

Três deles são silenciosos e cada um tem teste aqui:

1. **Controle igual à base.** Distância zero em todas as bases, e a sonda declara
   estrutura tensorial em qualquer modelo. Aconteceu na primeira versão do
   inventário, no par `hall`: `\\rho_{xy}` não tem índice mudo, então não existe
   edição no-op para ele, e eu pus a base no lugar do controle.
2. **Controle que não é no-op.** Aí os dois arms mudam a Física e a comparação
   perde o sentido — ela passa a medir qual mudança é maior, não qual é real.
3. **Par de ordem sobre tensor SIMÉTRICO.** Trocar os índices de `g_{\\mu\\nu}` é
   no-op, então o par entraria invertido: o "significativo" não muda nada.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

torch = pytest.importorskip("torch", reason="requer a venv de treino (.venv-treino)")
pytest.importorskip("transformers", reason="requer a venv de treino (.venv-treino)")

from phifm.eval import tensorial as t  # noqa: E402

VOCAB = 128


def _encoder(semente: int):
    """Um BERT minúsculo com tokenizer de caractere — o suficiente para a costura."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    letras = sorted({c for p in t.PARES
                     for c in (p.base + p.significativo + p.controle + t.MOLDURA)})
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for c in letras:
        vocab.setdefault(c, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]",
                                  unk_token="[UNK]", cls_token="[CLS]",
                                  sep_token="[SEP]", mask_token="[MASK]")

    torch.manual_seed(semente)
    mod = BertModel(BertConfig(vocab_size=len(vocab), hidden_size=32,
                               num_hidden_layers=2, num_attention_heads=2,
                               intermediate_size=48, max_position_embeddings=256)).eval()
    return mod, tok


# ── 1. o inventário de pares ────────────────────────────────────────────────

def test_nenhum_controle_e_igual_a_base():
    """O defeito que inflaria a sonda em TODAS as bases ao mesmo tempo.

    Controle idêntico à base dá distância zero, então `d_sig > d_ctl` vale sempre
    e a sonda declara estrutura tensorial em qualquer modelo, inclusive num de
    pesos aleatórios.
    """
    iguais = [p.nome for p in t.PARES if p.controle == p.base]
    assert not iguais, (
        f"o controle é igual à base em {iguais}: a distância sai zero e a sonda "
        "passa a aprovar qualquer modelo")


def test_nenhum_significativo_e_igual_a_base():
    iguais = [p.nome for p in t.PARES if p.significativo == p.base]
    assert not iguais, f"edição significativa que não edita nada: {iguais}"


def test_significativo_e_controle_sao_diferentes_entre_si():
    iguais = [p.nome for p in t.PARES if p.significativo == p.controle]
    assert not iguais, f"os dois arms do par são a mesma cadeia: {iguais}"


def test_todo_par_declara_a_fisica_que_o_justifica():
    """Um par sem justificativa defensável não deve estar no inventário."""
    for p in t.PARES:
        assert len(p.porque) > 20, f"{p.nome} sem justificativa: {p.porque!r}"
        assert p.categoria in {"ordem", "posicao", "contracao"}, p.categoria


def test_a_metrica_simetrica_NAO_esta_entre_os_pares_de_ordem():
    """`g_{\\mu\\nu}` é simétrica: trocar os índices é no-op, e o par entraria
    invertido. É o exemplo mais citado no ESTADO.md, e por isso o mais fácil de
    incluir sem pensar."""
    for p in t.PARES:
        if p.categoria == "ordem":
            assert "g_{" not in p.base, (
                f"{p.nome} usa a métrica num par de ordem; ela é simétrica")


def test_os_textos_saem_na_ordem_que_sondar_espera():
    """`sondar` indexa por `3*i`, `3*i+1`, `3*i+2`. Um desencontro aqui compararia
    a base de um par com o controle de outro, sem nada avisar."""
    textos = t.textos_da_sonda(com_moldura=False)
    assert len(textos) == 3 * len(t.PARES)
    for i, p in enumerate(t.PARES):
        assert textos[3 * i] == p.base
        assert textos[3 * i + 1] == p.significativo
        assert textos[3 * i + 2] == p.controle


# ── 2. a distância de edição, que é a confusão declarada ────────────────────

def test_edicao_conta_o_que_diz_contar():
    assert t._edicao("abc", "abc") == 0
    assert t._edicao("abc", "abd") == 1
    assert t._edicao("", "abc") == 3


def test_a_confusao_de_superficie_fica_registrada(tmp_path):
    """A média de edição de cada arm vai para o artefato. Sem ela, um leitor não
    consegue distinguir «tem estrutura tensorial» de «mexeu mais caracteres»."""
    mod, tok = _encoder(0)
    s = t.sondar(mod, tok, "teste", max_tokens=96)
    assert s.edicao_significativo > 0 and s.edicao_controle > 0
    for p in s.por_par:
        assert "edicao_significativo" in p and "edicao_controle" in p


# ── 3. a sondagem ───────────────────────────────────────────────────────────

def test_sondar_devolve_uma_linha_por_par(tmp_path):
    mod, tok = _encoder(0)
    s = t.sondar(mod, tok, "teste", max_tokens=96)
    assert len(s.por_par) == len(t.PARES)
    assert s.bases_a_favor + s.bases_contra <= len(t.PARES)
    assert -1.0 <= s.indice <= 1.0


def test_o_indice_nao_estoura_com_distancias_minusculas():
    """Normalizar pela soma e não pelo controle: com duas distâncias ~1e-6 uma
    razão explodiria, e o regime normal desta sonda é exatamente esse."""
    s = t.Sondagem(nome="x", dist_significativo=0.0, dist_controle=0.0,
                   bases_a_favor=0, bases_contra=0, p=1.0,
                   edicao_significativo=0.0, edicao_controle=0.0)
    assert s.indice == 0.0


def test_mover_MAIS_no_controle_e_reportado_como_defeito():
    """O desfecho que significa «a representação responde a caracteres»."""
    s = t.Sondagem(nome="x", dist_significativo=0.01, dist_controle=0.05,
                   bases_a_favor=0, bases_contra=8, p=0.0078,
                   edicao_significativo=2.0, edicao_controle=3.0)
    assert "MAIS no par de controle" in t._leitura(s)


def test_sem_discordantes_e_INCONCLUSIVO():
    s = t.Sondagem(nome="x", dist_significativo=0.0, dist_controle=0.0,
                   bases_a_favor=0, bases_contra=0, p=1.0,
                   edicao_significativo=0.0, edicao_controle=0.0)
    assert t._leitura(s).startswith("INCONCLUSIVO")


# ── 4. a comparação entre braços ────────────────────────────────────────────

def test_comparar_bracos_reconhece_a_ablacao():
    mod_c, tok = _encoder(0)
    mod_t, _ = _encoder(1)
    d = t.comparar_bracos({"controle": t.sondar(mod_c, tok, "controle", max_tokens=96),
                           "tratado": t.sondar(mod_t, tok, "tratado", max_tokens=96)})
    assert "ablacao" in d
    assert "delta_indice" in d["ablacao"]
    assert d["pares"] == len(t.PARES)


def test_sem_os_dois_bracos_nao_ha_secao_de_ablacao():
    mod, tok = _encoder(0)
    d = t.comparar_bracos({"physbert": t.sondar(mod, tok, "physbert", max_tokens=96)})
    assert "ablacao" not in d, "comparou ablação com um braço só"


def test_a_leitura_da_ablacao_declara_a_falta_de_poder():
    """Com 8 pares o teste exato mal separa — dizer isso é melhor que reportar
    «empate» como se a amostra bastasse."""
    teste = {"discordantes": 4, "p": 0.6}
    assert "poder" in t._leitura_ablacao(teste, 2, 2, len(t.PARES))


def test_a_ressalva_sobre_pares_sintetizados_acompanha_o_resultado():
    mod, tok = _encoder(0)
    d = t.comparar_bracos({"x": t.sondar(mod, tok, "x", max_tokens=96)})
    assert "SINTETIZADOS" in d["ressalva"]


# ── 5. o tokenizer que colapsa, e a atribuição da culpa ─────────────────────

def _tokenizer_cego():
    """Mapeia tudo para `[UNK]`: nenhum par sobrevive à tokenização."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.WordLevel(vocab={"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]",
                                   pad_token="[PAD]")


def test_tokenizer_que_colapsa_e_denunciado_como_tal():
    """A distinção que a sonda existe para não borrar.

    Distância zero tem duas causas possíveis — «o modelo não distingue» e «a
    entrada nunca carregou a distinção» — e atribuí-la à representação quando a
    culpa é da tokenização culparia a camada errada. É exatamente o defeito que o
    DOC-05 §8 existe para evitar, então medi-lo aqui também testa o tokenizer.
    """
    tok = _tokenizer_cego()
    assert set(t.colapsados_pelo_tokenizer(tok)) == {p.nome for p in t.PARES}


def test_sondagem_com_tudo_colapsado_nao_vira_veredito():
    mod, _ = _encoder(0)
    s = t.sondar(mod, _tokenizer_cego(), "cego", max_tokens=48)
    assert s.colapsados and not s.por_par
    leitura = t._leitura(s)
    assert leitura.startswith("INCONCLUSIVO") and "TOKENIZER" in leitura


def test_ablacao_com_tudo_colapsado_recusa_comparar():
    mod, _ = _encoder(0)
    tok = _tokenizer_cego()
    d = t.comparar_bracos({"controle": t.sondar(mod, tok, "c", max_tokens=48),
                           "tratado": t.sondar(mod, tok, "t", max_tokens=48)})
    assert d["ablacao"].get("erro"), (
        "comparou dois conjuntos vazios e chamou de «nada a decidir»")
