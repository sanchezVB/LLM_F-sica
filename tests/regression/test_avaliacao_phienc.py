"""A avaliação de recuperação do ΦEnc, e as guardas que a tornam legível.

O ΦEnc não carrega por `AutoModel.from_pretrained`: ele é um `state_dict` cru do
laço de pré-treino com um tokenizer que é JSON do `tokenizers`. Esse caminho novo
tem quatro formas de falhar **em silêncio**, e cada uma tem um teste aqui:

1. **`token_type_ids`.** O ModernBERT não os aceita; um tokenizer BERT os produz.
   `_codificar` passava `**b` cru. A correção restringe às duas chaves que todo
   encoder aceita — e ela só é legítima se NÃO mudar os números já publicados do
   G1, o que `test_token_type_ids_nao_mudam_o_vetor` **mede** em vez de afirmar.
2. **`pad_token_id` divergente** entre modelo e tokenizer. O ModernBERT usa esse
   id para desempacotar; divergentes, ele descarta posições válidas sem erro.
3. **Braços que diferem em mais do que o tratamento.** Foi a disciplina que o
   T1c aplicou com `--base`, e aqui vale igual: um hiperparâmetro escorregado
   ainda produz números, e são eles que alguém copia para uma tabela.
4. **`fracao_tratada` baixa.** O braço tratado recai em MLM aleatório e a
   ablação passa a comparar aleatório com aleatório. Já aconteceu: o defeito do
   `^` no padrão de display deu 0 documentos tratados em 120.

O quinto teste é o que o pré-registro do T1c não tinha: guarda reprovada tem de
**substituir** o veredito, não ser anotada ao lado dele.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

torch = pytest.importorskip("torch", reason="requer a venv de treino (.venv-treino)")
pytest.importorskip("transformers", reason="requer a venv de treino (.venv-treino)")
pl = pytest.importorskip("polars")

from phifm.eval import phienc as mod  # noqa: E402
from phifm.models.encoder.config import ESPECIAIS, ConfigEnc  # noqa: E402
from phifm.models.encoder.modelo import construir  # noqa: E402

# Minúsculo de propósito: o que se testa é a costura, não a qualidade do modelo.
# `contexto % janela_local == 0` e `d_model % cabecas == 0` são invariantes que o
# próprio `ConfigEnc` impõe.
VOCAB = 64
TINY = ConfigEnc(nome="tiny-teste", camadas=2, d_model=32, cabecas=2, ffn=48,
                 vocab=VOCAB, contexto=128, janela_local=128, global_a_cada=2)


def _tokenizer(tmp: Path, tamanho: int = VOCAB) -> Path:
    """Um WordLevel com os cinco especiais nos ids que o `config.py` reserva."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {}
    for nome, ident in sorted(ESPECIAIS.items(), key=lambda kv: kv[1]):
        vocab[f"[{nome.upper()}]"] = ident
    for i in range(len(vocab), tamanho):
        vocab[f"w{i}"] = i

    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    caminho = tmp / f"tok_{tamanho}.json"
    tk.save(str(caminho))
    return caminho


def _metricas(p_equacao: float, fracao_tratada: float = 0.9, **troca) -> dict:
    """O `phienc.json` que o laço de pré-treino grava, no mínimo que importa."""
    base = {
        "modelo": TINY.como_dict(),
        "treino": {"lr_pico": 1e-3, "total_passos": 100, "acumulacao": 1},
        "dados": {"semente": 17, "sequencias": 1},
        "mascara": {"taxa": 0.30, "p_equacao": p_equacao, "p_mask": 0.8,
                    "p_aleatorio": 0.1, "semente": 17},
        "mascaramento": {"exemplos": 100, "tratados": int(100 * fracao_tratada),
                         "fracao_tratada": fracao_tratada, "taxa_efetiva": 0.30},
    }
    for chave, valor in troca.items():
        base[chave] = {**base[chave], **valor} if isinstance(valor, dict) else valor
    return base


def _braco(tmp: Path, nome: str, p_equacao: float, semente: int = 0, **kw) -> Path:
    """Grava um checkpoint completo: pesos + `phienc.json`."""
    d = tmp / nome
    d.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(semente)
    modelo = construir(TINY)
    torch.save({"passo": 100, "modelo": modelo.state_dict()}, d / mod.NOME_ESTADO)
    (d / mod.NOME_METRICAS).write_text(
        json.dumps(_metricas(p_equacao, **kw), ensure_ascii=False), encoding="utf-8")
    return d


def _val(n: int = 40):
    """Pares com âncora e positivo únicos — `preparar_pool` exige teto 1,0."""
    return pl.DataFrame({
        "ancora": [f"w{10 + i % 40} w{11 + i % 30} consulta numero {i}" for i in range(n)],
        "positivo": [f"w{12 + i % 35} w{13 + i % 25} alvo distinto {i}" for i in range(n)],
    })


# ── 1. a correção que tornou o caminho compartilhado possível ────────────────

def test_token_type_ids_nao_mudam_o_vetor():
    """A afirmação feita em `encoders.py`, medida em vez de suposta.

    Restringir a entrada a `input_ids`/`attention_mask` só é legítimo se não
    alterar os números já publicados do G1. Para entrada de segmento único,
    `token_type_ids` todo zero é o default de quem os omite — e é isto que este
    teste confere num BERT de verdade, não em argumento.
    """
    from transformers import BertConfig, BertModel

    from phifm.eval.encoders import media_mascarada

    torch.manual_seed(0)
    m = BertModel(BertConfig(vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2,
                             num_attention_heads=2, intermediate_size=48,
                             max_position_embeddings=64)).eval()
    ids = torch.randint(5, VOCAB, (2, 12))
    att = torch.ones_like(ids)
    tti = torch.zeros_like(ids)

    with torch.no_grad():
        com = media_mascarada(m(input_ids=ids, attention_mask=att,
                                token_type_ids=tti).last_hidden_state, att)
        sem = media_mascarada(m(input_ids=ids, attention_mask=att).last_hidden_state, att)
    assert torch.allclose(com, sem, atol=1e-6), (
        "omitir token_type_ids mudou a representação — a restrição de entradas em "
        "`_codificar` NÃO é neutra e invalidaria os números já publicados do G1")


# ── 2. o carregamento ───────────────────────────────────────────────────────

def test_carrega_o_tronco_e_nao_a_cabeca(tmp_path):
    d = _braco(tmp_path, "controle", 0.0)
    enc, tok, meta = mod.carregar(d, _tokenizer(tmp_path))
    assert type(enc).__name__ == "ModernBertModel", (
        "quem recupera é a representação; a cabeça de MLM não participa")
    assert meta["mascara"]["p_equacao"] == 0.0
    assert tok.pad_token_id == ESPECIAIS["pad_token_id"]


def test_tokenizer_de_outra_variante_levanta(tmp_path):
    """O erro de um caractere: o tokenizer é argumento separado do checkpoint.

    A primeira versão desta guarda conferia `pad_token_id` do tokenizer contra o
    do modelo — e era tautológica, porque os nomes dos especiais são derivados
    desses mesmos ids. Este teste existiu antes do conserto e reprovou, que é o
    desfecho que uma guarda deve produzir quando não guarda nada.

    O sintoma que de fato denuncia o tokenizer errado é o tamanho do vocabulário.
    E note a assimetria: com vocabulário MENOR nada estoura — o modelo só nunca
    vê os ids que faltam, e a métrica sai pior sem explicação.
    """
    d = _braco(tmp_path, "controle", 0.0)
    with pytest.raises(ValueError, match="vocabulário"):
        mod.carregar(d, _tokenizer(tmp_path, tamanho=VOCAB // 2))


def test_sem_phienc_json_a_mensagem_diz_o_que_falta(tmp_path):
    d = _braco(tmp_path, "controle", 0.0)
    (d / mod.NOME_METRICAS).unlink()
    with pytest.raises(FileNotFoundError, match="uma-variável|phienc.json"):
        mod.carregar(d, _tokenizer(tmp_path))


def test_config_vem_do_checkpoint_nao_do_default(tmp_path):
    """Avaliar com outra arquitetura carregaria pesos numa estrutura errada."""
    cfg = mod.config_do_checkpoint(_metricas(0.5))
    assert (cfg.camadas, cfg.d_model, cfg.vocab) == (TINY.camadas, TINY.d_model, VOCAB)


# ── 3. a métrica é a MESMA de `encoders.py` ─────────────────────────────────

def test_avalia_e_devolve_posicoes_por_item(tmp_path):
    d = _braco(tmp_path, "controle", 0.0)
    r, meta = mod.avaliar(d, _tokenizer(tmp_path), _val(), n=20, max_tokens=16, lote=8)
    assert len(r.posicoes) == 20, "sem posições por item não há comparação pareada"
    assert 0.0 <= r.ndcg_10 <= 1.0
    assert r.nosso is True
    assert meta["mascaramento"]["fracao_tratada"] == 0.9


# ── 4. as guardas da ablação ────────────────────────────────────────────────

def test_uma_variavel_aceita_so_p_equacao(tmp_path):
    assert mod.conferir_uma_variavel(_metricas(0.0), _metricas(0.5)) == []


def test_uma_variavel_acusa_hiperparametro_escorregado():
    a = _metricas(0.0)
    b = _metricas(0.5, treino={"lr_pico": 3e-4})
    fora = mod.conferir_uma_variavel(a, b)
    assert any("treino" in f and "lr_pico" in f for f in fora), fora


def test_uma_variavel_acusa_taxa_de_mascara_diferente():
    """Orçamento de máscara desigual mediria «trata equações» E «mascara mais»."""
    fora = mod.conferir_uma_variavel(_metricas(0.0), _metricas(0.5, mascara={"taxa": 0.5}))
    assert any("mascara.taxa" in f for f in fora), fora


def test_fracao_tratada_baixa_torna_INCONCLUSIVO(tmp_path):
    tok = _tokenizer(tmp_path)
    c = _braco(tmp_path, "c", 0.0, semente=0)
    t = _braco(tmp_path, "t", 0.5, semente=1, fracao_tratada=0.01)
    d = mod.comparar_bracos(c, t, tok, _val(), n=20, max_tokens=16, lote=8)
    assert d["veredito"].startswith("INCONCLUSIVO"), d["veredito"]
    assert "fracao_tratada" in d["veredito"]


def test_controle_tratado_por_engano_torna_INCONCLUSIVO(tmp_path):
    tok = _tokenizer(tmp_path)
    c = _braco(tmp_path, "c", 0.3, semente=0)
    t = _braco(tmp_path, "t", 0.5, semente=1)
    d = mod.comparar_bracos(c, t, tok, _val(), n=20, max_tokens=16, lote=8)
    assert d["veredito"].startswith("INCONCLUSIVO"), d["veredito"]
    assert "p_equacao=0" in d["veredito"]


def test_guarda_reprovada_SUBSTITUI_o_veredito(tmp_path):
    """O buraco exato do pré-registro do T1c.

    Lá, com um braço ausente, o script imprimiu a leitura que EXIGIA aquele braço
    ter rodado e perdido — porque a lógica olhava só quem venceu. Aqui o pareado
    continua sendo calculado e gravado, mas ele não pode virar veredito enquanto
    houver impedimento.
    """
    tok = _tokenizer(tmp_path)
    c = _braco(tmp_path, "c", 0.0, semente=0)
    t = _braco(tmp_path, "t", 0.5, semente=1, fracao_tratada=0.0)
    d = mod.comparar_bracos(c, t, tok, _val(), n=20, max_tokens=16, lote=8)
    assert d["impedimentos"], "o impedimento tem de ficar registrado"
    assert d["pareado"], "o pareado continua sendo gravado, para auditoria"
    assert "AJUDA" not in d["veredito"] and "ATRAPALHA" not in d["veredito"], (
        "um veredito sobre a hipótese saiu apesar de a guarda ter reprovado")


def test_bracos_saudaveis_produzem_veredito_legivel(tmp_path):
    tok = _tokenizer(tmp_path)
    c = _braco(tmp_path, "c", 0.0, semente=0)
    t = _braco(tmp_path, "t", 0.5, semente=1)
    d = mod.comparar_bracos(c, t, tok, _val(), n=20, max_tokens=16, lote=8)
    assert not d["impedimentos"], d["impedimentos"]
    assert not d["veredito"].startswith("INCONCLUSIVO") or "discordantes" in d["veredito"]
    assert d["protocolo"]["n"] == 20
    assert d["p_equacao"] == {"controle": 0.0, "tratado": 0.5}


def test_a_ressalva_sobre_MLM_cru_acompanha_o_resultado(tmp_path):
    """O número absoluto é fraco por construção. Sem a ressalva no artefato,
    alguém compara com a tabela do G1 e conclui que o ΦEnc é ruim."""
    tok = _tokenizer(tmp_path)
    c = _braco(tmp_path, "c", 0.0, semente=0)
    t = _braco(tmp_path, "t", 0.5, semente=1)
    d = mod.comparar_bracos(c, t, tok, _val(), n=20, max_tokens=16, lote=8)
    assert "G1" in d["ressalva"] and "FRACA" in d["ressalva"]
