"""A avaliação de MLM em texto denso em equações, e o que a torna legível.

O risco central desta avaliação não é errar a conta — é medir a coisa certa no
dado errado, ou medir duas tarefas diferentes e chamar de comparação.

1. **Dado visto.** O `Fluxo` do pré-treino permuta TODAS as sequências do
   binário: não existe divisão de validação. O conjunto de avaliação e o de
   treino têm o mesmo formato e o mesmo nome de arquivo, então apontar o script
   para `phienc_dados` por engano não produz erro — produz perplexidade baixa e
   um veredito animador sobre dado que o modelo já viu.
2. **Máscaras diferentes entre os braços.** Aí os dois modelos respondem tarefas
   distintas e a diferença de perda não significa nada.
3. **Prefixo em vez de sorteio.** As sequências estão na ordem em que as
   partições foram lidas; `range(n)` é um punhado de partições inteiras — amostra
   por conglomerado disfarçada de aleatória. Seria a sétima ocorrência da
   armadilha que a §5.2 do artigo cataloga.
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
np = pytest.importorskip("numpy")

from phifm.eval import mlm  # noqa: E402
from phifm.models.encoder.config import ConfigEnc  # noqa: E402
from phifm.models.encoder.modelo import construir  # noqa: E402
from phifm.training.pretrain.dados import (  # noqa: E402
    NOME_MANIFESTO,
    NOME_MARCAS,
    NOME_TOKENS,
    marcas_de,
)

VOCAB = 64
# ⚠️ 256, e não 64. `_escolher_equacao` exige equação de display com pelo menos
# MIN_TOKENS_TRATAMENTO=20 tokens que ainda caiba no orçamento `taxa × contexto`.
# A 64 tokens o orçamento é 19 e as duas condições se excluem — o regime de equação
# recairia INTEIRO em aleatório, que foi como este teste achou a guarda de
# configuração hoje em `mascaras_do_regime`.
CONTEXTO = 256
EQUACAO_TOKENS = 24
TINY = ConfigEnc(nome="tiny-mlm", camadas=2, d_model=32, cabecas=2, ffn=48,
                 vocab=VOCAB, contexto=CONTEXTO, janela_local=128, global_a_cada=2)


def _dados(tmp: Path, nome: str, partes: list[str], n_seq: int = 12) -> Path:
    """Um preparo sintético com equações em display de verdade nas marcas."""
    d = tmp / nome
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    total = n_seq * CONTEXTO
    ids = rng.integers(5, VOCAB, size=total).astype(np.uint16)

    # Uma equação em display por sequência, acima do mínimo de tratamento.
    ide = np.full(total, -1, dtype=np.int32)
    disp = np.zeros(total, dtype=bool)
    for s in range(n_seq):
        a = s * CONTEXTO + 40
        ide[a:a + EQUACAO_TOKENS] = s
        disp[a:a + EQUACAO_TOKENS] = True
    marcas = marcas_de(ide, disp)

    ids.tofile(d / NOME_TOKENS)
    marcas.tofile(d / NOME_MARCAS)
    (d / NOME_MANIFESTO).write_text(json.dumps({
        "tokens": int(total), "contexto": CONTEXTO, "partes_usadas": partes,
        "tokenizer": "sha-de-teste",
    }), encoding="utf-8")
    return d


def _modelo(semente: int):
    torch.manual_seed(semente)
    return construir(TINY).eval()


# ── 1. a guarda de disjunção ────────────────────────────────────────────────

def test_particoes_compartilhadas_levantam(tmp_path):
    """O erro que não produz sintoma: perplexidade medida em dado visto."""
    treino = _dados(tmp_path, "treino", ["part-00.parquet", "part-01.parquet"])
    aval = _dados(tmp_path, "aval", ["part-01.parquet", "part-02.parquet"])
    with pytest.raises(ValueError, match="compartilham"):
        mlm.conferir_disjuncao(treino, aval)


def test_particoes_disjuntas_passam(tmp_path):
    treino = _dados(tmp_path, "treino", ["part-00.parquet"])
    aval = _dados(tmp_path, "aval", ["part-01.parquet"])
    d = mlm.conferir_disjuncao(treino, aval)
    assert d["disjuntas"] is True
    assert d["particoes_treino"] == 1


def test_manifesto_sem_lista_de_partes_levanta(tmp_path):
    """Contagem não basta: com sorteio, «9 de 44» não diz QUAIS."""
    treino = _dados(tmp_path, "treino", ["part-00.parquet"])
    aval = _dados(tmp_path, "aval", ["part-01.parquet"])
    man = json.loads((aval / NOME_MANIFESTO).read_text())
    del man["partes_usadas"]
    (aval / NOME_MANIFESTO).write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(ValueError, match="partes_usadas"):
        mlm.conferir_disjuncao(treino, aval)


def test_sem_manifesto_levanta(tmp_path):
    treino = _dados(tmp_path, "treino", ["part-00.parquet"])
    aval = _dados(tmp_path, "aval", ["part-01.parquet"])
    (aval / NOME_MANIFESTO).unlink()
    with pytest.raises(FileNotFoundError, match="disjunta"):
        mlm.conferir_disjuncao(treino, aval)


# ── 2. a amostragem ─────────────────────────────────────────────────────────

def test_as_sequencias_sao_SORTEADAS_e_nao_um_prefixo(tmp_path):
    """Prefixo seria a sétima ocorrência da armadilha catalogada na §5.2."""
    fluxo = mlm.carregar_fluxo(_dados(tmp_path, "aval", ["p0"], n_seq=40))
    escolhidas = mlm.escolher_sequencias(fluxo, 8, semente=17)
    assert escolhidas != list(range(8)), "voltou a pegar o prefixo"
    assert len(set(escolhidas)) == 8, "sorteou com reposição"
    assert escolhidas == mlm.escolher_sequencias(fluxo, 8, semente=17), (
        "o sorteio não é reproduzível pela semente")


def test_pedir_mais_sequencias_do_que_existe_levanta(tmp_path):
    fluxo = mlm.carregar_fluxo(_dados(tmp_path, "aval", ["p0"], n_seq=5))
    with pytest.raises(ValueError, match="tem 5"):
        mlm.escolher_sequencias(fluxo, 50, semente=17)


# ── 3. as máscaras são as MESMAS para os dois braços ────────────────────────

def test_mascaras_calculadas_uma_vez_e_compartilhadas(tmp_path):
    """Não «mesma semente»: o mesmo array. A igualdade é estrutural."""
    fluxo = mlm.carregar_fluxo(_dados(tmp_path, "aval", ["p0"]))
    ind = mlm.escolher_sequencias(fluxo, 4, semente=17)
    lotes, _ = mlm.mascaras_do_regime(fluxo, ind, "aleatorio", taxa=0.3,
                                      semente=17, n_vocab=VOCAB)
    a = mlm.medir(_modelo(0), lotes, "a", "aleatorio", 0.0)
    b = mlm.medir(_modelo(1), lotes, "b", "aleatorio", 0.0)
    assert a.tokens_avaliados == b.tokens_avaliados, (
        "os dois braços avaliaram números diferentes de tokens — as máscaras "
        "divergiram e a comparação é entre tarefas distintas")
    assert len(a.perda_por_sequencia) == len(b.perda_por_sequencia)


def test_o_regime_de_equacao_de_fato_trata(tmp_path):
    """`fracao_tratada` zero significa recaída em MLM aleatório — a ablação
    passaria a comparar aleatório com aleatório e reportaria empate."""
    fluxo = mlm.carregar_fluxo(_dados(tmp_path, "aval", ["p0"]))
    ind = mlm.escolher_sequencias(fluxo, 6, semente=17)
    _, ft = mlm.mascaras_do_regime(fluxo, ind, "equacao", taxa=0.3,
                                   semente=17, n_vocab=VOCAB)
    assert ft > 0.0, "nenhuma sequência tratada apesar de todas terem display"


def test_regime_desconhecido_levanta(tmp_path):
    fluxo = mlm.carregar_fluxo(_dados(tmp_path, "aval", ["p0"]))
    with pytest.raises(ValueError, match="desconhecido"):
        mlm.mascaras_do_regime(fluxo, [0], "equacoes", taxa=0.3, semente=1,
                               n_vocab=VOCAB)


# ── 4. a decomposição, que carrega a evidência de mecanismo ─────────────────

def test_equacao_mais_prosa_fecha_o_total(tmp_path):
    fluxo = mlm.carregar_fluxo(_dados(tmp_path, "aval", ["p0"]))
    ind = mlm.escolher_sequencias(fluxo, 6, semente=17)
    lotes, ft = mlm.mascaras_do_regime(fluxo, ind, "aleatorio", taxa=0.3,
                                       semente=17, n_vocab=VOCAB)
    m = mlm.medir(_modelo(0), lotes, "a", "aleatorio", ft)
    assert m.tokens_equacao + m.tokens_prosa == m.tokens_avaliados
    assert m.tokens_equacao > 0, (
        "nenhum token de equação sorteado sob mascaramento aleatório — a célula "
        "de mecanismo ficaria vazia")
    assert m.perplexidade == pytest.approx(float(np.exp(m.perda)), rel=1e-6)


# ── 5. o pareado ────────────────────────────────────────────────────────────

def test_modelos_identicos_nao_produzem_discordantes(tmp_path):
    """Mesmo modelo nos dois lados: toda sequência empata, nada a decidir."""
    fluxo = mlm.carregar_fluxo(_dados(tmp_path, "aval", ["p0"]))
    ind = mlm.escolher_sequencias(fluxo, 6, semente=17)
    lotes, ft = mlm.mascaras_do_regime(fluxo, ind, "aleatorio", taxa=0.3,
                                       semente=17, n_vocab=VOCAB)
    m = _modelo(0)
    cmp = mlm.comparar_medidas(mlm.medir(m, lotes, "c", "aleatorio", ft),
                               mlm.medir(m, lotes, "t", "aleatorio", ft))
    assert cmp["discordantes"] == 0
    assert cmp["p"] == 1.0


def test_poucos_discordantes_sao_INCONCLUSIVO(tmp_path):
    cmp = {"discordantes": 5, "p": 0.01, "controle_melhor": 5, "tratado_melhor": 0}
    assert mlm._leitura("aleatorio", cmp).startswith("INCONCLUSIVO")


def test_a_leitura_nomeia_de_QUEM_e_o_objetivo_em_cada_regime():
    """Sem isso, um ganho no regime de equação é lido como vitória — e ele
    favorece o tratado por construção."""
    cmp = {"discordantes": 40, "p": 0.001, "controle_melhor": 5, "tratado_melhor": 35}
    assert "CONTROLE" in mlm._leitura("aleatorio", cmp)
    assert "construção" in mlm._leitura("equacao", cmp)


def test_regime_de_equacao_que_recaiu_e_INCONCLUSIVO():
    """Guarda achada por um teste: contexto pequeno faz o regime recair
    inteiro em aleatório, e aí o 2×2 reporta dois regimes idênticos."""
    cmp = {"discordantes": 40, "p": 0.001, "controle_melhor": 5,
           "tratado_melhor": 35}
    assert mlm._leitura("equacao", cmp, fracao_tratada=0.0).startswith(
        "INCONCLUSIVO")


# ── 6. o 2×2 inteiro ────────────────────────────────────────────────────────

def test_ablacao_produz_as_quatro_celulas_e_o_mecanismo(tmp_path):
    fluxo = mlm.carregar_fluxo(_dados(tmp_path, "aval", ["p0"]))
    d = mlm.avaliar_ablacao({"controle": _modelo(0), "tratado": _modelo(1)},
                            fluxo, sequencias=6, semente=17, n_vocab=VOCAB)
    assert set(d["celulas"]) == {"aleatorio", "equacao"}
    for regime in d["celulas"]:
        assert set(d["celulas"][regime]) == {"controle", "tratado"}
    assert "delta" in d["mecanismo"]
    assert d["mecanismo"]["tokens"] > 0
    assert "CONSTRUÇÃO" in d["ressalva"]


def test_ablacao_exige_os_dois_bracos(tmp_path):
    fluxo = mlm.carregar_fluxo(_dados(tmp_path, "aval", ["p0"]))
    with pytest.raises(ValueError, match="tratado"):
        mlm.avaliar_ablacao({"controle": _modelo(0)}, fluxo, sequencias=4,
                            n_vocab=VOCAB)
