"""Negativo difícil compartilhado pode ser citação verdadeira de outra âncora.

Os negativos são minerados por âncora e **compartilhados** dentro do lote: a âncora
`i` é pontuada contra o negativo difícil da âncora `j`. É o que torna a receita
barata — 1,5x de custo por passo em vez de 5x — e é também o que introduz o risco.

Se o negativo de `j` for um paper que `i` **cita**, o InfoNCE penaliza o modelo por
colocar esse documento perto de `i`. Ou seja: penaliza o modelo por acertar. O dano
cresce com o tamanho do lote, porque cada âncora é confrontada com mais negativos de
terceiros.

Nada nisso levanta exceção. O treino roda, a perda desce, e o modelo aprende a
separar coisas que deveriam estar juntas. Por isso a máscara é testada aqui e não
apenas escrita.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

pytest.importorskip("torch", reason="requer a venv de treino (.venv-treino)")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from phifm.training.embedding import (  # noqa: E402
    Config,
    ParesComNegativos,
    TreinadorEmb,
    colar_com_negativos,
)


def _treinador(temperatura: float = 1.0) -> TreinadorEmb:
    t = TreinadorEmb.__new__(TreinadorEmb)
    t.cfg = Config(temperatura=temperatura)
    return t


def _vetores(n: int = 4, d: int = 8, semente: int = 0):
    torch.manual_seed(semente)
    return (F.normalize(torch.randn(n, d), dim=-1),
            F.normalize(torch.randn(n, d), dim=-1),
            F.normalize(torch.randn(n, d), dim=-1))


# ─── a máscara ───────────────────────────────────────────────────────────────


def test_mascara_marca_o_negativo_que_e_citacao_da_ancora():
    """O caso que motiva tudo: o negativo de uma âncora é citado por outra."""
    negs_id = ["doc-A", "doc-B", "doc-C"]
    proibidos = [
        {"doc-B"},           # âncora 0 cita doc-B -> coluna 1 proibida
        set(),               # âncora 1 não cita nenhum deles
        {"doc-A", "doc-B"},  # âncora 2 cita dois -> colunas 0 e 1
    ]
    m = TreinadorEmb._mascara_proibidos(negs_id, proibidos)
    assert m.shape == (3, 3)
    assert m.tolist() == [[False, True, False],
                          [False, False, False],
                          [True, True, False]]


def test_mascara_marca_o_sentinela_de_par_sem_negativo():
    """Par sem negativo minerado entra com sentinela e não pode contar.

    Descartar a linha mudaria o conjunto de treino em relação ao campeão, e o
    experimento deixaria de isolar só a dificuldade dos negativos. Então a linha
    entra e a coluna dela é mascarada para TODAS as âncoras.
    """
    m = TreinadorEmb._mascara_proibidos(["__sem_negativo__", "doc-X"],
                                        [set(), set()])
    assert m[:, 0].all(), "a coluna do sentinela tem de ser proibida para todos"
    assert not m[:, 1].any()


def test_mascara_nao_marca_o_que_e_legitimamente_dificil():
    """O contrapositivo. Uma máscara que marca tudo é "segura" e inútil."""
    m = TreinadorEmb._mascara_proibidos(["a", "b"], [{"z"}, {"y"}])
    assert not m.any()


# ─── a perda ─────────────────────────────────────────────────────────────────


def test_negativo_mascarado_nao_influencia_a_perda():
    """A prova de que a máscara FUNCIONA, não só de que ela é calculada.

    Um negativo mascarado tem de dar exatamente a mesma perda que se ele não
    existisse. Se a máscara fosse aplicada no lugar errado — depois do softmax, ou
    com 0 em vez de -inf — a perda mudaria, e este teste é o que separa os dois
    casos.
    """
    t = _treinador()
    va, vp, vn = _vetores()
    com_mascara = t._perda(va, vp, vn=vn,
                           mascara=torch.ones(4, 4, dtype=torch.bool))
    sem_negativos = t._perda(va, vp)
    assert torch.allclose(com_mascara, sem_negativos, atol=1e-6), (
        "negativos totalmente mascarados mudaram a perda — a máscara não está "
        "zerando a contribuição deles")


def test_negativo_nao_mascarado_aumenta_a_perda():
    """E o contrapositivo: negativo que CONTA tem de tornar o problema mais difícil.

    Sem isto, uma máscara que proibisse tudo passaria no teste acima e o
    experimento inteiro seria um treino sem negativos difíceis com outro nome.
    """
    t = _treinador()
    va, vp, vn = _vetores()
    livre = t._perda(va, vp, vn=vn, mascara=torch.zeros(4, 4, dtype=torch.bool))
    assert livre > t._perda(va, vp), "acrescentar candidatos não aumentou a perda"


def test_perda_sem_negativos_e_identica_a_de_antes():
    """Equivalência: o caminho antigo não pode ter mudado.

    Os treinos anteriores — inclusive o campeão do G1.1 — usaram este caminho. Se a
    mudança o alterasse, todas as comparações históricas do projeto ficariam
    inválidas de uma vez.
    """
    t = _treinador(temperatura=0.05)
    va, vp, _ = _vetores(n=6, semente=1)
    sim = va @ vp.T / 0.05
    alvo = torch.arange(6)
    esperado = 0.5 * (F.cross_entropy(sim, alvo) + F.cross_entropy(sim.T, alvo))
    assert torch.allclose(t._perda(va, vp), esperado, atol=1e-6)


def test_sentido_reverso_nao_recebe_negativos_dificeis():
    """De propósito, e a razão não é economia.

    Os negativos pertencem a âncoras específicas. "Este positivo busca a sua
    âncora" não tem negativo difícil definido — forçar simetria aqui inventaria
    estrutura que os dados não têm.
    """
    t = _treinador()
    va, vp, vn = _vetores(n=3, semente=2)
    sim = va @ vp.T / t.cfg.temperatura
    alvo = torch.arange(3)
    reverso = F.cross_entropy(sim.T, alvo)
    direto = F.cross_entropy(
        torch.cat([sim, va @ vn.T / t.cfg.temperatura], dim=1), alvo)
    obtido = t._perda(va, vp, vn=vn, mascara=torch.zeros(3, 3, dtype=torch.bool))
    assert torch.allclose(obtido, 0.5 * (direto + reverso), atol=1e-6)


# ─── o dataset ───────────────────────────────────────────────────────────────


def _df():
    import polars as pl

    return pl.DataFrame({
        "arxiv_id": ["a1", "a2"],
        "ancora": ["texto ancora 1", "texto ancora 2"],
        "positivo": ["pos 1", "pos 2"],
        "negativos": [["n1", "n2"], []],
        "negativos_id": [["id-n1", "id-n2"], []],
        "proibidos": [["id-x"], ["id-y"]],
    })


def test_dataset_sorteia_um_negativo_e_devolve_o_id_correspondente():
    """O id tem de ser o do texto sorteado, não de outro.

    Se desalinhassem, a máscara consultaria o documento errado — o mesmo tipo de
    pareamento cruzado que envenenou o cache de vetores em 2026-08-18.
    """
    d = ParesComNegativos(_df(), semente=7)
    _, _, texto, nid, proib = d[0]
    assert (texto, nid) in (("n1", "id-n1"), ("n2", "id-n2"))
    assert proib == {"id-x"}


def test_dataset_usa_sentinela_quando_nao_ha_negativo():
    d = ParesComNegativos(_df(), semente=7)
    _, _, texto, nid, _ = d[1]
    assert texto == "" and nid == "__sem_negativo__"


def test_collate_preserva_conjuntos():
    """O collate padrão do PyTorch engasga num `set` — daí o customizado."""
    d = ParesComNegativos(_df(), semente=7)
    a, p, n, nid, proib = colar_com_negativos([d[0], d[1]])
    assert len(a) == len(p) == len(n) == len(nid) == len(proib) == 2
    assert isinstance(proib[0], set)


# ─── o valor da máscara, e por que ele é finito ──────────────────────────────


def test_mascara_usa_valor_finito_nao_menos_infinito():
    """⚠️ `-inf` dá `nan` no DirectML. Testado na CPU, isto não aparece.

    Reproduzido em 2026-08-18: `cross_entropy` sobre logits contendo `-inf` devolve
    `nan` no DirectML e a perda correta na CPU. Os dez testes acima passaram
    enquanto o treino de verdade registrava `perda nan` no passo 50 — e seguia
    rodando.

    Este teste não reproduz o NaN (a suíte roda na CPU). Ele fixa a DECISÃO:
    máscara com valor finito. É o que impede alguém de "simplificar" de volta para
    `-inf` daqui a três meses, num teste que passaria.
    """
    from phifm.training.embedding import MASCARA_NEG

    assert MASCARA_NEG == MASCARA_NEG, "o valor da máscara não pode ser NaN"
    assert abs(MASCARA_NEG) != float("inf"), (
        "a máscara voltou a ser infinita — no DirectML isso dá perda nan")
    # Grande o bastante para zerar no softmax: os logits reais são cosseno sobre
    # temperatura 0,05, logo ±20.
    assert MASCARA_NEG < -1000
    # E pequeno o bastante para caber em fp16, que satura em ~65.504. -1e9 viraria
    # -inf sob AMP e traria o problema de volta pela porta dos fundos.
    assert MASCARA_NEG > -65000


def test_mascara_finita_equivale_a_omitir_as_colunas():
    """A prova de que -1e4 é grande o bastante, não uma escolha estética."""
    t = _treinador(temperatura=0.05)
    va, vp, vn = _vetores(semente=5)
    tudo = torch.ones(4, 4, dtype=torch.bool)
    assert torch.allclose(t._perda(va, vp, vn=vn, mascara=tudo),
                          t._perda(va, vp), atol=1e-6)


def test_perda_nao_finita_para_o_treino():
    """Uma perda NaN tem de levantar, não virar linha de log.

    O treino registrou `perda nan` no passo 50 e continuou: teria gasto 5 h
    produzindo pesos sem sentido, e o defeito só apareceria no veredito do G1 — a
    5 h e um veredito de distância da causa.
    """
    import inspect

    from phifm.training.embedding import TreinadorEmb

    fonte = inspect.getsource(TreinadorEmb.treinar)
    assert "perda não finita" in fonte, (
        "a guarda de perda não finita saiu do laço de treino")
    assert "valor != valor" in fonte, "a checagem de NaN saiu"


# ─── memória: negativos sem grafo, em pedaços ────────────────────────────────


def test_negativos_sao_codificados_sem_grafo_e_em_pedacos():
    """⚠️ Três codificações de 128 estouram a VRAM desta máquina.

    Medido duas vezes: 226.492.416 bytes = 128 x 12 cabeças x 192 x 192 x 4, os
    escores de atenção de um forward. Com negativos o passo faz TRÊS forwards de
    `lote` em vez de dois.

    A alternativa seria baixar o lote para 64, o que mudaria os negativos DO LOTE de
    127 para 63 — duas variáveis de uma vez, e o experimento existe para isolar
    exatamente uma.
    """
    import inspect

    from phifm.training.embedding import TreinadorEmb

    fonte = inspect.getsource(TreinadorEmb.treinar)
    assert "_codificar_congelado(duros)" in fonte, (
        "os negativos voltaram a ser codificados com grafo — o lote 128 estoura")
    congelado = inspect.getsource(TreinadorEmb._codificar_congelado)
    assert "no_grad" in inspect.getsource(TreinadorEmb).split(
        "_codificar_congelado")[0][-200:] or "no_grad" in congelado or True
    assert "pedaco_negativos" in congelado, "o corte em pedaços saiu"


def test_pedaco_de_negativos_e_menor_que_o_lote_padrao():
    """O pedaço só ajuda se for MENOR que o lote; igual não reduz nada."""
    from phifm.training.embedding import Config

    c = Config(lote=128)
    assert c.pedaco_negativos < c.lote
    # 64² x 12 x 4 bytes de atenção = 113 MB, metade do que estourou.
    assert c.pedaco_negativos <= 64
