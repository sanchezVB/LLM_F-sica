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
