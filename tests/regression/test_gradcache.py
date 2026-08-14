"""GradCache tem de produzir os MESMOS gradientes que o caminho direto.

Implementado em 2026-08-14 para destravar o G1.2. O diagnóstico: perdemos do
GTE-large por 0,005 de nDCG@10, e o gargalo medido é **negativos no lote**, não
capacidade do modelo — o nosso de 110M ficou atrás do de 23M porque cabia lote 8
(7 negativos) contra 128 (127). GradCache desacopla o lote da memória.

## Por que este teste é o portão

GradCache faz DOIS forwards do mesmo pedaço: um para cachear a representação,
outro para propagar o gradiente. Se os dois divergirem — dropout sorteando
máscaras diferentes é a causa clássica — o gradiente injetado não corresponde à
ativação recomputada.

**E falha em silêncio.** O treino roda, a perda cai, as métricas se movem, e os
gradientes estão errados. Nada no log denuncia. É indistinguível de um treino que
simplesmente aprende menos, e a conclusão seria "lote maior não ajudou" — o oposto
do que o experimento quer testar.

Comparar com o caminho direto é a única forma de saber. Sem este teste passando,
não se lança o treino.

## O que cada teste fixa

| | |
|---|---|
`test_gradientes_batem` | os gradientes de TODO parâmetro coincidem |
`test_perda_bate` | o valor da perda coincide |
`test_um_pedaco_e_o_caminho_direto` | `sub_lote == lote` é o caso degenerado |
`test_pedacos_desiguais` | lote que não divide pelo sub-lote |
`test_negativos_sao_do_lote_LOGICO` | o ganho existe: mais negativos, não mais passos |
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

torch = pytest.importorskip("torch", reason="requer a venv de treino (.venv-treino)")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from phifm.training.embedding import Config, TreinadorEmb  # noqa: E402


class EncoderFalso(nn.Module):
    """Encoder minúsculo COM dropout — o dropout é o ponto do teste.

    Sem dropout, os dois forwards do GradCache coincidem trivialmente e o teste
    não exercita a parte que quebra. Com dropout, ele só passa se a semente por
    pedaço estiver funcionando.
    """

    def __init__(self, vocab: int = 40, dim: int = 16, p: float = 0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.lin = nn.Linear(dim, dim)
        self.drop = nn.Dropout(p)

    def forward(self, texto: list[str]) -> torch.Tensor:
        # "tokenização" determinística: soma dos códigos das letras
        ids = torch.tensor([[ord(c) % 40 for c in t[:8].ljust(8, "a")] for t in texto])
        h = self.drop(self.lin(self.emb(ids)))
        return F.normalize(h.mean(dim=1), dim=-1)


class TreinadorFalso(TreinadorEmb):
    """Reaproveita `_perda` e `_passo_gradcache` sem baixar modelo de verdade."""

    def __init__(self, sub_lote: int | None, semente: int = 17):
        self.cfg = Config(sub_lote=sub_lote, semente=semente, temperatura=0.05)
        self.dev = torch.device("cpu")
        torch.manual_seed(semente)
        self.mod = EncoderFalso()
        self.mod.train()

    def _codificar(self, textos):          # noqa: D102 — substitui o do pai
        return self.mod(list(textos))


def par_de_textos(n: int) -> tuple[list[str], list[str]]:
    return ([f"ancora{i:03d}" for i in range(n)],
            [f"positivo{i:03d}" for i in range(n)])


def grads_direto(a, p, sub=None, semente=17) -> tuple[float, dict[str, torch.Tensor]]:
    """Referência: o caminho INGÊNUO que o GradCache substitui.

    ⚠️ A referência tem de fatiar do mesmo jeito. Minha primeira versão fazia UM
    forward do lote inteiro com uma semente, e comparava com o GradCache, que faz
    um forward por pedaço com semente por pedaço. As máscaras de dropout diferem
    POR CONSTRUÇÃO, então a comparação não podia bater — e o teste acusava a
    implementação por um defeito dele. Só o caso de um pedaço passava, porque ali
    as duas coincidem.

    A referência certa mantém TODOS os grafos vivos ao mesmo tempo — que é
    exatamente o custo de memória que o GradCache existe para evitar. É o cálculo
    que queremos reproduzir, feito do jeito caro.
    """
    t = TreinadorFalso(sub_lote=None, semente=semente)
    sub = sub or len(a)
    sementes = [semente * 1_000_003 + i for i in range((len(a) + sub - 1) // sub)]
    ra, rp = [], []
    for j, i in enumerate(range(0, len(a), sub)):
        torch.manual_seed(sementes[j])
        ra.append(t._codificar(a[i:i + sub]))       # COM grafo
        rp.append(t._codificar(p[i:i + sub]))
    perda = t._perda(torch.cat(ra), torch.cat(rp))
    perda.backward()
    return perda.item(), {k: v.grad.clone() for k, v in t.mod.named_parameters()}


def grads_gradcache(a, p, sub, semente=17) -> tuple[float, dict[str, torch.Tensor]]:
    t = TreinadorFalso(sub_lote=sub, semente=semente)
    perda = t._passo_gradcache(a, p)
    return perda.item(), {k: v.grad.clone() for k, v in t.mod.named_parameters()}


# ─── o portão ────────────────────────────────────────────────────────────────

def test_um_pedaco_e_o_caminho_direto():
    """`sub_lote == lote`: um pedaço só, e tem de bater EXATAMENTE.

    É o caso degenerado, e o mais fácil de acertar — se este falhar, o problema
    está na mecânica das três fases, não no fatiamento.
    """
    a, p = par_de_textos(8)
    pd, gd = grads_direto(a, p, sub=8)
    pg, gg = grads_gradcache(a, p, sub=8)
    assert pg == pytest.approx(pd, rel=1e-5)
    for k in gd:
        assert torch.allclose(gd[k], gg[k], atol=1e-6), f"gradiente de {k} divergiu"


def test_gradientes_batem_com_varios_pedacos():
    """O teste que importa: 4 pedaços de 4, contra um lote direto de 16.

    Se o dropout sortear máscaras diferentes entre as fases 1 e 3, este falha.
    """
    a, p = par_de_textos(16)
    pd, gd = grads_direto(a, p, sub=4)
    pg, gg = grads_gradcache(a, p, sub=4)
    assert pg == pytest.approx(pd, rel=1e-5), "a perda divergiu"
    for k in gd:
        assert torch.allclose(gd[k], gg[k], atol=1e-5), (
            f"gradiente de {k} divergiu — provável máscara de dropout diferente "
            f"entre as duas fases; max |Δ| = {(gd[k]-gg[k]).abs().max():.2e}")


def test_pedacos_desiguais():
    """Lote 10 com sub-lote 4: pedaços de 4, 4 e 2. O último é o que quebra
    implementações que assumem tamanho fixo ao fatiar o gradiente."""
    a, p = par_de_textos(10)
    pd, gd = grads_direto(a, p, sub=4)
    pg, gg = grads_gradcache(a, p, sub=4)
    assert pg == pytest.approx(pd, rel=1e-5)
    for k in gd:
        assert torch.allclose(gd[k], gg[k], atol=1e-5), f"gradiente de {k} divergiu"


def test_sub_lote_maior_que_o_lote_nao_quebra():
    a, p = par_de_textos(5)
    pd, _ = grads_direto(a, p, sub=64)
    pg, _ = grads_gradcache(a, p, sub=64)
    assert pg == pytest.approx(pd, rel=1e-5)


# ─── o ganho tem de ser real ────────────────────────────────────────────────

def test_negativos_sao_do_lote_LOGICO_nao_do_sub_lote():
    """O ponto todo do GradCache.

    Um lote de 16 fatiado em 4 tem de dar a MESMA perda que um lote de 16 direto —
    e uma perda DIFERENTE de quatro lotes de 4 independentes. Se desse o mesmo que
    os quatro lotes pequenos, o GradCache não estaria comprando negativo nenhum e
    seria só um jeito lento de fazer acumulação de gradiente.
    """
    a, p = par_de_textos(16)
    perda_16_cache, _ = grads_gradcache(a, p, sub=4)
    perda_16_direto, _ = grads_direto(a, p, sub=4)

    # Quatro lotes de 4, independentes: cada um vê 3 negativos em vez de 15.
    perdas_pequenas = []
    for j, i in enumerate(range(0, 16, 4)):
        t = TreinadorFalso(sub_lote=None)
        torch.manual_seed(17 * 1_000_003 + j)     # a semente DAQUELE pedaço
        perdas_pequenas.append(
            t._perda(t._codificar(a[i:i + 4]), t._codificar(p[i:i + 4])).item())
    media_pequena = sum(perdas_pequenas) / len(perdas_pequenas)

    assert perda_16_cache == pytest.approx(perda_16_direto, rel=1e-5)
    assert perda_16_cache > media_pequena, (
        "a perda com 15 negativos não é maior que com 3 — o lote lógico não está "
        "sendo usado, e o GradCache virou acumulação de gradiente disfarçada")


def test_piso_da_perda_cresce_com_negativos():
    """InfoNCE tem piso ln(N) com N negativos: comparar perdas de lotes diferentes
    é erro de leitura, e este teste fixa o porquê.

    Já registrado no ESTADO.md: o MiniLM com lote 128 marcava ~1,5 contra 0,3–0,5
    do SciBERT com lote 8, e isso NÃO era pior.
    """
    import math
    a, p = par_de_textos(32)
    p8, _ = grads_gradcache(a[:8], p[:8], sub=4)
    p32, _ = grads_gradcache(a, p, sub=4)
    assert p32 > p8
    # O piso teórico não é alcançado num modelo não treinado, mas a ORDEM vale.
    assert math.log(32) > math.log(8)
