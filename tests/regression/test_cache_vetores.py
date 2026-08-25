"""Cache de embeddings validado por CONTAGEM parearia vetor com documento errado.

Regressão de 2026-08-18, e é o defeito mais perigoso que apareceu neste projeto —
não porque quebra, mas porque **não** quebra.

O minerador de negativos difíceis codifica 667 mil documentos (~35 min) e guarda os
vetores em cache para uma relançada não recomeçar. A primeira versão validava o
cache assim:

    if len(v) == len(textos):   # ERRADO
        return v

O pool vem de `unique()` do polars, que **não garante ordem**. Duas execuções com a
mesma contagem tinham ordens diferentes, então o cache devolvia o vetor do documento
`i` para o documento `j`. Consequência: negativos "difíceis" sorteados ao acaso, com
aparência perfeitamente plausível — nenhuma exceção, nenhum aviso, e um treino
aprendendo lixo enquanto as métricas de treino pareceriam normais.

Como foi pego: o número `descartados_por_serem_citacao_verdadeira` caiu de 131 para
**0** entre duas execuções idênticas. Esse contador existia só por precaução — para
provar que a exclusão de citações verdadeiras estava funcionando — e virou o único
sinal de um defeito completamente diferente. É o argumento a favor de instrumentar
o que se acredita já estar certo.

Os dois consertos, e os dois são necessários:
  1. `unique(..., maintain_order=True)` — ordem determinística.
  2. Cache guarda os IDS e confere um por um — protege mesmo se a ordem mudar por
     outro motivo qualquer, o que é o ponto: a correção 1 depende de uma promessa
     de biblioteca, a 2 não depende de nada.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

pytest.importorskip("torch", reason="o minerador vive na venv de treino")

from scripts.minerar_negativos import _codificar_tudo  # noqa: E402


class TokFalso:
    """Devolve tensores do tamanho certo sem tokenizar nada de verdade."""

    def __call__(self, textos, **kw):
        import torch

        n, t = len(textos), 8
        return {"input_ids": torch.zeros(n, t, dtype=torch.long),
                "attention_mask": torch.ones(n, t, dtype=torch.long)}


class ModFalso:
    """Codifica cada texto num vetor determinístico e DISTINTO por conteúdo.

    É o que permite detectar pareamento errado: se o cache devolver o vetor do
    documento errado, o valor não bate com o texto.
    """

    def __init__(self):
        self.chamadas = 0

    def __call__(self, **b):
        import torch

        n, t = b["input_ids"].shape
        self.chamadas += n
        # o valor não importa; só precisa existir e ter a forma certa
        h = torch.ones(n, t, 4)
        return type("S", (), {"last_hidden_state": h})()

    def eval(self):
        return self


def _codificar(textos, ids, cache, mod=None):
    import torch

    return _codificar_tudo(textos, mod or ModFalso(), TokFalso(),
                           torch.device("cpu"), lote=4, max_tokens=8,
                           rotulo="teste", cache=cache, ids=ids)


def test_cache_com_mesma_contagem_e_ids_diferentes_e_rejeitado(tmp_path):
    """O caso exato do defeito: mesma contagem, ordem diferente.

    Sem a conferência de ids, isto devolveria os vetores do primeiro conjunto para
    o segundo — e nada acusaria.
    """
    cache = tmp_path / "v.npz"
    _codificar(["a", "b", "c"], ["id-1", "id-2", "id-3"], cache)
    assert cache.exists()

    mod = ModFalso()
    _codificar(["x", "y", "z"], ["id-9", "id-8", "id-7"], cache, mod=mod)
    assert mod.chamadas == 3, (
        "o cache foi reusado para outros documentos — é o defeito de 2026-08-18, "
        "que produz negativos aleatórios com aparência plausível")


def test_cache_com_os_mesmos_ids_e_reusado(tmp_path):
    """O contrapositivo. Sem ele, um cache que nunca acerta passaria nos dois testes."""
    cache = tmp_path / "v.npz"
    _codificar(["a", "b", "c"], ["id-1", "id-2", "id-3"], cache)

    mod = ModFalso()
    _codificar(["a", "b", "c"], ["id-1", "id-2", "id-3"], cache, mod=mod)
    assert mod.chamadas == 0, "recodificou tendo cache válido — o cache não serve"


def test_ordem_diferente_dos_mesmos_ids_e_rejeitada(tmp_path):
    """Mesmo CONJUNTO de ids, ordem trocada, ainda tem de recodificar.

    Comparar conjuntos em vez de sequências seria um meio-conserto: o vetor da
    posição 0 pertence ao id da posição 0, e trocar a ordem já basta para desalinhar
    tudo.
    """
    cache = tmp_path / "v.npz"
    _codificar(["a", "b", "c"], ["id-1", "id-2", "id-3"], cache)

    mod = ModFalso()
    _codificar(["c", "b", "a"], ["id-3", "id-2", "id-1"], cache, mod=mod)
    assert mod.chamadas == 3, "aceitou o cache com a ordem trocada"


def test_cache_corrompido_recodifica_em_vez_de_derrubar(tmp_path):
    """Arquivo ilegível é aviso e recomputo — perder cache é barato, abortar não."""
    cache = tmp_path / "v.npz"
    cache.write_bytes(b"isto nao e um npz")

    mod = ModFalso()
    v = _codificar(["a", "b"], ["id-1", "id-2"], cache, mod=mod)
    assert mod.chamadas == 2 and len(v) == 2


def test_cache_guarda_os_ids_junto_dos_vetores(tmp_path):
    """A conferência só é possível porque os ids vão no arquivo.

    Guardar os ids num arquivo separado, ou derivá-los do nome, reintroduziria a
    chance de os dois divergirem — que é a origem do defeito.
    """
    cache = tmp_path / "v.npz"
    _codificar(["a", "b"], ["id-1", "id-2"], cache)
    z = np.load(cache, allow_pickle=False)
    assert set(z) == {"vetores", "ids"}
    assert [str(x) for x in z["ids"]] == ["id-1", "id-2"]
    assert z["vetores"].shape[0] == 2


def test_pool_do_minerador_tem_ordem_deterministica():
    """`unique()` sem `maintain_order=True` é a raiz do defeito.

    A conferência de ids protege o resultado de qualquer forma, mas ordem instável
    faria o cache ser invalidado a cada execução — 35 min de recodificação por
    relançada, o que anula a razão de o cache existir.
    """
    fonte = (RAIZ / "scripts" / "minerar_negativos.py").read_text(encoding="utf-8")
    assert 'unique(subset=["arxiv_citado"], maintain_order=True)' in fonte


def test_o_guarda_de_zero_descartes_continua_no_lugar():
    """Foi ele que denunciou o defeito do cache, não um teste.

    `descartados_por_serem_citacao_verdadeira == 0` significa ou que o campeão não
    recupera nenhuma citação verdadeira no top-K, ou que a exclusão parou de
    funcionar. Nos dois casos os negativos não valem. Instrumentar o que se acredita
    já estar certo é o que transformou um defeito silencioso em um número estranho.
    """
    fonte = (RAIZ / "scripts" / "minerar_negativos.py").read_text(encoding="utf-8")
    assert "if descartados == 0:" in fonte
    assert "ZERO descartes" in fonte
