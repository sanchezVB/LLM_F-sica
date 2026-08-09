"""Escolha do representante do cluster (DOC-04 §5.3).

Dado um cluster de quase-duplicatas, **qual cópia sobrevive?** É decisão de
projeto com consequência real, não desempate arbitrário.

A ordem de precedência, e o critério 1 é o não óbvio:

    1. licença mais permissiva      ★ aumenta o PhysCorpus-Open de graça
    2. tem journal-ref ou DOI         versão revisada por pares
    3. versão mais recente do arXiv   correções incorporadas
    4. melhor qualidade de parsing    menos degradação de LaTeX
    5. mais longo                     desempate; provavelmente mais completo

═══ Por que a licença vem primeiro ═══

Se o mesmo paper existe via arXiv (licença padrão, **não redistribuível**) e
via SCOAP³ (CC BY, **redistribuível**), manter o CC BY aumenta diretamente o
subconjunto publicável do ADR-0001 §6 — sem custo algum e sem perder uma
linha de conteúdo.

É uma escolha que só aparece porque a licença é cidadã de primeira classe do
schema (princípio P1 do DOC-01), e é um bom exemplo de por que P1 vale a
sobrecarga que impõe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from phifm.core.licensing.registry import Partition, resolve

# Quanto maior, melhor. `train_open` é o que interessa preservar.
_VALOR_PARTICAO = {
    Partition.TRAIN_OPEN: 3,   # treina E redistribui
    Partition.TRAIN_ONLY: 2,   # treina, não redistribui
    Partition.EVAL_ONLY: 1,    # nunca treina — sobrevive só se for o único
    Partition.EXCLUDED: 0,
}

_VERSAO = re.compile(r"v(\d+)$")


@dataclass(frozen=True)
class Candidato:
    """O mínimo que a escolha precisa saber sobre um documento."""

    indice: int
    licenca: str | None = None
    journal_ref: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    qualidade_parse: float = 0.0
    comprimento: int = 0


def _chave(c: Candidato) -> tuple:
    """Chave de ordenação decrescente — o maior vence."""
    r = resolve(c.licenca)
    versao = int(m.group(1)) if (m := _VERSAO.search(c.arxiv_id or "")) else 1
    return (
        _VALOR_PARTICAO.get(r.partition, 0),          # 1. licença
        int(bool(c.journal_ref) or bool(c.doi)),      # 2. revisado por pares
        versao,                                        # 3. versão
        c.qualidade_parse,                             # 4. parsing
        c.comprimento,                                 # 5. comprimento
        -c.indice,                                     # determinismo: menor índice
    )


def escolher(candidatos: list[Candidato]) -> Candidato:
    """Devolve o representante. Determinístico: empate resolve pelo menor índice.

    Sem o desempate por índice, dois runs sobre o mesmo cluster poderiam
    escolher representantes diferentes, e o corpus deixaria de ser
    reconstruível a partir do manifesto (critério G1.5).
    """
    if not candidatos:
        raise ValueError("cluster vazio")
    return max(candidatos, key=_chave)


def ganho_de_licenca(candidatos: list[Candidato]) -> bool:
    """O critério 1 mudou alguma coisa neste cluster?

    Usado para medir quanto a regra rende na prática — se render zero, ela é
    complexidade sem retorno e o DOC-04 §5.3 precisa ser revisado.
    """
    if len(candidatos) < 2:
        return False
    escolhido = escolher(candidatos)
    melhor = _VALOR_PARTICAO.get(resolve(escolhido.licenca).partition, 0)
    outros = [_VALOR_PARTICAO.get(resolve(c.licenca).partition, 0)
              for c in candidatos if c.indice != escolhido.indice]
    return bool(outros) and melhor > min(outros)
