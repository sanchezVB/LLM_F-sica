"""Canonicalizador LaTeX (DOC-03 §3).

Consumido por deduplicação (DOC-04), recuperação de fórmulas (DOC-13) e
descontaminação por equação (DOC-12).

> **Canonicalização serve para COMPARAR, nunca para TREINAR** (DOC-03 §3.1).
> O modelo treina no LaTeX original do autor — diversidade notacional é sinal,
> não ruído. A forma canônica existe só para responder "estas duas equações
> são a mesma?".
"""

from __future__ import annotations

from phifm.core.latex.subscritos import (
    blindar_subscritos,
    identificador,
    normalizar_subscritos,
)

__all__ = ["blindar_subscritos", "identificador", "normalizar_subscritos"]
