"""Registro de licenças — o ADR-0001 imposto em código.

O DOC-01 §6 exige que `license.train_ok` seja "um booleano que o *loader*
respeita, não uma nota em planilha". Este módulo é onde isso acontece: cada
licença observada no corpus é resolvida em três direitos independentes, e o
roteamento de partição decorre deles mecanicamente.

Os três direitos vêm do ADR-0001 §2 e **não podem ser colapsados em um**:

    D1  acesso        — podemos obter e ler?
    D2  treinar       — podemos usar para treinar um modelo?     → train_ok
    D3  redistribuir  — podemos publicar os bytes?               → redistributable

Colapsá-los é o risco número um registrado no ADR: aplicar a regra de D3 ao
treino derrubaria o corpus treinável de ~30 B para ~8 B tokens e reprovaria o
Tier 2 antes de começar.

A cláusula não-comercial recebe tratamento próprio: sob a decisão Q3 (pesos
sob Apache-2.0), conteúdo NC fica **fora do treino** — ADR-0001 §4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class Partition(StrEnum):
    """Para onde o documento é fisicamente roteado (DOC-01 §6, ADR-0001 §5)."""

    TRAIN_OPEN = "train_open"      # treina e pode ser redistribuído  → PhysCorpus-Open
    TRAIN_ONLY = "train_only"      # treina, não redistribui           → PhysCorpus-Full
    EVAL_ONLY = "eval_only"        # NUNCA treina                      → PhysEval-Restricted
    EXCLUDED = "excluded"          # fora do corpus


@dataclass(frozen=True)
class LicenseRecord:
    spdx_id: str
    license_url: str | None
    train_ok: bool
    redistributable: bool
    commercial_ok: bool
    attribution_required: bool
    share_alike: bool
    non_commercial: bool
    note: str = ""

    @property
    def partition(self) -> Partition:
        if not self.train_ok:
            return Partition.EVAL_ONLY
        return Partition.TRAIN_OPEN if self.redistributable else Partition.TRAIN_ONLY


def _lic(spdx, url, *, train, redist, comm, attr=False, sa=False, nc=False, note="") -> LicenseRecord:
    return LicenseRecord(spdx, url, train, redist, comm, attr, sa, nc, note)


# ── Catálogo ──────────────────────────────────────────────────────────────
# `train_ok=False` para NC é decisão de projeto, não leitura literal da
# licença: a cláusula NC não proíbe treinar, mas publicar pesos comerciais
# (Apache-2.0) a partir dela é juridicamente não assentado. ADR-0001 §4
# adota a postura conservadora e registra a alternativa (opção C: dois modelos).

CATALOG: dict[str, LicenseRecord] = {
    "CC0-1.0": _lic("CC0-1.0", "https://creativecommons.org/publicdomain/zero/1.0/",
                    train=True, redist=True, comm=True),
    "CC-BY-4.0": _lic("CC-BY-4.0", "https://creativecommons.org/licenses/by/4.0/",
                      train=True, redist=True, comm=True, attr=True),
    "CC-BY-SA-4.0": _lic("CC-BY-SA-4.0", "https://creativecommons.org/licenses/by-sa/4.0/",
                         train=True, redist=True, comm=True, attr=True, sa=True,
                         note="Share-alike: obras derivadas do CORPUS herdam a licença. "
                              "Não afeta os pesos, que não são obra derivada do corpus."),
    "CC-BY-NC-SA-4.0": _lic("CC-BY-NC-SA-4.0", "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                            train=False, redist=False, comm=False, attr=True, sa=True, nc=True,
                            note="NC excluído do treino sob Q3 (ADR-0001 §4)."),
    "CC-BY-NC-ND-4.0": _lic("CC-BY-NC-ND-4.0", "https://creativecommons.org/licenses/by-nc-nd/4.0/",
                            train=False, redist=False, comm=False, attr=True, nc=True,
                            note="NC + ND. Excluído do treino sob Q3."),
    "arXiv-1.0": _lic("LicenseRef-arXiv-perpetual-nonexclusive",
                      "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
                      train=True, redist=False, comm=True,
                      note="Concede ao arXiv o direito de distribuir, NÃO a terceiros. "
                           "D2 sob argumento de TDM/uso legítimo; D3 negado."),
    "US-PD": _lic("LicenseRef-US-Government-Work", None,
                  train=True, redist=True, comm=True,
                  note="Obra do governo federal dos EUA — 17 U.S.C. §105. NASA NTRS, NIST, OSTI."),
    "PD-old": _lic("LicenseRef-Public-Domain", None,
                   train=True, redist=True, comm=True,
                   note="Domínio público por expiração (pré-1931 nos EUA)."),
    "COPYRIGHTED": _lic("LicenseRef-All-Rights-Reserved", None,
                        train=False, redist=False, comm=False,
                        note="Copyright ativo. Partição EVAL-ONLY (ADR-0001 §5). "
                             "Jackson, Landau, Sakurai, Peskin, MTW…"),
    "UNKNOWN": _lic("NOASSERTION", None,
                    train=True, redist=False, comm=False,
                    note="Licença não resolvida. Padrão conservador: treina, não redistribui. "
                         "Teses e repositórios institucionais caem aqui."),
}

# ── Resolução a partir da URL crua observada no metadado ───────────────────
_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"creativecommons\.org/publicdomain/zero", re.I), "CC0-1.0"),
    (re.compile(r"creativecommons\.org/licenses/by-nc-nd", re.I), "CC-BY-NC-ND-4.0"),
    (re.compile(r"creativecommons\.org/licenses/by-nc-sa", re.I), "CC-BY-NC-SA-4.0"),
    (re.compile(r"creativecommons\.org/licenses/by-nc(?![-a-z])", re.I), "CC-BY-NC-SA-4.0"),
    (re.compile(r"creativecommons\.org/licenses/by-sa", re.I), "CC-BY-SA-4.0"),
    (re.compile(r"creativecommons\.org/licenses/by(?![-a-z])", re.I), "CC-BY-4.0"),
    (re.compile(r"arxiv\.org/licenses/nonexclusive-distrib", re.I), "arXiv-1.0"),
    (re.compile(r"^arXiv-perpetual-nonexclusive$", re.I), "arXiv-1.0"),
]


def resolve(raw: str | None) -> LicenseRecord:
    """URL crua de licença → `LicenseRecord`.

    Nunca levanta exceção e nunca devolve ``None``: uma licença irreconhecível
    vira ``UNKNOWN``, cujo padrão é conservador (treina, não redistribui). O
    princípio A3 do DOC-02 exige que a ausência de resolução seja **visível**,
    não silenciosa — por isso `UNKNOWN` é um valor real e contável, não um nulo.
    """
    if not raw:
        return CATALOG["arXiv-1.0"]  # ausência no arXiv = licença padrão
    for pattern, key in _RULES:
        if pattern.search(raw):
            return CATALOG[key]
    return CATALOG["UNKNOWN"]


def resolve_spdx(raw: str | None) -> str:
    return resolve(raw).spdx_id


def resolve_partition(raw: str | None) -> str:
    return resolve(raw).partition.value
