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
from dataclasses import dataclass, replace
from enum import StrEnum
from urllib.parse import unquote


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
    "CC-BY-NC-4.0": _lic("CC-BY-NC-4.0", "https://creativecommons.org/licenses/by-nc/4.0/",
                         train=False, redist=False, comm=False, attr=True, nc=True,
                         note="NC excluído do treino sob Q3 (ADR-0001 §4)."),
    "CC-BY-NC-SA-4.0": _lic("CC-BY-NC-SA-4.0", "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                            train=False, redist=False, comm=False, attr=True, sa=True, nc=True,
                            note="NC excluído do treino sob Q3 (ADR-0001 §4)."),
    "CC-BY-NC-ND-4.0": _lic("CC-BY-NC-ND-4.0", "https://creativecommons.org/licenses/by-nc-nd/4.0/",
                            train=False, redist=False, comm=False, attr=True, nc=True,
                            note="NC + ND. Excluído do treino sob Q3."),
    "CC-PDDC": _lic("CC-PDDC", "https://creativecommons.org/licenses/publicdomain/",
                    train=True, redist=True, comm=True,
                    note="Dedicação ao domínio público da Creative Commons — a ferramenta "
                         "anterior ao CC0, aposentada em 2010. Mesmos direitos do CC0. "
                         "No arXiv: 1.660 registros, todos de 2008 a 2015."),
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

# ── Resolução a partir do valor cru observado no metadado ──────────────────
# ⚠️ A VERSÃO faz parte da licença. Até 2026-10-01 as regras olhavam só o tipo, e
# `by/3.0` saía rotulado `CC-BY-4.0`. Os direitos são os mesmos — a partição não
# mudava, e por isso nenhum teste acusava —, mas 7.631 registros do índice
# carregavam o identificador da licença errada, e atribuição correta é a condição
# de uso de uma CC-BY: um PhysCorpus-Open publicado assim citaria a licença errada
# em 2% dos documentos.
#
# A mesma leitura por tipo deixava `licenses/publicdomain/` sem regra nenhuma:
# 1.660 registros dedicados ao domínio público caíam em UNKNOWN e contavam como
# NÃO redistribuíveis. O erro era conservador, e por isso sobreviveu — a fração
# redistribuível do corpus saía 0,1 ponto abaixo do que é.
#
# ⚠️ A primeira versão desta correção trocou um defeito por outro, e foi um revisor
# independente que achou. Ela procurava (`search`) a primeira URL da Creative
# Commons na string, e uma string com DUAS licenças — `…/by/4.0/ …/by-nc/4.0/` —
# passou de `eval_only` para `train_open`: as regras antigas testavam NC primeiro
# sobre a string inteira, e a nova parava na primeira ocorrência. Daí as duas
# propriedades que este bloco tem agora, e que valem mais que qualquer regra:
#
#   1. Só se AFIRMA um identificador quando a string INTEIRA é uma licença — a
#      URL canônica, ou o rótulo SPDX. Texto que apenas contém uma URL, duas
#      licenças juntas, versão com lixo depois, jurisdição (`3.0/us`): nada disso
#      recebe rótulo, e sem rótulo não se redistribui.
#   2. NC ganha de tudo. Depois de resolver, se o resultado treinaria e a string
#      menciona a cláusula NC num contexto de Creative Commons, o resultado é
#      trocado por NC. É uma guarda sobre o RESULTADO, não mais uma regra na
#      fila — e por isso não depende da ordem nem da forma das outras.
_VERSOES = ("1.0", "2.0", "2.5", "3.0", "4.0")

# Tipo → entrada 4.0 do catálogo, que serve de molde dos direitos: entre as
# versões de um mesmo tipo muda o texto jurídico, não o que este projeto faz com
# o documento (ADR-0001 §2 decide por cláusula — BY, SA, NC, ND —, não por versão).
_MOLDE = {
    "by": "CC-BY-4.0",
    "by-sa": "CC-BY-SA-4.0",
    "by-nc": "CC-BY-NC-4.0",
    "by-nc-sa": "CC-BY-NC-SA-4.0",
    "by-nc-nd": "CC-BY-NC-ND-4.0",
}


def _pares() -> tuple[dict[tuple[str, str], LicenseRecord], dict[tuple[str, str], LicenseRecord]]:
    """Os pares (tipo, versão) que EXISTEM, por URL e por rótulo SPDX.

    Tabela fechada, e não o produto livre tipo × versão: a `by-nc-nd` só existe a
    partir da 2.0 — na 1.0 a mesma licença se chama `by-nd-nc`, e o SPDX a
    registra como `CC-BY-NC-ND-1.0`. O produto livre fabricava a URL
    `by-nc-nd/1.0/`, que a Creative Commons nunca publicou.
    """
    por_url: dict[tuple[str, str], LicenseRecord] = {}
    por_rotulo: dict[tuple[str, str], LicenseRecord] = {}
    for tipo, chave in _MOLDE.items():
        for versao in _VERSOES:
            caminho = tipo
            if tipo == "by-nc-nd" and versao == "1.0":
                caminho = "by-nd-nc"
            r = CATALOG[chave] if versao == "4.0" else replace(
                CATALOG[chave], spdx_id=f"CC-{tipo.upper()}-{versao}",
                license_url=f"https://creativecommons.org/licenses/{caminho}/{versao}/")
            por_url[(caminho, versao)] = r
            por_rotulo[(tipo, versao)] = r
    return por_url, por_rotulo


_CC_POR_URL, _CC_POR_ROTULO = _pares()
_POR_SPDX = {r.spdx_id.lower(): r for r in CATALOG.values()}

# A string INTEIRA é uma URL de licença da Creative Commons. O sufixo aceito é o
# das duas páginas que a própria CC serve para cada licença (`deed`, `legalcode`).
_CC_URL = re.compile(
    r"(?:https?://)?(?:www\.)?creativecommons\.org/licenses/"
    r"(?P<tipo>[a-z+]+(?:-[a-z+]+)*)"
    r"(?:/(?P<versao>\d+\.\d+))?"
    r"(?:/(?P<jurisdicao>[a-z]{2,3}))?"
    r"(?:/(?:deed|legalcode)(?:\.[a-z_-]+)?)?/?", re.I)
# A string INTEIRA é um rótulo: `CC-BY-NC-SA-3.0`, `cc-by`, `CC BY-NC 4.0`.
_CC_ROTULO = re.compile(
    r"cc[-_ ]?(?P<tipo>by(?:[-_ ](?:nc|sa|nd))*)(?:[-_ ]v?(?P<versao>\d+\.\d+))?", re.I)

_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?:https?://)?(?:www\.)?creativecommons\.org/publicdomain/zero/1\.0"
                r"(?:/(?:deed|legalcode)(?:\.[a-z_-]+)?)?/?", re.I), "CC0-1.0"),
    (re.compile(r"(?:https?://)?arxiv\.org/licenses/nonexclusive-distrib/1\.0/?", re.I),
     "arXiv-1.0"),
    (re.compile(r"arXiv-perpetual-nonexclusive", re.I), "arXiv-1.0"),
]

_CONTEXTO_CC = re.compile(r"creativecommons|(?<![a-z])cc(?![a-z])", re.I)
_TOKEN_NC = re.compile(r"(?<![a-z])nc(?![a-z])", re.I)


def _menciona_nc(texto: str) -> bool:
    """A string fala de Creative Commons e traz `nc` como palavra — em qualquer forma."""
    return bool(_CONTEXTO_CC.search(texto) and _TOKEN_NC.search(texto))


def _nc_sem_rotulo(raw: str) -> LicenseRecord:
    """Direitos de NC, identificador não afirmado.

    ⚠️ Não pode cair em `UNKNOWN`: o padrão conservador de lá é *treina, não
    redistribui*, e treinar em conteúdo NC é exatamente o que o ADR-0001 §4
    proíbe. Conservador, aqui, é não treinar.
    """
    return replace(CATALOG["CC-BY-NC-ND-4.0"], spdx_id="NOASSERTION", license_url=None,
                   note=f"Creative Commons com cláusula NC, em forma que o catálogo "
                        f"não rotula ({raw[:80]!r}). Tratada como NC: fora do treino.")


def _afirmar(texto: str) -> LicenseRecord:
    """A string inteira é UMA licença conhecida → o registro; senão, `UNKNOWN`."""
    m = _CC_URL.fullmatch(texto)
    if m and m["jurisdicao"] is None:
        tipo = m["tipo"].lower()
        if tipo == "publicdomain" and m["versao"] is None:
            return CATALOG["CC-PDDC"]
        r = _CC_POR_URL.get((tipo, m["versao"] or ""))
        if r is not None:
            return r
    m = _CC_ROTULO.fullmatch(texto)
    if m:
        tipo = re.sub(r"[_ ]", "-", m["tipo"].lower())
        r = _CC_POR_ROTULO.get((tipo, m["versao"] or ""))
        if r is not None:
            return r
    for padrao, chave in _RULES:
        if padrao.fullmatch(texto):
            return CATALOG[chave]
    return _POR_SPDX.get(texto.lower(), CATALOG["UNKNOWN"])


def resolve(raw: str | None) -> LicenseRecord:
    """Valor cru de licença → `LicenseRecord`.

    Nunca levanta exceção e nunca devolve ``None``: uma licença irreconhecível
    vira ``UNKNOWN``, cujo padrão é conservador (treina, não redistribui). O
    princípio A3 do DOC-02 exige que a ausência de resolução seja **visível**,
    não silenciosa — por isso `UNKNOWN` é um valor real e contável, não um nulo.
    """
    if not raw:
        return CATALOG["arXiv-1.0"]  # ausência no arXiv = licença padrão
    texto = unquote(raw).strip()
    r = _afirmar(texto)
    if r.train_ok and _menciona_nc(texto):
        return _nc_sem_rotulo(raw)
    return r


def resolve_spdx(raw: str | None) -> str:
    return resolve(raw).spdx_id


def resolve_partition(raw: str | None) -> str:
    return resolve(raw).partition.value
