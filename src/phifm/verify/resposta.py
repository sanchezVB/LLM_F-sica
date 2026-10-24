"""O contrato de resposta do RLVR (ADR-0005 §4.5).

Lê a resposta como o prompt a pede, antes de ela ir ao barramento: a última
`\\boxed{}`, sem envoltório nem pontuação no fim, e só o lado direito de
`grandeza = expressão`. O contrato é **fechado na falha**: o que ele não cobre
segue como está, e o `parse` o recusa ou o lê como relação — nunca vira uma
leitura mais generosa.

O que ele **não** faz, de propósito:

- **Tirar unidade.** Pela decisão do dono de 2026-10-19, resposta com unidade
  dá `None` no verify; o prompt do RLVR pede a resposta sem unidade, nos
  símbolos do enunciado. Responder com unidade viola o formato e vale −1.
- **Tirar decoração** (`\\vec`, `\\hat`). Vetor e escalar são grandezas
  diferentes, e `\\hat` é marcador de operador.
- **Ler cadeias** (`K = A = B`) ou outras relações (`\\propto`, `\\sim`, `\\leq`):
  dão relação, e a relação é incomparável com uma expressão.

O corretor do G2 tem leitura própria e mais generosa; ele é o verificador
reservado do DOC-09 §5.4, e por isso não compartilha este contrato.
"""

from __future__ import annotations

import re

from phifm.verify.symbolic import _GREGAS, traduzir_unicode

__all__ = ["ler_gabarito", "ler_resposta", "ultima_caixa"]

_CAIXA = re.compile(r"\\(?:boxed|fbox)\s*\{")
_ENVOLTORIOS = (("$$", "$$"), ("$", "$"), ("\\[", "\\]"), ("\\(", "\\)"))
_PONTUACAO_FINAL = ".,;"

# Grandeza solta: uma letra ou uma grega (ou `\hbar`, `\ell`), com subscrito
# opcional — entre chaves, de uma letra ou dígito, ou com macro de fonte — e
# linhas opcionais. Nunca uma aplicação de função: `K(v) = …` fica como está.
_MACRO_DE_GRANDEZA = "|".join(sorted(_GREGAS | {"hbar", "ell"}, key=len, reverse=True))
_GRANDEZA_SOLTA = re.compile(
    rf"^\s*(?:[A-Za-z]|\\(?:{_MACRO_DE_GRANDEZA})(?![A-Za-z]))\s*"
    r"(?:_\s*(?:\{[^{}]*\}|\\[A-Za-z]+\s*\{[^{}]*\}|[A-Za-z0-9]))?\s*'*\s*$"
)
_IGUAL = re.compile(r"(?<![<>!=\\])=(?!=)|\\approx(?![A-Za-z])|\\simeq(?![A-Za-z])")
_APROXIMADO_NO_INICIO = re.compile(r"^\s*\\(?:approx|simeq)(?![A-Za-z])\s*")


def _fecha(texto: str, abre: int) -> int | None:
    """O índice da chave que fecha a aberta em `abre`, ou None."""
    nivel = 0
    for j in range(abre, len(texto)):
        if texto[j] == "{":
            nivel += 1
        elif texto[j] == "}":
            nivel -= 1
            if nivel == 0:
                return j
    return None


def ultima_caixa(texto: str) -> str | None:
    """O conteúdo da ÚLTIMA `\\boxed{…}` de chaves balanceadas, ou None.

    Caixa dentro de caixa lê a de dentro: `\\boxed{\\boxed{x}}` é `x`.
    """
    for m in reversed(list(_CAIXA.finditer(texto))):
        fim = _fecha(texto, m.end() - 1)
        if fim is not None:
            conteudo = texto[m.end():fim]
            dentro = ultima_caixa(conteudo)
            return dentro if dentro is not None else conteudo
    return None


def _desembrulhar(s: str) -> str:
    """Tira `$…$`, `\\(…\\)`, `\\[…\\]` e a pontuação do fim, até estabilizar."""
    anterior = None
    while s != anterior:
        anterior = s
        s = s.strip().rstrip(_PONTUACAO_FINAL).strip()
        for a, b in _ENVOLTORIOS:
            if s.startswith(a) and s.endswith(b) and len(s) >= len(a) + len(b):
                s = s[len(a):len(s) - len(b)]
                break
    return s


def _partes_de_nivel_zero(s: str) -> list[str]:
    """Divide em `=`, `\\approx` e `\\simeq` fora de chaves e parênteses:
    o `=` de `\\sum_{n=1}` não é o da resposta."""
    partes, nivel, ultimo = [], 0, 0
    profundidade = []
    for c in s:
        if c in "{(":
            nivel += 1
        elif c in "})":
            nivel -= 1
        profundidade.append(nivel)
    for m in _IGUAL.finditer(s):
        if profundidade[m.start()] == 0:
            partes.append(s[ultimo:m.start()])
            ultimo = m.end()
    partes.append(s[ultimo:])
    return partes


def _canonizar(s: str) -> str:
    traduzido = traduzir_unicode(s)
    if traduzido is not None:
        s = traduzido
    s = _desembrulhar(s)
    s = _APROXIMADO_NO_INICIO.sub("", s, count=1)
    partes = _partes_de_nivel_zero(s)
    # Exatamente dois lados, e o esquerdo uma grandeza solta. Cadeia de
    # igualdades fica inteira: é relação, e relação não se compara.
    if len(partes) == 2 and _GRANDEZA_SOLTA.match(partes[0]):
        s = partes[1]
    return _desembrulhar(s)


def ler_resposta(texto: str) -> str | None:
    """A resposta de uma geração, pelo contrato. Sem `\\boxed{}`, None."""
    caixa = ultima_caixa(texto)
    if caixa is None:
        return None
    resposta = _canonizar(caixa)
    return resposta or None


def ler_gabarito(texto: str) -> str:
    """O gabarito pelo mesmo contrato, sem exigir a caixa."""
    caixa = ultima_caixa(texto)
    return _canonizar(caixa if caixa is not None else texto)
