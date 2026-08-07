"""Subscrito nomeado → identificador estável (DOC-03 §3).

Subscrito nomeado é a forma normal de distinguir grandezas homônimas em
Física: `E_{cin}` e `E_{pot}` são duas energias, e `E` sozinho é ambíguo entre
energia, campo elétrico e módulo de Young (ver `verify/dimensional.AMBIGUOS`).
É justamente o subscrito que desfaz a ambiguidade — então ele precisa chegar
inteiro a quem for resolver o símbolo.

O backend ANTLR do `sympy.parsing.latex` lê o conteúdo de `_{...}` como
**expressão**, não como nome:

    E_{cin}     →  E_{c*(i*n)}     e ainda solta `i` e `n` como símbolos livres
    \\rho_{xy}   →  rho_{x*y}
    v_{0}       →  Symbol('v_{0}')  ≠  Symbol('v_0') de `v_0`

A segunda linha é a grave, e o motivo é uma frase só: **produto comuta, nome
não.** `\\rho_{xy}` e `\\rho_{yx}` colapsam no mesmo símbolo, e resistividade
Hall é antissimétrica. `g_{\\mu\\nu}` e `g_{\\nu\\mu}` idem. Ou seja, o parser
estava executando por baixo exatamente a transformação que o DOC-03 §3.2 lista
como **rejeitada** — "Normalizar posição de índices" — e que o canonicalizador
se recusa a fazer de propósito.

A terceira linha é silenciosa e cara: `v_{0}` e `v_0` são a mesma grandeza e
viravam símbolos distintos, então a forma com chaves não casava com a tabela
`INEQUIVOCOS`.

Este módulo mora na camada de ingestão, não no verificador, porque o artefato
que ele produz é de ingestão: um identificador canônico por grandeza mais o
mapa de volta ao LaTeX do autor — o que o DOC-03 §5 pede para popular
`context["dimensions"]`, e o que o DOC-03 §3.1 exige ao separar
`equations[].latex` (treino) de `equations[].canonical_latex` (índices).

**Conservador por princípio** (DOC-03 §3.2, "na dúvida, não canonicalize"):
subscrito que não seja composto só de nomes — porque tem operador, macro
desconhecida ou `\\mathbf` — fica intacto. Um identificador a menos custa uma
dedup perdida; um identificador errado destrói distinção física no corpus
inteiro.
"""

from __future__ import annotations

import re
import string

__all__ = ["normalizar_subscritos", "blindar_subscritos", "identificador"]

# Macros aceitas dentro de um subscrito. A lista é fechada de propósito: macro
# fora dela faz o subscrito inteiro ser deixado como está, em vez de virar um
# nome que ninguém escreveu.
_GREGAS = frozenset(
    """alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota
    kappa lambda mu nu xi omicron pi varpi rho varrho sigma varsigma tau
    upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma
    Upsilon Phi Psi Omega ell hbar infty partial nabla dagger prime perp
    parallel star ast circ odot oplus otimes""".split()
)

# Fontes retas — puramente tipográficas em subscrito, e a forma usual de
# escrever nome: `E_{\rm cin}`, `E_{\mathrm{cin}}`, `E_{\text{cin}}`.
#
# ⚠️ `\mathbf` e `\vec` NÃO entram aqui. O DOC-03 §3.2 rejeita explicitamente
# unificar `\mathbf{B}` ↔ `B`: a distinção vetor/escalar é semântica. Subscrito
# com `\mathbf` cai na regra conservadora e fica intacto.
_ENVOLTORIO = re.compile(r"^\\(?:mathrm|text|textrm|textup|mbox)\s*\{(.*)\}$", re.S)
_INTERRUPTOR = re.compile(r"^\\(?:rm|it|sf|tt|up|scriptstyle|displaystyle)(?![A-Za-z])\s*")

# Base de um subscrito: macro ou letra solta. `T^{\mu}_{\nu}` não casa aqui
# porque a base do `_` seria `}` — e é bom que não case: índice contravariante
# é problema à parte, e mais difícil (ver as limitações no fim do módulo).
#
# ⚠️ Os dois ramos têm regras de vizinhança **diferentes**, e igualá-los é um
# defeito sutil. Macro começa em `\`, que já a delimita: exigir que não venha
# letra antes fazia `\hbar\omega_{max}` — LaTeX corriqueiro — não casar, porque
# antes do segundo `\` vem o `r` de `hbar`. Letra solta, essa sim precisa da
# restrição: sem ela, `sigma_SB` casaria com base `a` e viraria `sigm` + um
# identificador `a_SB` que ninguém escreveu.
_MACRO = r"\\[A-Za-z]+"
_LETRA = r"(?<![A-Za-z\\])[A-Za-z]"
_BASE = rf"({_MACRO}|{_LETRA})"

# `X_{...}` — o conteúdo é fatiado com contagem de chaves, porque
# `E_{\mathrm{cin}}` aninha e regex não fecha par aninhado.
_ABRE_CHAVE = re.compile(_BASE + r"\s*_\s*\{")

# `X_nome` já achatado — a ingestão pode já ter normalizado, e aí o ANTLR
# recebe `E_cin` e estilhaça do mesmo jeito (`E_{c}*(i*n)`, que ainda por cima
# colapsa `E_cin` com `E_cal` em `E_{c}`). Subscrito de um caractere fica de
# fora: `v_0` e `k_B` já parseiam certo.
_ACHATADO = re.compile(_BASE + r"_([A-Za-z0-9]{2,})(?![A-Za-z0-9_{])")


def _atomos(conteudo: str) -> list[str] | None:
    """Fatia o conteúdo de um subscrito em nomes, ou devolve ``None``.

    ``None`` significa "isto não é um nome" e é a resposta certa para
    `k_{n+1}`, `x_{\\mathbf{i}}` e qualquer coisa com operador: nesses casos o
    subscrito fica como está.
    """
    s = conteudo.strip()
    while (m := _ENVOLTORIO.match(s)) is not None:
        s = m.group(1).strip()
    s = _INTERRUPTOR.sub("", s).strip()

    atomos: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace() or c == ",":
            # Vírgula separa índice (`T_{i,j}`), não junta. Preservar a
            # separação é o que mantém `T_{i,j}` distinto de `T_{j,i}`.
            i += 1
            continue
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            if m is None or m.group(1) not in _GREGAS:
                return None
            atomos.append(m.group(1))
            i += m.end()
            continue
        if c.isalnum():
            m = re.match(r"[A-Za-z0-9]+", s[i:])
            atomos.append(m.group(0))
            i += m.end()
            continue
        return None  # `+`, `-`, `^`, chave aninhada solta…
    return atomos or None


def identificador(base: str, atomos: list[str]) -> str:
    """`\\rho` + `['xy']` → ``rho_xy``; `g` + `['mu','nu']` → ``g_mu_nu``.

    Os átomos entram na **ordem escrita**. É o que separa `g_mu_nu` de
    `g_nu_mu` — a distinção que o parser apagava.
    """
    return "_".join([base.lstrip("\\"), *atomos])


def _ocorrencias(texto: str) -> list[tuple[int, int, str]]:
    """(início, fim, identificador) de cada subscrito nomeado, sem sobrepor."""
    achados: list[tuple[int, int, str]] = []

    pos = 0
    while (m := _ABRE_CHAVE.search(texto, pos)) is not None:
        base, abre = m.group(1), m.end() - 1
        nivel, j = 0, abre
        while j < len(texto):
            if texto[j] == "{":
                nivel += 1
            elif texto[j] == "}":
                nivel -= 1
                if nivel == 0:
                    break
            j += 1
        if nivel != 0:  # chave sem par: LaTeX quebrado, não é problema nosso
            pos = m.end()
            continue
        atomos = _atomos(texto[abre + 1 : j])
        if atomos is not None:
            achados.append((m.start(), j + 1, identificador(base, atomos)))
        pos = j + 1

    for m in _ACHATADO.finditer(texto):
        if any(ini <= m.start() < fim for ini, fim, _ in achados):
            continue
        atomos = _atomos(m.group(2))
        if atomos is not None:
            achados.append((m.start(), m.end(), identificador(m.group(1), atomos)))

    achados.sort()
    return achados


def _cola(esquerda: str, direita: str) -> bool:
    """Os dois trechos fundiriam num token só se ficassem encostados?

    `\\hbar\\omega_{max}` reescrito sem cuidado vira `\\hbaromega_max` — uma
    macro que não existe, a partir de LaTeX perfeitamente válido. É o tipo de
    defeito que não levanta exceção: o parser apenas devolve outro símbolo.
    """
    if not esquerda or not direita:
        return False
    return (esquerda[-1].isalnum() or esquerda[-1] == "\\") and direita[0].isalnum()


def _substituir(texto: str, novo_de) -> tuple[str, list[tuple[str, str, str]]]:
    """Devolve o texto reescrito e as trocas ``(substituto, identificador,
    trecho original)`` — as três formas que os chamadores precisam mapear."""
    saida = ""
    trocas: list[tuple[str, str, str]] = []
    fim_anterior = 0
    for ini, fim, ident in _ocorrencias(texto):
        original = texto[ini:fim]
        substituto = novo_de(ident, original)
        saida += texto[fim_anterior:ini]
        # Espaço só onde faria falta. Em LaTeX e na sintaxe do SymPy ele
        # separa sem alterar o sentido — `m v` continua produto.
        if _cola(saida, substituto):
            saida += " "
        saida += substituto
        if _cola(substituto, texto[fim : fim + 1]):
            saida += " "
        trocas.append((substituto, ident, original))
        fim_anterior = fim
    return saida + texto[fim_anterior:], trocas


def normalizar_subscritos(latex: str) -> tuple[str, dict[str, str]]:
    """`E_{cin} = \\frac{1}{2}mv^2` → `E_cin = \\frac{1}{2}mv^2`, `{E_cin: 'E_{cin}'}`.

    Devolve o texto com cada subscrito nomeado reescrito como identificador
    único, e o mapa ``{identificador: trecho LaTeX original}``. O mapa é o que
    permite ao `context["dimensions"]` declarar `E_cin` e ainda saber que o
    autor escreveu `E_{\\rm cin}`.

    O texto original **não** é substituído no corpus: DOC-03 §3.1 manda treinar
    no LaTeX do autor e usar a forma canônica só para comparar e indexar.
    """
    texto, trocas = _substituir(latex, lambda ident, original: ident)
    return texto, {ident: original for _, ident, original in trocas}


# Base do marcador: letra + subscrito numérico é a única forma que o ANTLR
# atravessa sem estilhaçar (`W_{12}` → `Symbol('W_{12}')`). A base é escolhida
# entre as ausentes do texto, então o marcador não pode colidir com conteúdo.
_CANDIDATAS = tuple(string.ascii_uppercase + string.ascii_lowercase)


# Identificador já normalizado: `E_cin`, `rho_xy`, `g_mu_nu`, `v_0`.
#
# A base aqui é `[A-Za-z][A-Za-z0-9]*`, e não uma letra só, porque é
# exatamente o que `normalizar_subscritos` produz a partir de `\rho_{xy}`. Com
# base de uma letra, `rho_xy` escapava da blindagem e o ANTLR o recebia inteiro
# — de volta ao defeito que este módulo existe para fechar.
_IDENTIFICADOR = re.compile(
    r"(?<![A-Za-z0-9\\_])([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)(?![A-Za-z0-9_])"
)


def blindar_subscritos(latex: str) -> tuple[str, dict[str, str]]:
    """Troca identificadores compostos por marcadores que o ANTLR não quebra.

    Espera texto **já passado por `normalizar_subscritos`** — opera sobre os
    identificadores que aquela produz, não sobre `_{...}` cru.

    Devolve ``(texto, {nome_do_marcador: identificador})``. O chamador parseia
    o texto e desfaz a troca nos símbolos — ver `verify.symbolic.parse`.

    Existe separado de `normalizar_subscritos` porque resolve outro problema:
    aquela produz o identificador que vai para o schema, esta o protege de um
    parser específico. Se o backend LaTeX do SymPy for trocado um dia, esta
    função sai e aquela fica.

    Blinda inclusive o que o ANTLR atravessaria intacto (`v_0`): distinguir
    custaria uma segunda regra de "o que é seguro", e regra de segurança
    duplicada é onde o próximo defeito se esconde.
    """
    livre = next((c for c in _CANDIDATAS if f"{c}_" not in latex), None)
    if livre is None:
        # Precisaria de uma expressão usando as 52 letras com subscrito. Se
        # acontecer, devolver intacto perde a blindagem — mas blindar com base
        # colidente trocaria um símbolo do autor por outro, em silêncio.
        return latex, {}

    contador = iter(range(1000, 10000))
    mapa: dict[str, str] = {}

    def _marcar(m: re.Match) -> str:
        marcador = f"{livre}_{{{next(contador)}}}"
        mapa[marcador] = m.group(1)
        return marcador

    return _IDENTIFICADOR.sub(_marcar, latex), mapa


# ── Limitações declaradas ──────────────────────────────────────────────────
#
# Índice contravariante não é tratado, e já estava quebrado antes deste módulo:
# o ANTLR lê `T^{\mu}_{\nu}` como `T**mu` e **descarta o subscrito**, e
# `T_{\nu}^{\mu}` como `T_{nu}**mu`, potência em vez de índice. Consertar isso
# exige representar tensor com índices, o que é decisão de schema (DOC-03 §3.2
# trata posição de índice como informação a preservar, não a normalizar), não
# de parser. O que este módulo garante é não piorar: a base `_BASE` não casa
# depois de `}`, então construções com sobrescrito ficam como estavam.
#
# Também ficam de fora, de propósito:
#
#   `k_{n+1}`      subscrito com operador — não é nome, e tratá-lo como nome
#                  apagaria a aritmética do índice
#   `x_{\mathbf{i}}`  `\mathbf` é semântico (DOC-03 §3.2), não tipográfico
#   `mv_{max}`     letra colada em letra: `m·v_max` ou um símbolo `mv`? A regra
#                  conservadora não escolhe. Com o espaço que todo mundo
#                  escreve — `m v_{max}` — resolve normalmente.
