"""Verificador de coerência dimensional (DOC-10 §3.2).

Álgebra sobre as sete dimensões-base do SI ``[M L T I Θ N J]``, com projeção
para dimensão de massa em unidades naturais.

**Por que não `pint`.** O `pint` resolve unidades concretas (metro, joule), e o
que se verifica aqui são **dimensões de símbolos algébricos** — não há número
com unidade em ``F = m·a²``. Além disso o caso que mais importa, unidades
naturais, exige projetar as sete dimensões numa única potência de energia, e
isso não existe no modelo do `pint`. Um vetor de sete racionais faz o trabalho
inteiro em ~40 linhas, sem dependência nova.

**O problema difícil é resolução de símbolos, e ele não tem solução completa.**
Um ``E`` pode ser energia, campo elétrico ou módulo de Young — no mesmo
documento. A cascata do DOC-10 §3.2 é seguida à risca:

  1. dimensões declaradas em ``claim.context["dimensions"]``
  2. inferência pela estrutura da equação (um único símbolo desconhecido)
  3. `INCONCLUSIVE`

Símbolos genuinamente ambíguos estão em `AMBIGUOS` e **nunca** são resolvidos
por conta própria, mesmo que uma das leituras seja mais comum. A expectativa
calibrada do DOC-03 §5 é **20–35% das equações verificáveis** — um verificador
que alegasse mais estaria adivinhando.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from phifm.verify.bus import Claim, Verdict, VerificationResult
from phifm.verify.symbolic import parse

BASES = ("M", "L", "T", "I", "Theta", "N", "J")


@dataclass(frozen=True)
class Dim:
    """Expoentes das sete dimensões-base, em `Rational` (há expoentes 1/2)."""

    exp: tuple = (0,) * 7

    def __post_init__(self):
        object.__setattr__(self, "exp", tuple(sp.Rational(e) for e in self.exp))

    def __mul__(self, o: Dim) -> Dim:
        return Dim(tuple(a + b for a, b in zip(self.exp, o.exp, strict=False)))

    def __truediv__(self, o: Dim) -> Dim:
        return Dim(tuple(a - b for a, b in zip(self.exp, o.exp, strict=False)))

    def __pow__(self, n) -> Dim:
        r = sp.Rational(n)
        return Dim(tuple(a * r for a in self.exp))

    @property
    def adimensional(self) -> bool:
        return all(e == 0 for e in self.exp)

    @property
    def massa(self) -> sp.Rational:
        """Dimensão de massa em unidades naturais (ℏ = c = 1).

        Com ℏ = c = 1, comprimento e tempo viram E⁻¹ e massa vira E¹, então
        ``M^a L^b T^c → E^(a−b−c)``. As outras quatro bases não participam.

        A projeção acerta os casos que importam: energia ``M L²T⁻²`` → E¹;
        velocidade ``L T⁻¹`` → E⁰, que é exatamente o esperado, porque em
        unidades naturais toda velocidade é uma fração de c.
        """
        a, b, c = self.exp[0], self.exp[1], self.exp[2]
        return a - b - c

    def __str__(self) -> str:
        partes = [
            f"{n}^{e}" if e != 1 else n
            for n, e in zip(BASES, self.exp, strict=False) if e != 0
        ]
        return "·".join(partes) if partes else "1"


def _d(**kw) -> Dim:
    return Dim(tuple(kw.get(n, 0) for n in BASES))


ADIM = Dim()
MASSA = _d(M=1)
COMPRIMENTO = _d(L=1)
TEMPO = _d(T=1)
VELOCIDADE = _d(L=1, T=-1)
ACELERACAO = _d(L=1, T=-2)
FORCA = _d(M=1, L=1, T=-2)
ENERGIA = _d(M=1, L=2, T=-2)
MOMENTO = _d(M=1, L=1, T=-1)
ACAO = _d(M=1, L=2, T=-1)
POTENCIA = _d(M=1, L=2, T=-3)
CARGA = _d(I=1, T=1)
FREQUENCIA = _d(T=-1)

# ── Símbolos com leitura única na literatura de Física ─────────────────────
#
# Critério de admissão: o símbolo tem UMA leitura convencional forte o
# bastante para que a outra seja erro de notação, não ambiguidade real.
# Qualquer dúvida manda o símbolo para `AMBIGUOS`, não para cá — porque um
# falso PASS dimensional é pior que um INCONCLUSIVE.
INEQUIVOCOS: dict[str, Dim] = {
    # constantes
    "c": VELOCIDADE,
    "hbar": ACAO,
    "h": ACAO,               # Planck; ALTURA usa outro símbolo — ver AMBIGUOS
    "G": _d(L=3, M=-1, T=-2),
    "g": ACELERACAO,
    "k_B": ENERGIA / _d(Theta=1),
    "kB": ENERGIA / _d(Theta=1),
    "epsilon_0": _d(I=2, T=4, M=-1, L=-3),
    "mu_0": _d(M=1, L=1, T=-2, I=-2),
    "N_A": _d(N=-1),
    "sigma_SB": _d(M=1, T=-3, Theta=-4),
    # cinemática e dinâmica
    "m": MASSA, "m_0": MASSA, "m0": MASSA, "M_massa": MASSA,
    "t": TEMPO, "dt": TEMPO, "tau": TEMPO,
    "v": VELOCIDADE, "u": VELOCIDADE, "v_0": VELOCIDADE, "v0": VELOCIDADE,
    "a": ACELERACAO,
    "F": FORCA,
    "x": COMPRIMENTO, "y": COMPRIMENTO, "z": COMPRIMENTO,
    "r": COMPRIMENTO, "d": COMPRIMENTO, "s": COMPRIMENTO,
    "omega": FREQUENCIA, "f": FREQUENCIA, "nu": FREQUENCIA,
    "rho": _d(M=1, L=-3),
    "q": CARGA,
    "W": ENERGIA,            # trabalho
    "gamma": ADIM,           # fator de Lorentz
    "beta": ADIM,            # v/c
}

# ── Símbolos ambíguos: NUNCA resolvidos sem declaração ─────────────────────
#
# Cada entrada lista as leituras que competem. Presença aqui força a cascata
# a passar pela inferência estrutural ou terminar em INCONCLUSIVE.
AMBIGUOS: dict[str, tuple[str, ...]] = {
    "E": ("energia", "campo elétrico", "módulo de Young"),
    "k": ("constante elástica", "número de onda", "Boltzmann"),
    "T": ("período", "temperatura", "tensão"),
    "L": ("comprimento", "momento angular", "indutância", "lagrangiana"),
    "I": ("corrente", "momento de inércia", "intensidade"),
    "p": ("momento linear", "pressão"),
    "P": ("potência", "pressão", "peso"),
    "Q": ("calor", "carga", "fator de qualidade"),
    "A": ("área", "amplitude", "número de massa"),
    "R": ("raio", "resistência", "constante dos gases"),
    "lambda": ("comprimento de onda", "constante de decaimento", "densidade linear"),
    "n": ("índice de refração", "número de mols", "densidade numérica"),
    "N": ("número de partículas", "força normal"),
    "S": ("entropia", "área", "ação"),
    "C": ("capacitância", "capacidade térmica", "constante"),
    "B": ("campo magnético", "número bariônico"),
    "V": ("volume", "potencial elétrico", "energia potencial"),
    "U": ("energia interna", "energia potencial"),
    "phi": ("fluxo", "fase", "potencial"),
}


class Indeterminada(Exception):
    """Símbolo sem dimensão resolvível."""

    def __init__(self, simbolo: str, motivo: str = ""):
        self.simbolo = simbolo
        self.motivo = motivo
        super().__init__(f"{simbolo}: {motivo or 'dimensão desconhecida'}")


class Incoerente(Exception):
    """Soma de termos com dimensões diferentes — o defeito que se busca."""

    def __init__(self, a: Dim, b: Dim, onde: str):
        self.a, self.b, self.onde = a, b, onde
        super().__init__(f"[{a}] + [{b}] em {onde}")


def dimensao(expr: sp.Expr, tabela: dict[str, Dim]) -> Dim:
    """Dimensão de uma expressão. Levanta `Indeterminada` ou `Incoerente`."""
    if expr.is_number:
        return ADIM

    if isinstance(expr, sp.Symbol):
        nome = expr.name
        if nome in tabela:
            return tabela[nome]
        if nome in AMBIGUOS:
            raise Indeterminada(nome, "ambíguo: " + ", ".join(AMBIGUOS[nome]))
        raise Indeterminada(nome)

    if isinstance(expr, sp.Mul):
        d = ADIM
        for arg in expr.args:
            d = d * dimensao(arg, tabela)
        return d

    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        if not exp.is_number:
            # x^n com n simbólico só é dimensionalmente definido se x for
            # adimensional — e aí o resultado também é.
            if dimensao(base, tabela).adimensional:
                return ADIM
            raise Indeterminada(str(expr), "expoente simbólico sobre base dimensional")
        return dimensao(base, tabela) ** sp.Rational(exp)

    if isinstance(expr, sp.Add):
        dims = [dimensao(a, tabela) for a in expr.args]
        for d in dims[1:]:
            if d.exp != dims[0].exp:
                raise Incoerente(dims[0], d, str(expr))
        return dims[0]

    if isinstance(expr, sp.Function):
        # sin, cos, exp, log... exigem argumento adimensional e devolvem
        # adimensional. É uma restrição real: `sin(t)` com t em segundos é
        # erro de Física, não escolha de notação.
        for arg in expr.args:
            d = dimensao(arg, tabela)
            if not d.adimensional:
                raise Incoerente(d, ADIM, f"argumento de {expr.func.__name__}")
        return ADIM

    raise Indeterminada(str(expr), f"nó não suportado: {type(expr).__name__}")


def _tabela(claim: Claim) -> dict[str, Dim]:
    """Nível 1 da cascata: declaração vence sempre.

    ``context["dimensions"] = {"E": {"M": 1, "L": 2, "T": -2}}`` ou o atalho
    ``{"E": "energia"}`` com nome de grandeza conhecida.
    """
    tab = dict(INEQUIVOCOS)
    ATALHOS = {
        "adimensional": ADIM, "massa": MASSA, "comprimento": COMPRIMENTO,
        "tempo": TEMPO, "velocidade": VELOCIDADE, "aceleracao": ACELERACAO,
        "forca": FORCA, "energia": ENERGIA, "momento": MOMENTO, "acao": ACAO,
        "potencia": POTENCIA, "carga": CARGA, "frequencia": FREQUENCIA,
        "temperatura": _d(Theta=1), "pressao": _d(M=1, L=-1, T=-2),
        "area": _d(L=2), "volume": _d(L=3), "corrente": _d(I=1),
    }
    for nome, spec in (claim.context.get("dimensions") or {}).items():
        if isinstance(spec, str):
            if spec not in ATALHOS:
                raise Indeterminada(nome, f"grandeza declarada desconhecida: {spec}")
            tab[nome] = ATALHOS[spec]
        elif isinstance(spec, Dim):
            tab[nome] = spec
        else:
            tab[nome] = _d(**dict(dict(spec).items()))
    return tab


def _inferir(expr: sp.Expr, alvo: Dim, tabela: dict[str, Dim]) -> Dim | None:
    """Nível 2 da cascata: um único desconhecido num produto de potências.

    Se ``expr`` é ``k·x^2`` e sabe-se que o todo vale energia, então
    ``[k] = [energia]/[x]^2``. Restringe-se a produto de potências com
    **exatamente um** símbolo desconhecido: com dois, o sistema é
    indeterminado, e resolver por chute é justamente o que a cascata evita.

    Devolve a dimensão inferida do desconhecido, ou ``None``.
    """
    desconhecidos = [
        s for s in expr.free_symbols if s.name not in tabela
    ]
    if len(desconhecidos) != 1:
        return None
    alvo_sym = desconhecidos[0]

    # Fatorar expr em (parte conhecida) · alvo^n
    n = sp.degree(sp.Poly(expr, alvo_sym)) if expr.is_polynomial(alvo_sym) else None
    if not n:
        return None
    resto = sp.simplify(expr / alvo_sym**n)
    if resto.has(alvo_sym):
        return None
    try:
        d_resto = dimensao(resto, tabela)
    except (Indeterminada, Incoerente):
        return None
    return (alvo / d_resto) ** sp.Rational(1, n)


class DimensionalVerifier:
    id = "dimensional/si7@0.1.0"

    def applicable(self, claim: Claim) -> bool:
        """Precisa de dois lados para comparar, ou de uma soma para checar.

        ``F = m·a²`` compara lados. ``½mv² + mgh`` checa coerência interna
        sem gabarito nenhum.
        """
        if claim.rhs is not None:
            return True
        return any(op in claim.lhs for op in ("+", "-"))

    def verify(self, claim: Claim) -> VerificationResult:
        sistema = claim.context.get("unit_system")
        if sistema is None:
            # DOC-10 §3.2: não escolher sistema por conta própria. Assumir SI
            # num documento em unidades naturais reprovaria `E = m`, que está
            # certo lá; assumir naturais num de SI aprovaria vacuamente.
            return VerificationResult(
                Verdict.INCONCLUSIVE, self.id, 0.0,
                "sistema de unidades não declarado (physics.conventions nulo) — "
                "declarar context['unit_system'] entre SI, gaussian, natural, hep",
            )
        if sistema == "gaussian":
            # Carga e campos mudam de dimensão e aparece fator 4π. A tabela
            # atual é SI: usá-la aqui daria FAIL espúrio em Coulomb.
            return VerificationResult(
                Verdict.INCONCLUSIVE, self.id, 0.0,
                "sistema gaussiano ainda não mapeado — tabela vigente é SI",
            )
        if sistema not in ("SI", "natural", "hep"):
            return VerificationResult(
                Verdict.INCONCLUSIVE, self.id, 0.0, f"sistema desconhecido: {sistema}"
            )

        try:
            tabela = _tabela(claim)
        except Indeterminada as exc:
            return VerificationResult(Verdict.INCONCLUSIVE, self.id, 0.0, str(exc))

        # Símbolo declarado é símbolo, não constante do SymPy: quem declarou
        # `I` como corrente não quer a unidade imaginária de volta.
        locais = {n: sp.Symbol(n) for n in (claim.context.get("dimensions") or {})}
        a = parse(claim.lhs, locais)
        b = parse(claim.rhs, locais) if claim.rhs is not None else None
        if a is None or (claim.rhs is not None and b is None):
            return VerificationResult(Verdict.INCONCLUSIVE, self.id, 0.0, "não parseou")

        try:
            da = dimensao(a, tabela)
        except Incoerente as exc:
            return self._incoerencia(exc, sistema, "lado esquerdo")
        except Indeterminada as exc:
            da = None
            falha_a = exc

        if b is None:
            if da is None:
                return VerificationResult(Verdict.INCONCLUSIVE, self.id, 0.0, str(falha_a))
            return VerificationResult(
                Verdict.PASS, self.id, 0.9,
                f"soma dimensionalmente coerente: todos os termos são [{da}]",
            )

        try:
            db = dimensao(b, tabela)
        except Incoerente as exc:
            return self._incoerencia(exc, sistema, "lado direito")
        except Indeterminada as exc:
            db = None
            falha_b = exc

        # Nível 2: um lado resolvido permite inferir o desconhecido do outro.
        if da is not None and db is None:
            inf = _inferir(b, da, tabela)
            if inf is None:
                return VerificationResult(Verdict.INCONCLUSIVE, self.id, 0.0, str(falha_b))
            return VerificationResult(
                Verdict.INCONCLUSIVE, self.id, 0.0,
                f"consistente se o símbolo livre valer [{inf}] — inferido, "
                "não verificado: com o desconhecido livre a igualdade é "
                "satisfazível por construção",
            )
        if db is not None and da is None:
            inf = _inferir(a, db, tabela)
            if inf is None:
                return VerificationResult(Verdict.INCONCLUSIVE, self.id, 0.0, str(falha_a))
            return VerificationResult(
                Verdict.INCONCLUSIVE, self.id, 0.0,
                f"consistente se o símbolo livre valer [{inf}] — inferido, não verificado",
            )
        if da is None and db is None:
            return VerificationResult(
                Verdict.INCONCLUSIVE, self.id, 0.0, f"{falha_a}; {falha_b}"
            )

        if sistema in ("natural", "hep"):
            ma, mb = da.massa, db.massa
            if ma == mb:
                return VerificationResult(
                    Verdict.PASS, self.id, 0.75,
                    f"dimensão de massa concorda: E^{ma} em ambos os lados "
                    "(confiança reduzida: em unidades naturais a checagem é "
                    "escalar, logo mais frouxa que o vetor de sete bases)",
                )
            return VerificationResult(
                Verdict.FAIL, self.id, 0.95,
                f"dimensão de massa difere: E^{ma} vs E^{mb}",
                counterexample=f"{claim.lhs} → E^{ma} | {claim.rhs} → E^{mb}",
            )

        if da.exp == db.exp:
            return VerificationResult(
                Verdict.PASS, self.id, 0.95, f"dimensões idênticas: [{da}]"
            )
        return VerificationResult(
            Verdict.FAIL, self.id, 0.98,
            f"dimensões incompatíveis: [{da}] vs [{db}]",
            counterexample=f"{claim.lhs} → [{da}] | {claim.rhs} → [{db}]",
        )

    def _incoerencia(self, exc: Incoerente, sistema: str, onde: str) -> VerificationResult:
        if sistema in ("natural", "hep") and exc.a.massa == exc.b.massa:
            # Em unidades naturais os termos só precisam concordar em potência
            # de energia; discordar nas sete bases é esperado ali.
            return VerificationResult(
                Verdict.INCONCLUSIVE, self.id, 0.0,
                f"termos diferem nas bases SI mas concordam em E^{exc.a.massa}",
            )
        return VerificationResult(
            Verdict.FAIL, self.id, 0.98,
            f"soma incoerente no {onde}: [{exc.a}] + [{exc.b}]",
            counterexample=exc.onde,
        )
