"""Verificador de invariantes (DOC-10 §3.5).

Energia, momento linear e angular, carga, número bariônico e leptônico,
unitariedade. Duas famílias de checagem, com naturezas bem diferentes:

**Contínua** — para uma solução alegada, o invariante não varia:
``d/dt Q[solução] = 0``. Exige substituir a solução proposta na expressão do
invariante, e é aqui que se pega a solução que satisfaz a equação de movimento
por acaso num instante e viola conservação em geral.

**Discreta** — numa reação, os números quânticos somam igual dos dois lados.
Não há cálculo: é aritmética sobre uma tabela. E é justamente por ser barata
que vale a pena — ``n → p + e⁻`` (sem o antineutrino) viola número leptônico e
é reprovada em microssegundos, sem nenhum modelo de física nuclear.

**A tabela de partículas é incompleta por escolha.** Partícula ausente devolve
`INCONCLUSIVE`, nunca `PASS`: uma tabela que chutasse números quânticos de
hádrons exóticos daria FAIL espúrio em Física correta, e o custo de um falso
negativo em RLVR é o do DOC-10 §2.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from phifm.verify.bus import Claim, VerificationResult, Verdict
from phifm.verify.symbolic import _Timeout, _com_limite, parse

TOL = sp.Rational(1, 10**12)
AMOSTRAS = (sp.Rational(1, 3), sp.Rational(7, 5), sp.Rational(11, 4),
            sp.Rational(17, 3), sp.Rational(23, 2))


@dataclass(frozen=True)
class Numeros:
    """Números quânticos aditivos. `Q` em unidades de e."""

    Q: sp.Rational = sp.Integer(0)   # carga
    B: sp.Rational = sp.Integer(0)   # número bariônico
    Le: int = 0                      # leptônico eletrônico
    Lmu: int = 0
    Ltau: int = 0

    def __add__(self, o: Numeros) -> Numeros:
        return Numeros(self.Q + o.Q, self.B + o.B,
                       self.Le + o.Le, self.Lmu + o.Lmu, self.Ltau + o.Ltau)

    def __neg__(self) -> Numeros:
        return Numeros(-self.Q, -self.B, -self.Le, -self.Lmu, -self.Ltau)

    def itens(self):
        return (("carga", self.Q), ("número bariônico", self.B),
                ("número leptônico e", self.Le), ("número leptônico μ", self.Lmu),
                ("número leptônico τ", self.Ltau))


_t = sp.Rational(1, 3)  # carga de quark, para os hádrons abaixo

PARTICULAS: dict[str, Numeros] = {
    # bósons
    "gamma": Numeros(),
    "g": Numeros(),
    "Z0": Numeros(),
    "W+": Numeros(Q=1), "W-": Numeros(Q=-1),
    "H": Numeros(),
    # léptons carregados
    "e-": Numeros(Q=-1, Le=1),   "e+": Numeros(Q=1, Le=-1),
    "mu-": Numeros(Q=-1, Lmu=1), "mu+": Numeros(Q=1, Lmu=-1),
    "tau-": Numeros(Q=-1, Ltau=1), "tau+": Numeros(Q=1, Ltau=-1),
    # neutrinos
    "nu_e": Numeros(Le=1),     "nu_e_bar": Numeros(Le=-1),
    "nu_mu": Numeros(Lmu=1),   "nu_mu_bar": Numeros(Lmu=-1),
    "nu_tau": Numeros(Ltau=1), "nu_tau_bar": Numeros(Ltau=-1),
    # bárions
    "p": Numeros(Q=1, B=1),   "p_bar": Numeros(Q=-1, B=-1),
    "n": Numeros(B=1),        "n_bar": Numeros(B=-1),
    "Lambda0": Numeros(B=1),
    "Sigma+": Numeros(Q=1, B=1), "Sigma0": Numeros(B=1), "Sigma-": Numeros(Q=-1, B=1),
    "Xi0": Numeros(B=1), "Xi-": Numeros(Q=-1, B=1),
    "Omega-": Numeros(Q=-1, B=1),
    # mésons
    "pi+": Numeros(Q=1), "pi-": Numeros(Q=-1), "pi0": Numeros(),
    "K+": Numeros(Q=1), "K-": Numeros(Q=-1), "K0": Numeros(), "K0_bar": Numeros(),
    "eta": Numeros(), "rho+": Numeros(Q=1), "rho-": Numeros(Q=-1), "rho0": Numeros(),
}

# Núcleos escritos como `Z:A` — carga Z, número bariônico A. Cobre decaimento
# nuclear sem precisar tabelar isótopo por isótopo.
def _nucleo(nome: str) -> Numeros | None:
    if ":" not in nome:
        return None
    z, _, a = nome.partition(":")
    try:
        return Numeros(Q=sp.Integer(int(z)), B=sp.Integer(int(a)))
    except ValueError:
        return None


def _somar(lista: list[str]) -> tuple[Numeros, list[str]]:
    """Soma os números quânticos. Devolve também os nomes não reconhecidos."""
    total, faltando = Numeros(), []
    for nome in lista:
        n = nome.strip()
        mult = 1
        # Multiplicador só antes de NOME de partícula. A checagem de ":" vem
        # primeiro porque `6:14` (carbono-14) começa com dígito e seria lido
        # como "6 × :14" — núcleo nenhum, INCONCLUSIVE espúrio.
        if ":" not in n and len(n) > 1 and n[0].isdigit() and n[0] != "0" and not n[1].isdigit():
            mult, n = int(n[0]), n[1:]        # "2gamma" → 2 × gamma
        p = PARTICULAS.get(n) or _nucleo(n)
        if p is None:
            faltando.append(n)
            continue
        for _ in range(mult):
            total = total + p
    return total, faltando


class ConservationVerifier:
    id = "conservation/sympy@0.1.0"

    def applicable(self, claim: Claim) -> bool:
        return bool(claim.context.get("conserved") or claim.context.get("reaction"))

    def verify(self, claim: Claim) -> VerificationResult:
        if claim.context.get("reaction"):
            return self._reacao(claim.context["reaction"])
        return self._continua(claim, claim.context["conserved"])

    # ── discreta ──────────────────────────────────────────────────────────
    def _reacao(self, spec: dict) -> VerificationResult:
        antes, depois = spec.get("before") or [], spec.get("after") or []
        if not antes or not depois:
            return VerificationResult(
                Verdict.INCONCLUSIVE, self.id, 0.0, "reação sem 'before'/'after'"
            )

        na, fa = _somar(list(antes))
        nd, fd = _somar(list(depois))
        if fa or fd:
            return VerificationResult(
                Verdict.INCONCLUSIVE, self.id, 0.0,
                "partícula fora da tabela: " + ", ".join(sorted(set(fa + fd)))
                + " — tabela incompleta por escolha, ver docstring",
            )

        violadas = [(nome, va, vd) for (nome, va), (_, vd) in zip(na.itens(), nd.itens())
                    if va != vd]
        seta = f"{' + '.join(antes)} → {' + '.join(depois)}"
        if violadas:
            det = "; ".join(f"{n}: {va} → {vd}" for n, va, vd in violadas)
            return VerificationResult(
                Verdict.FAIL, self.id, 1.0,
                f"invariante(s) violado(s) em {seta} — {det}",
                counterexample=det,
            )
        return VerificationResult(
            Verdict.PASS, self.id, 1.0,
            f"{seta}: carga, número bariônico e os três leptônicos conservados",
        )

    # ── contínua ──────────────────────────────────────────────────────────
    def _continua(self, claim: Claim, spec: dict) -> VerificationResult:
        grandeza = spec.get("quantity", "invariante")
        var = sp.Symbol(spec.get("var", "t"))
        fonte = spec.get("expr") or claim.lhs
        expr = parse(str(fonte))
        if expr is None:
            return VerificationResult(Verdict.INCONCLUSIVE, self.id, 0.0, "não parseou")

        if grandeza in ("unitaridade", "unitarity", "norma"):
            return self._unitaridade(expr, spec)

        # Substituir a solução alegada. Sem isto o invariante é uma expressão
        # genérica e sua derivada não é nula por bom motivo nenhum.
        subs = {}
        for k, v in (spec.get("solution") or {}).items():
            alvo = parse(str(v))
            if alvo is None:
                return VerificationResult(
                    Verdict.INCONCLUSIVE, self.id, 0.0, f"solução de {k} não parseou"
                )
            subs[sp.Symbol(k)] = alvo
        if subs:
            expr = expr.subs(subs)

        if not expr.has(var):
            return VerificationResult(
                Verdict.PASS, self.id, 0.85,
                f"{grandeza} não depende de {var} — conservado trivialmente",
            )

        try:
            d = _com_limite(lambda: sp.simplify(sp.diff(expr, var)))
        except (_Timeout, Exception):
            d = None

        if d is not None and d == 0:
            return VerificationResult(
                Verdict.PASS, self.id, 0.95, f"d({grandeza})/d{var} = 0 simbolicamente"
            )

        # Cruzamento numérico: DOC-10 §3.5 prevê integração/avaliação numérica
        # quando o simbólico não fecha. Símbolos livres recebem racionais
        # primos entre si — valores redondos escondem cancelamento acidental.
        livres = sorted(expr.free_symbols - {var}, key=str)
        fixos = {s: sp.Rational(p, 7) for s, p in zip(livres, (3, 5, 11, 13, 17, 19, 23))}
        try:
            vals = [sp.N(expr.subs({var: a, **fixos}), 30) for a in AMOSTRAS]
        except Exception:
            vals = []
        if vals and all(v.is_number for v in vals):
            escala = max([abs(v) for v in vals] + [sp.Float(1)])
            desvio = max(abs(v - vals[0]) for v in vals) / escala
            if desvio < TOL:
                return VerificationResult(
                    Verdict.PASS, self.id, 0.75,
                    f"{grandeza} constante em {len(AMOSTRAS)} valores de {var} "
                    f"(desvio relativo {float(desvio):.1e}; o simbólico não fechou)",
                )
            return VerificationResult(
                Verdict.FAIL, self.id, 0.9,
                f"{grandeza} varia com {var}: desvio relativo {float(desvio):.2e}",
                counterexample=f"{var}={AMOSTRAS[0]} → {vals[0]} | "
                               f"{var}={AMOSTRAS[-1]} → {vals[-1]}",
            )

        if d is not None:
            return VerificationResult(
                Verdict.FAIL, self.id, 0.85,
                f"d({grandeza})/d{var} ≠ 0", counterexample=str(d),
            )
        return VerificationResult(
            Verdict.INCONCLUSIVE, self.id, 0.0,
            f"não foi possível decidir a conservação de {grandeza}",
        )

    def _unitaridade(self, psi: sp.Expr, spec: dict) -> VerificationResult:
        """‖ψ‖² = 1 sobre o domínio. Norma > 1 ou < 1 é estado não normalizado,
        e um modelo que devolve isso não resolveu o problema — só a forma."""
        var = sp.Symbol(spec.get("integrate", spec.get("var", "x")))
        dom = spec.get("domain", (-sp.oo, sp.oo))
        a, b = (_p(dom[0]), _p(dom[1]))
        try:
            norma = _com_limite(
                lambda: sp.simplify(sp.integrate(sp.Abs(psi) ** 2, (var, a, b)))
            )
        except (_Timeout, Exception):
            return VerificationResult(
                Verdict.INCONCLUSIVE, self.id, 0.0,
                f"integral de |ψ|² em {var} ∈ [{a}, {b}] não convergiu no tempo",
            )
        if norma.free_symbols:
            return VerificationResult(
                Verdict.INCONCLUSIVE, self.id, 0.0,
                f"norma depende de {', '.join(map(str, sorted(norma.free_symbols, key=str)))}"
                " — declarar os parâmetros para decidir",
            )
        if abs(sp.N(norma - 1)) < float(TOL):
            return VerificationResult(Verdict.PASS, self.id, 0.95, "‖ψ‖² = 1")
        return VerificationResult(
            Verdict.FAIL, self.id, 0.95, f"‖ψ‖² = {norma} ≠ 1",
            counterexample=f"norma {sp.N(norma, 12)}",
        )


def _p(v):
    if isinstance(v, str):
        return {"inf": sp.oo, "oo": sp.oo, "-inf": -sp.oo, "-oo": -sp.oo}.get(v, sp.sympify(v))
    return sp.sympify(v)
