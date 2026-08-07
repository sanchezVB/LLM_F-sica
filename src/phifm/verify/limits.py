"""Verificador de redução em casos-limite (DOC-10 §3.4).

Ataca **F3**: a teoria alegada precisa degenerar na teoria conhecida no regime
onde esta vale. É a checagem que pega o modelo que produz uma fórmula
relativística plausível e errada — se ela não vira ½mv² quando v ≪ c, está
errada, e nenhuma comparação com gabarito é necessária para saber disso.

**Dois modos, e confundi-los é o defeito silencioso deste verificador.**

  ``limite``    — ``lim(expr) = esperado``. É o que se quer para T → ∞,
                  r → ∞, ℏ → 0.
  ``dominante`` — primeiro termo não nulo da série. É o que se quer para
                  v/c → 0: o limite estrito da energia cinética relativística
                  quando v → 0 é **zero**, não ½mv². Testar `limite` ali
                  reprovaria a Física correta.

O modo ``auto`` tenta `limite` e, se não fechar, `dominante` — sempre
declarando no `evidence` qual dos dois decidiu, porque "passou" e "passou como
termo dominante" são afirmações diferentes.

Quando o limite simbólico não fecha, há verificação numérica cruzada sobre uma
sequência convergente (DOC-10 §3.4).
"""

from __future__ import annotations

import sympy as sp

from phifm.verify.bus import Claim, VerificationResult, Verdict
from phifm.verify.symbolic import _Timeout, _com_limite, parse

# Sequência do cruzamento numérico. Não desce além de 1e-6 porque abaixo disso
# o próprio cancelamento em ponto flutuante domina o resíduo que se mede.
SEQUENCIA = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6)
TOL_REL = 1e-6
ORDEM_SERIE = 6


# ── Biblioteca de reduções canônicas (DOC-10 §3.4) ────────────────────────
#
# Cada entrada fixa a variável, o ponto e o modo. `var` é sobreponível pelo
# contexto: o acoplamento se chama `g` numa teoria e `alpha` noutra.
REDUCOES: dict[str, dict] = {
    "v/c->0":       {"var": "v",    "to": 0,     "mode": "dominante",
                     "de": "cinemática relativística", "para": "newtoniana"},
    "hbar->0":      {"var": "hbar", "to": 0,     "mode": "auto",
                     "de": "mecânica quântica", "para": "clássica (Ehrenfest, WKB)"},
    "campo-fraco":  {"var": "v",    "to": 0,     "mode": "dominante",
                     "de": "relatividade geral", "para": "gravitação newtoniana"},
    "T->inf":       {"var": "T",    "to": sp.oo, "mode": "limite",
                     "de": "Fermi-Dirac / Bose-Einstein", "para": "Maxwell-Boltzmann"},
    "T->0":         {"var": "T",    "to": 0,     "mode": "limite",
                     "de": "estatística quântica", "para": "estado fundamental"},
    "r->inf":       {"var": "r",    "to": sp.oo, "mode": "limite",
                     "de": "solução de Schwarzschild", "para": "espaço-tempo plano"},
    "N->inf":       {"var": "N",    "to": sp.oo, "mode": "limite",
                     "de": "física estatística", "para": "limite termodinâmico"},
    "acoplamento->0": {"var": "g",  "to": 0,     "mode": "auto",
                       "de": "teoria interagente", "para": "teoria livre"},
}


def _ponto(v):
    if isinstance(v, str):
        return {"inf": sp.oo, "oo": sp.oo, "+inf": sp.oo, "-inf": -sp.oo}.get(v, sp.sympify(v))
    return sp.sympify(v)


def _expoente(t: sp.Expr, var: sp.Symbol):
    """Potência de `var` num monômio, aceitando expoente negativo.

    `sp.degree` sozinho não serve: `degree(1/T, T)` não é um grau de polinômio.
    A razão numerador/denominador dá −1, que é o que se quer.
    """
    if not t.has(var):
        return sp.Integer(0)
    p, q = sp.together(t).as_numer_denom()
    try:
        return sp.degree(p, var) - sp.degree(q, var)
    except Exception:
        return None


def _termo_dominante(expr: sp.Expr, var: sp.Symbol, to) -> sp.Expr | None:
    """Termo que domina a expansão em torno de `to` — a teoria-limite.

    **O lado que domina depende do ponto, e trocá-los é erro silencioso.**
    Perto de um ponto finito, quem domina é a MENOR potência do desvio: em
    v → 0, ½mv² supera v⁴. No infinito é a MAIOR: em T → ∞, a expansão de
    Bose-Einstein dá ``kT/ε − 1/2 + ε/(12kT) + …`` e quem domina é ``kT/ε``,
    não o ``−1/2``, que é o de menor grau.

    Perto de ponto finito a expansão se faz na variável deslocada
    ``u = var − to``, senão o grau medido em `var` não é o grau do desvio.
    """
    infinito = to in (sp.oo, -sp.oo)

    if infinito:
        try:
            s = _com_limite(lambda: sp.series(expr, var, to, ORDEM_SERIE).removeO())
        except (_Timeout, Exception):
            return None
        alvo, volta = var, None
    else:
        u = sp.Symbol("_u")
        try:
            s = _com_limite(
                lambda: sp.series(expr.subs(var, to + u), u, 0, ORDEM_SERIE).removeO()
            )
        except (_Timeout, Exception):
            return None
        alvo, volta = u, var - to

    s = sp.expand(s)
    termos = [t for t in (s.args if isinstance(s, sp.Add) else (s,)) if t != 0]
    graus = [(t, _expoente(t, alvo)) for t in termos]
    graus = [(t, g) for t, g in graus if g is not None]
    if not graus:
        return None

    dominante = (max if infinito else min)(graus, key=lambda tg: tg[1])[0]
    return dominante.subs(alvo, volta) if volta is not None else dominante


def _iguais(a: sp.Expr, b: sp.Expr) -> bool:
    try:
        return bool(_com_limite(lambda: sp.simplify(a - b) == 0))
    except (_Timeout, Exception):
        return False


def _cruzamento_numerico(expr: sp.Expr, esperado: sp.Expr, var: sp.Symbol, to) -> bool:
    """Avalia numa sequência convergente. Livres restantes recebem valores
    fixos e primos entre si — coincidência em valores redondos é comum."""
    livres = sorted((expr.free_symbols | esperado.free_symbols) - {var}, key=str)
    fixos = {s: sp.Rational(p, 7) for s, p in zip(livres, (3, 5, 11, 13, 17, 19, 23))}
    for eps in SEQUENCIA:
        ponto = sp.Rational(1) / sp.Rational(str(eps)) if to is sp.oo else _ponto(to) + sp.Rational(str(eps))
        try:
            va = sp.N(expr.subs({var: ponto, **fixos}), 30)
            vb = sp.N(esperado.subs({var: ponto, **fixos}), 30)
        except Exception:
            return False
        if not (va.is_number and vb.is_number):
            return False
        escala = max(abs(va), abs(vb), sp.Float(1))
        if abs(va - vb) / escala > TOL_REL:
            return False
    return True


class LimitsVerifier:
    id = "limits/sympy@0.1.0"

    def applicable(self, claim: Claim) -> bool:
        spec = claim.context.get("limit")
        return isinstance(spec, dict) and bool(spec.get("expected") or claim.rhs)

    def verify(self, claim: Claim) -> VerificationResult:
        spec = dict(claim.context["limit"])

        nome = spec.get("name")
        if nome:
            if nome not in REDUCOES:
                return VerificationResult(
                    Verdict.INCONCLUSIVE, self.id, 0.0,
                    f"redução desconhecida: {nome} — conhecidas: {', '.join(REDUCOES)}",
                )
                # (a lista fecha o loop para quem chamou com erro de digitação)
            spec = {**REDUCOES[nome], **spec}

        if "var" not in spec:
            return VerificationResult(
                Verdict.INCONCLUSIVE, self.id, 0.0, "limite sem variável declarada"
            )

        modo = spec.get("mode", "auto")
        to = _ponto(spec.get("to", 0))
        var = sp.Symbol(spec["var"])

        expr = parse(claim.lhs)
        esperado = parse(str(spec.get("expected") or claim.rhs or ""))
        if expr is None or esperado is None:
            return VerificationResult(Verdict.INCONCLUSIVE, self.id, 0.0, "não parseou")

        if not expr.has(var):
            # Sem a variável, o "limite" é a própria expressão. Passar aqui
            # seria aprovar uma redução que nunca foi testada.
            return VerificationResult(
                Verdict.INCONCLUSIVE, self.id, 0.0,
                f"a expressão não contém {var} — nada a fazer tender a {to}",
            )

        rotulo = f"{var}→{to}"
        if nome:
            r = REDUCOES[nome]
            rotulo = f"{nome} ({r['de']} → {r['para']})"

        # ── modo `limite` ──────────────────────────────────────────────────
        lim = None
        if modo in ("limite", "auto"):
            try:
                lim = _com_limite(lambda: sp.limit(expr, var, to))
            except (_Timeout, Exception):
                lim = None
            if lim is not None and _iguais(lim, esperado):
                return VerificationResult(
                    Verdict.PASS, self.id, 0.95, f"{rotulo}: limite simbólico = {esperado}"
                )

        # ── modo `dominante` ──────────────────────────────────────────────
        if modo in ("dominante", "auto"):
            dom = _termo_dominante(expr, var, to)
            if dom is not None and _iguais(dom, esperado):
                extra = f" (o limite estrito é {lim})" if modo == "auto" and lim is not None else ""
                return VerificationResult(
                    Verdict.PASS, self.id, 0.9,
                    f"{rotulo}: termo dominante da série = {esperado}{extra}",
                )

        # ── cruzamento numérico ───────────────────────────────────────────
        if _cruzamento_numerico(expr, esperado, var, to):
            return VerificationResult(
                Verdict.PASS, self.id, 0.7,
                f"{rotulo}: concordância numérica ao longo da sequência convergente "
                "(o simbólico não fechou — confiança reduzida)",
            )

        if lim is None and modo == "limite":
            return VerificationResult(
                Verdict.INCONCLUSIVE, self.id, 0.0,
                f"{rotulo}: sympy.limit não concluiu e o cruzamento numérico não confirmou",
            )

        obtido = lim if lim is not None else _termo_dominante(expr, var, to)
        return VerificationResult(
            Verdict.FAIL, self.id, 0.9,
            f"{rotulo}: não reduz ao esperado",
            counterexample=f"obtido {obtido}, esperado {esperado}",
        )
