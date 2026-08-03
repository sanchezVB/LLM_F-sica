"""Verificador por substituição numérica aleatória (DOC-10 §3.3).

Complemento **indispensável** do simbólico, não plano B: pelo teorema de
Richardson não existe verificador simbólico completo, então uma fração das
comparações só pode ser decidida numericamente.

Parâmetros e o porquê de cada um (DOC-10 §3.3):

  20 substituições  — uma coincidência numérica em 20 pontos aleatórios
                      independentes é improvável; em 1, não é.
  50 dígitos        — ruído de ponto flutuante de 64 bits geraria falsos
                      negativos em expressões com cancelamento.
  acordo em TODAS   — tolerância frouxa é vetor de reward hacking listado
                      em DOC-09 §5.4.

Cortes de ramo (`sqrt`, `log`, potências fracionárias) são a principal fonte
de falso negativo: duas expressões podem ser equivalentes num ramo e não em
outro. Discordância isolada vira `INCONCLUSIVE`, nunca `FAIL`.
"""

from __future__ import annotations

import random

import mpmath as mp
import sympy as sp

from phifm.verify.bus import Claim, VerificationResult, Verdict
from phifm.verify.symbolic import _tem_operadores, parse

N_SUBS = 20
PRECISAO = 50
# Tolerância relativa. Atingível porque as entradas são `mpf` de 50 dígitos
# (ver `_amostra`); com entradas float64 o piso seria ~1e-16 e este valor
# reprovaria respostas corretas.
TOL = mp.mpf("1e-35")
MAX_DISCORDANCIA_ISOLADA = 2  # acima disso, é diferença real, não corte de ramo


# Domínio de amostragem. Complexo parece mais rigoroso e é ERRADO em Física.
#
# Bug real encontrado em 2026-08-03: as duas formas da energia relativística
#     √((pc)² + (mc²)²)   e   mc²·√(1 + (p/mc)²)
# são idênticas (ambas valem c·√(p² + m²c²)), mas divergiram em 20/20
# substituições complexas. A causa é corte de ramo: √(AB) ≠ √A·√B no plano
# complexo, e quase toda manipulação algébrica de Física fatora raízes.
#
# Grandezas físicas — massa, momento, temperatura, comprimento — são REAIS e
# quase sempre POSITIVAS. Amostrar no complexo foi escolha de matemático,
# não de físico, e produzia falso negativo sistemático.
def _amostra(rng: random.Random, complexo: bool = False):
    """Amostra real positiva por padrão; complexa só sob pedido explícito.

    Devolve **`mp.mpf`/`mp.mpc`, não `float`**. Isso não é detalhe: alimentar
    o mpmath com `float` de 64 bits limita a saída a ~16 dígitos por mais que
    `mp.dps` esteja em 50, e a tolerância de 1e-30 passa a ser inatingível.

    Bug real encontrado em 2026-08-03: as duas formas da energia relativística
    concordavam até o 16º dígito e divergiam no 17º — puro ruído de float64 —
    e o verificador as reprovava. Num laço de RLVR isso puniria
    sistematicamente **toda resposta correta** que exigisse manipulação
    algébrica, ensinando o modelo a não simplificar. Falso negativo silencioso
    e devastador.

    Evita 0, 1 e vizinhanças: muitas expressões distintas coincidem ali, e
    coincidência nesses pontos não é evidência de equivalência.
    """
    if complexo:
        while True:
            re_, im_ = rng.uniform(-3, 3), rng.uniform(-3, 3)
            if abs(re_) > 0.15 and abs(im_) > 0.15 and abs(abs(re_) - 1) > 0.1:
                return mp.mpc(mp.mpf(repr(re_)), mp.mpf(repr(im_)))
    while True:
        x = rng.uniform(0.2, 5.0)
        if abs(x - 1) > 0.1:
            return mp.mpf(repr(x))


class NumericVerifier:
    id = "numeric/mpmath@0.1.0"

    def applicable(self, claim: Claim) -> bool:
        """Não aplicável a álgebra não-comutativa.

        Substituição numérica é *estruturalmente incapaz* de verificar
        operadores: atribuir números a ``Â`` e ``B̂`` faz ``ÂB̂ − B̂Â`` valer 0
        sempre, o que é falso em Mecânica Quântica. O simbólico já se abstém
        nesse caso (DOC-10 §3.1), mas isso não basta — sem esta guarda o
        numérico aprovava e o barramento concluía PASS.

        Defeito encontrado pela própria suíte golden em 2026-08-03. É o
        argumento do DOC-10 §5 em ação: proteger um verificador não protege o
        barramento.
        """
        return claim.rhs is not None and not _tem_operadores(claim)

    def verify(self, claim: Claim) -> VerificationResult:
        a, b = parse(claim.lhs), parse(claim.rhs or "")
        if a is None or b is None:
            return VerificationResult(Verdict.INCONCLUSIVE, self.id, 0.0, "não parseou")

        simbolos = sorted(a.free_symbols | b.free_symbols, key=str)
        if not simbolos:
            return self._constantes(a, b)

        rng = random.Random(17)  # semente fixa: verificação é determinística
        # Só amostra no complexo se a expressão realmente exigir.
        complexo = bool(claim.context.get("complex")) or any(
            e.has(sp.I) for e in (a, b)
        )
        mp.mp.dps = PRECISAO
        f_a, f_b = sp.lambdify(simbolos, a, "mpmath"), sp.lambdify(simbolos, b, "mpmath")

        ok = disc = indef = 0
        contra: str | None = None

        for _ in range(N_SUBS):
            vals = [_amostra(rng, complexo) for _ in simbolos]
            try:
                va, vb = f_a(*vals), f_b(*vals)
            except Exception:
                indef += 1
                continue
            try:
                if not (mp.isfinite(mp.re(va)) and mp.isfinite(mp.re(vb))):
                    indef += 1
                    continue
                escala = max(abs(va), abs(vb), mp.mpf(1))
                if abs(va - vb) / escala < TOL:
                    ok += 1
                else:
                    disc += 1
                    if contra is None:
                        atrib = ", ".join(f"{s}={v:.3g}" for s, v in zip(simbolos, vals))
                        contra = f"{atrib} → {va} ≠ {vb}"
            except Exception:
                indef += 1

        avaliadas = ok + disc
        if avaliadas < N_SUBS // 2:
            return VerificationResult(
                Verdict.INCONCLUSIVE, self.id, 0.0,
                f"apenas {avaliadas}/{N_SUBS} substituições avaliáveis "
                f"({indef} indefinidas — provável singularidade)",
            )

        if disc == 0:
            return VerificationResult(
                Verdict.PASS, self.id, 0.9,
                f"concordância em {ok}/{avaliadas} substituições a {PRECISAO} dígitos",
            )

        if complexo and disc <= MAX_DISCORDANCIA_ISOLADA and ok > disc:
            # Discordância isolada em meio a concordância ampla é assinatura de
            # corte de ramo, não de expressões diferentes.
            return VerificationResult(
                Verdict.INCONCLUSIVE, self.id, 0.0,
                f"{disc} discordância(s) isolada(s) em {avaliadas} — provável corte de ramo",
                counterexample=contra,
            )

        return VerificationResult(
            Verdict.FAIL, self.id, 0.95,
            f"divergiram em {disc}/{avaliadas} substituições",
            counterexample=contra,
        )

    def _constantes(self, a: sp.Expr, b: sp.Expr) -> VerificationResult:
        mp.mp.dps = PRECISAO
        try:
            va, vb = mp.mpf(str(sp.N(a, PRECISAO))), mp.mpf(str(sp.N(b, PRECISAO)))
        except Exception:
            try:
                va, vb = complex(sp.N(a, PRECISAO)), complex(sp.N(b, PRECISAO))
            except Exception:
                return VerificationResult(Verdict.INCONCLUSIVE, self.id, 0.0, "não avaliou")
        escala = max(abs(va), abs(vb), 1)
        if abs(va - vb) / escala < float(TOL):
            return VerificationResult(Verdict.PASS, self.id, 0.95, f"iguais: {va}")
        return VerificationResult(
            Verdict.FAIL, self.id, 1.0, "constantes diferentes",
            counterexample=f"{va} ≠ {vb}",
        )
