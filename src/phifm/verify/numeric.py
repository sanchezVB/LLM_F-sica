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

from phifm.verify.bus import (
    Claim,
    Motivo,
    Papel,
    Verdict,
    VerificationResult,
    tipo_da_afirmacao,
)
from phifm.verify.symbolic import (
    _com_limite,
    _declarados,
    _ler_os_dois_lados,
    _operadores,
    _Timeout,
    _vocabulario_extra,
)

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
    papel = Papel.EQUIVALENCIA

    def applicable(self, claim: Claim) -> bool:
        """Aplicável a toda equivalência; função só do ITEM.

        Substituição numérica é *estruturalmente incapaz* de verificar
        operadores: atribuir números a ``Â`` e ``B̂`` faz ``ÂB̂ − B̂Â`` valer 0
        sempre, o que é falso em Mecânica Quântica. Defeito encontrado pela
        própria suíte golden em 2026-08-03: sem uma guarda, o numérico aprovava
        e o barramento concluía PASS.

        ⚠️ A guarda agora é uma abstenção com motivo, em `verify`, e não esta
        aplicabilidade: quando ela lia o texto da resposta, um `\\sigma` só na
        resposta fazia o numérico sumir do vetor, e a resposta errada saía sem
        FAIL nenhum (ADR-0005).
        """
        return claim.rhs is not None and tipo_da_afirmacao(claim) == "equivalencia"

    def _abster(self, motivo: Motivo, evidencia: str, **extra) -> VerificationResult:
        return VerificationResult(
            Verdict.INCONCLUSIVE, self.id, 0.0, evidencia, motivo=motivo, **extra
        )

    def verify(self, claim: Claim) -> VerificationResult:
        no_item, na_resposta = _operadores(claim)
        if no_item:
            return self._abster(
                Motivo.OPERADORES,
                "o item tem operadores: substituição numérica não verifica não-comutatividade",
            )
        locais = _declarados(claim)
        a, b, abstencao = _ler_os_dois_lados(claim, locais, no_item, na_resposta)
        if abstencao is not None:
            return self._abster(*abstencao)
        extras = _vocabulario_extra(a, b, claim)

        simbolos = sorted(a.free_symbols | b.free_symbols, key=str)
        if not simbolos:
            return self._constantes(a, b)

        rng = random.Random(17)  # semente fixa: verificação é determinística
        # Só amostra no complexo se a expressão realmente exigir.
        complexo = bool(claim.context.get("complex")) or any(
            e.has(sp.I) for e in (a, b)
        )
        mp.mp.dps = PRECISAO
        pontos = [[_amostra(rng, complexo) for _ in simbolos] for _ in range(N_SUBS)]
        try:
            f_b = sp.lambdify(simbolos, b, "mpmath")
        except Exception as exc:
            return self._abster(Motivo.GABARITO_ILEGIVEL, f"o gabarito não lambdifica: {exc}"[:200])
        try:
            f_a = sp.lambdify(simbolos, a, "mpmath")
        except Exception as exc:
            return self._abster(
                Motivo.EXCECAO_NA_RESPOSTA, f"a resposta não lambdifica: {exc}"[:200]
            )

        # Cada lado sob o seu próprio teto: o gabarito já foi avaliado no
        # autoteste do item, então um estouro só na resposta é da resposta.
        try:
            vb = _com_limite(_avaliar, f_b, pontos)
        except _Timeout:
            return self._abster(Motivo.TEMPO, "a avaliação do gabarito estourou o teto")
        try:
            va = _com_limite(_avaliar, f_a, pontos)
        except _Timeout:
            return self._abster(
                Motivo.TEMPO_NA_RESPOSTA, "a avaliação da resposta estourou o teto"
            )

        ok = disc = 0
        contra: str | None = None
        for vals, x, y in zip(pontos, va, vb, strict=True):
            if x is None or y is None:
                continue
            if _iguais(x, y):
                ok += 1
            else:
                disc += 1
                if contra is None:
                    # `mp.nstr`: `f"{mpf:.3g}"` levanta, e no código antigo a
                    # exceção era engolida e a 1ª discordância contava como
                    # ponto indefinido.
                    atrib = ", ".join(f"{s}={mp.nstr(v, 3)}"
                                      for s, v in zip(simbolos, vals, strict=False))
                    contra = f"{atrib} → {mp.nstr(x, 12)} ≠ {mp.nstr(y, 12)}"

        avaliadas = ok + disc
        if avaliadas < N_SUBS // 2:
            no_gabarito = sum(y is not None for y in vb)
            if no_gabarito < N_SUBS // 2:
                motivo = Motivo.INDECIDIVEL
            elif extras:
                motivo = Motivo.VOCABULARIO
            else:
                # O gabarito avalia e a resposta não: `/0`, `0^{-m}`, função
                # indefinida. É a resposta que impede a verificação.
                motivo = Motivo.RESPOSTA_INDEFINIDA
            return self._abster(
                motivo,
                f"apenas {avaliadas}/{N_SUBS} substituições avaliáveis (gabarito "
                f"avaliável em {no_gabarito}, resposta em {sum(x is not None for x in va)})",
            )

        if disc == 0:
            return VerificationResult(
                Verdict.PASS, self.id, 0.9,
                f"concordância em {ok}/{avaliadas} substituições a {PRECISAO} dígitos",
            )

        if complexo and disc <= MAX_DISCORDANCIA_ISOLADA and ok > disc:
            # Discordância isolada em meio a concordância ampla é assinatura de
            # corte de ramo, não de expressões diferentes.
            return self._abster(
                Motivo.INDECIDIVEL,
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
            vb = _com_limite(_numero, b)
        except _Timeout:
            return self._abster(Motivo.TEMPO, "a avaliação do gabarito estourou o teto")
        if vb is None:
            return self._abster(Motivo.GABARITO_ILEGIVEL, "o gabarito constante não avaliou")
        try:
            va = _com_limite(_numero, a)
        except _Timeout:
            return self._abster(
                Motivo.TEMPO_NA_RESPOSTA, "a avaliação da resposta estourou o teto"
            )
        if va is None:
            return self._abster(Motivo.RESPOSTA_INDEFINIDA, "a resposta constante não avaliou")
        if _iguais(va, vb):
            return VerificationResult(Verdict.PASS, self.id, 0.95, f"iguais: {va}")
        return VerificationResult(
            Verdict.FAIL, self.id, 1.0, "constantes diferentes",
            counterexample=f"{va} ≠ {vb}",
        )


def _finito(v) -> bool:
    try:
        return bool(mp.isfinite(mp.re(v)) and mp.isfinite(mp.im(v)))
    except Exception:
        return False


def _avaliar(f, pontos: list[list]) -> list:
    """O valor em cada ponto, ou None onde não avalia ou não é finito."""
    saida = []
    for vals in pontos:
        try:
            v = f(*vals)
        except Exception:
            saida.append(None)
            continue
        saida.append(v if _finito(v) else None)
    return saida


def _numero(e: sp.Expr):
    """O valor de uma constante a `PRECISAO` dígitos, ou None."""
    try:
        v = sp.N(e, PRECISAO)
        re_, im_ = (mp.mpf(str(p)) for p in v.as_real_imag())
        v = mp.mpc(re_, im_) if im_ != 0 else re_
    except Exception:
        return None
    return v if _finito(v) else None


def _iguais(x, y) -> bool:
    """Igualdade relativa a `TOL`.

    ⚠️ Sem o piso absoluto de 1 que havia aqui: com ele, `0` contra um gabarito
    de 1,87·10⁻⁶⁴ — grandeza comum em SI — passava (|diferença| < 10⁻³⁵), e a
    resposta errada saía do grupo por discordância com o simbólico (ADR-0005).
    O piso só vale quando o gabarito é exatamente zero, onde o erro relativo
    não existe.
    """
    escala = max(abs(x), abs(y)) if y != 0 else max(abs(x), mp.mpf(1))
    if escala == 0:
        return True
    return bool(abs(x - y) / escala < TOL)
