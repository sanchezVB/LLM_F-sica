"""A recompensa do RLVR: o R1 do DOC-09 §5.1 (ADR-0005 §4.6).

Lê o VETOR de `VerifierBus.run` e o motivo de cada abstenção, não o veredito
combinado. Três valores:

- **+1** — um verificador de equivalência deu PASS e nenhum deu FAIL decisivo;
- **−1** — qualquer FAIL decisivo, **sem multiplicar pela confiança**, e toda
  abstenção causada pela resposta (ilegível, incomparável, indefinida,
  vocabulário novo, operador só na resposta, exceção ou estouro na resposta);
- **None** — tirar a amostra do grupo: abstenção causada pelo item ou pelo
  problema, discordância e contradição.

**Por que não 0.** O GRPO normaliza a vantagem dentro do grupo (DOC-09 §5.3):
num grupo de sete FAIL, uma amostra que vale 0 recebe a mesma vantagem de uma
resposta certa, +√7. Qualquer valor acima de −1 dá esse +√7, então uma
penalidade intermediária não fecha nada — só o sinal conta. O neutro de
verdade, o do DOC-10 §2.1, é a amostra sair do grupo: `None` mascara a amostra
inteira (perda, média, desvio e KL), e um grupo com menos de duas amostras
pontuadas sai do passo. Isso é contrato com quem implementar o laço do GRPO.

**Por que sem a confiança.** Com −0,95 para um FAIL e −1 para outro, sete
erradas a −0,95 ganhariam vantagem positiva cheia contra uma a −1: ruído de
0,05 viraria gradiente.

**Soma ponderada R1…R7.** Quem a montar tem de dar à amostra cujo R1 é −1 por
motivo da resposta o PIOR valor em cada componente aplicável; senão uma errada
legível que falha R2, R3 e R5 fica abaixo da ilegível, e o atalho volta.
"""

from __future__ import annotations

from dataclasses import dataclass

from phifm.verify.bus import (
    MOTIVOS_DA_RESPOSTA,
    MOTIVOS_DO_ITEM,
    Claim,
    Motivo,
    Papel,
    Regra,
    Verdict,
    VerificationResult,
    VerifierBus,
    classificar,
    falha_decisiva,
)
from phifm.verify.resposta import ler_gabarito, ler_resposta

__all__ = ["Recompensa", "admitir", "avaliar", "recompensa"]


@dataclass(frozen=True)
class Recompensa:
    """`valor` None é a amostra fora do grupo. `alerta` pede investigação."""

    valor: float | None
    motivo: str
    alerta: bool = False


def recompensa(resultados: list[VerificationResult], high_confidence: float = 0.9) -> Recompensa:
    """O R1 de uma amostra, pela tabela ordenada do ADR-0005 §4.6."""
    regra = classificar(resultados, high_confidence)
    if regra is Regra.VAZIO:
        return Recompensa(None, "nenhum verificador aplicável", alerta=True)
    if regra is Regra.DISCORDANCIA:
        return Recompensa(None, Motivo.DISCORDANCIA, alerta=True)
    if regra is Regra.CONTRADICAO:
        return Recompensa(None, Motivo.CONTRADICAO, alerta=True)
    if regra is Regra.FAIL:
        return Recompensa(-1.0, "fail")
    if regra is Regra.PASS:
        return Recompensa(1.0, "pass")
    if regra is Regra.PASS_DE_RELACAO:
        # No RLVR toda afirmação é de equivalência; relação aqui é engano de
        # quem montou o Claim.
        return Recompensa(None, "afirmação de relação não é recompensa", alerta=True)

    abstencoes = [r for r in resultados
                  if r.verdict is not Verdict.PASS and not falha_decisiva(r, high_confidence)]
    # O item primeiro: um gabarito que não lê não diz nada da resposta. Só os
    # motivos da equivalência — o dimensional sem sistema declarado dá
    # `sem_convencao` em todo item, e deixá-lo vencer reabria o atalho.
    do_item = {r.motivo for r in abstencoes if r.papel is Papel.EQUIVALENCIA} & MOTIVOS_DO_ITEM
    if do_item:
        return Recompensa(None, min(do_item), alerta=True)
    da_resposta = {r.motivo for r in abstencoes} & MOTIVOS_DA_RESPOSTA
    if da_resposta:
        return Recompensa(-1.0, min(da_resposta))
    if any(r.motivo in (Motivo.SEM_MOTIVO, Motivo.EXCECAO) for r in abstencoes):
        return Recompensa(None, "abstenção sem motivo ou verificador quebrado", alerta=True)
    motivos = sorted({r.motivo for r in abstencoes if r.motivo is not None})
    return Recompensa(None, motivos[0] if motivos else Motivo.INDECIDIVEL)


def _barramento(barramento: VerifierBus | None) -> VerifierBus:
    if barramento is not None:
        return barramento
    from phifm.verify import barramento_padrao  # aqui, para não fechar ciclo na importação

    return barramento_padrao()


def avaliar(texto: str, gabarito: str, contexto: dict | None = None,
            barramento: VerifierBus | None = None) -> Recompensa:
    """A recompensa de uma geração inteira: contrato de resposta e barramento.

    Sem `\\boxed{}` é violação do formato que o prompt pede, e vale −1.
    """
    resposta = ler_resposta(texto)
    if resposta is None:
        return Recompensa(-1.0, Motivo.RESPOSTA_ILEGIVEL)
    bus = _barramento(barramento)
    claim = Claim(lhs=resposta, rhs=ler_gabarito(gabarito),
                  context={**(contexto or {}), "tipo": "equivalencia"})
    return recompensa(bus.run(claim), bus.high_confidence)


def admitir(gabarito: str, contexto: dict | None = None,
            barramento: VerifierBus | None = None) -> tuple[bool, str]:
    """O item entra no RLVR? O gabarito tem de passar contra si mesmo.

    ⚠️ Não basta o PASS estrutural do simbólico, que o gabarito sempre tem
    contra si mesmo: o numérico precisa ser aplicável e dar PASS. Sem isso
    entravam itens em que o numérico nunca avalia, e neles toda resposta errada
    que deixasse o simbólico indeciso saía do grupo (ADR-0005).
    """
    bus = _barramento(barramento)
    g = ler_gabarito(gabarito)
    resultados = bus.run(Claim(lhs=g, rhs=g, context={**(contexto or {}), "tipo": "equivalencia"}))
    r = recompensa(resultados, bus.high_confidence)
    if r.valor != 1.0:
        return False, f"o gabarito contra si mesmo deu {r.valor} ({r.motivo})"
    numerico = [x for x in resultados if x.papel is Papel.EQUIVALENCIA and x.verifier_id.startswith("numeric")]
    if not numerico or numerico[0].verdict is not Verdict.PASS:
        return False, "o numérico não aprovou o gabarito contra si mesmo"
    if any(x.verdict is Verdict.ERROR for x in resultados):
        return False, "um verificador quebrou com o gabarito"
    return True, "admitido"
