"""Interação com o sistema operacional durante coletas longas.

O `run_harvest.sh` registra um achado de 2026-08-03 que custou caro: sem
`caffeinate -i`, o macOS suspendeu o processo e **60% do tempo de relógio virou
pausa** — 1,63 h perdidas em 2,72 h. O mesmo vale no Windows, onde o padrão de
economia de energia suspende o sistema por inatividade de *usuário*, e um
processo que só fala com a rede não conta como atividade.

O pedido é feito pelo próprio processo de coleta, não pelo lançador. Assim ele
vale para qualquer forma de invocação — script, terminal, agente — e é
liberado automaticamente quando o processo morre, sem deixar a máquina
permanentemente impedida de dormir.
"""

from __future__ import annotations

import ctypes
import logging
import sys

log = logging.getLogger(__name__)

# winbase.h
_ES_CONTINUOUS = 0x80000000       # o estado persiste até ser revogado
_ES_SYSTEM_REQUIRED = 0x00000001  # não suspender o sistema
_ES_AWAYMODE_REQUIRED = 0x00000040


def impedir_suspensao() -> bool:
    """Pede ao SO para não suspender enquanto este processo viver.

    **Não impede o desligamento da tela** — de propósito. O que se quer é a
    CPU e a rede vivas, não o monitor aceso por cinco horas.

    Devolve `True` se o pedido foi aceito. Fora do Windows devolve `False` sem
    reclamar: em Linux a suspensão por inatividade não afeta um processo de
    rede em servidor, e no macOS o lançador já usa `caffeinate`.
    """
    if not sys.platform.startswith("win"):
        return False
    try:
        # ES_AWAYMODE_REQUIRED cobre o caso de a política da máquina mandar
        # suspender de todo jeito: o sistema entra em away mode, que mantém o
        # processo rodando com a tela apagada.
        estado = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_AWAYMODE_REQUIRED
        if ctypes.windll.kernel32.SetThreadExecutionState(estado) == 0:
            # Away mode pode ser negado por política; o essencial ainda vale.
            estado = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
            if ctypes.windll.kernel32.SetThreadExecutionState(estado) == 0:
                log.warning("SetThreadExecutionState recusado — a máquina pode suspender")
                return False
        log.info("suspensão do sistema impedida enquanto a coleta durar")
        return True
    except Exception as exc:  # ctypes indisponível, ou build sem kernel32
        log.warning("não foi possível impedir a suspensão: %s", exc)
        return False


def liberar_suspensao() -> None:
    """Revoga o pedido. O SO já faz isso ao fim do processo; existe para quem
    queira soltar antes, e para deixar a simetria visível no código."""
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        except Exception:
            pass
