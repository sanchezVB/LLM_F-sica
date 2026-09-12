#!/usr/bin/env python3
"""Espera uma vaga de GPU no Kaggle e empurra o experimento. Fila de um lugar.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/enfileirar_kaggle.py \\
        --experimento t2a_e --so-notebook \\
        --esperar viaciclo/phifm-t1f-base-gte \\
        --esperar viaciclo/phifm-t2a-tokenizer-a

## Por que isto existe

O Kaggle aceita **no máximo 2 sessões de GPU simultâneas**, e recusa a terceira
com `Maximum batch GPU session count of 2 reached` — **saindo com código 0**.
Medido em 2026-09-11, com o T1f e o braço A do T2a rodando.

Sem uma fila, o terceiro experimento depende de alguém olhar o relógio, perceber
que um kernel terminou, e empurrar o próximo à mão. Numa cota semanal de 30 h, o
tempo entre "vagou" e "alguém viu" é cota parada.

## ⚠️ O que este script NÃO faz

Não decide nada. Ele empurra **o experimento que já foi mandado empurrar**, com
os mesmos argumentos do `publicar_kaggle.py`, quando houver vaga. Se o push
falhar por qualquer outro motivo, ele PARA e diz — não tenta de novo às cegas,
porque um push que falha por conteúdo falharia igual daqui a uma hora.

## ⚠️ E ele confere a vaga pela CAUSA, não pelo relógio

Esperar "N horas" seria adivinhar. Ele consulta o status dos kernels nomeados em
`--esperar` e só empurra quando nenhum deles está `RUNNING`. Um kernel que morre
aos 2 min — como o T1f morreu na primeira tentativa — libera a vaga na hora, e a
fila anda sem ninguém perceber que houve um erro.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8 as console_utf8  # noqa: E402
from phifm.core.kaggle import EXPERIMENTOS  # noqa: E402

console_utf8()

log = logging.getLogger("fila")

# Estados que ocupam uma vaga de GPU. Qualquer outro — COMPLETE, ERROR,
# CANCEL_ACKNOWLEDGED — significa que a sessão acabou.
OCUPADOS = ("RUNNING", "QUEUED")


def ocupado(kernel: str) -> bool | None:
    """`True` se o kernel ocupa vaga, `False` se não, `None` se não deu para ver.

    ⚠️ `None` e `False` são coisas diferentes. Uma consulta que falhou (rede, API
    fora) NÃO é uma vaga livre: empurrar nesse estado gastaria a tentativa e
    receberia a recusa de sessão que este script existe para evitar.
    """
    try:
        r = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "status", kernel],
            capture_output=True, text=True, timeout=120)
    except Exception as exc:  # noqa: BLE001
        log.warning("status de %s indisponível: %s", kernel, exc)
        return None
    saida = (r.stdout or "") + (r.stderr or "")
    if "status" not in saida:
        log.warning("resposta inesperada para %s: %s", kernel, saida.strip()[:120])
        return None
    return any(e in saida for e in OCUPADOS)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--experimento", required=True, choices=sorted(EXPERIMENTOS))
    p.add_argument("--esperar", action="append", default=[], required=True,
                   metavar="dono/slug",
                   help="kernel que precisa terminar antes; repetível")
    p.add_argument("--so-notebook", action="store_true",
                   help="repassado ao publicar_kaggle.py")
    p.add_argument("--intervalo", type=int, default=300,
                   help="segundos entre consultas (padrão 300)")
    p.add_argument("--limite-horas", type=float, default=12.0,
                   help="desiste depois disto, em vez de esperar para sempre")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    log.info("fila: %s espera %s", a.experimento, ", ".join(a.esperar))
    limite = time.monotonic() + a.limite_horas * 3600
    while True:
        estados = {k: ocupado(k) for k in a.esperar}
        if all(v is False for v in estados.values()):
            log.info("vaga livre — empurrando %s", a.experimento)
            break
        if time.monotonic() > limite:
            raise SystemExit(
                f"passaram {a.limite_horas} h e ainda há kernel ocupando vaga: "
                f"{ {k: v for k, v in estados.items() if v is not False} }.\n"
                "Não vou esperar indefinidamente — empurre à mão quando quiser.")
        ainda = [k for k, v in estados.items() if v is not False]
        log.info("ocupado: %s · nova consulta em %d s", ", ".join(ainda),
                 a.intervalo)
        time.sleep(a.intervalo)

    cmd = [sys.executable, str(RAIZ / "scripts/publicar_kaggle.py"),
           "--experimento", a.experimento, "--enviar"]
    if a.so_notebook:
        cmd.append("--so-notebook")
    log.info("$ %s", " ".join(cmd))
    # ⚠️ Sem captura: a saída do publicador vai direto para o log desta fila, e é
    # ela que diz se o push pegou. O publicador já recusa as falhas silenciosas
    # da CLI — inclusive a de sessão esgotada, que é a razão desta fila existir.
    r = subprocess.run(cmd, cwd=RAIZ)
    if r.returncode:
        raise SystemExit(
            f"o push de {a.experimento} falhou (código {r.returncode}). NÃO vou "
            "tentar de novo: um push que falha por conteúdo falharia igual daqui "
            "a uma hora, e repetir só esconderia a causa.")
    log.info("✅ %s empurrado", a.experimento)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
