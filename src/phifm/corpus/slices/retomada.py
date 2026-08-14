"""Retomada por manifesto explícito, compartilhada pelos coletores de fatia.

## O defeito que isto existe para não repetir

Duas fatias — RedPajama e as do HuggingFace — decidiam se a unidade `n` estava
feita testando se `part-{n-1}.parquet` existia. Isso trata **número da unidade de
entrada como índice do arquivo de saída**, e são contadores diferentes:

| contador | avança quando |
|---|---|
| unidade de entrada | um shard / arquivo termina |
| índice de saída | `FLUSH` registros são guardados |

No RedPajama um shard rende ~8.400 registros de Física e o flush é a cada 20.000 —
um parquet cobre ~2,4 shards. No OpenWebMath são ~7.400 aceitos por arquivo, ~2,7
arquivos por parquet. Em nenhum dos dois casos os contadores coincidem, e nos dois
a divergência começa no primeiro flush.

Medido no custo real, 2026-08-14: 77 shards concluídos produziram 34 parquets; ao
retomar, o código pulou 34 shards, reprocessou do 35 em diante e gravou **40.000
registros duplicados**. O log dizia "shard 36/100" com aparência de progresso
normal.

Corrigi no RedPajama e **não propaguei** para o filtro do HuggingFace — que tinha o
mesmo código e ia rodar minutos depois. É exatamente por isso que a lógica mora
aqui agora: mecanismo duplicado divergiu, e vai divergir de novo se voltar a ser
duplicado.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

MANIFESTO = "_unidades_feitas.json"
# ⚠️ Nome ANTERIOR, ainda lido. Renomeei o manifesto enquanto uma coleta rodava com
# o nome velho — se `feitas()` só olhasse o novo, a retomada dessa coleta veria
# zero unidades feitas e rebaixaria 81 GB. Renomear estado durável exige ler os
# dois nomes até o antigo não existir mais em lugar nenhum.
LEGADOS = ("_shards_feitos.json",)
# Chaves aceitas dentro do JSON, pela mesma razão.
CHAVES = ("unidades", "shards")


def feitas(destino: Path, nome: str = MANIFESTO) -> set[int]:
    """Unidades de entrada já concluídas.

    Manifesto ausente ou ilegível devolve conjunto vazio — refazer é caro mas
    correto, e estourar deixaria a coleta sem saída.
    """
    for candidato in (nome, *LEGADOS):
        m = destino / candidato
        if not m.exists():
            continue
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("manifesto %s ilegível (%s) — tentando o próximo", candidato, exc)
            continue
        for chave in CHAVES:
            if chave in d:
                if candidato != nome:
                    log.info("manifesto no nome antigo (%s) — lido normalmente", candidato)
                return set(d[chave])
    return set()


def marcar(destino: Path, feitas_: set[int], nome: str = MANIFESTO) -> None:
    """Grava ordenado, para o diff do arquivo ser legível."""
    destino.mkdir(parents=True, exist_ok=True)
    (destino / nome).write_text(
        json.dumps({"unidades": sorted(feitas_)}, indent=0), encoding="utf-8")


def proximo_indice(destino: Path, padrao: str = "part-*.parquet") -> int:
    """Próximo índice de SAÍDA, contado dos arquivos que existem.

    Este é o único uso legítimo de contar arquivos: o índice de saída É um
    contador de arquivos de saída. O erro era usá-lo para responder sobre a
    ENTRADA.
    """
    existentes = sorted(destino.glob(padrao)) if destino.exists() else []
    return 1 + max((int(x.stem.split("-")[1]) for x in existentes), default=-1)
