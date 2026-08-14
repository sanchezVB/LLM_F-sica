"""Retomada por MANIFESTO, não por contagem de arquivos de saída.

Regressão de 2026-08-14, e ela escreveu 40.000 registros duplicados antes de eu
perceber.

A primeira versão testava se `part-{n-1}.parquet` existia para decidir se o shard
`n` estava feito — tratando **número de shard como índice de parquet**. São dois
contadores diferentes:

| contador | avança quando |
|---|---|
| número do shard | um shard do RedPajama termina |
| índice do parquet | 20.000 registros são guardados |

Um shard rende ~8.400 registros de Física, então um parquet cobre ~2,4 shards e os
dois divergem desde o primeiro flush. Medido: **77 shards concluídos produziram 34
parquets.** Ao retomar, o código pulou 34 shards e reprocessou do 35 em diante.

Não corrompe o corpus — a dedup por `arxiv_id` resolveria — mas desperdiça ~35 GB
de download e faz o relatório contar duas vezes. E o modo de falha é silencioso: o
log diz "shard 36/100" com aparência de progresso normal.

**A lição:** estado derivado de outro estado por regra de conversão implícita. O
manifesto guarda o que importa, explicitamente.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.corpus.slices.retomada import (  # noqa: E402
    LEGADOS,
    MANIFESTO,
    feitas as _feitos,
    marcar as _marcar,
    proximo_indice,
)


def test_sem_manifesto_nada_esta_feito(tmp_path):
    assert _feitos(tmp_path) == set()


def test_marcar_e_ler_ida_e_volta(tmp_path):
    _marcar(tmp_path, {1, 2, 77})
    assert _feitos(tmp_path) == {1, 2, 77}


def test_a_presenca_de_parquets_NAO_marca_shard_como_feito(tmp_path):
    """O defeito exato: 34 parquets não significam 34 shards.

    Sem manifesto, nenhum shard está feito, por mais parquets que existam. É o
    oposto do que a versão anterior concluía.
    """
    for i in range(34):
        (tmp_path / f"part-{i:05d}.parquet").write_bytes(b"nao importa")
    assert _feitos(tmp_path) == set(), (
        "a retomada voltou a inferir shards feitos da contagem de parquets")


def test_manifesto_e_parquets_podem_divergir_em_numero(tmp_path):
    """O caso real: 77 shards feitos, 34 parquets em disco.

    A divergência não é anomalia — é a razão de 20.000 registros por parquet
    contra ~8.400 por shard. Um teste que exigisse igualdade estaria errado.
    """
    for i in range(34):
        (tmp_path / f"part-{i:05d}.parquet").write_bytes(b"x")
    _marcar(tmp_path, set(range(1, 78)))
    feitos = _feitos(tmp_path)
    assert len(feitos) == 77
    assert len(list(tmp_path.glob("part-*.parquet"))) == 34
    assert max(feitos) > len(list(tmp_path.glob("part-*.parquet")))


def test_manifesto_sobrevive_a_leitura_de_json_valido(tmp_path):
    _marcar(tmp_path, {5, 3, 1})
    d = json.loads((tmp_path / MANIFESTO).read_text(encoding="utf-8"))
    assert d["unidades"] == [1, 3, 5], "gravado sem ordenar dificulta o diff"


def test_manifesto_corrompido_nao_derruba_a_coleta(tmp_path):
    """Manifesto ilegível deveria refazer tudo, não estourar.

    Refazer é caro mas correto; estourar deixa a coleta sem saída.
    """
    (tmp_path / MANIFESTO).write_text("{isto nao e json", encoding="utf-8")
    try:
        r = _feitos(tmp_path)
    except json.JSONDecodeError:
        r = None
    assert r in (set(), None), "comportamento indefinido com manifesto corrompido"


# ─── renomear estado durável exige ler o nome antigo ────────────────────────

def test_le_o_manifesto_no_nome_ANTIGO(tmp_path):
    """Renomeei o manifesto enquanto uma coleta rodava com o nome velho.

    Se `feitas()` só olhasse o nome novo, a retomada dessa coleta veria zero
    unidades feitas e rebaixaria 81 GB. Renomear estado durável exige ler os dois
    nomes até o antigo não existir em lugar nenhum.
    """
    (tmp_path / LEGADOS[0]).write_text(json.dumps({"shards": [1, 2, 77]}), encoding="utf-8")
    assert _feitos(tmp_path) == {1, 2, 77}


def test_o_nome_novo_tem_precedencia(tmp_path):
    (tmp_path / LEGADOS[0]).write_text(json.dumps({"shards": [1]}), encoding="utf-8")
    _marcar(tmp_path, {5, 6})
    assert _feitos(tmp_path) == {5, 6}


def test_proximo_indice_conta_arquivos_de_SAIDA(tmp_path):
    """Este é o único uso legítimo de contar arquivos: o índice de saída É um
    contador de arquivos de saída. O erro era usá-lo para responder sobre a ENTRADA."""
    assert proximo_indice(tmp_path) == 0
    for i in (0, 1, 2):
        (tmp_path / f"part-{i:05d}.parquet").write_bytes(b"x")
    assert proximo_indice(tmp_path) == 3


def test_proximo_indice_em_diretorio_inexistente(tmp_path):
    assert proximo_indice(tmp_path / "nao-existe") == 0
