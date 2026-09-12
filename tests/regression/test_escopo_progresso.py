"""Contadores de escopos diferentes no mesmo artefato produzem leitura errada.

O `_progresso.json` da coleta do RedPajama dizia `shards_lidos: 100` com
`registros_vistos: 350.451`, e eu li isso como "o RedPajama-arXiv tem 350 mil
documentos". Ele tem entre 1,35 M e 1,93 M — os 828.601 de Física em disco
divididos pela taxa de casamento.

A causa é uma assimetria no laço: o shard PULADO por retomada incrementa
`shards_lidos` e não passa por `processar`, então `registros_vistos` e
`bytes_lidos` não crescem. Um campo é acumulado entre execuções; os outros são
desta execução.

⚠️ Com o número errado, o teto de ganho de acrescentar `math` e `cs` ao filtro
saiu **~5× menor** do que é — uma decisão de dados baseada num artefato que
misturava escopos.

E o `filtrar_hf.py` já gravava `escopo_dos_numeros: "esta execução, não o
acumulado"` pela MESMA razão. A lição estava paga e aplicada num arquivo só.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.corpus.slices.redpajama import Progresso  # noqa: E402


def test_o_shard_PULADO_nao_conta_como_processado():
    """É a assimetria que causou o erro: o pulado entra no acumulado e não traz
    registros nem bytes consigo."""
    p = Progresso()
    p.shards_vistos_no_indice += 1              # pulado por retomada
    assert p.shards_processados_agora == 0
    assert p.registros_vistos == 0
    assert p.bytes_lidos == 0


def test_o_artefato_SEPARA_os_dois_escopos():
    """⚠️ Um artefato que mistura escopos é pior que um incompleto: ele parece
    responder a pergunta que não responde."""
    p = Progresso(shards_vistos_no_indice=100, shards_processados_agora=26,
                  registros_vistos=350_451, registros_guardados=182_926)
    d = p.como_dict()
    assert "escopo_acumulado" in d and "escopo_desta_execucao" in d
    assert d["escopo_acumulado"]["shards_vistos_no_indice"] == 100
    assert d["escopo_desta_execucao"]["registros_vistos"] == 350_451
    assert d["escopo_desta_execucao"]["shards_processados_agora"] == 26
    # Nenhum campo de execução pode vazar para o bloco acumulado.
    assert "registros_vistos" not in d["escopo_acumulado"]


def test_o_artefato_DIZ_como_estimar_o_total_da_fonte():
    """A conta certa é `documentos em disco ÷ taxa`, e ela tem de estar escrita
    onde o erro foi cometido — senão o próximo leitor repete."""
    d = Progresso(registros_vistos=100, registros_guardados=52).como_dict()
    nota = d["como_estimar_o_total_da_fonte"]
    assert "÷" in nota or "dividido" in nota
    assert "NÃO use `registros_vistos` como" in nota


def test_a_linha_de_log_marca_o_escopo():
    """O log é o que se lê durante 11,7 h de coleta. Sem a marca, os números
    parecem todos do mesmo lugar."""
    linha = Progresso(shards_vistos_no_indice=100, shards_processados_agora=26,
                      registros_vistos=350_451,
                      registros_guardados=182_926).linha()
    assert "no índice" in linha and "processados agora" in linha
    assert "DESTA execução" in linha


def test_a_taxa_e_DESTA_execucao_e_a_docstring_diz():
    """A taxa é o único jeito de estimar o total da fonte, então o escopo dela
    tem de estar explícito."""
    doc = Progresso.taxa_fisica.fget.__doc__ or ""
    assert "DESTA execução" in doc
    assert "nunca use `registros_vistos` como se fosse o total" in doc
    p = Progresso(registros_vistos=1000, registros_guardados=522)
    assert abs(p.taxa_fisica - 0.522) < 1e-9


def test_o_nome_antigo_continua_LEGIVEL():
    """Artefatos já gravados e scripts de terceiros esperam `shards_lidos`.
    Renomear sem manter a leitura quebraria o que já está no disco."""
    p = Progresso(shards_vistos_no_indice=100)
    assert p.shards_lidos == 100
