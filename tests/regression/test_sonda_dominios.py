"""Um defeito no ÚLTIMO passo espera o trabalho caro terminar para aparecer.

A sonda de domínios leu 5 shards do RedPajama (4 GB de rede, 2 min), contou todos
os registros por domínio, e morreu em

    TypeError: Object of type int64 is not JSON serializable

**na gravação do artefato**. Os números ficaram só no log. A causa: `permutation`
do numpy devolve `np.int64`, e `json.dumps` não serializa isso.

É a mesma família do "guarda depois da escrita não é guarda" — e aqui o custo foi
refazer o download. Este teste monta o artefato com tipos de numpy dentro e exige
que ele serialize, sem tocar na rede.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

FONTE = (RAIZ / "scripts/sondar_dominios_redpajama.py").read_text(encoding="utf-8")


def test_o_artefato_SERIALIZA_com_tipos_de_numpy():
    """O artefato mistura contadores de `Counter` (int nativo) com índices que
    vêm de `permutation` (np.int64). Só um deles serializa."""
    artefato = {
        "protocolo": {"shards_sorteados": [int(x) for x in np.array([28, 49])],
                      "semente": 17},
        "na_amostra": {"documentos": {"math": 18_977, "cs": 15_283}},
        "extrapolado_para_100_shards": {
            "math_mais_cs_documentos": int(np.int64(633_700)),
            "fator": float(np.float64(20.0)),
        },
    }
    json.dumps(artefato)      # levanta TypeError se algum np.* escapou


def test_o_sorteio_converte_para_int_NATIVO():
    """Por AST: a conversão tem de estar na atribuição de `escolhidos`, que é de
    onde o np.int64 vinha."""
    arvore = ast.parse(FONTE)
    atrib = next(n for n in ast.walk(arvore) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "escolhidos"
                         for t in n.targets))
    texto = ast.unparse(atrib)
    assert "int(" in texto, (
        f"`escolhidos` não converte para int nativo: {texto}. `permutation` "
        "devolve np.int64 e o artefato não serializa.")


def test_os_shards_sao_SORTEADOS_e_nao_os_primeiros():
    """⚠️ As 8 primeiras partes do corpus têm 32,4% de matemática contra 42,7%
    das demais. Para uma sonda que existe para extrapolar, enviesar a amostra é
    enviesar a conclusão."""
    # ⚠️ Sobre o CÓDIGO, não sobre o arquivo: a docstring MENCIONA `urls[:n]`
    # para explicar o que não fazer, e um teste que proíbe a string proíbe a
    # explicação junto. (A primeira versão deste teste reprovou por isso.)
    from conftest import so_codigo
    codigo = so_codigo(FONTE)
    assert "permutation" in codigo
    assert "urls[:" not in codigo, (
        "o código fatia as urls em vez de sortear; as 8 primeiras partes têm "
        "32,4% de matemática contra 42,7% das demais")
    assert "32,4%" in FONTE, "a razão do sorteio não está escrita junto"


def test_os_ids_ja_no_corpus_SAEM_da_conta_do_ganho():
    """Um artigo cross-list que está em `math` E na tabela mestra de Física já foi
    guardado pelo filtro atual — contá-lo como ganho infla o resultado. Medido:
    123.418 ids de `math` também estão na Física."""
    assert "- fisica" in FONTE or "ids - fisica" in FONTE
    assert "não é ganho" in FONTE or "já está no disco" in FONTE


def test_a_extrapolacao_declara_que_NAO_estimou_erro_de_amostragem():
    """5 de 100 shards dão ordem de grandeza, não medida. Um número sem essa
    ressalva convida a tratá-lo como exato."""
    assert "erro de" in FONTE and "amostragem" in FONTE
    assert "ordem de grandeza, não medida" in FONTE


def test_a_sonda_tem_uma_REGUA_de_conferencia():
    """A Física extrapolada tem de bater com os 828.601 documentos que já estão em
    disco. Sem essa conferência, uma extrapolação errada passa sem aviso."""
    assert "fisica_conferencia" in FONTE
    assert "828.601" in FONTE or "828601" in FONTE


def test_a_sonda_NAO_guarda_texto():
    """Guardar o texto seria refazer a coleta, que é o que a sonda existe para
    decidir. Ela responde 'quantos e quão grandes', não 'quais'."""
    assert "Nada de texto é guardado" in FONTE
    assert "write_parquet" not in FONTE
