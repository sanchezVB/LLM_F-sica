"""O assistente, com o modelo de linguagem e a busca FALSOS.

O que se tranca aqui é o encanamento, que pode errar sem dar erro:

1. a busca recebe o texto HIPOTÉTICO do modelo, não a pergunta crua;
2. o prompt da resposta leva as fontes numeradas com o resumo inteiro, na ordem da busca;
3. o portão tira do texto toda citação a fonte que não foi fornecida, e marca as frases
   sem citação — sem mexer nas citações válidas.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.rag.assistente import (  # noqa: E402
    SISTEMA_HYDE,
    SISTEMA_RESPOSTA,
    Assistente,
    verificar_citacoes,
)
from phifm.rag.llm import chatml  # noqa: E402
from phifm.retrieval.indice import Resultado  # noqa: E402

HIPOTETICO = "Quantum decoherence in open systems. We study how environment coupling..."


class ModeloFalso:
    def __init__(self, resposta: str):
        self.resposta, self.chamadas = resposta, []

    def gerar(self, sistema, usuario, **kw):
        self.chamadas.append((sistema, usuario))
        return f"  {HIPOTETICO}\n" if sistema == SISTEMA_HYDE else self.resposta


class BuscaFalsa:
    def __init__(self, ids):
        self.ids, self.consultas = ids, []

    def buscar(self, consulta, k=10, excluir=None):
        self.consultas.append(consulta)
        return [Resultado(i, 0.9 - i / 10, a, f"Título {a}", 2020, "quant-ph", ["A"])
                for i, a in enumerate(self.ids[:k])]


def _spine(tmp_path):
    ids = [f"2001.{i:05d}" for i in range(5)]
    pl.DataFrame({"arxiv_id": ids, "title": [f"Título {a}" for a in ids],
                  "abstract": [f"Resumo  do\n artigo {a}." for a in ids],
                  "year": [2020] * 5}).write_parquet(tmp_path / "spine.parquet")
    return tmp_path / "spine.parquet", ids


# ── o portão de citações ────────────────────────────────────────────────────


def test_citacao_a_fonte_INEXISTENTE_sai_do_texto_e_a_valida_fica():
    texto, citadas, removidas, _ = verificar_citacoes(
        "A constante vale 1/137 [1]. Ela acopla o campo [2, 9]. Isto vem de [7].", 6)
    assert texto == "A constante vale 1/137 [1]. Ela acopla o campo [2]. Isto vem de."
    assert citadas == [1, 2] and removidas == [7, 9]


def test_intervalos_e_listas_viram_citacoes_separadas():
    texto, citadas, removidas, _ = verificar_citacoes("Vários trabalhos [1-3; 5] e [2–2].", 4)
    assert texto == "Vários trabalhos [1][2][3] e [2]."
    assert citadas == [1, 2, 3] and removidas == [5]


def test_frase_longa_SEM_citacao_e_marcada_e_a_admissao_de_falta_nao():
    _, _, _, sem = verificar_citacoes(
        "O efeito foi medido em redes ópticas com átomos frios [1]. "
        "Ele também aparece em supercondutores de alta temperatura crítica. "
        "As fontes fornecidas não tratam do caso relativístico em detalhe. Ok.", 3)
    assert sem == ["Ele também aparece em supercondutores de alta temperatura crítica."]


# ── o encanamento ───────────────────────────────────────────────────────────


def test_a_busca_recebe_o_texto_HIPOTETICO_e_o_prompt_leva_as_fontes(tmp_path):
    spine, ids = _spine(tmp_path)
    modelo = ModeloFalso("Decoerência é a perda de coerência [2]. Ver também [4].")
    busca = BuscaFalsa(ids)
    r = Assistente(busca, modelo, spine, k=3).responder("O que é decoerência?")

    assert busca.consultas == [HIPOTETICO]
    assert r.consulta == HIPOTETICO
    sistema, usuario = modelo.chamadas[-1]
    assert sistema == SISTEMA_RESPOSTA
    assert usuario.endswith("Question: O que é decoerência?")
    for n, a in enumerate(ids[:3], start=1):
        assert f"[{n}] Título {a} (2020, arXiv:{a})\nResumo do artigo {a}." in usuario
    assert "[4]" not in usuario.split("Question:")[0]
    assert [f.arxiv_id for f in r.fontes] == ids[:3]
    assert r.texto == "Decoerência é a perda de coerência [2]. Ver também."
    assert r.citadas == [2] and r.removidas == [4]


def test_o_prompt_do_qwen_desliga_o_raciocinio_longo():
    p = chatml("S", "U")
    assert p.startswith("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n")
    assert p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
