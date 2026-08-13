"""Comparação pareada de encoders — o que duas proporções soltas não enxergam.

Motivação de 2026-08-10. O ΦEmb media recall@1 0,402 e o MiniLM 0,398 em 256
itens. Como proporções independentes, a margem de 0,004 desaparece num erro
padrão de ±0,031, e o veredito foi "empate estatístico". Correto — mas
desperdiça informação: os dois foram medidos nos **mesmos itens**.

A maioria dos itens os dois acertam ou os dois erram, e esses não dizem nada
sobre a diferença. Só os **discordantes** informam. Se de 256 itens 40 são
discordantes com placar de 32 a 8, isso é evidência forte que o teste não
pareado joga no lixo.

É o teste de McNemar, aqui na versão exata (binomial), sem aproximação normal —
que é ruim justamente quando os números são pequenos, o caso em que se precisa
dela.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

pytest.importorskip("torch", reason="requer a venv de treino (.venv-treino, Python 3.12)")

from phifm.eval.encoders import Resultado, comparar_pareado  # noqa: E402


def res(nome: str, posicoes: list[int], nosso: bool = False) -> Resultado:
    n = len(posicoes)
    return Resultado(
        nome=nome, caminho=nome,
        recall_1=sum(1 for p in posicoes if p == 1) / n,
        recall_10=sum(1 for p in posicoes if p <= 10) / n,
        mrr=sum(1 / p for p in posicoes) / n,
        # nDCG@10 com um relevante por consulta. É a métrica que o G1.1 nomeia;
        # aqui entra por completude, porque o que este arquivo testa é o pareado.
        ndcg_10=sum(1 / math.log2(1 + p) for p in posicoes if p <= 10) / n,
        parametros_m=0.0, segundos=0.0, nosso=nosso, posicoes=posicoes,
    )


def test_placar_conta_so_os_discordantes():
    """20 itens: 5 só A acerta, 3 só B acerta, 12 concordantes."""
    a = res("A", [1] * 5 + [9] * 3 + [1] * 6 + [7] * 6)
    b = res("B", [4] * 5 + [1] * 3 + [1] * 6 + [7] * 6)
    c = comparar_pareado(a, b)
    assert c["ganha_a"] == 5
    assert c["ganha_b"] == 3
    assert c["discordantes"] == 8, "concordantes entraram na conta"


def test_margem_pequena_mas_consistente_e_detectada():
    """O caso que motivou tudo, exagerado para caber num teste.

    Diferença de 0,04 em recall@1 — que como proporções soltas seria ruído —
    torna-se significativa quando TODOS os discordantes vão para o mesmo lado.
    """
    # 100 itens: 96 concordantes, 4 só A acerta, 0 só B. Depois amplia para 30/0.
    a = res("A", [1] * 30 + [1] * 50 + [5] * 20)
    b = res("B", [5] * 30 + [1] * 50 + [5] * 20)
    c = comparar_pareado(a, b)
    assert c["discordantes"] == 30
    assert c["p"] < 0.05, "placar de 30 a 0 devia ser significativo"
    assert "A vence" in c["veredito"]


def test_empate_real_nao_e_declarado_vitoria():
    a = res("A", [1] * 20 + [5] * 20 + [1] * 60)
    b = res("B", [5] * 20 + [1] * 20 + [1] * 60)
    c = comparar_pareado(a, b)
    assert c["ganha_a"] == c["ganha_b"] == 20
    assert c["p"] == pytest.approx(1.0)
    assert "empate" in c["veredito"]


def test_poucos_discordantes_e_indeciso_nao_empate():
    """Diferença de amostra pequena não é empate: é ausência de dado. Chamar de
    empate esconderia que o teste não foi capaz de decidir."""
    a = res("A", [1] * 3 + [1] * 97)
    b = res("B", [5] * 3 + [1] * 97)
    c = comparar_pareado(a, b)
    assert c["discordantes"] == 3
    assert "indeciso" in c["veredito"], f"disse: {c['veredito']}"


def test_identicos_nao_tem_o_que_decidir():
    p = [1, 2, 1, 8, 1]
    c = comparar_pareado(res("A", p), res("B", list(p)))
    assert c["discordantes"] == 0
    assert c["p"] == 1.0
    assert "idênticos" in c["veredito"]


def test_tamanhos_diferentes_sao_recusados():
    """Comparar conjuntos de tamanhos diferentes é comparar coisas diferentes."""
    c = comparar_pareado(res("A", [1] * 10), res("B", [1] * 20))
    assert "erro" in c
    assert "tamanhos diferentes" in c["erro"]


def test_sem_posicoes_avisa_em_vez_de_mentir():
    """Resultado de uma versão antiga não tem `posicoes`. Devolver zero seria
    dizer 'empate' onde o dado não existe."""
    a = Resultado("A", "A", 0.4, 0.9, 0.5, 0.6, 0.0, 0.0)
    b = Resultado("B", "B", 0.3, 0.8, 0.4, 0.5, 0.0, 0.0)
    c = comparar_pareado(a, b)
    assert "erro" in c
    assert "posições" in c["erro"]


def test_p_e_bicaudal():
    """Placar de 8 a 0 em 8 discordantes: 2 × (1/2)^8 = 0,0078."""
    a = res("A", [1] * 8 + [1] * 92)
    b = res("B", [5] * 8 + [1] * 92)
    c = comparar_pareado(a, b)
    assert c["p"] == pytest.approx(2 * 0.5 ** 8, rel=1e-6)


def test_simetria():
    """Trocar a ordem inverte o placar e mantém o p — senão a conclusão
    dependeria de quem foi escrito primeiro."""
    a = res("A", [1] * 12 + [7] * 4 + [1] * 84)
    b = res("B", [7] * 12 + [1] * 4 + [1] * 84)
    ab, ba = comparar_pareado(a, b), comparar_pareado(b, a)
    assert ab["ganha_a"] == ba["ganha_b"]
    assert ab["ganha_b"] == ba["ganha_a"]
    assert ab["p"] == pytest.approx(ba["p"])
