"""O ΦRank do sistema é o que a medição elegeu, e o motivo tem de estar no código.

O T1b deixou o reranqueador FORA do sistema por um ano de trabalho: ele empatava com
a fusão (p=0,118) porque partia da mesma base do recuperador. O T1c mediu três bases
com a mesma receita e uma venceu:

    base                        params  acc@1  nDCG    p(k=10)
    all-MiniLM-L6-v2 (= ΦEmb)      23M  0,498  0,1483   0,1458  empate
    thenlper/gte-base             109M  0,510  0,1530   0,6371  empate
    thellert/physbert_cased       109M  0,566  0,1666   0,0062  VENCE

Estes testes existem porque as duas configurações que sustentam esse resultado — a
base padrão de treino e o modelo padrão de avaliação — são uma linha cada, e uma
linha se reverte sem que nada quebre. O sistema voltaria a compor com um
reranqueador que a medição diz ser um no-op, e a métrica cairia 0,009 sem nenhum
teste vermelho.

⚠️ Nada aqui lê `models/`, que é gitignored. Um teste que exigisse os pesos passaria
nesta máquina e derrubaria o CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from conftest import so_codigo_de  # noqa: E402

VENCEDOR = "thellert/physbert_cased"
PERDEDORAS = ("sentence-transformers/all-MiniLM-L6-v2", "thenlper/gte-base")


def test_a_base_padrao_do_reranqueador_e_a_que_venceu():
    """A asserção central, e ela roda na suíte RÁPIDA de propósito.

    ⚠️ Ler o fonte em vez de importar `phifm.training.rerank`, que arrasta torch. Um
    `importorskip("torch")` faria este teste PULAR no CI — e um guarda que não roda
    no CI não guarda a linha que ele existe para proteger. É a mesma razão pela qual
    `amostragem.py` mora fora de `rerank.py`.
    """
    codigo = so_codigo_de(RAIZ / "src/phifm/training/rerank.py")
    assert f'BASE_PADRAO = {VENCEDOR!r}' in codigo, (
        f"a base padrão não é mais {VENCEDOR}. As medidas do T1c dizem que só ela "
        "bate a fusão (p=0,0062); as outras duas empatam.")
    assert "base: str = BASE_PADRAO" in codigo, (
        "a config não usa mais BASE_PADRAO — mudar a constante deixou de ter efeito")
    for perdedora in PERDEDORAS:
        assert f"BASE_PADRAO = {perdedora!r}" not in codigo


def test_a_base_padrao_vale_de_verdade_na_config():
    """O mesmo, exercitando o objeto. Pula sem torch, e é por isso que o teste de
    cima existe: este não roda no CI."""
    import pytest

    pytest.importorskip("torch", reason="requer a venv de treino (.venv-treino)")
    from phifm.training.rerank import BASE_PADRAO, ConfigRank

    assert BASE_PADRAO == VENCEDOR
    assert ConfigRank().base == VENCEDOR


def test_o_motivo_esta_no_codigo_com_os_numeros():
    """Um default sem justificativa medida ao lado é um default que alguém reverte.

    O comentário tem de carregar os três p, porque é a comparação — e não o valor
    absoluto — que sustenta a escolha: o `gte-base` tem o MESMO tamanho do vencedor
    e não acrescenta nada, então não é capacidade nem diversidade.
    """
    fonte = (RAIZ / "src/phifm/training/rerank.py").read_text(encoding="utf-8")
    for p in ("0,0062", "0,6371", "0,1458"):
        assert p in fonte, (
            f"o p={p} saiu do comentário que justifica a base padrão; sem os três a "
            "escolha parece arbitrária")
    assert "gte-base" in fonte and "MESMO tamanho" in fonte, (
        "o argumento de que não é capacidade nem diversidade desapareceu")


def test_o_default_do_avaliador_aponta_para_o_modelo_instalado():
    """A composição do T1b tem de compor com o vencedor, não com o no-op."""
    fonte = (RAIZ / "scripts/avaliar_t1b.py").read_text(encoding="utf-8")
    assert 'default=Path("models/phirank-physbert-melhor")' in fonte, (
        "o avaliador voltou a compor com outro ΦRank")
    assert "phirank-minilm-melhor" not in fonte, (
        "o caminho do reranqueador que empata com a fusão voltou como default")


def test_o_aviso_de_nao_usar_esta_base_no_recuperador():
    """A assimetria medida, e ela é contraintuitiva o suficiente para ser dita.

    O PhysBERT é o MELHOR reranqueador e um recuperador RUIM: nDCG 0,2752 contra
    0,4657 do ΦEmb, que é a margem de +0,190 do G1.1. Alguém que leia "a base de
    domínio venceu" e aplique isso ao `train_embedding.py` perderia 0,19 de nDCG.
    """
    fonte = (RAIZ / "src/phifm/training/rerank.py").read_text(encoding="utf-8")
    assert "0,2752" in fonte and "0,4657" in fonte, (
        "os números da assimetria saíram do comentário")
    assert "train_embedding" in fonte, (
        "falta o aviso explícito de não levar este default para o recuperador")

    # E o recuperador tem de continuar com a base DELE. Fonte, não import: ver
    # `test_a_base_padrao_do_reranqueador_e_a_que_venceu`.
    codigo_emb = so_codigo_de(RAIZ / "src/phifm/training/embedding.py")
    assert f"BASE_PADRAO = {VENCEDOR!r}" not in codigo_emb, (
        "a base do recuperador virou a do reranqueador — são medições diferentes, "
        "e no recuperador esta base perde por 0,190")


def test_a_proveniencia_gravada_usa_a_base_de_verdade():
    """Até 2026-09-03 esta string era fixa e afirmava MiniLM para qualquer base.

    O `phirank.json` do modelo de PhysBERT saiu do treino dizendo que ele veio do
    MiniLM — proveniência errada, gravada com a mesma confiança da certa, dentro do
    diretório do modelo. `scripts/train_rerank.py` já tinha sido parametrizado; esta
    cópia dentro do treinador escapou.
    """
    # ⚠️ `so_codigo` porque a asserção de ausência abaixo reprovaria o COMENTÁRIO
    # que explica o erro — quarta ocorrência dessa armadilha neste repositório. Ver
    # `tests/conftest.py`.
    codigo = so_codigo_de(RAIZ / "src/phifm/training/rerank.py")
    assert "inicializado de {self.cfg.base}" in codigo, (
        "a proveniência gravada no diretório do modelo voltou a ser uma string fixa")
    assert "inicializado do MiniLM, a mesma base" not in codigo, (
        "a string fixa que afirmava MiniLM para qualquer base voltou")


def test_o_instalador_exige_justificativa_medida():
    """Pesos que aparecem em `models/` sem proveniência são um modelo órfão."""
    fonte = (RAIZ / "scripts/instalar_phirank.py").read_text(encoding="utf-8")
    assert '"--nota", required=True' in fonte, (
        "instalar um ΦRank sem a medição que justifica voltou a ser possível")
    assert "o_que_a_medicao_NAO_diz" in fonte, (
        "o manifesto tem de registrar os limites, não só o resultado")
    assert "gravar_manifesto_etapa" in fonte


def test_o_instalador_registra_a_divergencia_em_vez_de_reescrever():
    """Reescrever a proveniência de um artefato depois do fato é pior que registrar.

    O `phirank.json` do modelo instalado tem a linha errada e FICA com ela: é o que a
    execução produziu. Quem manda é o manifesto de etapa, que aponta o erro pelo nome.
    """
    fonte = (RAIZ / "scripts/instalar_phirank.py").read_text(encoding="utf-8")
    assert "divergencia_de_proveniencia" in fonte
    assert "não é corrigido" in fonte, (
        "a docstring deixou de dizer que o artefato não é reescrito")
    for reescrita in ("phirank.json\").write_text", "json.dump(meta"):
        assert reescrita not in fonte, (
            f"o instalador passou a reescrever o artefato ({reescrita!r})")
