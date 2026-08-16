"""Uma morte no passo 150 tem de deixar algo para retomar.

Regressão de 2026-08-16. O treino de 1,5 M pares morreu em silêncio no passo 150
— sem traceback, sem evento no log do Windows — e o diretório de saída estava
VAZIO. Não porque a morte foi estranha, mas porque o estado retomável era salvo
DENTRO do bloco de avaliação, e `passos_aval` era 500. O primeiro estado só sairia
no passo 500, então 150 passos de trabalho não tinham para onde voltar.

As duas coisas têm custo muito diferente: avaliar leva ~15 s (codifica 2.000
textos), salvar o estado leva ~2 s (pesos e momentos do Adam). Amarrar a barata na
cara foi economia no lugar errado.

O segundo fato aqui é o `progresso.json`. O supervisor é escrito em PowerShell e
não sabe abrir um `.pt`; sem um marcador legível de fora, o progresso que ele lia
durante um treino era sempre -1, a guarda de "morre sempre no mesmo ponto" ficava
DESLIGADA, e ele relançaria 40 vezes um treino que morre sempre no 150 — laço
infinito com aparência de resiliência.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# `torch` vive só na venv de TREINO (Python 3.12) — o `torch-directml` não
# suporta o 3.14 da venv principal. Salto com motivo declarado; a verificação de
# que estes testes PASSAM é feita rodando a suíte na venv de treino.
torch = pytest.importorskip("torch", reason="requer a venv de treino (.venv-treino)")


def _treinador(tmp: Path):
    """Um treinador com o mínimo que `salvar_estado` e `retomar` tocam.

    Um `nn.Linear` em vez do MiniLM: o que se testa é a durabilidade do estado,
    não o encoder — carregar SciBERT aqui custaria 40 s e não mediria nada a mais.
    """
    from phifm.training.embedding import Config, TreinadorEmb

    t = TreinadorEmb.__new__(TreinadorEmb)
    t.cfg = Config(passos_estado=100, passos_aval=500)
    t.dev = torch.device("cpu")
    t.mod = torch.nn.Linear(4, 4)
    t.opt = torch.optim.AdamW(t.mod.parameters(), lr=1e-3)
    t._melhor_ndcg = -1.0
    return t


def test_estado_no_passo_150_e_retomavel(tmp_path):
    """O caso exato: morre no 150, com avaliação a cada 500."""
    saida = tmp_path / "phiemb-minilm-1m5"
    t = _treinador(tmp_path)
    t.salvar_estado(saida, 150)

    t2 = _treinador(tmp_path)              # simula o relançamento
    assert t2.retomar(saida) == 150, "a retomada não achou o estado do passo 150"


def test_cadencia_do_estado_e_mais_curta_que_a_da_avaliacao():
    """O defeito era estrutural: uma cadência servindo a dois custos diferentes.

    Sem esta desigualdade o estado volta a depender da avaliação, e um treino com
    `--passos-aval 500` perde meia hora de trabalho por queda.
    """
    from phifm.training.embedding import Config

    c = Config()
    assert c.passos_estado < c.passos_aval, (
        "o estado não pode ser mais raro que a avaliação — ele é o que se perde")


def test_salvar_estado_nao_esta_dentro_do_bloco_de_avaliacao():
    """A asserção que os outros testes NÃO fazem, e que é o defeito de verdade.

    `test_estado_no_passo_150_e_retomavel` chama `salvar_estado` diretamente, então
    passaria mesmo com a chamada de volta para dentro do `if passo % passos_aval`
    — que é precisamente o defeito de 2026-08-16. O que estava errado não era o
    mecanismo, era o LUGAR de onde ele é chamado.

    Rodar o laço de verdade exigiria dataset, tokenizer e encoder; a estrutura do
    laço é o fato menor que decide a mesma coisa, e é verificável de graça.

    ⚠️ A primeira versão deste teste olhava TODOS os `salvar_estado` de `treinar` e
    era vazia: existe um save final DEPOIS do laço, que nunca está sob
    `passos_aval`, e ele só sozinho já satisfazia a asserção. Verificado por
    mutação — com a chamada devolvida para dentro do bloco de avaliação, o teste
    passava. Só o que está DENTRO do laço decide o que sobrevive a uma queda.
    """
    import ast
    import inspect

    from phifm.training.embedding import TreinadorEmb

    arvore = ast.parse(inspect.getsource(TreinadorEmb.treinar).lstrip())
    laco = next((n for n in ast.walk(arvore) if isinstance(n, ast.For)), None)
    assert laco is not None, "`treinar` não tem laço de passos"

    def condicoes_acima(no, dentro=()):
        """Cada `salvar_estado` com a pilha de `if`s que o cerca."""
        for filho in ast.iter_child_nodes(no):
            if isinstance(filho, ast.Call) and getattr(
                    filho.func, "attr", None) == "salvar_estado":
                yield dentro
            proximo = dentro
            if isinstance(no, ast.If) and filho in no.body:
                proximo = (*dentro, ast.unparse(no.test))
            yield from condicoes_acima(filho, proximo)

    chamadas = list(condicoes_acima(laco))
    assert chamadas, "o laço de `treinar` não salva estado — uma queda perde tudo"
    fora = [c for c in chamadas if not any("passos_aval" in t for t in c)]
    assert fora, (
        "todo `salvar_estado` DO LAÇO está sob uma condição de `passos_aval` — o "
        f"estado voltou a depender da avaliação. Condições vistas: {chamadas}")


def test_progresso_legivel_de_fora(tmp_path):
    """O supervisor é PowerShell e não abre `.pt`. Sem isto ele fica cego."""
    saida = tmp_path / "phiemb"
    t = _treinador(tmp_path)
    t.salvar_estado(saida, 300)

    p = saida / "progresso.json"
    assert p.exists(), "salvar_estado não deixou marcador de progresso legível"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["passo"] == 300
    assert d["ts"], "sem horário não se distingue progresso de arquivo esquecido"


def test_progresso_avanca_entre_duas_gravacoes(tmp_path):
    """É a COMPARAÇÃO de dois valores que o supervisor usa, não o valor.

    Um marcador que não muda faz o supervisor concluir "não avança" e abortar um
    treino saudável — que é o modo de falha que ele já cometeu uma vez, em
    2026-08-07 09:55, abortando por engano no exato caso que existia para cobrir.
    """
    saida = tmp_path / "phiemb"
    t = _treinador(tmp_path)
    t.salvar_estado(saida, 100)
    primeiro = json.loads((saida / "progresso.json").read_text(encoding="utf-8"))["passo"]
    t.salvar_estado(saida, 200)
    segundo = json.loads((saida / "progresso.json").read_text(encoding="utf-8"))["passo"]
    assert segundo > primeiro == 100


def test_estado_nunca_fica_meio_gravado(tmp_path):
    """Gravação atômica: uma queda DURANTE o save não pode deixar `.pt` truncado.

    O `.tmp` é o mecanismo; o que se verifica é que ele não sobra — um `.tmp`
    órfão ao lado do estado bom é ruído que faz o próximo diagnóstico duvidar do
    arquivo certo.
    """
    saida = tmp_path / "phiemb"
    t = _treinador(tmp_path)
    t.salvar_estado(saida, 100)
    assert not list(saida.glob("*.tmp")), "sobrou temporário depois de gravar"
    assert (saida / "estado_treino.pt").exists()
