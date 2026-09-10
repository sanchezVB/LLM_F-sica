"""O T1e gasta 2h50 de cota escassa, e estes testes é que impedem de gastá-la à toa.

O T1e mede a curva de profundidade da cadeia. Em 2026-09-10 a cota era de **4h40**,
então uma execução perdida custa a semana. O que os testes fixam:

  1. **a chave `--sub-profundidades` é conferida ANTES de medir.** Ela é o
     experimento inteiro; um tarball de commit anterior a ela produziria uma linha
     só, depois de 2h50;
  2. **as três profundidades saem da MESMA passagem** — nada de um segundo run,
     que compararia sessões diferentes (a armadilha do T1b2);
  3. **os três desfechos escritos antes do número**, o de derrota incluído;
  4. **a segunda leitura é independente** do confronto entre profundidades;
  5. **reusa o dataset do T1d** e não publica um novo: uma versão nova quebraria a
     assinatura que a célula do T1d confere.

⚠️ Nada aqui lê `models/` nem `data/processed/`, que são gitignored.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "kaggle"))

import t1e_profundidade  # noqa: E402

from conftest import so_codigo  # noqa: E402
from phifm.core.kaggle import obter  # noqa: E402

CELULA = t1e_profundidade.CELULA
DOC = t1e_profundidade.__doc__ or ""
CODIGO = so_codigo(CELULA)


def test_a_celula_e_python_valido():
    """O extrator do publicador é um regex sobre este arquivo; se o formato
    mudar, o notebook gerado sai vazio e a descoberta custa a cota da semana."""
    ast.parse(CELULA)


def test_a_chave_do_experimento_e_conferida_ANTES_de_medir():
    """⚠️ `--sub-profundidades` É o experimento.

    Um tarball de commit anterior a ela faria o `avaliar_t1b.py` recusá-la — ou,
    pior, ignorá-la — e a sessão produziria uma profundidade só, depois de 2h50.
    A conferência custa um `--help`.
    """
    i_conf = CELULA.index('"--sub-profundidades" in _ajuda.stdout')
    i_medir = CELULA.index("# ── 5.")
    assert i_conf < i_medir, "a conferência da chave está depois da medição"
    assert '"--help"' in CELULA


def test_as_TRES_profundidades_saem_da_MESMA_passagem():
    """Um segundo run custaria 85 min e compararia sessões diferentes — a
    armadilha que o T1b2 documentou e pagou."""
    arvore = ast.parse(CELULA)
    chamadas = [n for n in ast.walk(arvore)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "_rodar"]
    assert len(chamadas) == 1, (
        f"{len(chamadas)} execuções do avaliador; as sub-profundidades existem "
        "justamente para que baste uma")
    assert 'SUB = "50,100"' in CELULA
    assert "PROFUNDIDADE = 200" in CELULA


def test_a_sub_profundidade_nao_excede_a_profundidade():
    """O avaliador recusa, mas recusar no Kaggle custa a sessão."""
    subs = [int(x) for x in
            CELULA.split('SUB = "')[1].split('"')[0].split(",")]
    prof = int(CELULA.split("PROFUNDIDADE = ")[1].split("\n")[0])
    assert all(s < prof for s in subs), (subs, prof)
    assert 100 in subs, (
        "sem o @100 não há o que confrontar: ele é a profundidade de hoje")


def test_os_TRES_desfechos_estao_escritos_antes_do_numero():
    """Incluindo a derrota. Um desfecho não previsto vira interpretação depois do
    fato, que é como um resultado nulo se transforma em "quase deu"."""
    i_regra = CELULA.index("REGRA, no pareado")
    i_medir = CELULA.index("# ── 5.")
    assert i_regra < i_medir
    trecho = CELULA[i_regra:i_medir]
    assert "@200 VENCE" in trecho
    assert "EMPATE" in trecho
    assert "@200 PERDE" in trecho
    # ⚠️ A assimetria: empate NÃO mantém o estágio, porque o T1d já o deixou sem
    # evidência em @100. Sem isto, "empate" viraria "fica como está".
    assert "SAI do sistema" in trecho


def test_a_segunda_leitura_e_INDEPENDENTE():
    """Se nenhuma profundidade vencer o ΦEmb sozinho, o estágio não paga o custo
    em profundidade alguma — e isso vale seja qual for o vencedor entre elas."""
    assert "INDEPENDENTE" in CELULA
    assert "não paga o próprio custo em" in CELULA


def test_a_faixa_esperada_esta_declarada_antes():
    """Escrever a expectativa antes é o que impede de racionalizar o resultado
    depois. 46,6% de taxa condicional, +0,045 no melhor caso, 13,4% de folga."""
    i_medir = CELULA.index("# ── 5.")
    trecho = CELULA[:i_medir]
    assert "46,6%" in trecho
    assert "13,4%" in trecho


def test_confere_hash_e_assinatura_ANTES_de_medir():
    i_hash = CELULA.index("conferidos por blake3")
    i_assin = CELULA.index("assinatura do bundle confere")
    i_medir = CELULA.index("# ── 5.")
    assert i_hash < i_medir and i_assin < i_medir


def test_exige_gpu_em_vez_de_cair_para_cpu():
    assert 'assert torch.cuda.is_available(), "sem GPU' in CELULA


def test_avaliacao_que_falha_derruba_o_notebook():
    """⚠️ Por AST: um avaliador que sai com código diferente de zero tem de
    levantar, não seguir para uma leitura de artefato que não existe."""
    achou = False
    for no in ast.walk(ast.parse(CELULA)):
        if not isinstance(no, ast.If):
            continue
        if not (isinstance(no.test, ast.Compare)
                and any(isinstance(c, ast.NotEq) for c in no.test.ops)):
            continue
        if any(isinstance(x, ast.Raise) for x in ast.walk(no)):
            achou = True
    assert achou, "nenhum `if <codigo> != 0` levanta na célula"


def test_o_log_vai_para_arquivo_E_stdout():
    """Mandar só para o arquivo custou 33 h de cegueira em 2026-09-07."""
    assert "bufsize=1" in CODIGO
    assert "fh.write(linha)" in CODIGO
    assert "print(linha, end=" in CODIGO


def test_usa_sem_fusao():
    """A composição decidida no T1b2, e sem ela cada consulta custaria duas
    passagens — 5h40 em vez de 2h50, o dobro da cota disponível."""
    assert '"--sem-fusao"' in CELULA


def test_o_registro_declara_o_t1e_REUSANDO_o_dataset_do_t1d():
    """⚠️ Republicar criaria uma versão nova do dataset e quebraria a
    `assinatura_do_manifesto` que a célula do T1d confere — o notebook do T1d
    passaria a recusar o próprio dado."""
    e, dono = obter("t1e"), obter("t1d")
    e.conferir()
    assert e.reusa_dados_de == "t1d"
    assert e.slug_dados == dono.slug_dados
    assert e.pacote == dono.pacote
    assert e.slug_notebook != dono.slug_notebook
    assert e.fonte_celula == "kaggle/t1e_profundidade.py"
    # E não precisa do script de treino: este experimento não treina nada.
    assert e.scripts == ("avaliar_t1b.py",)


def test_o_publicador_RECUSA_publicar_o_dataset_de_quem_reusa():
    """A guarda tem de estar no publicador, não só na intenção de quem roda."""
    fonte = (RAIZ / "scripts" / "publicar_kaggle.py").read_text(encoding="utf-8")
    assert "reusa_dados_de" in fonte
    assert "--so-notebook" in fonte


def test_o_codigo_vem_do_GITHUB_num_sha():
    assert "codeload.github.com" in CELULA
    assert "__SHA__" in CELULA
    assert "assinatura_do_manifesto" in CODIGO
