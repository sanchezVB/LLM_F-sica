"""O T1f gasta 2h39 para responder se a base importa mais que o ajuste.

O GTE-base **zero-shot** já empata com o nosso ΦEmb ajustado no top-10 (r@10 0,2755
contra 0,2810, p=0,545) sem ter visto uma linha do nosso dado. O T1f ajusta o
GTE-base nos MESMOS 400 mil pares sorteados, com os MESMOS hiperparâmetros: uma
variável, a base.

O que estes testes fixam:

  1. **os hiperparâmetros são os do T1a e não se mexem.** Cada um deles é uma
     variável que não pode mudar junto com a base — o `--lote` define quantos
     negativos o InfoNCE vê, e mudá-lo mudaria a dificuldade da tarefa;
  2. **o dataset é o do T1a, conferido pelo nome do experimento.** O braço de
     referência treinou nestes bytes; com outro pacote mudariam base *e* dado;
  3. **a métrica primária está declarada E justificada.** Ela mudou de significado
     quando o T1e tirou o ΦRank do sistema: nada lê o fundo hoje, então o r@200 —
     onde o GTE perde — virou diagnóstico. Declarar isso depois do resultado seria
     escolher a métrica que dá a resposta desejada;
  4. **os três desfechos escritos antes**, e nenhum deles é "troca a base": o GTE
     custa 4,4× para embutir, e essa conta é permanente;
  5. **a célula treina UM braço e para.** A comparação é local, com o avaliador do
     G1 nos dois checkpoints na mesma sessão.

⚠️ Nada aqui lê `models/` nem `data/processed/`, que são gitignored.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "kaggle"))

import t1f_base_gte  # noqa: E402

from conftest import so_codigo  # noqa: E402
from phifm.core.kaggle import obter  # noqa: E402

CELULA = t1f_base_gte.CELULA
CODIGO = so_codigo(CELULA)
REGRA = " ".join(
    CELULA[CELULA.index("A REGRA, escrita ANTES"):CELULA.index("# ── 5.")].split())


def test_a_celula_e_python_valido():
    """O extrator do publicador é um regex sobre este arquivo; se o formato
    mudar, o notebook gerado sai vazio e a descoberta custa a sessão."""
    ast.parse(CELULA)


def test_os_hiperparametros_sao_os_do_T1A_e_estao_em_CONSTANTES():
    """⚠️ Cada um é uma variável que não pode mudar junto com a base.

    O `--lote 128` define os 127 negativos do InfoNCE, e mudá-lo mudaria a
    dificuldade da tarefa. O `--max-tokens 192` é o do campeão — e o T1e mediu que
    ler 256 ou 384 não muda o recall, então 192 não deixa nada na mesa. A semente
    17 governa o sorteio dos pares E o pool de avaliação.
    """
    esperado = {"LOTE": 128, "MAX_TOKENS": 192, "SEMENTE": 17,
                "N_CANDIDATOS": 1000, "PASSOS_AVAL": 200}
    achado = {}
    for no in ast.walk(ast.parse(CELULA)):
        if not isinstance(no, ast.Assign):
            continue
        for alvo in no.targets:
            if isinstance(alvo, ast.Name) and alvo.id in esperado:
                achado[alvo.id] = ast.literal_eval(no.value)
    assert achado == esperado, (
        f"os hiperparâmetros divergem dos do T1a: {achado} contra {esperado}. "
        "Mudar qualquer um deles junto com a base deixaria o resultado sem dono.")


def test_a_base_e_o_gte_base_e_esta_numa_constante():
    arvore = ast.parse(CELULA)
    base = next(ast.literal_eval(n.value) for n in ast.walk(arvore)
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "BASE"
                        for t in n.targets))
    assert base == "thenlper/gte-base", base


def test_a_celula_EXIGE_que_o_dataset_seja_o_do_t1a():
    """⚠️ O reúso é a PREMISSA do experimento, não economia de banda.

    O braço de referência (`phiemb-minilm-t4-sorteado-melhor`) treinou nestes bytes
    exatos. Um dataset novo, mesmo com a mesma semente e o mesmo volume, mudaria
    base *e* dado na mesma rodada.
    """
    i_conf = CELULA.index('assert man["experimento"] == "t1a"')
    i_treino = CELULA.index("# ── 5.")
    assert i_conf < i_treino, "a conferência do dataset está depois do treino"
    assert "mudaria base E dado" in CELULA


def test_a_metrica_primaria_esta_declarada_E_justificada():
    """⚠️ Ela MUDOU de significado quando o T1e tirou o ΦRank do sistema.

    O GTE-base zero-shot empata no top-10 e perde fundo (r@200 0,645 contra
    0,730). Até o T1e isso seria decisivo — a cadeia era ΦEmb → RRF → ΦRank sobre
    um pool profundo. Agora a cadeia é ΦEmb → top-10 e nada lê o fundo, então o
    r@200 é diagnóstico. Declarar isso DEPOIS do resultado seria escolher a
    métrica que dá a resposta desejada.
    """
    assert "MEDIDA PRIMÁRIA: nDCG@10" in REGRA
    assert "NADA lê o fundo hoje" in REGRA
    assert "r@200" in REGRA
    assert "diagnóstico" in REGRA
    i_regra = CELULA.index("A REGRA, escrita ANTES")
    assert i_regra < CELULA.index("# ── 5."), "a regra está depois do treino"


def test_os_TRES_desfechos_estao_escritos_antes_do_numero():
    for desfecho in ("GTE VENCE", "EMPATE", "MiniLM VENCE"):
        assert desfecho in REGRA, f"falta o desfecho {desfecho}"


def test_vitoria_do_GTE_nao_e_automaticamente_troca_de_base():
    """⚠️ O custo de serviço é permanente: 954 s contra 217 s para embutir o mesmo
    universo, 4,4×. A margem tem de pagar isso para sempre, e essa decisão é do
    dono do projeto — não deste número."""
    assert "4,4x" in REGRA or "4,4×" in REGRA
    assert "NÃO é" in REGRA and "troca de base" in REGRA
    assert "954" in CELULA and "217" in CELULA


def test_a_regra_diz_que_isto_e_uma_SONDA_a_400_mil():
    """⚠️ O recuperador do sistema treinou em 6 M. A ordem entre duas bases a 400
    mil pode não sobreviver a 6 M — bases maiores costumam precisar de mais dado
    para se separar. Sem esta linha, um empate aqui viraria "as bases empatam"."""
    assert "SONDA" in REGRA
    assert "6 M" in REGRA
    assert "não dá para ver" in REGRA
    assert "39 h" in REGRA or "39 h" in CELULA


def test_a_celula_treina_UM_braco_e_para():
    """A referência não é retreinada: mesma T4, mesmo sorteio, mesmos 400 mil
    pares, já no disco. Retreinar gastaria 36 min de cota para reproduzir o que
    está lá — e a comparação acontece local, no mesmo avaliador."""
    chamados = [ast.unparse(n.args[0]) for n in ast.walk(ast.parse(CELULA))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "_rodar" and n.args]
    assert len(chamados) == 1, f"esperava um treino só, achei {chamados}"
    assert "train_embedding" in chamados[0]
    assert "A COMPARAÇÃO NÃO ACONTECE AQUI" in CELULA


def test_o_pool_INTERNO_do_treino_nao_e_confundido_com_o_do_G1():
    """⚠️ O `ndcg_10` do `melhor.json` sai do pool interno (n=1000), e o do G1 sai
    de um pool DESDUPLICADO com teto 1,0. São duas réguas, e comparar este número
    com o do MiniLM seria o erro do T1b2 outra vez."""
    assert "pool INTERNO" in CELULA
    assert "duas réguas" in CELULA


def test_o_plano_B_de_memoria_preserva_os_NEGATIVOS():
    """O GTE-base tem 109 M contra os 23 M do MiniLM. Se estourar, a saída é
    `--sub-lote --sem-amp`: o GradCache preserva o lote LÓGICO, então continuam
    127 negativos e o que muda é fp16→fp32. Reduzir o `--lote` mudaria a
    dificuldade da tarefa junto com a base."""
    assert "--sub-lote" in CELULA and "--sem-amp" in CELULA
    assert "lote LÓGICO" in CELULA
    assert "109 M" in CELULA


def test_confere_hash_e_assinatura_ANTES_de_treinar():
    i_hash = CELULA.index("conferidos por blake3")
    i_assin = CELULA.index("assinatura do bundle confere")
    i_treino = CELULA.index("# ── 5.")
    assert i_hash < i_treino and i_assin < i_treino


def test_exige_gpu_em_vez_de_cair_para_cpu():
    assert 'assert torch.cuda.is_available(), "sem GPU' in CELULA


def test_treino_que_falha_derruba_o_notebook():
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


def test_a_ausencia_do_melhor_derruba_a_celula():
    """Sem `-melhor` não há o que baixar, e a sessão foi gasta. Descobrir isso
    lendo o log depois é descobrir tarde."""
    assert "assert melhor.exists()" in CELULA
    assert "a sessão foi gasta" in CELULA


def test_o_codigo_vem_do_GITHUB_num_sha():
    assert "codeload.github.com" in CELULA
    assert "__SHA__" in CELULA
    assert "assinatura_do_manifesto" in CODIGO


# ── o registro ──────────────────────────────────────────────────────────────

def test_o_registro_declara_o_t1f_REUSANDO_o_dataset_do_t1a():
    e, dono = obter("t1f"), obter("t1a")
    e.conferir()
    assert e.reusa_dados_de == "t1a"
    assert e.slug_dados == dono.slug_dados
    assert e.pacote == dono.pacote
    assert e.slug_notebook != dono.slug_notebook
    assert e.fonte_celula == "kaggle/t1f_base_gte.py"
    assert e.max_pares == dono.max_pares == 400_000, (
        "o volume tem de ser o mesmo do braço de referência")


def test_o_t1f_nao_sobe_modelo_nenhum():
    """O `gte-base` é público e o Kaggle o baixa do HuggingFace. E o braço de
    REFERÊNCIA não vai porque não é retreinado lá — ele fica em casa, que é onde
    a comparação acontece."""
    assert obter("t1f").modelos == ()


def test_o_publicador_RECUSA_publicar_o_dataset_de_quem_reusa():
    fonte = (RAIZ / "scripts" / "publicar_kaggle.py").read_text(encoding="utf-8")
    assert "reusa_dados_de" in fonte
    assert "--so-notebook" in fonte
