"""O T1d é um experimento de UMA variável, e estes testes é que garantem isso.

O T1d pergunta se treinar o ΦRank nos negativos da composição que ele **vê**
(top-50 do ΦEmb, decidida em 2026-09-08) recupera o ganho marginal que encolheu de
p=0,0081 para p=0,086. A resposta só significa algo se nada mais mudar junto.

O que estes testes fixam:

  1. **`--max-grupos` é o do T1c.** Os negativos novos têm 16.391 grupos e o T1c
     usou 12.500 — subir o limite mudaria distribuição *E* volume na mesma rodada,
     e o ganho não seria atribuível. Este é o eixo em que o T1d escorregaria;
  2. **os dois reranqueadores na MESMA sessão.** A pergunta é "quanto mudou", e em
     2026-09-08 comparar contra o número histórico teria reportado SETE VEZES o
     efeito real da troca do recuperador;
  3. **o pareado que a regra nomeia é calculado pela célula.** No T1b2 a regra
     pedia um confronto entre cadeias, o run mediu as duas e o teste não pôde ser
     feito depois — as posições não eram gravadas. Aqui saem do artefato e o
     `mcnemar_em` roda na célula;
  4. **`--sem-fusao`**, sem o qual dois braços não caberiam numa sessão;
  5. **os três desfechos escritos antes do número**, o de refutação incluído.

⚠️ Nada aqui lê `models/` nem `data/processed/`, que são gitignored. Um teste que
depende de artefato não versionado passa nesta máquina e derruba o CI.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "kaggle"))

import t1d_phirank_denso  # noqa: E402

from conftest import so_codigo  # noqa: E402
from phifm.core.kaggle import obter  # noqa: E402

CELULA = t1d_phirank_denso.CELULA
DOC = t1d_phirank_denso.__doc__ or ""
CODIGO = so_codigo(CELULA)

# ⚠️ Os hiperparâmetros do T1c, como LITERAIS. Lidos de `models/` fariam este
# teste passar aqui e falhar no CI — foi o que aconteceu em 2026-08-27.
T1C = {"--max-grupos": "12500", "--grupos": "2", "--n-negativos": "7",
       "--lr": "2e-5", "--max-tokens": "384", "--semente": "17"}


def test_a_celula_e_python_valido():
    """O extrator do publicador é um regex sobre este arquivo; se o formato
    mudar, o notebook gerado sai vazio e a descoberta custa uma sessão de GPU."""
    ast.parse(CELULA)


def test_SO_a_distribuicao_dos_negativos_muda(tmp_path=None):
    """⚠️ A asserção central: `--max-grupos` continua 12.500.

    Os negativos densos têm 16.391 grupos e o T1c usou 12.500. Subir o limite
    mudaria distribuição E volume na mesma rodada — o ganho deixaria de ser
    atribuível à distribuição, que é a hipótese sob teste.
    """
    for chave, valor in T1C.items():
        assert f'"{chave}", {valor}' in CELULA or f'"{chave}", {valor},' in CELULA, (
            f"{chave} não aparece com o valor {valor} do T1c")
    # E o HIPER é uma lista única, não valores espalhados por várias chamadas.
    assert CODIGO.count("HIPER = [") == 1
    assert "*HIPER" in CODIGO, "os hiperparâmetros não são reusados de HIPER"


def test_os_2102_grupos_de_fora_estao_DECLARADOS():
    """Deixar dado de treino de fora é uma escolha, e uma escolha não declarada
    parece um descuido para quem lê o resultado depois."""
    assert "2.102" in CELULA
    assert "16.391" in CELULA
    for texto in (DOC, CELULA):
        assert "volume" in texto


def test_os_DOIS_reranqueadores_na_mesma_sessao():
    """A pergunta é "quanto mudou", e em 2026-09-08 comparar contra o número
    histórico teria reportado SETE VEZES o efeito real da troca do recuperador."""
    # Por AST: `so_codigo` normaliza aspas no `ast.unparse`, entao uma
    # assercao por texto sobre o dicionario quebraria sozinha.
    arvore = ast.parse(CELULA)
    bracos = next(
        (n.value for n in ast.walk(arvore)
         if isinstance(n, ast.Assign)
         and any(getattr(x, "id", None) == "BRACOS" for x in n.targets)), None)
    assert isinstance(bracos, ast.Dict), "BRACOS nao e um dicionario literal"
    chaves = {k.value for k in bracos.keys}
    assert chaves == {"novo", "antigo"}, chaves
    valores = {getattr(v, "id", None) for v in bracos.values}
    assert valores == {"NOVO", "ANTIGO"}, valores
    assert "for nome, rank in BRACOS.items()" in CODIGO
    assert "sete vezes" in DOC.lower() or "SETE VEZES" in CELULA


def test_o_pareado_da_REGRA_e_calculado_pela_celula():
    """⚠️ No T1b2 a regra pedia um confronto entre cadeias, o run mediu as duas e
    o teste não pôde ser feito depois: as posições não eram gravadas.

    Aqui o `mcnemar_em` roda na célula, sobre as `posicoes` do artefato — o
    veredito sai da execução, não de uma conta feita depois com o que sobrou.
    """
    assert "from phifm.eval.hibrido import mcnemar_em" in CODIGO
    arvore = ast.parse(CELULA)
    chamadas = [n for n in ast.walk(arvore)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "mcnemar_em"]
    assert chamadas, "a célula não roda nenhum McNemar"
    # E os dois lados são os DOIS BRAÇOS, não um braço contra a referência
    # interna dele.
    fonte = ast.unparse(chamadas[0])
    assert "antigo" in fonte and "novo" in fonte, fonte
    assert "_posicoes(" in CODIGO, "o confronto não lê as posições do artefato"


def test_o_confronto_sai_nos_DOIS_k():
    """k=1 e k=10. Só o k=10 esconderia uma troca que melhora o topo e piora o
    resto, ou o contrário."""
    assert "for k in (1, 10)" in CODIGO


def test_usa_sem_fusao_e_diz_por_que():
    """Sem a chave, cada braço reordena duas vezes e dois braços não cabem numa
    sessão — e a composição extra é a que a regra do T1b2 já descartou."""
    assert '"--sem-fusao"' in CELULA
    assert "uma" in DOC and "duas" in DOC


def test_os_TRES_desfechos_estao_escritos_antes_do_numero():
    """Incluindo a refutação. Um desfecho não previsto vira interpretação depois
    do fato, que é como um resultado nulo se transforma em "quase deu"."""
    i_regra = CELULA.index("REGRA, no pareado")
    i_treino = CELULA.index("5. Treinar o novo")
    assert i_regra < i_treino, "a regra está depois do treino"
    trecho = CELULA[i_regra:i_treino]
    assert "NOVO VENCE" in trecho
    assert "EMPATE" in trecho
    assert "NOVO PERDE" in trecho
    assert "REFUTADA" in trecho
    # ⚠️ O empate NÃO instala: trocar peça do sistema sem evidência é risco sem
    # retorno, e a assimetria tem de estar escrita.
    assert "NÃO substitui" in trecho


def test_a_segunda_leitura_e_INDEPENDENTE_do_confronto():
    """Se nenhuma das duas cadeias vencer o ΦEmb sozinho, o estágio de
    reordenação não paga o próprio custo — e isso vale seja qual for o vencedor
    do confronto entre elas."""
    assert "INDEPENDENTE" in CELULA
    assert "não paga o próprio" in CELULA


def test_confere_hash_ANTES_de_treinar():
    """Treinar sobre upload truncado gastaria a sessão para um número
    incomparável."""
    i_hash = CELULA.index("conferidos por blake3")
    assert i_hash < CELULA.index("train_rerank.py")


def test_exige_gpu_em_vez_de_cair_para_cpu():
    """Em CPU o treino levaria dias e o notebook morreria no limite de sessão,
    depois de horas."""
    assert 'assert torch.cuda.is_available(), "sem GPU' in CELULA


def test_treino_que_falha_derruba_o_notebook():
    """⚠️ Asserção sobre COMPORTAMENTO, por AST: um treino que sai com código
    diferente de zero tem de levantar, não seguir para a avaliação de um modelo
    que não existe.

    A versão por texto deste teste quebrou três vezes em refatorações legítimas.
    """
    arvore = ast.parse(CELULA)
    achou = False
    for no in ast.walk(arvore):
        if not isinstance(no, ast.If):
            continue
        if not (isinstance(no.test, ast.Compare)
                and any(isinstance(c, ast.NotEq) for c in no.test.ops)):
            continue
        if any(isinstance(x, ast.Raise) for x in ast.walk(no)):
            achou = True
    assert achou, "nenhum `if <codigo> != 0` levanta na célula"
    # E o melhor checkpoint é conferido antes de avaliar.
    assert 'assert (NOVO / "model.safetensors").exists()' in CELULA


def test_o_log_vai_para_arquivo_E_stdout():
    """Mandar só para o arquivo custou 33 h de cegueira em 2026-09-07: uma
    execução saudável ficava indistinguível de uma travada."""
    assert "bufsize=1" in CODIGO
    assert "fh.write(linha)" in CODIGO
    assert "print(linha, end=" in CODIGO


def test_o_codigo_vem_do_GITHUB_num_sha():
    """O Kaggle FIXA a versão do dataset no anexo e `kernels push` não
    re-resolve: em 2026-09-03 um notebook rodou 15 min sobre código antigo."""
    assert "codeload.github.com" in CELULA
    assert "__SHA__" in CELULA
    assert "assinatura_do_manifesto" in CODIGO
    assert "ASSINATURA_ESPERADA" in CODIGO


def test_o_registro_declara_o_t1d_coerente_com_a_celula():
    e = obter("t1d")
    e.conferir()
    assert e.fonte_celula == "kaggle/t1d_phirank_denso.py"
    assert "pares_do_recuperador_denso_limpos.parquet" in e.arquivos
    assert "train_rerank.py" in e.scripts and "avaliar_t1b.py" in e.scripts
    # ⚠️ O ΦRank ANTIGO tem de viajar: é o braço de referência.
    assert "models/phirank-physbert-melhor" in e.modelos
    assert "models/phiemb-do-sistema" in e.modelos
    assert e.slug_dados != e.slug_notebook, (
        "com o mesmo slug nos dois, um erro de digitação aponta para o objeto "
        "errado sem avisar")
    # Os negativos vão NOMEADOS na célula, porque são a variável do experimento.
    assert "pares_do_recuperador_denso_limpos.parquet" in CELULA
    # ⚠️ O `pares_validacao.parquet` NÃO aparece por nome, e está certo: o
    # `avaliar_t1b.py` o lê de `--pares DADOS`, o diretório. Exigir o nome aqui
    # foi a primeira versão deste teste, e ela estava errada — o arquivo é usado,
    # só não é nomeado. O que importa é o diretório chegar ao avaliador.
    assert '"--pares", DADOS' in CELULA


def test_a_base_e_a_que_o_T1c_elegeu():
    """PhysBERT é a única das três bases medidas que bate a fusão (p=0,0062).
    Trocar a base junto com os negativos mediria duas coisas."""
    assert 'BASE = "thellert/physbert_cased"' in CELULA
    assert "gte" not in CODIGO.lower(), "outra base entrou na rodada"


def test_o_arquivo_de_negativos_pertence_a_IDENTIDADE_do_experimento():
    """⚠️ A guarda contra o mispackage invisível.

    O `--negativos` do empacotador tinha default fixo apontando para os negativos
    da FUSÃO. Montar o `t1d` sem a bandeira empacotaria os antigos sob o nome do
    experimento novo, e o manifesto atestaria o arquivo errado com a cara certa.

    No T1d isso seria fatal e invisível: a hipótese sob teste É a distribuição
    dos negativos. O pacote diria "densos" e conteria os da fusão, o resultado
    sairia igual ao do T1c, e a leitura seria "a distribuição não importa".
    """
    t1c, t1d = obter("t1c"), obter("t1d")
    assert t1c.negativos and t1d.negativos, "algum dos dois não declara negativos"
    assert t1c.negativos != t1d.negativos, (
        "os dois experimentos apontam para o MESMO arquivo de negativos; o t1d "
        "existe justamente para testar outros")
    assert "denso" in t1d.negativos
    assert "denso" not in t1c.negativos
    # E o arquivo declarado é o que a lista de `arquivos` promete.
    assert Path(t1d.negativos).name in t1d.arquivos
    assert Path(t1c.negativos).name in t1c.arquivos


def test_o_empacotador_nao_tem_default_fixo_de_negativos():
    """Um default fixo é o que fazia o t1d empacotar o arquivo do t1c. A
    verificação é sobre a INTERFACE: o valor tem de vir do experimento."""
    import subprocess

    ajuda = subprocess.run(
        [sys.executable, str(RAIZ / "scripts" / "empacotar_kaggle.py"), "--help"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(RAIZ)).stdout
    assert "--negativos" in ajuda
    assert "pares_do_recuperador_limpos.parquet" not in ajuda, (
        "a ajuda ainda anuncia um arquivo de negativos como default")
    assert "declarado pelo experimento" in ajuda
