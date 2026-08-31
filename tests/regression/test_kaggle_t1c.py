"""O T1c é um experimento de UMA variável, e estes testes é que garantem isso.

O T1c pergunta se o reranqueador precisa de uma base com pré-treino diferente do
recuperador. A resposta só significa algo se **nada mais** mudar junto: este
repositório já perdeu um experimento inteiro por trocar base e lote na mesma corrida
e não saber qual dos dois explicava o resultado.

Então os testes fixam três coisas que, se escorregarem, transformam o resultado em
número sem valor:

  1. os hiperparâmetros são byte a byte os do controle — só `--base` muda;
  2. a regra de decisão está escrita ANTES de qualquer número, com o limiar de
     Bonferroni para duas variantes;
  3. o controle é REAVALIADO no mesmo protocolo, e não copiado do JSON antigo.

⚠️ Nada aqui lê `models/` nem `data/processed/`, que são gitignored. Um teste que
depende de artefato não versionado passa nesta máquina e derruba o CI — foi
exatamente o que aconteceu em 2026-08-27, e o commit que documentava a lição foi o
que a repetiu.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "kaggle"))

import t1c_phirank  # noqa: E402

CELULA = t1c_phirank.CELULA
DOC = t1c_phirank.__doc__ or ""

# ⚠️ Os hiperparâmetros do CONTROLE, copiados de `phirank.json` do
# `phirank-rrf-melhor` (treinado em 2026-08-24, acerto@1 0,498). Estão aqui como
# literais e não lidos do arquivo porque `models/` é gitignored — ler dali faria
# este teste passar aqui e falhar no CI.
CONTROLE = {"--max-grupos": "12500", "--grupos": "2", "--n-negativos": "7",
            "--lr": "2e-5", "--max-tokens": "384", "--semente": "17"}


def test_a_celula_e_python_valido():
    """O extrator do publicador é um regex sobre este arquivo; se o formato mudar,
    o notebook gerado sai vazio e a descoberta custa uma sessão de GPU."""
    import ast

    ast.parse(CELULA)


def test_chama_os_scripts_em_vez_de_reimplementar():
    """Dois laços de treino no projeto divergem em silêncio: os dois rodam."""
    assert "scripts/train_rerank.py" in CELULA
    assert "scripts/avaliar_t1b.py" in CELULA
    for sinal in ("backward()", "cross_entropy", "for passo", "AutoModelFor"):
        assert sinal not in CELULA, (
            f"a célula parece reimplementar treino ou avaliação ({sinal!r})")


def test_so_a_base_muda():
    """A asserção central deste arquivo.

    Se alguém "otimizar" o lote porque a T4 comporta mais, o experimento passa a
    medir base E lote ao mesmo tempo, e nenhum dos dois fica atribuível.
    """
    for bandeira, valor in CONTROLE.items():
        assert f'"{bandeira}", {valor}' in CELULA or \
               f'"{bandeira}", {valor.rstrip("0")}' in CELULA or \
               f"{bandeira}\", {valor}" in CELULA, (
            f"{bandeira} não está fixado em {valor}, que é o valor do controle. "
            "Mudar isto junto com a base faria o experimento medir duas coisas.")


def test_os_hiperparametros_aparecem_uma_vez_cada():
    """Duas ocorrências da mesma bandeira significam duas variantes com valores
    diferentes — e aí `--base` não é mais a única coisa que muda."""
    for bandeira in CONTROLE:
        assert CELULA.count(f'"{bandeira}"') == 1, (
            f"{bandeira} aparece {CELULA.count(chr(34) + bandeira + chr(34))} vezes; "
            "com mais de uma, as variantes não são comparáveis entre si")


def test_a_regra_de_decisao_vem_antes_dos_numeros():
    """Escolher o limiar depois de ver o p é o modo mais fácil de mentir para si.

    A regra tem de estar impressa antes da primeira chamada a `avaliar`, e o limiar
    de Bonferroni tem de ser explícito: duas variantes são dois testes.
    """
    assert "ALFA_BONFERRONI = 0.025" in CELULA, (
        "duas variantes são dois testes; 0,05 sem correção infla a taxa de falso "
        "positivo para ~0,10")
    i_regra = CELULA.index("ALFA_BONFERRONI")
    i_primeira_medida = CELULA.index('resultados["minilm (controle)"]')
    assert i_regra < i_primeira_medida, (
        "a regra de decisão tem de ser impressa ANTES de qualquer medição")
    # E a leitura de cada desfecho possível também é pré-registrada, incluindo o
    # desfecho em que a hipótese cai.
    for leitura in ("diversidade", "domínio", "capacidade", "NENHUMA"):
        assert leitura in CELULA, f"falta a leitura pré-registrada de {leitura!r}"


def test_o_desfecho_negativo_esta_previsto_e_nomeado():
    """Um pré-registro que só descreve o sucesso não é pré-registro.

    Se nenhuma variante vencer, a conclusão já está escrita: a redundância
    informacional não é a restrição que manda, e o próximo lugar a olhar é outro.
    """
    assert "não é a restrição" in CELULA
    assert "objetivo de treino" in CELULA


def test_o_controle_e_reavaliado_e_nao_copiado():
    """O número antigo do ΦRank saiu de 1.000 consultas e deu p=0,118.

    Comparar uma variante nova em 2.000 consultas contra o controle em 1.000
    misturaria tamanho de efeito com poder estatístico.
    """
    assert "N_CONSULTAS = 2000" in CELULA
    assert 'avaliar("controle", CONTROLE)' in CELULA
    # Uma constante única para as três avaliações: dois números diferentes aqui
    # reintroduziriam exatamente o problema.
    assert CELULA.count("N_CONSULTAS = ") == 1
    assert "1000" not in CELULA.split("N_CONSULTAS = 2000")[1].split("def ")[0]


def test_o_controle_roda_primeiro_e_diz_por_que():
    """23 M avaliam em ~12 min; 109 M treinam em ~35. Um encanamento quebrado
    descoberto pelo caminho barato custa 12 min, não 45."""
    i_controle = CELULA.index('avaliar("controle"')
    i_variantes = CELULA.index("for nome, base in BASES.items()")
    assert i_controle < i_variantes
    assert "mais barata" in DOC or "mais barato" in DOC


def test_uma_variante_que_morre_nao_leva_a_outra():
    """Um id de modelo errado ou um OOM custaria a sessão inteira, e a variante que
    já treinou continua sendo um resultado."""
    assert "except BaseException as exc:" in CELULA
    assert "falhas[nome]" in CELULA
    # E a falha tem de APARECER na tabela final, não sumir num except silencioso.
    assert "FALHOU" in CELULA


def test_confere_hash_antes_de_treinar():
    """Upload truncado produz um número que parece comparável e não é."""
    assert "MANIFESTO.json" in CELULA
    assert "hash difere" in CELULA
    assert CELULA.index("hash difere") < CELULA.index("scripts/train_rerank.py")


def test_exige_gpu_em_vez_de_cair_para_cpu():
    assert "assert torch.cuda.is_available()" in CELULA


def test_treino_que_falha_derruba_o_notebook():
    """`COMPLETE` tem de significar que treinou.

    Medido em 2026-08-26: `check=False` transformou um treino morto num notebook
    COMPLETE, com `codigo/` na saída e NENHUM modelo.
    """
    assert "if r.returncode != 0:" in CELULA
    assert "raise SystemExit" in CELULA
    # ⚠️ Precisão importa duas vezes aqui, e eu errei as duas na primeira tentativa.
    #
    # 1. Há UM `check=False` legítimo na célula, no `pip install blake3`, que tem
    #    fallback documentado para sha256. Proibir a string inteira reprovaria o
    #    fallback.
    # 2. E procurar a string no TEXTO de `_rodar` também reprova: a docstring dele
    #    explica por que o `check=False` de antes foi um erro. É o mesmo tropeço do
    #    `test_kaggle_t1a.py`, que buscava "InfoNCE" e o achava na explicação de por
    #    que não usar DataParallel — um teste que confunde o código com o comentário
    #    reprova o comentário por ser bom.
    #
    # Então a checagem é na ÁRVORE: toda `subprocess.run` que não seja um pip install
    # tem de deixar o código de saída visível.
    import ast

    arvore = ast.parse(CELULA)
    rodadas = [n for n in ast.walk(arvore)
               if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute) and n.func.attr == "run"]
    assert rodadas, "nenhuma chamada a subprocess.run — a célula não roda nada"
    for n in rodadas:
        fonte = ast.dump(n)
        if "'pip'" in fonte or '"pip"' in fonte:
            continue  # instalar blake3 pode falhar; há fallback registrado
        engole = [k for k in n.keywords
                  if k.arg == "check" and isinstance(k.value, ast.Constant)
                  and k.value.value is False]
        assert not engole, (
            "uma subprocess.run de trabalho voltou a engolir o código de saída com "
            "check=False; foi assim que um treino morto passou por COMPLETE")


def test_o_log_vai_para_arquivo_e_o_subprocesso_recebe_pythonpath():
    """A API do Kaggle devolveu log de 0 bytes em três execuções seguidas, e o
    `sys.path` da célula não vale para o processo filho."""
    assert "stderr=subprocess.STDOUT" in CELULA
    assert '"PYTHONPATH": str(CODIGO)' in CELULA
    assert "env=AMBIENTE" in CELULA


def test_as_bases_sao_as_duas_que_separam_os_mecanismos():
    """Uma base só não separaria "diversidade" de "domínio"."""
    assert "thenlper/gte-base" in CELULA
    assert "thellert/physbert_cased" in CELULA
    assert "all-MiniLM" not in CELULA.split("BASES = ")[1].split("\n")[0], (
        "a base do ΦEmb não pode estar entre as variantes — ela é o CONTROLE, e é "
        "justamente a redundância que o experimento testa")


def test_a_ressalva_do_lr_esta_escrita():
    """O `lr 2e-5` não foi re-ajustado para 109 M de parâmetros.

    Se as duas variantes falharem, "a taxa estava errada para este tamanho" segue
    sendo explicação viva — e dizer isso ANTES é o que separa uma ressalva de uma
    desculpa.
    """
    assert "não foi re-ajustado" in DOC or "NÃO foi re-ajustado" in DOC
    assert "ressalva" in CELULA


def test_o_teto_do_reranker_e_registrado():
    """recall@50 de 0,446 contra nDCG@10 de 0,1584 é a folga que justifica o
    experimento. Sem o teto no JSON, um ganho não se interpreta."""
    assert "teto_recall" in CELULA
    assert "teto_do_reranker" in CELULA


# ─── o registro e o empacotador ──────────────────────────────────────────────


def test_o_registro_declara_o_t1c_coerente_com_a_celula():
    from phifm.core.kaggle import T1C, obter

    assert obter("t1c") is T1C
    for nome in T1C.scripts:
        assert f"scripts/{nome}" in CELULA, (
            f"o registro zipa {nome} e a célula não o chama — ou o contrário")
    for arq in T1C.arquivos:
        if arq.endswith(".parquet"):
            assert arq in CELULA, f"a célula não usa {arq}, que o pacote carrega"


def test_o_t1c_carrega_o_emb_do_t1b_e_nao_o_do_t4():
    """Trocar o recuperador junto com o reranqueador mediria duas coisas.

    O nDCG 0,1584 de referência foi medido com `phiemb-minilm-melhor`.
    """
    from phifm.core.kaggle import T1C

    assert "models/phiemb-minilm-melhor" in T1C.modelos
    assert not any("t4" in m for m in T1C.modelos), (
        "o ΦEmb do T4 é outro treino; usá-lo aqui trocaria o recuperador")
    assert "models/phirank-rrf-melhor" in T1C.modelos, "falta o controle"


def test_zipar_modelos_poe_o_nome_final_na_raiz_e_deixa_o_estado_fora(tmp_path):
    """O notebook busca `MODELOS / "phiemb-minilm-melhor"`, e `estado_rank.pt` são
    centenas de MB de estado de otimizador que ninguém lê na inferência."""
    from empacotar_kaggle import _zipar_modelos

    d = tmp_path / "models" / "phiemb-minilm-melhor"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"pesos")
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "estado_rank.pt").write_bytes(b"x" * 1000)

    destino = tmp_path / "modelos.zip.bin"
    n = _zipar_modelos(tmp_path, destino, ("models/phiemb-minilm-melhor",))
    with zipfile.ZipFile(destino) as z:
        nomes = z.namelist()
    assert n == 2
    assert "phiemb-minilm-melhor/model.safetensors" in nomes
    assert not any(n_.startswith("models/") for n_ in nomes), (
        "o prefixo `models/` vazou — o notebook procura o nome final na raiz")
    assert not any(n_.endswith(".pt") for n_ in nomes), (
        "estado de otimizador entrou no pacote")


def test_zipar_modelos_quebra_alto_se_o_modelo_nao_existir(tmp_path):
    """Um pacote sem os pesos sobe inteiro e morre no assert do notebook, longe daqui."""
    import pytest

    from empacotar_kaggle import _zipar_modelos

    with pytest.raises(SystemExit, match="não é um diretório"):
        _zipar_modelos(tmp_path, tmp_path / "z.bin", ("models/nao-existe",))


def test_zipar_fonte_quebra_alto_se_um_script_declarado_faltar(tmp_path):
    """O registro declara os scripts; um nome errado daria FileNotFoundError na GPU,
    depois do upload inteiro."""
    import pytest

    from empacotar_kaggle import _zipar_fonte

    with pytest.raises(SystemExit, match="não existe"):
        _zipar_fonte(RAIZ, tmp_path / "z.bin", ("script_que_nao_existe.py",))


def test_o_publicador_gera_o_notebook_do_t1c():
    """O extrator, o gerador de `.ipynb` e o registro, no caminho de verdade."""
    import ast
    import importlib.util

    from phifm.core.kaggle import T1C

    spec = importlib.util.spec_from_file_location(
        "publicar_kaggle", RAIZ / "scripts/publicar_kaggle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    codigo = "".join(
        mod._ipynb(mod._celula(RAIZ / T1C.fonte_celula))["cells"][0]["source"])
    ast.parse(codigo)
    assert "scripts/train_rerank.py" in codigo
    assert "ALFA_BONFERRONI" in codigo
