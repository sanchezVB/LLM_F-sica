"""O notebook do Kaggle não pode reimplementar o treino.

A tentação, num notebook, é colar o laço de treino direto na célula — é mais rápido
de escrever e roda. O custo aparece depois: dois laços de treino no projeto,
divergindo em silêncio. Os dois rodariam, dariam números diferentes, e nada
apontaria qual está certo.

Então o notebook chama `scripts/train_embedding.py` a partir de um ZIP do pacote, e
estes testes fixam isso — junto com as três coisas que, se faltarem, transformam um
resultado do Kaggle em número sem valor:

  1. a conferência de hash dos dados de entrada;
  2. a exigência de GPU (rodar em CPU levaria dias e "funcionaria");
  3. o aviso de que 1.000 candidatos não é o protocolo do veredito.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

# ⚠️ A CÉLULA e a DOCSTRING são conferidas separadamente.
#
# A primeira versão deste arquivo lia o `.py` inteiro numa string e procurava
# "InfoNCE" nela — e encontrou, na docstring que explica POR QUE não usar
# `DataParallel`. Um teste que confunde o código com a explicação do código reprova
# a explicação por ser boa.
sys.path.insert(0, str(RAIZ / "kaggle"))
import t1a_phiemb  # noqa: E402

CELULA = t1a_phiemb.CELULA
DOC = t1a_phiemb.__doc__ or ""


def test_notebook_chama_o_treino_em_vez_de_reimplementar():
    """A asserção central deste arquivo.

    Se alguém colar o laço aqui, o projeto passa a ter dois — e a divergência entre
    eles é invisível, porque os dois rodam.
    """
    assert "scripts/train_embedding.py" in CELULA
    for sinal in ("for passo", "backward()", "cross_entropy", "InfoNCE"):
        assert sinal not in CELULA, (
            f"a célula parece reimplementar o treino ({sinal!r}). Ela deve CHAMAR "
            "`train_embedding.py` a partir do ZIP do pacote.")


def test_confere_hash_antes_de_treinar():
    """Upload truncado produz um número que parece comparável e não é.

    O Kaggle não garante nada sobre o conteúdo do input — só que existe.
    """
    assert "MANIFESTO.json" in CELULA
    assert "hash difere" in CELULA
    i_hash = CELULA.index("hash difere")
    i_treino = CELULA.index("train_embedding.py")
    assert i_hash < i_treino, "a conferência de hash tem de vir ANTES do treino"


def test_exige_gpu_em_vez_de_cair_para_cpu():
    """Um treino que "funciona" em CPU por 40 h é pior que um que não começa."""
    assert "torch.cuda.is_available()" in CELULA
    assert "assert torch.cuda.is_available()" in CELULA


def test_avisa_que_mil_candidatos_nao_e_o_veredito():
    """O protocolo do G1 é 2.000 candidatos.

    Este projeto já elegeu um campeão errado por comparar métricas de protocolos
    diferentes. O aviso existe para o número do log do Kaggle não virar alegação.
    """
    assert "2.000 candidatos" in CELULA
    assert "NÃO é comparável" in CELULA


def test_usa_uma_gpu_e_diz_por_que():
    """`DataParallel` num lote contrastivo corta os negativos pela metade.

    Cada réplica calcularia o InfoNCE só sobre a sua fatia: 127 negativos por âncora
    viram 63, sem nada avisar. É o mesmo tipo de erro silencioso que o GradCache
    existe para não cometer, e o comentário tem de estar lá para ninguém "otimizar"
    ligando as duas placas.
    """
    # Na DOCSTRING, que é onde a decisão é explicada; a célula só usa uma placa.
    assert "DataParallel" in DOC
    assert "63" in DOC
    assert "DataParallel" not in CELULA, "a célula não deve usar DataParallel"


def test_lote_e_o_do_campeao():
    """Mudar lote junto com dispositivo faria a comparação medir duas coisas.

    É o erro que este projeto cometeu ao mudar base e lote no mesmo treino, e que
    custou um experimento inteiro.
    """
    assert '"--lote", "128"' in CELULA


# ─── o empacotador ───────────────────────────────────────────────────────────


def test_o_volume_tem_justificativa_MEDIDA_e_nao_e_numero_redondo():
    """⚠️ A justificativa mudou de lugar E de conteúdo em 2026-09-07.

    O número mora agora no registro (`max_pares` do experimento), não num default
    do empacotador — ver `test_o_volume_vem_do_EXPERIMENTO_...`.

    E a justificativa original — "mais que isso a medição diz que não compra nada
    (p=0,950)" — está em DÚVIDA: ela vinha do run de 1,5 M interrompido em 38% por
    platô, e o platô era artefato do prefixo. Manter a frase antiga seria repetir
    uma afirmação que a própria medição deste repositório minou; o registro tem de
    dizer que ela está sob revisão, e é para isso que o `t1a15` existe.
    """
    doc = (RAIZ / "src/phifm/core/kaggle.py").read_text(encoding="utf-8")
    assert "max_pares=400_000" in doc
    assert "p=0,950" in doc, (
        "o volume de 400 mil precisa citar a medição que o justificou, mesmo "
        "agora que ela está em dúvida")
    assert "DUVIDA" in doc or "DÚVIDA" in doc, (
        "a justificativa do volume foi refutada em parte; o registro tem de "
        "dizer isso, senão alguém a cita como se valesse")


def test_pacote_gerado_tem_o_que_o_notebook_espera(tmp_path):
    """Contrato entre o empacotador e a célula, testado nos dois lados.

    Se o manifesto perder a chave `arquivos`, a célula falha no Kaggle — a 211 MB
    de upload de distância.

    ⚠️ Desde 2026-09-06 o pacote NÃO leva `phifm_src.zip.bin`: o código vem do
    GitHub num SHA, porque o Kaggle fixa a versão do dataset no anexo e `kernels
    push` não re-resolve. Ver a nota em `phifm.core.kaggle`.
    """
    import polars as pl

    pares = tmp_path / "pares"
    pares.mkdir()
    d = pl.DataFrame({"arxiv_id": ["a"], "arxiv_citado": ["b"],
                      "ancora": ["x"], "positivo": ["y"]})
    d.write_parquet(pares / "pares_treino.parquet")
    d.write_parquet(pares / "pares_validacao.parquet")

    import subprocess

    saida = tmp_path / "saida"
    r = subprocess.run(
        [sys.executable, str(RAIZ / "scripts" / "empacotar_kaggle.py"),
         "--pares", str(pares), "--out", str(saida), "--max-pares", "1"],
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(RAIZ / "src"),
             "PYTHONUTF8": "1"})
    assert r.returncode == 0, r.stderr[-800:]

    man = json.loads((saida / "MANIFESTO.json").read_text(encoding="utf-8"))
    assert set(man["arquivos"]) == {"pares_treino.parquet",
                                    "pares_validacao.parquet"}
    assert not (saida / "phifm_src.zip.bin").exists(), (
        "o zip de fonte voltou ao pacote; com `repo` no registro o código vem do "
        "GitHub, e código no dataset é código que pode ficar velho sem avisar")
    for v in man["arquivos"].values():
        assert len(v["blake3"]) == 64 and v["bytes"] > 0
    assert man["hash_algo"] == "blake3"
    # ⚠️ A procedência do SORTEIO. Sem ela, um pacote feito por `head` e um feito
    # por sorteio são indistinguíveis depois do fato — e a diferença entre os dois
    # foi 17.844 contra 191.300 documentos citados distintos.
    assert man["semente_do_sorteio"] == 17
    assert man["documentos_distintos"] >= 1
    assert man["linhas_disponiveis"] >= man["linhas_treino"]
    assert man["codigo_de"] == "sanchezVB/LLM_F-sica"


def test_zip_traz_o_ponto_de_entrada_e_o_pacote(tmp_path):
    """O ZIP tem de conter `scripts/train_embedding.py` E o pacote `phifm`.

    Só o pacote não basta: quem invoca o script pelo caminho falharia com "arquivo
    não encontrado" depois do upload.

    ⚠️ Testa `_zipar_fonte` DIRETAMENTE, e não o pacote da T1a: desde 2026-09-06 a
    T1a não leva zip de fonte (o código vem do GitHub). A função continua no
    caminho de qualquer experimento sem `repo`, então a garantia continua valendo —
    só não tem mais um pacote onde observá-la.
    """
    sys.path.insert(0, str(RAIZ / "scripts"))
    from empacotar_kaggle import _zipar_fonte

    destino = tmp_path / "fonte.zip.bin"
    n = _zipar_fonte(RAIZ, destino, ("train_embedding.py",))
    assert n > 0
    with zipfile.ZipFile(destino) as z:
        nomes = z.namelist()
    assert "scripts/train_embedding.py" in nomes
    assert "phifm/training/embedding.py" in nomes
    assert "phifm/training/amostragem.py" in nomes, (
        "o `train_embedding.py` importa `amostragem`; sem ele o zip dá "
        "ModuleNotFoundError depois do upload")
    assert not any("__pycache__" in n for n in nomes)


# ─── publicação: os dois metadados e o notebook gerado ───────────────────────


T1A_FONTE = RAIZ / "kaggle/t1a_phiemb.py"


def _publicar():
    """Importa `scripts/publicar_kaggle.py` sem executá-lo."""
    import importlib.util

    raiz = Path(__file__).resolve().parents[2]
    caminho = raiz / "scripts/publicar_kaggle.py"
    spec = importlib.util.spec_from_file_location("publicar_kaggle", caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_o_slug_do_dataset_sai_de_um_lugar_so():
    """Os dois metadados carregam o slug, e divergentes dão erro só no Kaggle.

    O `id` do dataset e o `dataset_sources` do notebook têm de casar. Se não
    casarem, o notebook sobe, roda, e morre no `assert` de que não achou o
    `MANIFESTO.json` — depois de consumir cota de GPU.

    ⚠️ O lugar único deixou de ser uma constante em cada script e passou a ser
    `phifm.core.kaggle`, para que empacotador e publicador não possam discordar. O
    teste segue o registro; afrouxá-lo por causa da mudança de lugar seria perder o
    invariante junto com a constante.
    """
    from phifm.core.kaggle import EXPERIMENTOS, T1A

    assert T1A.slug_dados and T1A.slug_notebook
    assert T1A.slug_dados != T1A.slug_notebook, (
        "dataset e notebook não podem ter o mesmo slug")
    # E o publicador tem de LER dali, em vez de reintroduzir a constante.
    fonte = (RAIZ / "scripts/publicar_kaggle.py").read_text(encoding="utf-8")
    assert "from phifm.core.kaggle import" in fonte
    assert "SLUG_DADOS =" not in fonte, (
        "a constante voltou para o script — é assim que os dois metadados divergem")
    # Nenhum par de experimentos pode compartilhar slug: o segundo sobrescreveria o
    # dataset do primeiro sem avisar.
    slugs = [s for e in EXPERIMENTOS.values()
             for s in (e.slug_dados, e.slug_notebook)]
    assert len(slugs) == len(set(slugs)), f"slugs repetidos entre experimentos: {slugs}"


def test_a_celula_extraida_e_python_valido():
    """O extrator é um regex sobre um arquivo nosso; se o formato mudar, quebra."""
    import ast

    ast.parse(_publicar()._celula(T1A_FONTE))


def test_o_notebook_gerado_nao_reimplementa_o_treino():
    """A mesma garantia do arquivo-fonte, agora depois de passar pelo gerador.

    Um gerador que perdesse a chamada ao `train_embedding.py` produziria um
    notebook que não treina nada, e o teste do arquivo-fonte não veria.
    """
    m = _publicar()
    codigo = "".join(m._ipynb(m._celula(T1A_FONTE))["cells"][0]["source"])
    assert "scripts/train_embedding.py" in codigo
    assert "InfoNCE" not in codigo, "o laço de treino vazou para o notebook"
    assert "backward()" not in codigo, "o laço de treino vazou para o notebook"


def test_o_ipynb_tem_uma_celula_de_codigo_e_nbformat_4():
    m = _publicar()
    nb = m._ipynb("print(1)\n")
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) == 1
    assert nb["cells"][0]["cell_type"] == "code"
    assert nb["cells"][0]["outputs"] == [], "notebook gerado não deve trazer saída"


def test_extrator_quebra_alto_se_o_formato_mudar(tmp_path):
    """Gerar um notebook vazio em silêncio custaria uma sessão de GPU para descobrir."""
    import pytest

    m = _publicar()
    falso = tmp_path / "sem_celula.py"
    falso.write_text('"""nada aqui."""\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="CELULA"):
        m._celula(falso)


def test_o_fonte_nao_sai_com_extensao_zip():
    """⚠️ O Kaggle DESCOMPACTA `.zip` no upload, e isso quebrou o T1a duas vezes.

    Medido em 2026-08-24: `phifm_src.zip` chegou no dataset como o diretório
    `phifm_src/`. O notebook morreu em `FileNotFoundError` aos 26 s — e pior, o hash
    do CÓDIGO deixou de ser conferível, porque o arquivo que o manifesto descreve não
    existia mais no dataset.

    Uma extensão que o Kaggle não reconhece como arquivo preserva os bytes, e o
    `zipfile` abre pelo conteúdo, não pela extensão.
    """
    from empacotar_kaggle import SUFIXO_ZIP
    from phifm.core.kaggle import EXPERIMENTOS

    assert SUFIXO_ZIP == ".zip.bin", (
        "voltou a gravar .zip — o Kaggle vai descompactar e o hash do código morre")
    # Todo zip DECLARADO tem de usar esse sufixo: um que declarasse `.zip` passaria
    # pelo empacotador e quebraria no notebook.
    #
    # ⚠️ Não exigir que cada experimento declare ALGUM zip. A T1a não declara mais
    # nenhum desde 2026-09-06 — nem fonte (vem do GitHub) nem modelos (os pesos são
    # públicos) —, e a versão anterior deste teste exigia pelo menos um, o que
    # transformaria o conserto em falha.
    for e in EXPERIMENTOS.values():
        for n in [x for x in e.arquivos if ".zip" in x]:
            assert n.endswith(".zip.bin"), f"{e.nome} declara {n}, que o Kaggle abre"
        if e.repo:
            assert not any("phifm_src" in x for x in e.arquivos), (
                f"{e.nome} busca o código no GitHub E declara o zip de fonte; o "
                "zip só reintroduz a chance de rodar código velho")


def test_a_celula_busca_o_codigo_no_GITHUB_num_sha():
    """⚠️ Substitui o teste do zip, e o motivo é a versão fixada no anexo.

    Medido em 2026-09-03 na T1c: o Kaggle fixa a versão do dataset no momento em
    que ela é anexada ao kernel e `kernels push` não re-resolve para a mais
    recente. O conserto do fp16 subiu numa versão nova, `datasets status` disse
    `ready`, e o notebook rodou 15 min sobre o código ANTIGO.

    O notebook, ao contrário do dataset, é reempurrado a cada publicação — então um
    SHA injetado na célula é sempre o do commit atual, e não há versão a fixar.
    """
    celula = CELULA
    assert "codeload.github.com" in celula
    assert 'SHA = "__SHA__"' in celula and 'REPO = "__REPO__"' in celula
    assert "tarfile" in celula
    # E o caminho antigo tem de ter SAÍDO: aceitar os dois deixaria o notebook
    # rodar código do dataset quando o download falhasse, que é o defeito de volta.
    from conftest import so_codigo

    codigo = so_codigo(CELULA)
    assert "phifm_src" not in codigo, (
        "a célula ainda lê o código do dataset; com o GitHub no caminho isso "
        "reintroduz a possibilidade de rodar código velho")


def test_a_celula_LEVANTA_se_o_bundle_de_dados_nao_for_o_publicado():
    """A conferência de blake3 do passo 2 não pega bundle da versão errada.

    Ela compara os arquivos com o manifesto que veio no MESMO dataset, e um bundle
    velho é internamente consistente: os hashes dele batem com os arquivos dele. Só
    um valor vindo de FORA distingue os dois, e ele é injetado na publicação.

    Isto existe porque a T1a foi retreinada com pares SORTEADOS e o dataset antigo
    continua anexável — rodar sobre ele reproduziria justamente o run que se queria
    substituir, e o único sinal seria uma linha de log que alguém precisaria ler.
    """
    celula = CELULA
    assert 'ASSINATURA_ESPERADA = "__ASSINATURA_DADOS__"' in celula
    assert "assinatura_do_manifesto" in celula
    assert "assert _obtida == ASSINATURA_ESPERADA" in celula, (
        "a assinatura tem de ser um `assert`; imprimir depende de leitura humana")


def test_parquets_ausentes_derrubam_antes_do_treino():
    """Hash não conferido é aviso; dado faltando é parada.

    Depois que a conferência passou a TOLERAR arquivo ausente (para aceitar o fonte
    extraído), nada impediria seguir sem os parquets — e a falha apareceria no meio
    do laço, longe da causa.
    """
    celula = CELULA
    assert 'for obrigatorio in ("pares_treino.parquet", "pares_validacao.parquet")' \
        in celula


def test_manifesto_descreve_o_que_a_etapa_produziu_nao_o_que_sobrou_no_disco(tmp_path):
    """Arquivo obsoleto no diretório de saída não pode entrar na atestação.

    Medido em 2026-08-24: ao renomear o fonte para `.zip.bin`, o `phifm_src.zip`
    antigo ficou no diretório e o manifesto passou a descrever OS DOIS — 175 KB de
    código velho atestados como parte do pacote, e enviados ao Kaggle.

    Um manifesto montado de `iterdir()` descreve o que sobrou no disco, não o que a
    etapa produziu. Isso não atesta nada.
    """
    fonte = (RAIZ / "scripts/empacotar_kaggle.py").read_text(encoding="utf-8")
    # A lista declarada mudou de lugar (constante -> `Experimento.arquivos`), e o
    # invariante é o mesmo: o manifesto descreve a lista, não o `iterdir()`.
    assert "exp.arquivos" in fonte, "a lista declarada de conteúdo desapareceu"
    assert "f.name not in exp.arquivos" in fonte, (
        "voltou a aceitar qualquer arquivo do diretório no manifesto")
    assert "for f in sorted(out.iterdir()):" in fonte, (
        "a limpeza do que não pertence ao pacote desapareceu")


def test_o_notebook_pede_acelerador_pelo_campo_que_vale():
    """`enable_gpu` sozinho não liga a GPU, e o Kaggle não reclama.

    Medido em 2026-08-24: o push subiu com `enable_gpu: True`, o Kaggle guardou e
    devolveu esse valor no `kernels pull -m`, e a sessão rodou SEM acelerador —
    `torch.cuda.is_available()` False, e o assert do notebook derrubou aos 18 s.

    Quem decide é `accelerator`. Conferir o metadado de volta não pega isto, porque
    o Kaggle ecoa `enable_gpu` de qualquer jeito; só a execução revela.
    """
    fonte = (RAIZ / "scripts/publicar_kaggle.py").read_text(encoding="utf-8")
    assert '"machine_shape": "NvidiaTeslaT4"' in fonte, (
        "sem `machine_shape` o notebook roda em CPU — e em CPU este treino levaria "
        "dias e 'funcionaria'. `enable_gpu` e `accelerator` NAO servem: o primeiro e "
        "aceito e ignorado, o segundo nem e lido")


def test_a_imagem_docker_nao_fica_presa_a_execucao_antiga():
    """O Kaggle FIXA a imagem docker do kernel, e o pin sobrevive a novos pushes.

    Medido em 2026-08-24/25: quatro execuções rodaram com `torch 2.10.0+cpu` mesmo
    com `machine_shape: NvidiaTeslaT4` confirmado pelo `kernels pull -m`. O kernel
    seguia preso à imagem da PRIMEIRA execução, que foi sem acelerador — e GPU
    alocada não muda o PyTorch instalado dentro de uma imagem CPU-only.

    `latest` manda o Kaggle escolher a imagem atual apropriada ao acelerador.
    `original` é o comportamento que criou o problema.
    """
    fonte = (RAIZ / "scripts/publicar_kaggle.py").read_text(encoding="utf-8")
    assert '"docker_image_pinning_type": "latest"' in fonte, (
        "sem isto o kernel arrasta a imagem de CPU da primeira execução e o treino "
        "roda sem GPU mesmo com o acelerador pedido")


def test_a_celula_acha_o_dataset_nos_dois_layouts_do_kaggle():
    """A imagem nova do Kaggle mudou onde o dataset é montado.

    Medido em 2026-08-26, logo depois que o pin da imagem de CPU foi solto:

        AssertionError: nenhum dataset com MANIFESTO.json em /kaggle/input.
                        Encontrados: ['datasets']

    A antiga montava em `/kaggle/input/<slug>/`, a nova em
    `/kaggle/input/datasets/<dono>/<slug>/`. O pin escondia isso: manter a imagem
    velha mantinha o layout velho junto.

    A busca vai de raso para fundo e para no primeiro nível que casa, então
    funciona nos dois — e o `assert` imprime o que encontrou, que foi o que
    permitiu identificar a mudança numa execução só.
    """
    assert 'for profundidade in range(' in CELULA, (
        "voltou a olhar só os filhos diretos de /kaggle/input")
    assert "ENTRADA.iterdir()" in CELULA, (
        "a mensagem de erro precisa listar o que existe, senão a próxima mudança "
        "de layout custa uma execução para ser diagnosticada")


def test_busca_em_profundidade_encontra_nos_dois_layouts(tmp_path):
    """Exercita a lógica de verdade, nos dois layouts, sem depender do Kaggle."""
    def achar(entrada):
        for profundidade in range(0, 5):
            padrao = "/".join(["*"] * profundidade + ["MANIFESTO.json"])
            c = sorted({m.parent for m in entrada.glob(padrao)})
            if c:
                return c
        return []

    antigo = tmp_path / "antigo"
    (antigo / "meu-slug").mkdir(parents=True)
    (antigo / "meu-slug" / "MANIFESTO.json").write_text("{}", encoding="utf-8")
    assert achar(antigo) == [antigo / "meu-slug"]

    novo = tmp_path / "novo"
    fundo = novo / "datasets" / "viaciclo" / "meu-slug"
    fundo.mkdir(parents=True)
    (fundo / "MANIFESTO.json").write_text("{}", encoding="utf-8")
    assert achar(novo) == [fundo]

    assert achar(tmp_path / "vazio") == []


def test_treino_que_falha_derruba_o_notebook():
    """`COMPLETE` tem de significar que treinou.

    Medido em 2026-08-26: o notebook rodava o treino com `check=False` e só
    imprimia o código de saída. O treino morreu, a célula terminou normalmente, e o
    Kaggle marcou a execução como COMPLETE — com `codigo/` na saída e NENHUM modelo.

    A justificativa original do `check=False` ("uma sessão que cai deixa o estado
    para a próxima retomar") estava errada: `/kaggle/working` persiste como saída do
    notebook independentemente de a célula levantar.

    ⚠️ A asserção é ESTRUTURAL, não textual. A primeira versão exigia a string
    `"if r.returncode != 0:"` e reprovou quando o `subprocess.run` virou `Popen` e
    a variável passou a se chamar `saida_treino` — um teste que quebra ao renomear
    uma variável não está guardando a propriedade, está guardando o nome.

    A propriedade é: existe um `if <algo> != 0` cujo corpo levanta.
    """
    import ast as _ast

    def levanta_se_nao_zero(no) -> bool:
        if not isinstance(no, _ast.If):
            return False
        teste = no.test
        if not (isinstance(teste, _ast.Compare) and len(teste.ops) == 1
                and isinstance(teste.ops[0], _ast.NotEq)):
            return False
        if not any(isinstance(c, _ast.Constant) and c.value == 0
                   for c in teste.comparators):
            return False
        return any(isinstance(n, _ast.Raise) for n in _ast.walk(no))

    arvore = _ast.parse(CELULA)
    assert any(levanta_se_nao_zero(n) for n in _ast.walk(arvore)), (
        "o notebook voltou a engolir o código de saída do treino: não há um "
        "`if <saída> != 0` que levante")


def test_a_saida_do_treino_vai_para_stdout_TAMBEM_e_nao_so_para_o_arquivo():
    """⚠️ Mandar só para o arquivo custou 33 h de cegueira.

    Medido em 2026-09-07: entre o lançamento do treino (aos 18,9 s) e o fim dele
    NADA aparecia no log da célula, porque o `stdout` do subprocesso ia inteiro
    para `treino.log` e o `print` só acontecia depois. Uma execução saudável ficou
    indistinguível de uma travada, e eu não soube dizer se o kernel estava na fila
    do Kaggle ou morto — estava na fila.

    O arquivo continua: o `kernels output` da API devolveu 0 byte em três
    execuções seguidas em 2026-08-26. Os dois destinos, não um.
    """
    celula = CELULA
    assert "subprocess.Popen" in celula, (
        "voltou o `subprocess.run` com stdout no arquivo; sem streaming não há "
        "sinal de progresso durante a corrida")
    assert "for linha in proc.stdout:" in celula
    assert "fh.write(linha)" in celula and "flush=True" in celula, (
        "sem flush o pipe acumula e a cegueira volta com outra cara")
    assert "bufsize=1" in celula, "sem bufsize=1 a leitura não é linha a linha"


def test_log_do_treino_vai_para_arquivo_baixavel():
    """A API do Kaggle devolveu log de 0 bytes em três execuções seguidas.

    Sem o log não há diagnóstico. Um arquivo em `/kaggle/working` desce junto com a
    saída do notebook mesmo quando `kernels output` não traz o log.
    """
    assert "TREINO_LOG" in CELULA
    assert "stderr=subprocess.STDOUT" in CELULA, (
        "stderr precisa ir para o mesmo arquivo — o traceback vive nele")


def test_o_subprocesso_recebe_pythonpath():
    """O `sys.path` da célula NÃO vale para o processo filho.

    Medido em 2026-08-26, no primeiro log que a correção anterior tornou legível:

        ModuleNotFoundError: No module named 'phifm'
          em /kaggle/working/codigo/scripts/train_embedding.py

    E o `sys.path.insert(parents[1] / "src")` que o próprio script faz não cobre
    todo layout.

    ⚠️ O alvo MUDOU em 2026-09-06, e é por isso que este teste é específico. Com o
    zip era `CODIGO`, porque `_zipar_fonte` gravava `phifm/` na raiz. Com o tarball
    do GitHub o `src/` é preservado, então o pacote vive em `<raiz>/src/phifm` e o
    PYTHONPATH tem de ser `CODIGO / "src"` — apontar para `CODIGO` daria
    ModuleNotFoundError depois de o Kaggle alocar a GPU.
    """
    assert '"PYTHONPATH": str(FONTE)' in CELULA, (
        "sem PYTHONPATH no ambiente do subprocesso o treino não importa phifm")
    assert 'FONTE = CODIGO / "src"' in CELULA, (
        "o tarball do GitHub preserva `src/`; FONTE tem de apontar para lá")
    assert "env=AMBIENTE" in CELULA, "o env montado não está sendo passado"


def test_o_zip_grava_o_pacote_na_raiz_e_nao_sob_src(tmp_path):
    """O layout do zip, para quem ainda o usa.

    ⚠️ A T1a NÃO depende mais disto: o código dela vem do tarball do GitHub, que
    preserva `src/`, e o PYTHONPATH dela aponta para `CODIGO / "src"`. Este teste
    fixa o layout de `_zipar_fonte` para um experimento futuro sem `repo` — os dois
    layouts existem e confundi-los é o que dá ModuleNotFoundError no Kaggle.
    """
    import sys as _sys

    _sys.path.insert(0, str(RAIZ / "scripts"))
    import zipfile as _zip

    from empacotar_kaggle import _zipar_fonte

    destino = tmp_path / "fonte.zip.bin"
    _zipar_fonte(RAIZ, destino, ("train_embedding.py",))
    with _zip.ZipFile(destino) as z:
        nomes = z.namelist()
    assert any(n.startswith("phifm/") for n in nomes), "pacote saiu de `phifm/`"
    assert not any(n.startswith("src/") for n in nomes), (
        "o zip passou a preservar `src/` — o PYTHONPATH do notebook precisa "
        "apontar para `CODIGO / 'src'` agora")


# ─── o experimento de volume ─────────────────────────────────────────────────


def test_o_volume_vem_do_EXPERIMENTO_e_nao_de_um_default_solto():
    """⚠️ O volume é a variável do t1a15, então pertence à identidade dele.

    Com `--max-pares` tendo um default fixo no empacotador, rodar
    `empacotar_kaggle.py --experimento t1a15` sem a bandeira montaria 400 mil
    pares sob o nome do experimento de 1,5 M — e o manifesto atestaria o número
    errado com a cara certa.
    """
    from phifm.core.kaggle import EXPERIMENTOS

    assert EXPERIMENTOS["t1a"].max_pares == 400_000
    assert EXPERIMENTOS["t1a15"].max_pares == 1_500_000

    fonte = (RAIZ / "scripts/empacotar_kaggle.py").read_text(encoding="utf-8")
    bloco = fonte.split('"--max-pares"')[1].split("p.add_argument")[0]
    assert "default=None" in bloco, (
        "o empacotador voltou a ter um volume default próprio; ele tem de cair "
        "no `max_pares` do experimento")
    assert "exp.max_pares" in fonte


def test_os_dois_experimentos_de_volume_nao_compartilham_slug():
    """Slugs próprios, e não versões novas dos do t1a.

    O Kaggle fixa a versão do dataset no momento do anexo e `kernels push` não
    re-resolve: subir 1,5 M como versão nova de `phifm-t1a-pares-sorteados` faria
    a assinatura do bundle LEVANTAR na célula — a guarda funcionando, e o run
    bloqueado. Notebook separado também preserva a execução de 400 mil como
    registro em vez de sobrescrevê-la.
    """
    from phifm.core.kaggle import EXPERIMENTOS

    a, b = EXPERIMENTOS["t1a"], EXPERIMENTOS["t1a15"]
    assert a.slug_dados != b.slug_dados
    assert a.slug_notebook != b.slug_notebook
    assert a.pacote != b.pacote
    # E o resto é o MESMO, porque o volume é a única variável do experimento.
    assert a.fonte_celula == b.fonte_celula
    assert a.arquivos == b.arquivos
    assert a.scripts == b.scripts
    assert a.repo == b.repo


def test_o_montador_do_t1a_serve_os_dois():
    """Duplicar o montador duplicaria também cada lição já paga nele — o sorteio,
    o `.zip.bin`, a procedência no manifesto — e as cópias divergiriam."""
    sys.path.insert(0, str(RAIZ / "scripts"))
    from empacotar_kaggle import MONTADORES

    assert MONTADORES["t1a"] is MONTADORES["t1a15"]
