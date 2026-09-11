"""O T2a custa 15 h de T4 e pode produzir um NULO FABRICADO. Daí estes testes.

O T2a pergunta se o regex de pré-tokenização da §8 do DOC-05 — o que torna `\\frac`,
`\\begin{…}` e `^{`/`_{` pré-tokens atômicos — compra alguma coisa. A variante E é a
A sem ele. O §11.2 chama esta de a comparação mais importante das seis: *"se E
empatar com A, a §8 está errada e o documento precisa ser revisado"*.

⚠️ **Este experimento tem um modo de falha que produz exatamente o resultado
esperado pela hipótese nula.** Se os dois braços treinarem sobre a MESMA fatia — por
um empacotamento que duplicou um diretório, por uma variante não injetada na célula,
por um manifesto que nomeia o tokenizer errado — o resultado é empate perfeito, e a
leitura natural é "a §8 não vale nada". Um nulo se escreve como "tentamos, não
funciona" e fecha a linha; um positivo convida a replicar. Por isso o nulo precisa de
MAIS guarda que o positivo, não menos — é a lição que o artigo §8 pagou duas vezes.

O que estes testes fixam:

  1. **a variante é conferida contra o MANIFESTO da fatia**, na célula e no
     empacotador, antes de qualquer GPU ser gasta;
  2. **o orçamento não depende da variante** — a mesma célula nos dois braços, com
     tokens, contexto, lote e passos como literais;
  3. **`p_equacao=0,0` nos dois**, senão o experimento mede tokenizer *e* a
     interação dele com o mascaramento do §2.3;
  4. **a comparação NÃO acontece no Kaggle.** Os braços treinam em sessões
     separadas; medir cada um na sua sessão repetiria a armadilha do T1b2;
  5. **o checkpoint é exportado**, senão as três medidas do §11.2 não o abrem;
  6. **o registro declara dois braços de uma variável só**, com E reusando o
     dataset de A — um dataset, uma assinatura.

⚠️ Nada aqui lê `models/` nem `data/processed/`, que são gitignored.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "kaggle"))
sys.path.insert(0, str(RAIZ / "scripts"))

import t2a_tokenizer  # noqa: E402

from conftest import so_codigo  # noqa: E402
from phifm.core.kaggle import BRACOS_DO_T2A, obter  # noqa: E402

CELULA = t2a_tokenizer.CELULA
DOC = t2a_tokenizer.__doc__ or ""
CODIGO = so_codigo(CELULA)
LACO = (RAIZ / "src/phifm/training/pretrain/laco.py").read_text(encoding="utf-8")


# ── a célula ────────────────────────────────────────────────────────────────

def test_a_celula_e_python_valido():
    """O extrator do publicador é um regex sobre este arquivo; se o formato
    mudar, o notebook gerado sai vazio e a descoberta custa 7,5 h de sessão."""
    ast.parse(CELULA)


def test_a_variante_e_validada_logo_no_topo():
    """`__VARIANTE__` é substituído na publicação. Se a substituição não
    acontecer, a célula recebe a string literal — e sem este assert ela seguiria
    até o `glob` do dataset falhar com uma mensagem que não menciona a variante."""
    i_assert = CELULA.index('assert VARIANTE in ("A", "E")')
    i_treino = CELULA.index("# ── 6.")
    assert i_assert < i_treino
    assert "__VARIANTE__" in CELULA, "o marcador da publicação sumiu da célula"


def test_a_fatia_montada_e_conferida_contra_o_TOKENIZER_dela():
    """⚠️ A guarda contra o nulo fabricado, do lado do Kaggle.

    O dataset traz as duas fatias achatadas e a célula remonta a da variante por
    nome de arquivo. Um nome errado — no empacotador, no dataset, na injeção —
    faria os dois braços treinarem no mesmo dado, e o empate resultante se leria
    como refutação da §8.

    O manifesto da fatia nomeia o tokenizer com que ela foi construída, e é um
    dado que veio de FORA da montagem: ele não pode concordar por acidente.
    """
    i_conf = CELULA.index('assert f"variante_{VARIANTE}" in man_fatia["tokenizer"]')
    i_treino = CELULA.index("# ── 6.")
    assert i_conf < i_treino, "a conferência da fatia está depois do treino"
    assert "treinariam no mesmo dado" in CELULA


def test_o_orcamento_NAO_depende_da_variante():
    """⚠️ A mesma célula roda os dois braços, e é isso que garante tokens iguais.

    Por AST: nenhuma das quatro constantes de orçamento pode ser atribuída dentro
    de um `if`, nem derivar de `VARIANTE`. Calcular o orçamento por braço seria o
    jeito de eles divergirem sem ninguém ver — e com orçamentos diferentes o
    resultado mede volume, não tokenizer.
    """
    arvore = ast.parse(CELULA)
    orcamento = {"TOKENS", "CONTEXTO", "SEQUENCIAS", "ACUMULACAO", "PASSOS"}
    vistos = set()
    for no in ast.walk(arvore):
        if not isinstance(no, ast.Assign):
            continue
        alvos = {t.id for t in no.targets if isinstance(t, ast.Name)}
        if not (alvos & orcamento):
            continue
        vistos |= alvos & orcamento
        nomes = {n.id for n in ast.walk(no.value) if isinstance(n, ast.Name)}
        assert "VARIANTE" not in nomes, (
            f"{alvos & orcamento} deriva de VARIANTE; o orçamento tem de ser "
            "idêntico nos dois braços")
    assert vistos == orcamento, f"faltou atribuir {orcamento - vistos}"

    # E nenhuma delas dentro de um `if`: um ramo por braço daria o mesmo estrago
    # sem passar pelo teste acima.
    for no in ast.walk(arvore):
        if isinstance(no, ast.If):
            dentro = {t.id for f in ast.walk(no) if isinstance(f, ast.Assign)
                      for t in f.targets if isinstance(t, ast.Name)}
            assert not (dentro & orcamento), (
                f"{dentro & orcamento} atribuído dentro de um `if`")


def test_o_mascaramento_e_PADRAO_nos_dois_bracos():
    """⚠️ `--p-equacao 0.0`, e não é preguiça.

    O tratamento de equação inteira do DOC-07 §2.3 depende de o tokenizer MARCAR
    equações. Com E, `\\frac` está estilhaçado, então o mascaramento por span se
    comportaria de outro jeito — e o experimento mediria "tokenizer + interação"
    em vez do tokenizer. A ablação do §2.3 é outro experimento, e precisa deste
    resolvido antes.
    """
    assert '"--p-equacao", 0.0' in CELULA
    assert CELULA.count('"--p-equacao"') == 1, (
        "mais de uma passagem de --p-equacao: um dos braços poderia divergir")
    assert "mediria tokenizer + interação" in CELULA or \
           "mediria a interação" in CELULA or "interação" in CELULA


def test_a_COMPARACAO_nao_acontece_no_kaggle():
    """⚠️ A lição do T1b2, aplicada a um experimento que não cabe numa sessão.

    15 h não cabem nas 9 h de uma sessão, então os braços treinam separados. Medir
    cada um na própria sessão traria de volta exatamente o que o T1b2 pagou:
    números de sessões diferentes comparados como se fossem pareados. A célula
    produz um checkpoint e para; as três medidas do §11.2 rodam local, com os dois
    modelos no mesmo processo.
    """
    assert "A COMPARAÇÃO NÃO ACONTECE AQUI" in CELULA
    # Nenhum avaliador é chamado: a célula só treina e exporta.
    chamados = [n.args[0] for n in ast.walk(ast.parse(CELULA))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "_rodar" and n.args]
    fontes = [ast.unparse(x) for x in chamados]
    assert len(fontes) == 2, f"esperava treinar e exportar, achei {fontes}"
    assert any("train_phienc" in f for f in fontes), fontes
    assert any("exportar_phienc" in f for f in fontes), fontes
    for f in fontes:
        assert "avaliar" not in f, (
            f"{f} avalia dentro da sessão do braço; a comparação é local")


def test_o_checkpoint_e_EXPORTADO():
    """O laço grava `state_dict` cru. Sem `config.json` ao lado, nenhuma das três
    medidas do §11.2 abre o checkpoint — e a descoberta seria depois das 7,5 h,
    com o resto da cota já gasto no outro braço."""
    i_treino = CELULA.index("# ── 6.")
    i_export = CELULA.index("# ── 7.")
    assert i_treino < i_export
    assert "exportar_phienc.py" in CELULA
    assert '"--tokenizer", DADOS /' in CELULA, (
        "o tokenizer tem de vir do DATASET: o caminho que o manifesto do run "
        "nomeia é um caminho da máquina local, que não existe no Kaggle")


def test_o_efeito_esperado_esta_escrito_ANTES_do_numero():
    """+12,6% de tokens por documento, medido nas fatias de verdade, contra os
    +37,7% da métrica intrínseca. Escrever a expectativa antes é o que impede de
    racionalizar o resultado depois."""
    i_treino = CELULA.index("# ── 6.")
    trecho = CELULA[:i_treino]
    assert "12,6%" in trecho
    assert "12,5%" in trecho
    assert "37,7%" in trecho, (
        "sem o número da métrica intrínseca ao lado, um resultado de +12% pareceria "
        "uma falha da §8 quando é a métrica que superestima por 3x")


def test_o_confundidor_de_capacidade_esta_nomeado():
    """⚠️ Por que A×E e não A×C ou A×D.

    A 48 M de parâmetros a tabela de embedding é 43,7% do modelo. C (V=32.768) e D
    (V=65.536) têm contagens diferentes de A, então comparar com elas mediria
    capacidade junto com tokenizer. A e E compartilham vocabulário e contagem.
    """
    assert "43,7%" in CELULA
    assert "ÚNICO par limpo" in CELULA


def test_confere_hash_e_assinatura_ANTES_de_treinar():
    i_hash = CELULA.index("conferidos por blake3")
    i_assin = CELULA.index("assinatura do bundle confere")
    i_treino = CELULA.index("# ── 6.")
    assert i_hash < i_treino and i_assin < i_treino


def test_exige_gpu_em_vez_de_cair_para_cpu():
    assert 'assert torch.cuda.is_available(), "sem GPU' in CELULA


def test_etapa_que_falha_derruba_o_notebook():
    """Por AST: um subprocesso que sai com código diferente de zero tem de
    levantar, não seguir para uma exportação de checkpoint que não existe."""
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


def test_o_codigo_vem_do_GITHUB_num_sha():
    assert "codeload.github.com" in CELULA
    assert "__SHA__" in CELULA
    assert "assinatura_do_manifesto" in CODIGO


# ── o registro ──────────────────────────────────────────────────────────────

def test_os_dois_bracos_diferem_SO_no_que_identifica_o_braco():
    """Duas entradas quase idênticas divergem em silêncio — foi para não duplicar
    lição paga que `phifm.core.kaggle` nasceu. Aqui a divergência custaria a
    comparabilidade dos braços."""
    a, e = (obter(n) for n in BRACOS_DO_T2A)
    for campo in ("titulo_dados", "slug_dados", "pacote", "fonte_celula",
                  "arquivos", "scripts", "modelos", "repo"):
        assert getattr(a, campo) == getattr(e, campo), (
            f"os braços divergem em {campo}: {getattr(a, campo)!r} contra "
            f"{getattr(e, campo)!r}")
    assert (a.variante, e.variante) == ("A", "E")
    assert a.slug_notebook != e.slug_notebook
    assert a.titulo_notebook != e.titulo_notebook


def test_o_braco_E_REUSA_o_dataset_do_A():
    """⚠️ Um dataset, uma assinatura. Com datasets separados nada acusaria dois
    preparos diferentes, e o resultado sairia com uma variável a mais dentro dele.
    Republicar também quebraria a assinatura que a célula do A confere."""
    a, e = (obter(n) for n in BRACOS_DO_T2A)
    assert e.reusa_dados_de == "t2a_a"
    assert a.reusa_dados_de is None
    assert e.slug_dados == a.slug_dados
    assert e.pacote == a.pacote


def test_o_pacote_traz_as_DUAS_fatias_e_os_DOIS_tokenizers():
    """A célula do braço E remonta a fatia E a partir do mesmo dataset que a do A
    conferiu. Faltando um arquivo, a descoberta é no Kaggle."""
    a = obter("t2a_a")
    for v in ("A", "E"):
        for nome in (f"tokens_{v}.u16.bin", f"marcas_{v}.u8.bin",
                     f"MANIFESTO_{v}.json", f"variante_{v}.json"):
            assert nome in a.arquivos, f"{nome} não está no pacote"


def test_o_publicador_injeta_a_VARIANTE_e_recusa_o_marcador_sobrando():
    """⚠️ Sem a injeção, os dois kernels rodariam o braço A: 15 h de T4 para
    produzir o mesmo modelo duas vezes, e uma comparação que dá empate perfeito."""
    fonte = (RAIZ / "scripts" / "publicar_kaggle.py").read_text(encoding="utf-8")
    assert 'celula.replace("__VARIANTE__", exp.variante)' in fonte
    assert '"__VARIANTE__"' in fonte.split("for marcador in")[1][:200], (
        "o marcador não está na lista de sobras conferidas")


def test_o_publicador_RECUSA_publicar_o_dataset_de_quem_reusa():
    fonte = (RAIZ / "scripts" / "publicar_kaggle.py").read_text(encoding="utf-8")
    assert "reusa_dados_de" in fonte
    assert "--so-notebook" in fonte


# ── o empacotador: as guardas contra o nulo fabricado ───────────────────────

def _fatia(raiz: Path, variante: str, **campos) -> Path:
    """Uma fatia mínima: os dois binários e o manifesto que o montador lê."""
    d = raiz / f"data/processed/t2a_{variante}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "tokens.u16.bin").write_bytes(b"\x00\x01" * 8)
    (d / "marcas.u8.bin").write_bytes(b"\x00" * 16)
    man = {"tokenizer": f"data/processed/tokenizer/variante_{variante}.json",
           "tokenizer_sha": f"sha_{variante}",
           "tokens": 800_000_000,
           "documentos_nesta_execucao": 100_000,
           "partes_usadas": ["part-00000.parquet"],
           "corpus": "data/processed/redpajama_fisica",
           "semente_do_sorteio": 17, "max_tokens": 1024, "em_ordem": False}
    man.update(campos)
    (d / "MANIFESTO_DADOS.json").write_text(json.dumps(man), encoding="utf-8")
    tok = raiz / "data/processed/tokenizer"
    tok.mkdir(parents=True, exist_ok=True)
    (tok / f"variante_{variante}.json").write_text("{}", encoding="utf-8")
    return d


def _montar(raiz: Path, out: Path):
    from empacotar_kaggle import _montar_t2a
    out.mkdir(parents=True, exist_ok=True)
    return _montar_t2a(obter("t2a_a"), raiz, out, object())


def test_o_empacotador_monta_as_duas_fatias_achatadas(tmp_path):
    """O Kaggle achata a estrutura do upload, então o nome é que carrega a
    variante — e a célula refaz os links canônicos do outro lado."""
    _fatia(tmp_path, "A")
    _fatia(tmp_path, "E")
    out = tmp_path / "pacote"
    extra = _montar(tmp_path, out)
    for v in ("A", "E"):
        for nome in (f"tokens_{v}.u16.bin", f"marcas_{v}.u8.bin",
                     f"MANIFESTO_{v}.json", f"variante_{v}.json"):
            assert (out / nome).exists(), nome
    assert set(extra["fatias"]) == {"A", "E"}
    assert extra["efeito_disponivel"]["tokens_por_doc_A"] == 8000.0


def test_o_empacotador_RECUSA_a_mesma_fatia_sob_dois_nomes(tmp_path):
    """⚠️ O teste central deste arquivo.

    Preparar a fatia "E" com o tokenizer A é um erro de uma palavra na linha de
    comando, e produz um pacote que parece certo: dois diretórios, quatro
    arquivos, tamanhos plausíveis. Os dois braços treinariam no mesmo dado, o
    resultado seria empate perfeito, e a leitura seria "a §8 não vale nada" —
    um nulo fabricado, que fecha a linha em vez de convidar a replicar.
    """
    _fatia(tmp_path, "A")
    _fatia(tmp_path, "E",
           tokenizer="data/processed/tokenizer/variante_A.json",
           tokenizer_sha="sha_A")
    with pytest.raises(SystemExit, match="empate por construção"):
        _montar(tmp_path, tmp_path / "pacote")


def test_o_empacotador_RECUSA_tokenizer_sha_igual(tmp_path):
    """O nome do arquivo pode estar certo e o conteúdo ser o mesmo: o `sha` é o
    que compara o que foi realmente usado."""
    _fatia(tmp_path, "A")
    _fatia(tmp_path, "E", tokenizer_sha="sha_A")
    with pytest.raises(SystemExit, match="não varia"):
        _montar(tmp_path, tmp_path / "pacote")


def test_o_empacotador_RECUSA_orcamentos_de_token_diferentes(tmp_path):
    """O protocolo iguala TOKENS. Com orçamentos diferentes o resultado mediria
    volume de treino, e o efeito procurado (12,6%) é menor que quase qualquer
    descuido de volume."""
    _fatia(tmp_path, "A")
    _fatia(tmp_path, "E", tokens=600_000_000)
    with pytest.raises(SystemExit, match="mede volume, não tokenizer"):
        _montar(tmp_path, tmp_path / "pacote")


@pytest.mark.parametrize("campo,valor", [
    ("corpus", "data/processed/pes2o_fisica"),
    ("semente_do_sorteio", 23),
    ("max_tokens", 512),
])
def test_o_empacotador_RECUSA_fatias_que_divergem_em_outra_coisa(
        tmp_path, campo, valor):
    """A variável é o tokenizer. Qualquer outra diferença entraria no resultado
    sem nome — e o corpus é o pior deles: as 8 primeiras partes têm 32,4% de
    matemática contra 42,7% das demais."""
    _fatia(tmp_path, "A")
    _fatia(tmp_path, "E", **{campo: valor})
    with pytest.raises(SystemExit, match="uma variável"):
        _montar(tmp_path, tmp_path / "pacote")


def test_o_empacotador_diz_o_que_rodar_quando_a_fatia_nao_existe(tmp_path):
    """Uma mensagem que nomeia o comando poupa a viagem até o script."""
    _fatia(tmp_path, "A")
    with pytest.raises(SystemExit, match="preparar_dados_phienc.py"):
        _montar(tmp_path, tmp_path / "pacote")


# ── o teto de sessão ────────────────────────────────────────────────────────

def test_a_celula_passa_o_teto_de_horas_da_sessao():
    """⚠️ Dimensionar por FLOPs é supor MFU, e a suposição é frouxa a 48 M: 15%
    contra 25% é 8,9 h contra 5,9 h — os dois lados de uma sessão de 9 h. O laço
    projeta na segunda janela de log e aborta se não couber."""
    assert '"--limite-horas", LIMITE_H' in CELULA
    limite = float(CELULA.split("LIMITE_H = ")[1].split("\n")[0])
    assert 0 < limite < 9.0, (
        f"teto de {limite} h; a sessão do Kaggle morre às 9 h e o laço não retoma "
        "entre sessões")


def test_o_laco_ABORTA_quando_a_vazao_medida_nao_cabe():
    """`horas_estimadas` existia e nao era chamada em lugar nenhum: o laco sabia
    projetar o custo e nunca fazia nada com a projecao.

    Por AST e nao por import: a suite rapida roda sem torch, e `laco.py` importa
    torch no topo. Um teste que so passa no venv de treino nao roda no CI.
    """
    arvore = ast.parse(LACO)
    cfg = next(n for n in ast.walk(arvore) if isinstance(n, ast.ClassDef)
               and n.name == "ConfigTreino")
    campos = {n.target.id: n for n in cfg.body if isinstance(n, ast.AnnAssign)}
    assert "limite_horas" in campos, "ConfigTreino nao tem limite_horas"
    assert ast.unparse(campos["limite_horas"].value) == "None", (
        "a guarda tem de vir desligada: em maquina propria um run longo e normal")

    treinador = next(n for n in ast.walk(arvore) if isinstance(n, ast.ClassDef)
                     and n.name == "Treinador")
    metodos = {n.name: n for n in treinador.body
               if isinstance(n, ast.FunctionDef)}
    guarda = metodos.get("_conferir_orcamento_de_sessao")
    assert guarda is not None, "nenhuma guarda de orcamento no Treinador"
    assert any(isinstance(n, ast.Raise) for n in ast.walk(guarda)), (
        "a guarda avisa em vez de abortar; a sessao morre no relogio de qualquer "
        "jeito, e seguir gasta as 9 h para chegar ao mesmo lugar")
    fontes = {ast.unparse(n.func) for n in ast.walk(guarda)
              if isinstance(n, ast.Call)}
    assert "self.horas_estimadas" in fontes, (
        "a guarda nao usa a projecao — era exatamente esse o defeito")

    # E ela e chamada de dentro do laco, nao so definida.
    chamada = {ast.unparse(n.func) for n in ast.walk(metodos["treinar"])
               if isinstance(n, ast.Call)}
    assert "self._conferir_orcamento_de_sessao" in chamada, (
        "a guarda existe e o laco nao a chama")


def test_a_guarda_espera_a_SEGUNDA_janela_de_log():
    """A primeira carrega autotune do cuDNN e primeira alocação: ela mede devagar
    por construção, e abortar nela reprovaria runs que cabem."""
    corpo = LACO.split("def _conferir_orcamento_de_sessao")[1].split("\n    def ")[0]
    assert "len(m.historico) < 2" in corpo


def test_a_guarda_avisa_que_encolher_UM_braco_quebra_o_pareamento():
    """A saída natural de "não cabe" é reduzir os passos. Num experimento
    pareado isso invalida o braço que já rodou — e quem lê a mensagem no Kaggle
    está a 7,5 h de distância de perceber sozinho."""
    corpo = LACO.split("def _conferir_orcamento_de_sessao")[1].split("\n    def ")[0]
    assert "BRAÇO" in corpo
    assert "orçamento igual" in corpo


def test_o_empacotador_exige_que_as_partes_sejam_PREFIXO(tmp_path):
    """⚠️ Prefixo, não igualdade.

    As duas fatias sorteiam as partes com a mesma semente, então a ordem é a
    mesma; o que muda é onde cada uma para. E gasta mais tokens por documento e
    enche o orçamento antes, então o conjunto de documentos de E é um PREFIXO do
    de A. Exigir igualdade reprovaria o caso normal; não exigir nada deixaria
    passar dois corpora embaralhados diferentes.
    """
    _fatia(tmp_path, "A", partes_usadas=["p35", "p14", "p02", "p09"])
    _fatia(tmp_path, "E", partes_usadas=["p35", "p14", "p02"])
    _montar(tmp_path, tmp_path / "pacote")  # prefixo: passa

    _fatia(tmp_path, "E", partes_usadas=["p14", "p35", "p02"])
    with pytest.raises(SystemExit, match="prefixo"):
        _montar(tmp_path, tmp_path / "pacote2")


def test_a_espera_pelo_dataset_ESCALA_com_o_tamanho():
    """⚠️ Os 900 s fixos foram calibrados em pacotes de 200 a 780 MB; o T2a sobe
    5,4 GB.

    Errar por baixo não é só esperar de novo: a mensagem de timeout convida a
    republicar, e republicar cria uma versão NOVA do dataset — que quebraria a
    `assinatura_do_manifesto` publicada no notebook, e o notebook passaria a
    recusar o próprio dado.
    """
    sys.path.insert(0, str(RAIZ / "scripts"))
    from publicar_kaggle import _limite_de_espera
    assert _limite_de_espera(int(0.78e9)) >= 900, "não pode encolher o que já valia"
    grande = _limite_de_espera(int(5.4e9))
    assert grande > 1500, (
        f"{grande} s para 5,4 GB; o T2a bateria no teto e o timeout convidaria a "
        "republicar")
    assert grande < 3600, f"{grande} s é espera demais para um erro de verdade"
