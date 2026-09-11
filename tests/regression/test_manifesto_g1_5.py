"""O manifesto raiz tem de PEGAR adulteração, não só existir.

O G1.5 (DOC-00 §5) pede o corpus "reprodutível ponta a ponta a partir de um único
hash de manifesto". Um sistema de manifesto que sempre responde "✅ OK" satisfaz o
critério na aparência e não na função — e é o modo de falha mais fácil de construir
sem perceber, porque o caminho felizes passa nos dois casos.

Então cada teste aqui **estraga** algo e exige que a verificação acuse. São os
quatro modos que importam:

  1. parquet alterado          → só a verificação PROFUNDA pega
  2. manifesto de etapa mexido → a RASA pega, e por isso ela existe
  3. arquivo do corpus apagado → profunda
  4. arquivo a mais            → profunda; corpus com parquet extra não é o corpus
                                 manifestado, e um treino que o leia mede outra coisa

E um teste que a suíte precisa mais que os outros: a verificação rasa **não** pode
ser lida como "corpus íntegro". Ela confere manifestos, não conteúdo, e o teste
`test_rasa_nao_pega_parquet_alterado` documenta isso como comportamento esperado em
vez de deixar alguém descobrir depois.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.core.schema.reprodutibilidade import (  # noqa: E402
    Entrada,
    EtapaRef,
    ManifestoEtapa,
    ManifestoRaiz,
    hash_arquivo,
    indexar,
    verificar,
)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Um corpus mínimo com duas etapas, manifestos e raiz — como o de verdade."""
    (tmp_path / "fatia").mkdir()
    (tmp_path / "fatia" / "part-00000.parquet").write_bytes(b"conteudo A" * 100)
    (tmp_path / "fatia" / "part-00001.parquet").write_bytes(b"conteudo B" * 100)
    (tmp_path / "spine.parquet").write_bytes(b"spine" * 500)

    refs = []
    for etapa, raiz in (("fatia", tmp_path / "fatia"),
                        ("tabela mestra", tmp_path / "spine.parquet")):
        idx = (indexar(raiz) if raiz.is_dir()
               else {raiz.name: hash_arquivo(raiz)})
        me = ManifestoEtapa(
            etapa=etapa, descricao=f"etapa {etapa}",
            raiz=raiz.relative_to(tmp_path).as_posix(),
            entradas=[Entrada(caminho="fonte", nota="teste")],
            checksum_index=idx, git_sha="abc1234").selar()
        destino = (raiz / "_manifesto_etapa.json" if raiz.is_dir()
                   else raiz.parent / f"{raiz.name}_manifesto_etapa.json")
        destino.write_text(me.model_dump_json(indent=2), encoding="utf-8")
        refs.append(EtapaRef(tipo="processamento",
                             caminho=destino.relative_to(tmp_path).as_posix(),
                             etapa=etapa, manifesto_id=me.manifesto_id,
                             hash_manifesto=hash_arquivo(destino)))

    raiz_json = tmp_path / "MANIFESTO-RAIZ.json"
    raiz_json.write_text(
        ManifestoRaiz(git_sha="abc1234", etapas=refs).selar().model_dump_json(indent=2),
        encoding="utf-8")
    return tmp_path


def _verificar(base: Path, **kw):
    return verificar(base / "MANIFESTO-RAIZ.json", base=base, **kw)


def test_corpus_intacto_passa_nos_dois_niveis(corpus):
    """Sem isto, todos os outros testes passariam com um verificador que só reprova."""
    assert _verificar(corpus).ok
    assert _verificar(corpus, profundo=True).ok


def test_profunda_pega_parquet_alterado(corpus):
    """UM byte. É o caso que motiva a existência do índice de checksums."""
    p = corpus / "fatia" / "part-00000.parquet"
    b = bytearray(p.read_bytes())
    b[0] ^= 0x01
    p.write_bytes(bytes(b))

    r = _verificar(corpus, profundo=True)
    assert not r.ok
    assert [d.tipo for d in r.divergencias] == ["alterado"]
    assert "part-00000.parquet" in r.divergencias[0].onde


def test_rasa_nao_pega_parquet_alterado(corpus):
    """Comportamento ESPERADO, documentado como teste em vez de descoberto depois.

    A rasa confere a cadeia de manifestos, não o conteúdo. Chamar o resultado dela
    de "corpus verificado" seria ausência de erro lida como sucesso — o defeito que
    este projeto passou a semana corrigindo em outros lugares.
    """
    p = corpus / "fatia" / "part-00000.parquet"
    p.write_bytes(p.read_bytes() + b"lixo")

    assert _verificar(corpus).ok, "a rasa deveria passar — ela não olha conteúdo"
    assert not _verificar(corpus, profundo=True).ok, "a profunda tem de pegar"


def test_rasa_pega_manifesto_de_etapa_mexido(corpus):
    """Trocar o hash de um arquivo DENTRO do manifesto para 'legalizar' a fraude.

    É o ataque óbvio contra o índice de checksums: se alterar o parquet é pego,
    altere o índice também. A raiz guarda o hash do ARQUIVO de manifesto, então
    isso quebra um nível acima — e custa milissegundos descobrir.
    """
    m = corpus / "fatia" / "_manifesto_etapa.json"
    d = json.loads(m.read_text(encoding="utf-8"))
    d["checksum_index"]["part-00000.parquet"] = "0" * 64
    m.write_text(json.dumps(d, indent=2), encoding="utf-8")

    r = _verificar(corpus)
    assert not r.ok
    assert r.divergencias[0].tipo == "manifesto_alterado"


def test_raiz_editada_a_mao_e_pega(corpus):
    """Reescrever a lista de etapas na raiz sem recalcular o hash raiz."""
    rj = corpus / "MANIFESTO-RAIZ.json"
    d = json.loads(rj.read_text(encoding="utf-8"))
    d["etapas"][0]["hash_manifesto"] = "0" * 64
    rj.write_text(json.dumps(d, indent=2), encoding="utf-8")

    r = _verificar(corpus)
    assert not r.ok
    assert any(x.tipo == "raiz_alterada" for x in r.divergencias)


def test_profunda_pega_arquivo_apagado(corpus):
    (corpus / "fatia" / "part-00001.parquet").unlink()
    r = _verificar(corpus, profundo=True)
    assert not r.ok
    assert any(d.tipo == "ausente" for d in r.divergencias)


def test_profunda_pega_arquivo_a_mais(corpus):
    """Corpus com um parquet extra não é o corpus manifestado.

    Não é paranoia: foi assim que 40.000 duplicatas entraram no RedPajama quando a
    retomada tratou número de shard como índice de parquet. Um índice que ignora
    arquivo novo não teria acusado.
    """
    (corpus / "fatia" / "part-99999.parquet").write_bytes(b"intruso")
    r = _verificar(corpus, profundo=True)
    assert not r.ok
    assert any(d.tipo == "extra" and "part-99999" in d.onde for d in r.divergencias)


def test_manifesto_de_etapa_ausente_e_pego(corpus):
    (corpus / "fatia" / "_manifesto_etapa.json").unlink()
    r = _verificar(corpus)
    assert not r.ok
    assert any(d.tipo == "etapa_ausente" for d in r.divergencias)


def test_identidade_da_etapa_e_estavel_e_sensivel(corpus):
    """A identidade é reconstruível do arquivo, e muda quando algo real muda.

    Estável: reler o manifesto e recalcular dá o mesmo `manifesto_id`. Sem isso a
    verificação não poderia recomputar nada e teria de confiar no valor gravado —
    que é exatamente o que um adulterador editaria.

    Sensível: mudar um parâmetro muda o id. Um identificador estável por ser
    insensível não identifica; é constante.
    """
    m = json.loads((corpus / "fatia" / "_manifesto_etapa.json").read_text(encoding="utf-8"))
    antes = m["manifesto_id"]
    m.pop("manifesto_id")
    depois = ManifestoEtapa.model_validate(m).identidade()
    assert antes == depois

    # E o inverso: mudar algo REAL tem de mudar o id.
    m["parametros"] = {"limiar": 0.5}
    assert ManifestoEtapa.model_validate(m).identidade() != antes


def test_construir_duas_vezes_da_o_mesmo_hash(tmp_path, monkeypatch):
    """Idempotência: reconstruir o manifesto de um corpus INTACTO não muda o hash.

    É a propriedade sem a qual o G1.5 não serve para nada. Um hash que muda a cada
    construção não pode ir na capa de um paper, e — pior — treina quem lê a ignorar
    diferenças de hash, que é justamente o sinal que o critério existe para dar.

    Duas coisas ameaçavam isso, e as duas foram encontradas ao escrever este teste:
    `gerado_em` dentro da identidade da etapa, e o manifesto da construção anterior
    entrando no índice da seguinte.
    """
    import scripts.manifesto_corpus as mc

    (tmp_path / "data" / "processed" / "fatia").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "fatia" / "part-00000.parquet").write_bytes(b"x" * 50)
    (tmp_path / "data" / "raw").mkdir(parents=True)

    monkeypatch.setattr(mc, "ETAPAS", [{
        "etapa": "fatia", "descricao": "fatia de teste",
        "raiz": "data/processed/fatia", "entradas": [], "parametros": {"limiar": 0.9},
    }])

    a = mc.construir(tmp_path, rede=False)
    b = mc.construir(tmp_path, rede=False)
    assert a.hash_raiz == b.hash_raiz, (
        "o hash raiz mudou sem o corpus mudar — provavelmente algo volátil entrou "
        "na identidade, ou o manifesto da construção anterior entrou no índice")

    # E o contrapositivo: mexer no corpus TEM de mudar o hash.
    (tmp_path / "data" / "processed" / "fatia" / "part-00001.parquet").write_bytes(b"novo")
    assert mc.construir(tmp_path, rede=False).hash_raiz != a.hash_raiz


def test_etapa_declarada_e_ausente_e_erro_nao_silencio(tmp_path, monkeypatch):
    """Um manifesto de corpus INCOMPLETO que confere é o pior resultado possível.

    Se a tabela mestra faltar e o construtor apenas a omitir, o hash raiz é válido, a
    verificação passa, e o corpus não tem tabela mestra. A declaração explícita em
    `ETAPAS` existe para que ausência seja erro.
    """
    import scripts.manifesto_corpus as mc

    (tmp_path / "data" / "raw").mkdir(parents=True)
    monkeypatch.setattr(mc, "ETAPAS", [{
        "etapa": "tabela mestra_que_nao_existe", "descricao": "—",
        "raiz": "data/processed/nao_existe.parquet", "entradas": [], "parametros": {},
    }])
    with pytest.raises(SystemExit, match="ausentes no disco"):
        mc.construir(tmp_path, rede=False)


def test_manifesto_capturado_na_execucao_nao_e_sobrescrito(tmp_path, monkeypatch):
    """O construtor roda muito mais vezes que as etapas.

    Se ele sobrescrevesse, trocaria parâmetros de VERDADE por parâmetros
    reconstruídos do código a cada construção — proveniência boa perdida para
    proveniência adivinhada, sem aviso. É o defeito mais fácil de introduzir aqui,
    porque o caminho felizes é indistinguível.
    """
    import scripts.manifesto_corpus as mc

    from phifm.core.schema.reprodutibilidade import gravar_manifesto_etapa

    raiz = tmp_path / "data" / "processed" / "fatia"
    raiz.mkdir(parents=True)
    (raiz / "part-00000.parquet").write_bytes(b"dado" * 50)
    (tmp_path / "data" / "raw").mkdir(parents=True)

    capturado = gravar_manifesto_etapa(
        etapa="fatia", descricao="capturada na execução", raiz=raiz, base=tmp_path,
        parametros={"limiar": 0.87, "argumento_real": "só quem rodou sabe"},
        registros=42)
    assert capturado.parametros_reconstruidos is False

    monkeypatch.setattr(mc, "ETAPAS", [{
        "etapa": "fatia", "descricao": "descrição do construtor",
        "raiz": "data/processed/fatia", "entradas": [],
        "parametros": {"limiar": 0.9},   # o valor ERRADO, reconstruído
    }])
    mc.construir(tmp_path, rede=False)

    d = json.loads((raiz / "_manifesto_etapa.json").read_text(encoding="utf-8"))
    assert d["parametros"]["limiar"] == 0.87, "o construtor sobrescreveu o capturado"
    assert d["parametros"]["argumento_real"] == "só quem rodou sabe"
    assert d["parametros_reconstruidos"] is False
    assert d["registros"] == 42


def test_manifesto_capturado_mas_desatualizado_e_refeito(tmp_path, monkeypatch):
    """Preservar cegamente seria pior que sobrescrever.

    Se a etapa rodou de novo sem gravar o manifesto, ou alguém mexeu nos arquivos,
    o manifesto "capturado" descreve outro corpus. Aí o reconstruído é o melhor
    disponível, e preservar o antigo faria o hash raiz atestar bytes que não estão
    mais lá.
    """
    import scripts.manifesto_corpus as mc

    from phifm.core.schema.reprodutibilidade import gravar_manifesto_etapa

    raiz = tmp_path / "data" / "processed" / "fatia"
    raiz.mkdir(parents=True)
    (raiz / "part-00000.parquet").write_bytes(b"dado")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    gravar_manifesto_etapa(etapa="fatia", descricao="—", raiz=raiz, base=tmp_path,
                           parametros={"limiar": 0.87})

    (raiz / "part-00000.parquet").write_bytes(b"OUTRO CONTEUDO")   # a etapa rodou de novo

    monkeypatch.setattr(mc, "ETAPAS", [{
        "etapa": "fatia", "descricao": "—", "raiz": "data/processed/fatia",
        "entradas": [], "parametros": {"limiar": 0.9},
    }])
    mc.construir(tmp_path, rede=False)

    d = json.loads((raiz / "_manifesto_etapa.json").read_text(encoding="utf-8"))
    assert d["parametros_reconstruidos"] is True, (
        "manifesto obsoleto foi preservado — o hash raiz atestaria bytes ausentes")
    assert d["parametros"]["limiar"] == 0.9


def test_indice_usa_barra_normal_em_qualquer_sistema(tmp_path):
    """Índice com `\\` não confere contra índice com `/`.

    O corpus é construído no Windows e o G1.5 precisa ser verificável em Linux,
    senão o critério vale numa máquina só.
    """
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "c.parquet").write_bytes(b"x")
    idx = indexar(tmp_path)
    assert list(idx) == ["a/b/c.parquet"]
    assert not any("\\" in k for k in idx)


def test_transitorios_ficam_fora_do_indice(tmp_path):
    """`.tmp` de gravação atômica e log de execução não são conteúdo do corpus.

    Sem isto a verificação acusaria "extra" a cada treino rodando ao lado, e um
    verificador que dá alarme falso é um verificador que se aprende a ignorar.
    """
    (tmp_path / "part-00000.parquet").write_bytes(b"dado")
    (tmp_path / "estado_treino.pt.tmp").write_bytes(b"meio gravado")
    (tmp_path / "execucao.log").write_text("linha", encoding="utf-8")
    (tmp_path / "progresso.json").write_text("{}", encoding="utf-8")
    assert list(indexar(tmp_path)) == ["part-00000.parquet"]

def test_manifesto_de_outra_etapa_nao_e_sobrescrito_em_silencio(tmp_path):
    """Duas etapas gravando na MESMA raiz apagavam a proveniência uma da outra.

    Aconteceu em 2026-08-24: um teste de fumaça de `minerar_do_recuperador.py` com
    400 âncoras sobrescreveu o manifesto que estava em
    `data/processed/negativos_dificeis/`. O diretório não é versionado, então não
    havia como recuperar.

    Perder proveniência sem aviso é o oposto do que este módulo faz, então ele para.
    """
    import pytest

    from phifm.core.schema.reprodutibilidade import gravar_manifesto_etapa

    raiz = tmp_path / "saida"
    raiz.mkdir()
    (raiz / "dados.parquet").write_bytes(b"x" * 32)

    gravar_manifesto_etapa(etapa="primeira", descricao="—", raiz=raiz,
                           base=tmp_path, registros=1)
    with pytest.raises(RuntimeError, match="primeira"):
        gravar_manifesto_etapa(etapa="segunda", descricao="—", raiz=raiz,
                               base=tmp_path, registros=1)


def test_regravar_o_manifesto_da_propria_etapa_continua_livre(tmp_path):
    """Reexecutar uma etapa e regravar o manifesto dela é o fluxo normal.

    A guarda acima não pode transformar idempotência em erro — senão retomar uma
    coleta viraria falha.
    """
    from phifm.core.schema.reprodutibilidade import gravar_manifesto_etapa

    raiz = tmp_path / "saida"
    raiz.mkdir()
    (raiz / "dados.parquet").write_bytes(b"y" * 16)

    a = gravar_manifesto_etapa(etapa="mesma", descricao="—", raiz=raiz,
                               base=tmp_path, registros=1)
    b = gravar_manifesto_etapa(etapa="mesma", descricao="—", raiz=raiz,
                               base=tmp_path, registros=2)
    assert a.etapa == b.etapa == "mesma"
    assert b.registros == 2


# ── a lacuna de 18 GB, e a guarda contra a próxima ──────────────────────────

def test_TODA_fonte_do_filtrar_hf_tem_etapa_no_manifesto():
    """⚠️ O peS2o ficou 16 dias fora do manifesto raiz, e nada acusou.

    Ele foi filtrado em 2026-08-26 e dobrou o corpus — 14,60 B tokens dos 27,75 B,
    18 GB em disco. Ninguém o acrescentou a `ETAPAS`, e o manifesto raiz seguiu
    atestando **21,79 GB** de um corpus de ~40 GB. O ESTADO.md dizia "o corpus por
    um hash"; era 55% do corpus por um hash.

    Nada acusava porque o construtor só percorre `ETAPAS`: uma etapa que não está
    lá não existe para ele, e o relatório sai ✅ sobre o que ele conhece. **Um
    verificador que passa não prova que o que ficou de fora está íntegro — prova
    que ele não olhou.**

    Esta guarda é código contra código: as duas tabelas são constantes de módulo,
    então ela roda sem `data/processed/` existir e pega a lacuna no CI, não meses
    depois numa contagem de bytes feita à mão.
    """
    import scripts.filtrar_hf as fh
    import scripts.manifesto_corpus as mc

    etapas = {e["etapa"] for e in mc.ETAPAS}
    faltando = {f"{fonte}_fisica" for fonte in fh.FONTES
                if f"{fonte}_fisica" not in etapas}
    assert not faltando, (
        f"{sorted(faltando)} saem de `filtrar_hf.FONTES` e não têm entrada em "
        "`manifesto_corpus.ETAPAS`. O manifesto raiz atestaria um corpus menor "
        "que o do disco, e o relatório de verificação diria ✅ assim mesmo.")


def test_as_etapas_do_filtrar_hf_declaram_a_FONTE_que_as_produz():
    """O `--fonte` é o que liga a entrada de `ETAPAS` ao comando que a gerou.

    Sem ele, duas etapas produzidas pelo mesmo script ficam indistinguíveis nos
    parâmetros reconstruídos — e foi por elas parecerem a mesma coisa que a
    segunda passou despercebida.
    """
    import scripts.filtrar_hf as fh
    import scripts.manifesto_corpus as mc

    porp = {e["etapa"]: e["parametros"] for e in mc.ETAPAS}
    for fonte in fh.FONTES:
        params = porp.get(f"{fonte}_fisica")
        assert params is not None, fonte
        assert params.get("script") == "scripts/filtrar_hf.py", params
        assert params.get("fonte") == fonte, (
            f"{fonte}_fisica não diz de que `--fonte` veio: {params}")


def test_o_prefixo_do_pes2o_esta_nos_PARAMETROS():
    """⚠️ `data/v2/` é a diferença entre 14,6 B tokens e um corpus com metade
    duplicada: o repositório publica v1 e v2 da MESMA coleção, e o S3b mediu que
    duplicação em pré-treino degrada 16,6%.

    Um parâmetro que decide isso não pode existir só dentro do script.
    """
    import scripts.filtrar_hf as fh
    import scripts.manifesto_corpus as mc

    _, _, prefixo = fh.FONTES["pes2o"]
    assert prefixo == "data/v2/"
    etapa = next(e for e in mc.ETAPAS if e["etapa"] == "pes2o_fisica")
    assert etapa["parametros"].get("prefixo") == prefixo, (
        "o prefixo do código e o do manifesto divergiram")


def test_o_filtrar_hf_GRAVA_o_proprio_manifesto_ao_terminar():
    """É isto que derruba `parametros_reconstruidos` para as duas fatias de HF.

    Por AST: importar o script executaria o `sys.path.insert` e os imports
    pesados, e o que interessa é se a chamada existe no fim do `main`.
    """
    import ast

    raiz = Path(__file__).resolve().parents[2]
    fonte = (raiz / "scripts/filtrar_hf.py").read_text(encoding="utf-8")
    main = next(n for n in ast.walk(ast.parse(fonte))
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    chamadas = {ast.unparse(n.func) for n in ast.walk(main)
                if isinstance(n, ast.Call)}
    assert "gravar_manifesto_etapa" in chamadas, (
        "sem esta chamada os parâmetros voltam a ser reconstruídos do código")
    # E os parâmetros que mais importam vão para lá: o limiar e o prefixo.
    chamada = next(n for n in ast.walk(main) if isinstance(n, ast.Call)
                   and ast.unparse(n.func) == "gravar_manifesto_etapa")
    params = next(k.value for k in chamada.keywords if k.arg == "parametros")
    chaves = {ast.literal_eval(c) for c in params.keys if c is not None}
    for exigida in ("limiar", "prefixo", "revisao", "fonte", "dataset"):
        assert exigida in chaves, f"{exigida} não é capturado: {sorted(chaves)}"


# ── a captura na execução, o que fecha a metade aberta do G1.5 ──────────────

def test_TODO_script_de_etapa_grava_o_proprio_manifesto():
    """⚠️ Sem esta chamada os parâmetros voltam a ser reconstruídos do código.

    E reconstruir não é uma imprecisão formal: `precision=0.95` do classificador
    governa o que entra no corpus, e o `prefixo` do peS2o é a diferença entre
    14,6 B tokens e um corpus com metade duplicada. Um parâmetro reconstruído dá
    o valor de HOJE, não o da execução que produziu o artefato — e um valor
    errado assim não tem com o que ser confrontado.

    Por AST: importar estes scripts carregaria polars e torch, e o que interessa
    é se a chamada existe no `main`.
    """
    import ast

    raiz = Path(__file__).resolve().parents[2]
    scripts = ["build_spine.py", "build_pairs.py", "train_classifier.py",
               "coletar_redpajama.py", "filtrar_hf.py"]
    sem_captura = []
    for nome in scripts:
        fonte = (raiz / "scripts" / nome).read_text(encoding="utf-8")
        main = next((n for n in ast.walk(ast.parse(fonte))
                     if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
        assert main is not None, f"{nome} não tem main()"
        chamadas = {ast.unparse(n.func) for n in ast.walk(main)
                    if isinstance(n, ast.Call)}
        if "gravar_manifesto_etapa" not in chamadas:
            sem_captura.append(nome)
    assert not sem_captura, (
        f"{sem_captura} produzem etapas do corpus e não gravam o próprio "
        "manifesto — os parâmetros delas continuam sendo adivinhados do código.")


def test_as_entradas_usam_entrada_de_para_formar_a_CADEIA():
    """⚠️ É o `manifesto_id` que faz a cadeia, e `Entrada(caminho=...)` cru não o
    tem.

    O `coletar_redpajama.py` capturava o próprio manifesto desde o começo e
    montava a entrada sem id: a etapa parecia bem atestada e a proveniência dela
    terminava na primeira aresta. Um elo assim não permite ir da saída até a
    fonte.
    """
    import ast

    raiz = Path(__file__).resolve().parents[2]
    for nome in ("build_spine.py", "build_pairs.py", "train_classifier.py",
                 "coletar_redpajama.py"):
        fonte = (raiz / "scripts" / nome).read_text(encoding="utf-8")
        main = next(n for n in ast.walk(ast.parse(fonte))
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        chamada = next(n for n in ast.walk(main) if isinstance(n, ast.Call)
                       and ast.unparse(n.func) == "gravar_manifesto_etapa")
        entradas = next((k.value for k in chamada.keywords
                         if k.arg == "entradas"), None)
        assert entradas is not None, f"{nome} não declara `entradas`"
        texto = ast.unparse(entradas)
        assert "entrada_de" in texto, (
            f"{nome} monta entradas sem `entrada_de`: {texto}. Sem o "
            "`manifesto_id` a cadeia para aqui.")


def test_entrada_de_acha_o_id_nas_TRES_formas(tmp_path):
    """O repositório grava manifesto de três jeitos, e a cadeia tem de encontrar
    os três: `_manifest.json` dos coletores, `_manifesto_etapa.json` dentro de um
    diretório, e `<arquivo>_manifesto_etapa.json` ao lado de um parquet único."""
    from phifm.core.schema.reprodutibilidade import entrada_de

    # 1. coletor: `manifest_id`, com a grafia inglesa
    d1 = tmp_path / "bruto"
    d1.mkdir()
    (d1 / "_manifest.json").write_text(json.dumps({"manifest_id": "aaa111"}))
    assert entrada_de(d1).manifesto_id == "aaa111"

    # 2. etapa em diretório
    d2 = tmp_path / "fatia"
    d2.mkdir()
    (d2 / "_manifesto_etapa.json").write_text(json.dumps({"manifesto_id": "bbb222"}))
    assert entrada_de(d2).manifesto_id == "bbb222"

    # 3. etapa cuja saída é UM arquivo: o manifesto fica ao lado
    arq = tmp_path / "spine.parquet"
    arq.write_bytes(b"x")
    (tmp_path / "spine.parquet_manifesto_etapa.json").write_text(
        json.dumps({"manifesto_id": "ccc333"}))
    assert entrada_de(arq).manifesto_id == "ccc333"


def test_entrada_de_sem_manifesto_DIZ_que_nao_tem(tmp_path):
    """⚠️ Entrada externa ao pipeline é legítima; o que não pode é ser calada.

    Devolver a entrada sem id e sem nota faria a cadeia parecer completa quando
    ela termina ali.
    """
    from phifm.core.schema.reprodutibilidade import entrada_de

    solto = tmp_path / "de_fora"
    solto.mkdir()
    e = entrada_de(solto)
    assert e.manifesto_id is None
    assert e.nota and "sem manifesto" in e.nota


def test_entrada_de_nao_LEVANTA_com_json_corrompido(tmp_path):
    """Um manifesto ilegível a montante não pode derrubar a etapa no fim, depois
    do trabalho feito — ele vira "sem manifesto", que é a verdade."""
    from phifm.core.schema.reprodutibilidade import entrada_de

    d = tmp_path / "quebrado"
    d.mkdir()
    (d / "_manifest.json").write_text("{isso nao e json")
    e = entrada_de(d)
    assert e.manifesto_id is None
