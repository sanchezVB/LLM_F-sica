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
    (tmp_path / "espinha.parquet").write_bytes(b"espinha" * 500)

    refs = []
    for etapa, raiz in (("fatia", tmp_path / "fatia"),
                        ("espinha", tmp_path / "espinha.parquet")):
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

    Se a espinha faltar e o construtor apenas a omitir, o hash raiz é válido, a
    verificação passa, e o corpus não tem espinha. A declaração explícita em
    `ETAPAS` existe para que ausência seja erro.
    """
    import scripts.manifesto_corpus as mc

    (tmp_path / "data" / "raw").mkdir(parents=True)
    monkeypatch.setattr(mc, "ETAPAS", [{
        "etapa": "espinha_que_nao_existe", "descricao": "—",
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
