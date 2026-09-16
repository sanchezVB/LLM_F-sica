"""O fluxo de dados sem estado do DOC-08 §7.2.

O documento diz que retomar a posição no fluxo é "onde a maioria das implementações
falha silenciosamente — e o efeito é revisitar ou pular dados, quebrando a política
de épocas e tornando a execução irreprodutível".

Silenciosamente é a palavra que justifica estes testes: um fluxo que pula 3% dos
dados treina normalmente, converge normalmente, e o número final é irreprodutível
sem que nada apareça no log.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.training.pretrain.dados import (  # noqa: E402
    BIT_DISPLAY,
    BIT_INICIO,
    BIT_MATH,
    NOME_MANIFESTO,
    NOME_MARCAS,
    NOME_TOKENS,
    ConfigDados,
    Fluxo,
    desempacotar,
    marcas_de,
)


def _corpus(tmp_path: Path, n_tokens: int = 8192 * 12, semente: int = 5) -> Path:
    """Um corpus sintético em disco, no formato que a preparação grava."""
    raiz = tmp_path / "dados"
    raiz.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(semente)
    ids = rng.integers(5, 40960, size=n_tokens, dtype=np.uint16)
    # Equações a cada ~500 tokens, de 40 tokens, alternando display e inline.
    ide = np.full(n_tokens, -1, dtype=np.int32)
    disp = np.zeros(n_tokens, dtype=bool)
    for k, i in enumerate(range(300, n_tokens - 60, 500)):
        ide[i:i + 40] = k
        disp[i:i + 40] = k % 2 == 0
    (raiz / NOME_TOKENS).write_bytes(ids.tobytes())
    (raiz / NOME_MARCAS).write_bytes(marcas_de(ide, disp).tobytes())
    (raiz / NOME_MANIFESTO).write_text(
        json.dumps({"tokens": int(n_tokens), "git_sha": "abc1234",
                    "tokenizer": "variante_A.json"}), encoding="utf-8")
    return raiz


# ─── os três bits ────────────────────────────────────────────────────────────


def test_ida_e_volta_preserva_o_agrupamento_e_o_display():
    """Os ids podem ser RENUMERADOS — o `desempacotar` conta inícios —, mas a
    partição dos tokens e o tipo de cada equação têm de sobreviver."""
    rng = np.random.default_rng(11)
    for _ in range(50):
        n = int(rng.integers(20, 500))
        ide = np.full(n, -1, dtype=np.int32)
        disp = np.zeros(n, dtype=bool)
        k = 0
        i = 0
        while i < n - 5:
            if rng.random() < 0.4:
                tam = int(rng.integers(2, 30))
                ide[i:i + tam] = k
                disp[i:i + tam] = bool(rng.random() < 0.5)
                k += 1
                i += tam
            i += int(rng.integers(1, 20))
        ide2, disp2 = desempacotar(marcas_de(ide, disp))
        assert ((ide >= 0) == (ide2 >= 0)).all(), "a partição math/não-math mudou"
        assert (disp == disp2).all(), "o tipo display/inline mudou"
        b1 = np.flatnonzero(np.diff(ide, prepend=-99) != 0)
        b2 = np.flatnonzero(np.diff(ide2, prepend=-99) != 0)
        assert b1.tolist() == b2.tolist(), "as fronteiras entre equações mudaram"


def test_equacao_cortada_pela_janela_fica_FORA_do_tratamento():
    """⚠️ A propriedade que os três bits dão de graça.

    Se a janela começa no meio de uma equação, aquele trecho não tem o bit de
    início e recebe id −1. Mascarar "a equação inteira" quando só metade dela está
    na janela seria mascarar metade e chamar de inteira — e o alvo teria uma
    continuação que o modelo não pode ver nem deduzir.
    """
    # equação inteira dos tokens 0..29 do corpus original
    ide = np.full(60, -1, dtype=np.int32)
    disp = np.zeros(60, dtype=bool)
    ide[0:30] = 0
    disp[0:30] = True
    marcas = marcas_de(ide, disp)

    # a janela pega de 10 em diante: perdeu o bit de início
    ide_cortado, disp_cortado = desempacotar(marcas[10:])
    assert (ide_cortado[:20] == -1).all(), (
        "a equação truncada recebeu id válido e entraria no tratamento")
    # e o display continua marcado, para a estatística não mentir
    assert disp_cortado[:20].all()

    # a janela que pega o início inteiro trata normalmente
    ide_ok, _ = desempacotar(marcas)
    assert (ide_ok[0:30] == 0).all()


def test_os_bits_sao_independentes():
    ide = np.array([-1, 0, 0, 1], dtype=np.int32)
    disp = np.array([False, True, True, False])
    m = marcas_de(ide, disp)
    assert m[0] == 0
    assert m[1] & BIT_MATH and m[1] & BIT_DISPLAY and m[1] & BIT_INICIO
    assert m[2] & BIT_MATH and m[2] & BIT_DISPLAY and not m[2] & BIT_INICIO
    assert m[3] & BIT_MATH and not m[3] & BIT_DISPLAY and m[3] & BIT_INICIO


# ─── o determinismo, que é o ponto do §7.2 ───────────────────────────────────


def test_o_mesmo_passo_da_o_mesmo_lote_em_execucoes_diferentes(tmp_path):
    """A asserção central. É isto que faz retomar do passo 60.000 produzir a mesma
    sequência que uma execução contínua produziria."""
    raiz = _corpus(tmp_path)
    cfg = ConfigDados(raiz=raiz, contexto=8192, sequencias=2)
    a, b = Fluxo(cfg), Fluxo(cfg)
    for passo in (0, 1, 13, 97, 500):
        assert a.indices_do_passo(passo) == b.indices_do_passo(passo)
        ia, _, _ = a.lote(passo)
        ib, _, _ = b.lote(passo)
        assert (ia == ib).all()


def test_uma_epoca_e_exatamente_uma_passagem(tmp_path):
    """⚠️ Permutação por época, não sorteio por passo.

    Com sorteio independente o modelo veria a mesma sequência duas vezes antes de
    ver outras, e a política de épocas do DOC-06 §2.4 seria só um nome. Com
    permutação, uma época é exatamente uma passagem — e o teste conta.
    """
    raiz = _corpus(tmp_path)
    f = Fluxo(ConfigDados(raiz=raiz, contexto=8192, sequencias=1))
    vistos = [f.indices_do_passo(p)[0] for p in range(f.n_seq)]
    assert sorted(vistos) == list(range(f.n_seq)), (
        "uma época não cobriu cada sequência exatamente uma vez")
    # e a época seguinte é outra ordem
    seguintes = [f.indices_do_passo(p)[0] for p in range(f.n_seq, 2 * f.n_seq)]
    assert sorted(seguintes) == list(range(f.n_seq))
    assert vistos != seguintes, "a segunda época repetiu a ordem da primeira"


def test_sementes_diferentes_dao_ordens_diferentes(tmp_path):
    raiz = _corpus(tmp_path)
    a = Fluxo(ConfigDados(raiz=raiz, contexto=8192, semente=17))
    b = Fluxo(ConfigDados(raiz=raiz, contexto=8192, semente=18))
    assert [a.indices_do_passo(p) for p in range(20)] != \
           [b.indices_do_passo(p) for p in range(20)]


def test_retomar_no_meio_nao_pula_nem_repete(tmp_path):
    """Simula a queda de sessão: um Fluxo novo, começando do passo 40, tem de
    devolver a mesma cauda que o Fluxo contínuo devolveria."""
    raiz = _corpus(tmp_path)
    cfg = ConfigDados(raiz=raiz, contexto=8192, sequencias=2)
    continuo = [Fluxo(cfg).indices_do_passo(p) for p in range(60)]
    retomado = Fluxo(cfg)  # instância nova, sem estado
    assert [retomado.indices_do_passo(p) for p in range(40, 60)] == continuo[40:60]


def test_a_epoca_do_passo_cresce_e_conta_certo(tmp_path):
    raiz = _corpus(tmp_path)
    f = Fluxo(ConfigDados(raiz=raiz, contexto=8192, sequencias=1))
    assert f.epoca_do_passo(f.n_seq - 1) == pytest.approx(1.0)
    assert f.epoca_do_passo(2 * f.n_seq - 1) == pytest.approx(2.0)


def test_passo_negativo_levanta(tmp_path):
    f = Fluxo(ConfigDados(raiz=_corpus(tmp_path), contexto=8192))
    with pytest.raises(ValueError, match="negativo"):
        f.indices_do_passo(-1)


# ─── as guardas contra dado corrompido ───────────────────────────────────────


def test_manifesto_ausente_diz_o_que_rodar(tmp_path):
    raiz = tmp_path / "vazio"
    raiz.mkdir()
    with pytest.raises(SystemExit, match="preparar_dados_phienc"):
        Fluxo(ConfigDados(raiz=raiz))


def test_tokens_e_marcas_de_tamanhos_diferentes_levantam(tmp_path):
    """Os dois são gravados juntos; um desencontro é preparação interrompida no
    meio, e treinar sobre isso alinharia marcas com os tokens errados."""
    raiz = _corpus(tmp_path)
    m = np.frombuffer((raiz / NOME_MARCAS).read_bytes(), dtype=np.uint8)
    (raiz / NOME_MARCAS).write_bytes(m[:-100].tobytes())
    with pytest.raises(SystemExit, match="marcas"):
        Fluxo(ConfigDados(raiz=raiz))


def test_manifesto_que_declara_outro_total_levanta(tmp_path):
    """Treinar sobre um prefixo silencioso mudaria a política de épocas sem avisar."""
    raiz = _corpus(tmp_path)
    man = json.loads((raiz / NOME_MANIFESTO).read_text(encoding="utf-8"))
    man["tokens"] = man["tokens"] + 1000
    (raiz / NOME_MANIFESTO).write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(SystemExit, match="declara"):
        Fluxo(ConfigDados(raiz=raiz))


def test_corpus_menor_que_uma_sequencia_levanta(tmp_path):
    raiz = _corpus(tmp_path, n_tokens=100)
    with pytest.raises(SystemExit, match="não dão uma sequência"):
        Fluxo(ConfigDados(raiz=raiz, contexto=8192))


def test_config_invalida_levanta(tmp_path):
    for kw in ({"contexto": 0}, {"sequencias": 0}):
        with pytest.raises(ValueError, match="positivos"):
            ConfigDados(raiz=tmp_path, **kw)


# ─── a forma do lote ─────────────────────────────────────────────────────────


def test_o_lote_tem_a_forma_e_os_tipos_certos(tmp_path):
    raiz = _corpus(tmp_path)
    f = Fluxo(ConfigDados(raiz=raiz, contexto=8192, sequencias=3))
    ids, ide, disp = f.lote(0)
    assert ids.shape == ide.shape == disp.shape == (3, 8192)
    assert ids.dtype == np.int64, "os ids têm de sair em int64 para o embedding"
    assert disp.dtype == bool
    assert f.tokens_por_passo() == 3 * 8192


def test_como_dict_carrega_a_proveniencia_da_preparacao(tmp_path):
    """O JSON do treino tem de dizer de que preparação os dados vieram: um corpus
    retokenizado com outro tokenizer daria outra perda pelo mesmo passo."""
    d = Fluxo(ConfigDados(raiz=_corpus(tmp_path), contexto=8192)).como_dict()
    assert d["preparacao"] == "abc1234"
    assert d["tokenizer"] == "variante_A.json"
    assert d["tokens_por_micro_passo"] == 8192


# ─── o corte no FIM da janela (2026-09-16) ───────────────────────────────────


def test_equacao_cortada_no_FIM_fica_fora_do_tratamento():
    """⚠️ O caso que escapava: a última equação da janela tem o bit de início e
    recebia id válido. Medido a 1.024 de contexto, 9,5% das escolhidas."""
    ide = np.full(20, -1, dtype=np.int32)
    disp = np.zeros(20, dtype=bool)
    ide[2:6], disp[2:6] = 0, True      # inteira
    ide[15:20], disp[15:20] = 1, True  # vai além da janela
    marcas = marcas_de(ide, disp)

    cortado, disp_c = desempacotar(marcas, continua_depois=True)
    assert (cortado[15:20] == -1).all(), "a equação cortada no fim entraria como inteira"
    assert (cortado[2:6] == 0).all(), "a correção não pode tocar a equação inteira"
    # as marcas de display continuam, para a estatística não mentir
    assert disp_c[15:20].all()

    sem_corte, _ = desempacotar(marcas, continua_depois=False)
    assert (sem_corte[15:20] == 1).all()


def test_continua_depois_nao_faz_nada_se_a_janela_termina_em_PROSA():
    ide = np.full(10, -1, dtype=np.int32)
    ide[2:5] = 0
    ide2, _ = desempacotar(marcas_de(ide, np.zeros(10, dtype=bool)),
                           continua_depois=True)
    assert (ide2[2:5] == 0).all()


def _corpus_com_equacao_na_fronteira(tmp_path: Path, seguinte_comeca_outra: bool) -> Path:
    raiz = tmp_path / "fronteira"
    raiz.mkdir()
    n = 64
    ids = np.arange(5, 5 + n, dtype=np.uint16)
    ide = np.full(n, -1, dtype=np.int32)
    disp = np.zeros(n, dtype=bool)
    if seguinte_comeca_outra:
        ide[28:32], disp[28:32] = 0, True   # termina exatamente na fronteira
        ide[32:36], disp[32:36] = 1, True   # e outra começa logo depois
    else:
        ide[28:36], disp[28:36] = 0, True   # atravessa a fronteira em 32
    (raiz / NOME_TOKENS).write_bytes(ids.tobytes())
    (raiz / NOME_MARCAS).write_bytes(marcas_de(ide, disp).tobytes())
    (raiz / NOME_MANIFESTO).write_text(json.dumps({"tokens": n}), encoding="utf-8")
    return raiz


def test_o_Fluxo_ve_a_equacao_que_ATRAVESSA_a_fronteira(tmp_path):
    f = Fluxo(ConfigDados(raiz=_corpus_com_equacao_na_fronteira(tmp_path, False),
                          contexto=32))
    _, ide, _ = f.sequencia(0)
    assert (ide[28:32] == -1).all()
    # e a janela seguinte já a descartava pelo corte no começo
    _, ide1, _ = f.sequencia(1)
    assert (ide1[0:4] == -1).all()


def test_o_Fluxo_NAO_descarta_equacao_que_termina_na_fronteira(tmp_path):
    """Terminar exatamente no fim da janela, com outra equação começando no token
    seguinte, é equação inteira. Descartá-la seria jogar tratamento fora."""
    f = Fluxo(ConfigDados(raiz=_corpus_com_equacao_na_fronteira(tmp_path, True),
                          contexto=32))
    _, ide, _ = f.sequencia(0)
    assert (ide[28:32] == 0).all()
    _, ide1, _ = f.sequencia(1)
    assert (ide1[0:4] == 0).all()


def test_o_CONTROLE_nao_muda_com_a_correcao():
    """⚠️ O braço de controle (E) já foi treinado com o código antigo. Com
    `p_equacao=0` a seleção nem é chamada e o gerador não é consumido: a mesma
    janela tem de dar a MESMA máscara com e sem a correção, byte a byte."""
    from phifm.training.pretrain.mascaramento import ConfigMascara, mascarar

    rng_dados = np.random.default_rng(3)
    for caso in range(50):
        n = 256
        ids = rng_dados.integers(5, 40960, size=n).astype(np.int64)
        ide = np.full(n, -1, dtype=np.int32)
        disp = np.zeros(n, dtype=bool)
        ide[200:256], disp[200:256] = 0, True
        marcas = marcas_de(ide, disp)
        saidas = []
        for continua in (False, True):
            ide_w, disp_w = desempacotar(marcas, continua_depois=continua)
            saidas.append(mascarar(
                ids, ide_w, disp_w, cfg=ConfigMascara(p_equacao=0.0),
                rng=np.random.default_rng((17, caso)), id_mask=4, n_vocab=40960,
                ids_especiais=frozenset(range(5))))
        assert (saidas[0][0] == saidas[1][0]).all()
        assert (saidas[0][1] == saidas[1][1]).all()
