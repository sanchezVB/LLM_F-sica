"""A medida que decide o T2a, e os dois jeitos de ela mentir.

Bits por byte existe porque acurácia de MLM **não compara tokenizers**: quem parte
o texto em pedaços menores acerta mais sem ser modelo melhor. Estes testes fixam
que a conta usa TEXTO no denominador, que o sinal dos log-probs é conferido, e que
o pareamento é por documento.

⚠️ O teste central é `test_a_acuracia_favorece_o_tokenizer_pior`: ele constrói o
caso em que acurácia e bits por byte discordam, que é o caso pelo qual a medida
foi trocada.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval.bits_por_byte import (  # noqa: E402
    Acumulador,
    bootstrap_pareado,
    confronto,
    posicoes_nas_unidades,
    sortear_unidades,
    unidades_comuns,
)


def _doc(acum: Acumulador, probs, bytes_tok, certo=None) -> None:
    lp = np.log(np.asarray(probs, dtype=np.float64))
    bt = np.asarray(bytes_tok, dtype=np.int64)
    ct = np.ones_like(bt) if certo is None else np.asarray(certo)
    acum.somar(lp, bt, ct)


def test_bits_por_byte_divide_por_TEXTO_e_nao_por_token():
    """Dois modelos igualmente bons, um com tokens do dobro do tamanho: em bits
    por TOKEN o de tokens longos parece pior, em bits por BYTE eles empatam."""
    # Longo: 2 tokens de 8 bytes, p=0,25 cada -> 2 bits cada -> 4 bits / 16 bytes
    longo = Acumulador()
    _doc(longo, [0.25, 0.25], [8, 8])
    # Curto: 4 tokens de 4 bytes, p=0,5 cada -> 1 bit cada -> 4 bits / 16 bytes
    curto = Acumulador()
    _doc(curto, [0.5, 0.5, 0.5, 0.5], [4, 4, 4, 4])

    dl, dc = longo.como_dict(), curto.como_dict()
    assert dl["bits_por_byte"] == dc["bits_por_byte"] == 0.25
    # E em bits por token eles NÃO empatam — que é o artefato evitado.
    assert dl["bits_por_token"] != dc["bits_por_token"]


def test_a_acuracia_favorece_o_tokenizer_pior():
    """⚠️ O caso pelo qual a medida foi trocada.

    O tokenizer de tokens curtos acerta mais posições — a vizinhança deixa mais
    redundância — e mesmo assim gasta MAIS bits para reconstruir o mesmo texto.
    Quem lê acurácia conclui o contrário de quem lê bits por byte, e quem está
    certo é bits por byte, porque o texto reconstruído é o mesmo.
    """
    # Tokens longos: acerta 1 de 2, mas com probabilidade alta no que acerta.
    longo = Acumulador()
    _doc(longo, [0.9, 0.35], [8, 8], certo=[1, 0])
    # Tokens curtos: acerta 3 de 4 (acurácia maior), com probabilidades medianas.
    curto = Acumulador()
    _doc(curto, [0.6, 0.6, 0.6, 0.3], [4, 4, 4, 4], certo=[1, 1, 1, 0])

    dl, dc = longo.como_dict(), curto.como_dict()
    assert dc["acuracia_diagnostica"] > dl["acuracia_diagnostica"], (
        "o caso construído exige que o de tokens curtos acerte mais")
    assert dl["bits_por_byte"] < dc["bits_por_byte"], (
        "e que ele gaste mais bits pelo mesmo texto — é essa discordância que "
        "a troca de medida resolve")


def test_a_nota_avisa_que_a_acuracia_nao_decide():
    d = Acumulador()
    _doc(d, [0.5], [4])
    assert "NÃO compara tokenizers" in d.como_dict()["nota"]
    assert "acuracia_diagnostica" in d.como_dict()


def test_log_prob_positivo_e_RECUSADO():
    """⚠️ Com o sinal trocado o melhor modelo sairia como o pior, e a conta não
    daria erro nenhum — só inverteria o resultado do experimento."""
    a = Acumulador()
    with pytest.raises(ValueError, match="log-prob positivo"):
        a.somar(np.array([0.7]), np.array([4]), np.array([1]))


def test_formas_desalinhadas_sao_RECUSADAS():
    a = Acumulador()
    with pytest.raises(ValueError, match="formas diferentes"):
        a.somar(np.array([-0.7, -0.2]), np.array([4]), np.array([1, 0]))


def test_a_media_global_pesa_documentos_pelo_TEXTO_que_esconderam():
    """Um documento que escondeu 10 bytes não pode pesar o mesmo que um que
    escondeu 1.000 numa média de bits por byte."""
    a = Acumulador()
    _doc(a, [0.5], [1])            # 1 bit / 1 byte  = 1,0
    _doc(a, [0.5] * 99, [1] * 99)  # 99 bits / 99 bytes = 1,0
    assert a.como_dict()["bits_por_byte"] == 1.0
    b = Acumulador()
    _doc(b, [0.25], [1])           # 2 bits / 1 byte = 2,0 num doc minúsculo
    _doc(b, [0.5] * 99, [1] * 99)  # 1,0 em 99 bytes
    # A média ponderada por bytes fica perto de 1,0, e não de 1,5.
    assert 1.0 < b.como_dict()["bits_por_byte"] < 1.02


def test_o_bootstrap_reamostra_DOCUMENTOS_e_e_pareado():
    """A mesma reamostra vale para os dois braços: eles viram os mesmos
    documentos, e reamostrar cada um por conta própria jogaria fora o
    pareamento — de onde vem o poder."""
    rng = np.random.default_rng(3)
    base = rng.normal(1.0, 0.2, size=200)
    a = base
    b = base + 0.05  # b é consistentemente pior por pouco
    r = bootstrap_pareado(a, b, semente=17)
    assert r["vence"] == "a", "a diferença é pequena mas sistemática"
    assert not r["cruza_zero"]
    assert r["diferenca_media"] < 0

    # Sem pareamento, um efeito desse tamanho sumiria no ruído entre documentos.
    independente = rng.normal(1.05, 0.2, size=200)
    solto = bootstrap_pareado(a, independente, semente=17)
    assert solto["cruza_zero"] or abs(solto["diferenca_media"]) > 0.02


def test_o_empate_e_quando_o_IC_cruza_zero():
    """⚠️ O sinal do IC decide, não a diferença pontual: uma diferença grande com
    IC que cruza zero é empate."""
    rng = np.random.default_rng(5)
    a = rng.normal(1.0, 1.0, size=30)
    b = a + rng.normal(0.0, 1.0, size=30)
    r = bootstrap_pareado(a, b, semente=17)
    if r["cruza_zero"]:
        assert r["vence"] is None
    assert "MENOR é melhor" in r["nota"]


def test_menor_e_melhor_esta_dito_na_saida():
    """Bits por byte é custo. Sem essa linha, `vence` se lê ao contrário."""
    r = bootstrap_pareado(np.array([1.0, 1.0]), np.array([2.0, 2.0]))
    assert r["vence"] == "a"
    assert "MENOR é melhor" in r["nota"]


def test_o_confronto_RECUSA_contagens_de_documento_diferentes():
    """⚠️ Se um braço pulou um documento por erro de tokenização, o pareamento
    passa a comparar textos diferentes e nada mais acusaria."""
    a, b = Acumulador(), Acumulador()
    _doc(a, [0.5], [4])
    _doc(a, [0.5], [4])
    _doc(b, [0.5], [4])
    with pytest.raises(ValueError, match="documentos"):
        confronto("A", a, "E", b)


def test_o_confronto_nomeia_o_vencedor_pelo_ROTULO():
    a, b = Acumulador(), Acumulador()
    for _ in range(40):
        _doc(a, [0.5], [4])     # 1 bit / 4 bytes = 0,25
        _doc(b, [0.25], [4])    # 2 bits / 4 bytes = 0,50
    r = confronto("variante A", a, "variante E", b)
    assert r["vencedor"] == "variante A"
    assert r["bits_por_byte"]["variante A"] < r["bits_por_byte"]["variante E"]


# ── o segundo instrumento: unidades comuns ──────────────────────────────────
#
# ⚠️ Adicionado em 2026-09-14, depois de o mascaramento por token dar E à frente
# por +0,078. Aquele instrumento favorece o tokenizer de tokens curtos; este
# favorece o de tokens longos. Os testes abaixo travam as três promessas que
# tornam o segundo legível: nenhum token partido, bytes idênticos, contexto igual.


def _segmentacao(rng, n: int) -> list[tuple[int, int]]:
    """Uma partição aleatória de `[0, n)` em tokens, às vezes pulando um caractere
    (o espaço que um pré-tokenizador descarta)."""
    offs, i = [], 0
    while i < n:
        if rng.random() < 0.1:
            i += 1
            continue
        j = min(n, i + int(rng.integers(1, 6)))
        offs.append((i, j))
        i = j
    return offs


def test_a_unidade_comum_NAO_parte_token_de_braco_nenhum():
    """`\\frac` inteiro em A, `\\fr` + `ac` em E: a unidade é a palavra toda."""
    A = [(0, 5)]
    E = [(0, 3), (3, 5)]
    assert unidades_comuns([A, E], 5) == [(0, 5)]


def test_texto_que_um_braco_NAO_representa_fica_fora():
    """O espaço em `ab cd` não tem token em nenhum braço: não há como escondê-lo."""
    A = [(0, 2), (3, 5)]
    E = [(0, 1), (1, 2), (3, 5)]
    assert unidades_comuns([A, E], 5) == [(0, 2), (3, 5)]


def test_com_um_braco_so_e_RECUSADO():
    with pytest.raises(ValueError, match="entre BRAÇOS"):
        unidades_comuns([[(0, 3)]], 3)


def test_os_bytes_escondidos_sao_IDENTICOS_nos_bracos():
    """⚠️ A promessa inteira do instrumento, sobre segmentações aleatórias e texto
    com caracteres de vários bytes. O script confere o mesmo em produção."""
    rng = np.random.default_rng(3)
    for caso in range(200):
        n = int(rng.integers(5, 80))
        texto = "".join(rng.choice(list(r"ab \{}αβ∂"), size=n))
        bracos = [_segmentacao(rng, n) for _ in range(3)]
        unid = unidades_comuns(bracos, n)
        esc = [unid[k] for k in sortear_unidades(len(unid), 17, caso, 0.3)]
        somas = {int(posicoes_nas_unidades(b, esc, texto)[1].sum()) for b in bracos}
        assert len(somas) <= 1, (caso, somas)


def test_toda_posicao_da_unidade_e_escondida_e_NENHUMA_de_fora():
    rng = np.random.default_rng(5)
    for caso in range(200):
        n = int(rng.integers(5, 80))
        texto = "x" * n
        A, E = _segmentacao(rng, n), _segmentacao(rng, n)
        unid = unidades_comuns([A, E], n)
        esc = [unid[k] for k in sortear_unidades(len(unid), 17, caso, 0.3)]
        for offs in (A, E):
            pos, _ = posicoes_nas_unidades(offs, esc, texto)
            dentro = {i for i, (a, b) in enumerate(offs)
                      if any(lo <= a < hi for lo, hi in esc)}
            assert set(pos.tolist()) == dentro
            for i in pos:
                a, b = offs[int(i)]
                assert any(lo <= a and b <= hi for lo, hi in esc)


def test_especiais_com_offset_zero_nunca_sao_escondidos():
    """[CLS] e [SEP] chegam com `(0, 0)`; escondê-los mediria outra tarefa."""
    offs = [(0, 0), (0, 3), (3, 5), (0, 0)]
    pos, byt = posicoes_nas_unidades(offs, [(0, 5)], "abcde")
    assert pos.tolist() == [1, 2]
    assert byt.tolist() == [5, 0]


def test_offsets_de_OUTRO_braco_sao_recusados():
    """Um token que atravessa a unidade prova que os cortes vieram de outro braço."""
    with pytest.raises(ValueError, match="atravessa"):
        posicoes_nas_unidades([(0, 4)], [(0, 2)], "abcd")
    with pytest.raises(ValueError, match="não é comum"):
        posicoes_nas_unidades([(0, 2)], [(3, 4)], "abcd")


def test_o_sorteio_e_DETERMINISTICO_e_nunca_vazio():
    a = sortear_unidades(100, 17, 4, 0.15)
    assert a.tolist() == sortear_unidades(100, 17, 4, 0.15).tolist()
    assert a.tolist() != sortear_unidades(100, 17, 5, 0.15).tolist()
    assert sortear_unidades(3, 17, 0, 0.0).size == 1
    assert sortear_unidades(0, 17, 0, 0.15).size == 0


def test_o_viés_por_unidade_PENALIZA_quem_parte_mais():
    """⚠️ Por que este instrumento favorece A, e não é neutro.

    Dois caracteres perfeitamente correlacionados — `xy` ou `zw`, meio a meio. A
    entropia real da unidade é 1 bit. Um braço que a tem como UM token paga 1 bit;
    um que a parte em dois, previstos independentemente com marginais ótimas (1/2
    cada), paga 2 — sem ser modelo pior. Vitória do braço que parte mais, aqui, é
    vitória apesar do instrumento."""
    inteiro, partido = Acumulador(), Acumulador()
    _doc(inteiro, [0.5], [2])
    _doc(partido, [0.5, 0.5], [2, 0])
    assert inteiro.como_dict()["bits_por_byte"] == 0.5
    assert partido.como_dict()["bits_por_byte"] == 1.0


def test_a_leitura_NAO_manda_mais_para_a_PLL_original():
    """A docstring mandava desfazer a ambiguidade com a pseudo-verossimilhança
    original, que tem o mesmo viés, mais forte. Fica travado que o artefato
    aponta para o instrumento de viés oposto."""
    fonte = (RAIZ / "scripts/avaliar_bits_por_byte.py").read_text(encoding="utf-8")
    assert "pedem a pseudo-verossimilhança canônica" not in fonte
    assert "Kauf & Ivanova" in fonte
