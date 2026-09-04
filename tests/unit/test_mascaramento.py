"""O mascaramento consciente de equações é a hipótese central do ΦEnc.

O DOC-07 §2.3 chama isso de "a única adição específica de Física ao objetivo de
treino", e é a razão de treinar do zero em vez de ajustar um modelo existente. Uma
ablação sobre isso só significa algo se três coisas forem verdadeiras, e são elas que
estes testes fixam:

  1. **o orçamento de máscara é igual nos dois braços** — senão a comparação mede
     "consciente de equações" E "mascara mais", e nada fica atribuível;
  2. **o que é mascarado no braço tratado é uma equação de DISPLAY inteira** — a
     mediana do inline é 7 tokens, e mascarar uma variável não testa hipótese nenhuma;
  3. **as recaídas são contadas** — um braço tratado que recai em aleatório reportaria
     empate sem que o tratamento tivesse acontecido.

⚠️ O teste `test_display_e_detectada_fora_do_inicio_da_string` guarda um bug real que
custou uma sonda inteira para ser achado. Ver a docstring dele.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.training.pretrain.mascaramento import (  # noqa: E402
    MIN_TOKENS_TRATAMENTO,
    ConfigMascara,
    Contadores,
    marcar_equacoes,
    mascarar,
    spans_de_equacao,
)

ESPECIAIS = frozenset({0, 1, 2, 3, 4})
ID_MASK, N_VOCAB = 4, 100


def _exemplo(n: int = 400, n_eq: int = 60, inicio_eq: int = 100,
            display: bool = True):
    """Um exemplo sintético: ids fora dos especiais, com uma equação no meio."""
    ids = np.arange(10, 10 + n, dtype=np.int64)
    marca = np.full(n, -1, dtype=np.int32)
    disp = np.zeros(n, dtype=bool)
    marca[inicio_eq:inicio_eq + n_eq] = 0
    disp[inicio_eq:inicio_eq + n_eq] = display
    return ids, marca, disp


# ─── 1. o orçamento ──────────────────────────────────────────────────────────


def test_o_orcamento_de_mascara_e_igual_nos_dois_bracos():
    """A asserção mais importante deste arquivo.

    Se o braço tratado mascarar mais tokens que o controle, a ablação compara duas
    coisas ao mesmo tempo. Este repositório já perdeu um experimento por trocar base
    e lote na mesma corrida e não saber qual dos dois explicava o resultado.
    """
    ids, marca, disp = _exemplo()
    for taxa in (0.15, 0.30, 0.45):
        contagens = []
        for p_eq in (0.0, 1.0):
            cfg = ConfigMascara(taxa=taxa, p_equacao=p_eq)
            _, alvos = mascarar(ids, marca, disp, cfg=cfg,
                                rng=np.random.default_rng(7), id_mask=ID_MASK,
                                n_vocab=N_VOCAB, ids_especiais=ESPECIAIS)
            contagens.append(int((alvos != -100).sum()))
        assert contagens[0] == contagens[1], (
            f"taxa {taxa}: controle mascarou {contagens[0]} e tratado "
            f"{contagens[1]} — a ablação passou a medir duas coisas")
        assert contagens[0] == round(taxa * ids.size)


def test_a_taxa_efetiva_bate_com_a_pedida_em_muitos_exemplos():
    cfg = ConfigMascara(taxa=0.30, p_equacao=0.5)
    c = Contadores()
    rng = np.random.default_rng(3)
    ids, marca, disp = _exemplo()
    for _ in range(200):
        mascarar(ids, marca, disp, cfg=cfg, rng=rng, id_mask=ID_MASK,
                 n_vocab=N_VOCAB, ids_especiais=ESPECIAIS, contadores=c)
    assert c.taxa_efetiva() == pytest.approx(0.30, abs=0.005)


# ─── 2. o que é mascarado no braço tratado ───────────────────────────────────


def test_o_tratamento_mascara_a_equacao_INTEIRA():
    """Meia equação mascarada entrega metade da resposta e dilui o efeito."""
    ids, marca, disp = _exemplo(n=400, n_eq=60, inicio_eq=100)
    cfg = ConfigMascara(taxa=0.30, p_equacao=1.0)
    entrada, alvos = mascarar(ids, marca, disp, cfg=cfg,
                              rng=np.random.default_rng(1), id_mask=ID_MASK,
                              n_vocab=N_VOCAB, ids_especiais=ESPECIAIS)
    eq = slice(100, 160)
    assert (alvos[eq] != -100).all(), "sobrou token de equação sem perda"
    # ⚠️ E a equação vai TODA para [MASK], sem o 80/10/10: decisão 2 do módulo.
    assert (entrada[eq] == ID_MASK).all(), (
        "algum token da equação escapou do [MASK] — o 80/10/10 vazou para o "
        "tratamento e entrega pedaços da resposta")


def test_o_controle_nao_mascara_a_equacao_toda():
    """Com p_equacao=0 é MLM padrão: a chance de a equação inteira cair por sorteio
    é desprezível, e se cair o teste avisa que o controle não é controle."""
    ids, marca, disp = _exemplo()
    cfg = ConfigMascara(taxa=0.30, p_equacao=0.0)
    entrada, _ = mascarar(ids, marca, disp, cfg=cfg,
                          rng=np.random.default_rng(1), id_mask=ID_MASK,
                          n_vocab=N_VOCAB, ids_especiais=ESPECIAIS)
    assert not (entrada[100:160] == ID_MASK).all()


def test_o_inline_nunca_e_escolhido_para_tratamento():
    """A mediana do inline é 7 tokens — é uma variável, não uma equação.

    Um documento só com inline tem de RECAIR e ser contado, não ser tratado como se
    a hipótese estivesse sendo testada.
    """
    ids, marca, disp = _exemplo(n_eq=60, display=False)
    cfg = ConfigMascara(taxa=0.30, p_equacao=1.0)
    c = Contadores()
    entrada, _ = mascarar(ids, marca, disp, cfg=cfg,
                          rng=np.random.default_rng(1), id_mask=ID_MASK,
                          n_vocab=N_VOCAB, ids_especiais=ESPECIAIS, contadores=c)
    assert c.tratados == 0
    assert c.recaida_sem_equacao == 1
    assert not (entrada[100:160] == ID_MASK).all()


def test_equacao_curta_recai_e_e_contada():
    """Abaixo de MIN_TOKENS_TRATAMENTO não é equação o suficiente para a hipótese."""
    ids, marca, disp = _exemplo(n_eq=MIN_TOKENS_TRATAMENTO - 1)
    c = Contadores()
    mascarar(ids, marca, disp, cfg=ConfigMascara(p_equacao=1.0),
             rng=np.random.default_rng(1), id_mask=ID_MASK, n_vocab=N_VOCAB,
             ids_especiais=ESPECIAIS, contadores=c)
    assert c.tratados == 0 and c.recaida_equacao_curta == 1


def test_equacao_maior_que_o_orcamento_recai_e_e_contada():
    """Mascarar uma equação de 200 tokens num orçamento de 30 estouraria o
    orçamento — e é isso que a decisão 1 do módulo proíbe."""
    ids, marca, disp = _exemplo(n=100, n_eq=80, inicio_eq=10)  # orçamento = 30
    c = Contadores()
    mascarar(ids, marca, disp, cfg=ConfigMascara(taxa=0.30, p_equacao=1.0),
             rng=np.random.default_rng(1), id_mask=ID_MASK, n_vocab=N_VOCAB,
             ids_especiais=ESPECIAIS, contadores=c)
    assert c.tratados == 0 and c.recaida_equacao_grande == 1


def test_escolhe_entre_varias_equacoes_e_nenhuma_estoura():
    """Com três display, alguma cabe — e a escolhida tem de sair inteira."""
    n = 600
    ids = np.arange(10, 10 + n, dtype=np.int64)
    marca = np.full(n, -1, dtype=np.int32)
    disp = np.zeros(n, dtype=bool)
    for k, (i, tam) in enumerate([(50, 40), (200, 300), (450, 60)]):
        marca[i:i + tam] = k
        disp[i:i + tam] = True
    cfg = ConfigMascara(taxa=0.30, p_equacao=1.0)  # orçamento = 180
    vistos = set()
    for s in range(30):
        c = Contadores()
        entrada, alvos = mascarar(ids, marca, disp, cfg=cfg,
                                  rng=np.random.default_rng(s), id_mask=ID_MASK,
                                  n_vocab=N_VOCAB, ids_especiais=ESPECIAIS,
                                  contadores=c)
        assert c.tratados == 1
        assert int((alvos != -100).sum()) == 180
        for k, (i, tam) in enumerate([(50, 40), (450, 60)]):
            if (entrada[i:i + tam] == ID_MASK).all():
                vistos.add(k)
        # a de 300 tokens NUNCA pode ser escolhida: não cabe em 180
        assert not (entrada[200:500] == ID_MASK).all()
    assert vistos == {0, 1}, (
        f"só {vistos} foram escolhidas em 30 sorteios — a escolha está enviesada "
        "para uma equação, e o tratamento fica menos diverso do que parece")


# ─── 3. as invariantes do MLM ────────────────────────────────────────────────


def test_especiais_nunca_sao_mascarados():
    """Mascarar [CLS] ensina a prever [CLS]; mascarar [PAD] gasta perda em nada."""
    ids = np.array([2] + list(range(10, 60)) + [3] + [0] * 20, dtype=np.int64)
    marca = np.full(ids.size, -1, dtype=np.int32)
    disp = np.zeros(ids.size, dtype=bool)
    entrada, alvos = mascarar(ids, marca, disp, cfg=ConfigMascara(taxa=0.9),
                              rng=np.random.default_rng(1), id_mask=ID_MASK,
                              n_vocab=N_VOCAB, ids_especiais=ESPECIAIS)
    for pos in [0, 51, *range(52, 72)]:
        assert alvos[pos] == -100, f"posição especial {pos} entrou na perda"
        assert entrada[pos] == ids[pos]


def test_alvos_sao_menos_cem_fora_das_posicoes_mascaradas():
    """−100 é a convenção do CrossEntropyLoss. Usar 0 treinaria a prever [PAD]."""
    ids, marca, disp = _exemplo()
    entrada, alvos = mascarar(ids, marca, disp, cfg=ConfigMascara(),
                              rng=np.random.default_rng(1), id_mask=ID_MASK,
                              n_vocab=N_VOCAB, ids_especiais=ESPECIAIS)
    mascarados = alvos != -100
    assert (alvos[mascarados] == ids[mascarados]).all(), (
        "o alvo tem de ser o token ORIGINAL, não o que ficou na entrada")
    assert (entrada[~mascarados] == ids[~mascarados]).all(), (
        "posição não mascarada foi alterada na entrada")


def test_a_entrada_nao_e_toda_mask_por_causa_do_oitenta_dez_dez():
    """10% intactos impedem o modelo de aprender "onde não há [MASK], confie"."""
    ids, marca, disp = _exemplo(n=4000, n_eq=0, inicio_eq=0)
    cfg = ConfigMascara(taxa=0.30, p_equacao=0.0)
    entrada, alvos = mascarar(ids, marca, disp, cfg=cfg,
                              rng=np.random.default_rng(5), id_mask=ID_MASK,
                              n_vocab=N_VOCAB, ids_especiais=ESPECIAIS)
    m = alvos != -100
    frac_mask = float((entrada[m] == ID_MASK).mean())
    frac_intacta = float((entrada[m] == ids[m]).mean())
    assert frac_mask == pytest.approx(0.8, abs=0.03), frac_mask
    assert frac_intacta == pytest.approx(0.1, abs=0.03), frac_intacta


def test_o_mesmo_gerador_da_o_mesmo_resultado():
    """Sem determinismo, um spike não se reproduz e o (seed, step) do DOC-08 §7.2
    deixa de identificar o batch ofensor."""
    ids, marca, disp = _exemplo()
    saidas = [mascarar(ids, marca, disp, cfg=ConfigMascara(p_equacao=0.5),
                       rng=np.random.default_rng(11), id_mask=ID_MASK,
                       n_vocab=N_VOCAB, ids_especiais=ESPECIAIS)
              for _ in range(2)]
    assert (saidas[0][0] == saidas[1][0]).all()
    assert (saidas[0][1] == saidas[1][1]).all()


def test_ids_e_marca_de_tamanhos_diferentes_levantam():
    ids = np.arange(10, dtype=np.int64)
    with pytest.raises(ValueError, match="diferem"):
        mascarar(ids, np.full(5, -1, dtype=np.int32), np.zeros(5, dtype=bool),
                 cfg=ConfigMascara(), rng=np.random.default_rng(1),
                 id_mask=ID_MASK, n_vocab=N_VOCAB, ids_especiais=ESPECIAIS)


def test_config_invalida_levanta():
    for kw, msg in (({"taxa": 0.0}, "taxa"), ({"taxa": 1.0}, "taxa"),
                    ({"p_equacao": 1.5}, "p_equacao"),
                    ({"p_mask": 0.8, "p_aleatorio": 0.5}, "não pode ser negativa")):
        with pytest.raises(ValueError, match=msg):
            ConfigMascara(**kw)


# ─── 4. onde está a matemática ───────────────────────────────────────────────


def test_display_e_detectada_fora_do_inicio_da_string():
    """⚠️ REGRESSÃO de um bug que zerou o tratamento inteiro.

    A primeira versão usava `re.compile(r"^(?:\\\\begin\\{|...)")` com
    `DISPLAY.match(texto, i)`. `Pattern.match(s, pos)` já ancora em `pos`, **mas o
    `^` continua se referindo ao início REAL da string** — então o padrão nunca
    casava para nenhuma equação que não começasse no caractere 0.

    Medido: 0 de 120 documentos tratados, com `recaida_sem_equacao` em 120, contra
    91,7% de documentos com display medidos por uma sonda que fatiava a string antes
    de casar. Foi a discordância entre as duas sondas que localizou o erro — e sem a
    `fracao_tratada` nos contadores a ablação teria rodado inteira comparando
    aleatório com aleatório.
    """
    texto = "prosa antes " + r"\begin{equation} F = ma \end{equation}" + " prosa"
    spans = spans_de_equacao(texto)
    assert len(spans) == 1
    assert spans[0][2] is True, (
        "display não detectada fora da posição 0 — o `^` voltou para o padrão")

    for abre, fecha in ((r"\[", r"\]"), ("$$", "$$")):
        t = "prosa " + abre + " x = 1 " + fecha
        assert spans_de_equacao(t)[0][2] is True, f"{abre} não reconhecido"


def test_inline_nao_e_display():
    assert spans_de_equacao("antes $E = mc^2$ depois")[0][2] is False


def test_dolar_duplo_vem_antes_do_simples():
    """Se `$…$` casar primeiro, `$$x$$` vira duas equações degeneradas."""
    spans = spans_de_equacao("antes $$E = mc^2$$ depois")
    assert len(spans) == 1 and spans[0][2] is True


def test_dolar_solto_nao_engole_paragrafos():
    """Um `$` de moeda ou mal escapado marcaria prosa como equação — pior que não
    marcar, porque o tratamento passaria a mascarar prosa achando que é matemática."""
    texto = ("custou $5 milhoes e o resto do paragrafo segue por muito tempo.\n\n"
             "outro paragrafo inteiro aqui, sem nenhuma matematica de verdade.\n\n"
             "e um terceiro, tambem sem nada.")
    for ini, fim, _ in spans_de_equacao(texto):
        assert "\n\n" not in texto[ini:fim], (
            f"um span atravessou parágrafos: {texto[ini:fim][:80]!r}")


def test_ambiente_fecha_com_o_mesmo_nome():
    """`\\begin{align}…\\end{equation}` não é um ambiente; casar isso juntaria duas
    equações distintas numa só e o orçamento estouraria sem motivo."""
    texto = (r"\begin{align} a = 1 \end{align}" " texto "
             r"\begin{equation} b = 2 \end{equation}")
    spans = spans_de_equacao(texto)
    assert len(spans) == 2, [texto[i:f] for i, f, _ in spans]


def test_marcar_usa_sobreposicao_e_ignora_token_sem_extensao():
    """Um token de fronteira carrega o `$` ou o `\\begin`, que é justamente a pista
    de que ali havia matemática. E offsets `(0, 0)` são especiais."""
    texto = "ab " + r"\begin{equation}x=1\end{equation}" + " cd"
    offsets = [(0, 0), (0, 2), (3, 20), (20, 23), (23, 36), (37, 39), (0, 0)]
    marca, disp = marcar_equacoes(offsets, texto)
    assert marca[0] == -1 and marca[-1] == -1, "token especial foi marcado"
    assert (marca[2:5] == 0).all(), "o corpo da equação não foi marcado"
    assert disp[2:5].all()
    assert marca[1] == -1 and marca[5] == -1, "prosa foi marcada como equação"


def test_texto_sem_matematica_nao_marca_nada():
    texto = "um paragrafo em prosa, sem nenhuma matematica, nem inline."
    offsets = [(i, i + 1) for i in range(len(texto))]
    marca, disp = marcar_equacoes(offsets, texto)
    assert (marca == -1).all() and not disp.any()


def test_a_nota_dos_contadores_diz_o_que_invalida_a_ablacao():
    """O JSON do treino é lido depois, por quem não escreveu isto."""
    d = Contadores().como_dict()
    assert "fracao_tratada" in d
    assert "invalida" in d["nota"] and "§2.3" in d["nota"]


# ─── 5. a vetorização, e a propriedade que ela não pode perder ───────────────


def _marcar_referencia(offsets, texto):
    """A varredura linear, escrita AQUI como referência.

    ⚠️ Deliberadamente ingênua e O(spans × tokens): é a definição de "sobrepõe",
    traduzida direto. A versão de produção usa `searchsorted` e é 50× mais rápida —
    e é contra esta que ela tem de concordar, senão a otimização mudou a semântica.
    """
    marca = np.full(len(offsets), -1, dtype=np.int32)
    disp = np.zeros(len(offsets), dtype=bool)
    for k, (ini, fim, e_disp) in enumerate(spans_de_equacao(texto)):
        for t, (a, b) in enumerate(offsets):
            if a == b:
                continue
            if a < fim and b > ini:
                marca[t] = k
                disp[t] = e_disp
    return marca, disp


def test_a_versao_vetorizada_concorda_com_a_varredura_linear():
    """⚠️ A otimização que valeu 50× não pode ter mudado o que a função significa.

    Medido no corpus de verdade: 0 divergências em 1.181.249 tokens de 80
    documentos, e 62,4 mil tok/s → 3,1 M tok/s. Aqui a mesma comparação, em textos
    sintéticos que cobrem os casos de fronteira.
    """
    casos = [
        "prosa " + r"\begin{equation} F = ma \end{equation}" + " mais prosa",
        r"\begin{align} a=1 \end{align}" + " x " + r"\[ b=2 \]" + " y $c = 3$ z",
        "$$E = mc^2$$ no começo",
        r"\begin{equation}" + " no fim, sem fechar",
        "sem matemática nenhuma aqui",
        "$a$ $b$ $cc = dd$ inline em sequência",
    ]
    for texto in casos:
        # offsets de caractere: o pior caso para a busca binária, um token por char
        offs = [(i, i + 1) for i in range(len(texto))]
        m1, d1 = _marcar_referencia(offs, texto)
        m2, d2 = marcar_equacoes(offs, texto)
        assert (m1 == m2).all(), (texto, m1.tolist(), m2.tolist())
        assert (d1 == d2).all(), texto


def test_offsets_fora_de_ordem_levantam_em_vez_de_mentir():
    """A busca binária exige monotonicidade. Um tokenizer que emitisse tokens fora
    de ordem daria respostas erradas em SILÊNCIO — então a checagem levanta.

    Não há recaída para a varredura linear de propósito: ela é 50× mais lenta, e
    uma recaída silenciosa transformaria um bug de tokenizer numa preparação de 16
    horas que ninguém explicaria.
    """
    texto = "prosa " + r"\begin{equation} F = ma \end{equation}"
    fora_de_ordem = [(20, 25), (6, 10), (10, 20), (25, 40)]
    with pytest.raises(ValueError, match="não estão ordenados"):
        marcar_equacoes(fora_de_ordem, texto)


def test_tokens_sem_extensao_nas_duas_pontas_nao_quebram_a_busca():
    """Um `(0, 0)` no FIM da sequência quebraria a monotonicidade, e a busca
    binária responderia qualquer coisa. Eles saem do cálculo e voltam com −1."""
    texto = "ab " + r"\begin{equation}x=1\end{equation}" + " cd"
    offsets = [(0, 0), (0, 2), (3, 20), (20, 23), (23, 36), (37, 39), (0, 0)]
    m, d = marcar_equacoes(offsets, texto)
    m_ref, d_ref = _marcar_referencia(offsets, texto)
    assert (m == m_ref).all(), (m.tolist(), m_ref.tolist())
    assert (d == d_ref).all()
    assert m[0] == -1 and m[-1] == -1
