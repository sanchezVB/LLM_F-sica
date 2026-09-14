"""O Portão G1 media contra um teto de nDCG@10 0,7562, e não 1,0.

O protocolo é recuperação dentro do lote: `sim = ancoras @ positivos.T`, e a
resposta certa da linha *i* é a **coluna** *i*. Isso só é tarefa bem posta se cada
coluna for distinta. Duas coisas quebravam isso:

1. **`amostra = val.head(n)`.** A ordem de `pares_validacao.parquet` não é neutra —
   a mediana do comprimento da âncora cai de ~1.180 para ~870 do início ao fim, e o
   primeiro bloco de 2.000 fica no **percentil 94**. Pior: em `head(2000)` havia só
   **1.147 textos positivos distintos**, com **62% das linhas** num positivo
   repetido e um deles aparecendo **28 vezes**. Textos byte-idênticos dão cosseno
   idêntico, e o desempate do `argsort` é arbitrário.

2. **Âncoras repetidas.** Linhas com a mesma consulta compartilham um só ranking,
   então no máximo uma delas pode ter posto 1.

Teto de um modelo PERFEITO, medido:

    head(2000)          recall@1 0,5235   nDCG@10 0,7562   <- o que o G1 usou
    sample(2000)        recall@1 0,9364   nDCG@10 0,9761
    dedup + sample      recall@1 1,0000   nDCG@10 1,0000

O G1 reportou recall@1 **0,2620** contra um teto de **0,5235**. O empate era justo
entre modelos, e é por isso que a tabela parecia válida — o que ele destruía era a
**margem**, e o G1.2 se decide em +0,003 de nDCG@10.

Sétima ocorrência de `head()` sobre dado ordenado neste repositório.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

pl = pytest.importorskip("polars")

from conftest import so_codigo_de  # noqa: E402
from phifm.training.amostragem import (  # noqa: E402
    SEMENTE_POOL,
    preparar_pool,
    teto_do_pool,
)


def _pares(n_docs: int = 400, por_doc: int = 5) -> pl.DataFrame:
    """A MESMA patologia do parquet real: cada documento citado aparece várias
    vezes, com o texto do positivo idêntico, e as linhas vêm agrupadas."""
    linhas = []
    for d in range(n_docs):
        for k in range(por_doc):
            linhas.append({
                "arxiv_id": f"{d:04d}.{k:03d}",
                "arxiv_citado": f"cit{d:04d}",
                # a âncora varia; o positivo é o MESMO texto do documento citado
                "ancora": f"ancora do citante {d}-{k} com texto suficiente",
                "positivo": f"texto do documento citado {d}",
            })
    return pl.DataFrame(linhas)


# ─── o teto, que é a asserção central ────────────────────────────────────────


def test_o_teto_do_pool_preparado_e_UM():
    """⚠️ A asserção central deste arquivo. Ver a docstring do módulo.

    Se o teto não for 1,0, parte do número medido é desempate arbitrário, e a
    tabela sai com a cara certa.
    """
    pool = preparar_pool(_pares(), 300)
    t = teto_do_pool(pool)
    assert t["ndcg_10"] == pytest.approx(1.0), (
        f"teto do nDCG@10 é {t['ndcg_10']:.4f}: há empate no pool")
    assert t["recall_1"] == pytest.approx(1.0)
    assert pool.height == 300
    assert pool["positivo"].n_unique() == 300
    assert pool["ancora"].n_unique() == 300


def test_o_head_REPROVA_este_teste():
    """A prova de que o defeito era real, executando a regra antiga.

    Sem isto o teste acima poderia estar passando por outro motivo, e ninguém
    saberia se a versão anterior teria passado também.
    """
    t = teto_do_pool(_pares().head(300))
    assert t["ndcg_10"] < 0.85, (
        f"o `head` deixou de ter teto baixo (nDCG@10 {t['ndcg_10']:.4f}) — então "
        "ou ele nunca foi o defeito, ou este teste parou de reproduzi-lo")
    assert t["recall_1"] < 0.5


def test_o_teto_conta_ANCORA_repetida_tambem():
    """A metade que eu quase esqueci.

    Alvos únicos não bastam: se duas linhas têm a MESMA consulta, elas
    compartilham um ranking, e só uma pode ter posto 1. Um `teto_do_pool` que
    olhasse só as colunas devolveria 1,0 aqui e mentiria.
    """
    d = pl.DataFrame({
        "ancora": ["mesma consulta", "mesma consulta", "outra"],
        "positivo": ["alvo A", "alvo B", "alvo C"],
    })
    assert d["positivo"].n_unique() == 3, "os alvos são únicos, de propósito"
    t = teto_do_pool(d)
    assert t["recall_1"] < 1.0, (
        "com a consulta repetida o teto tem de cair; se não cai, `teto_do_pool` "
        "está olhando só as colunas")


# ─── as guardas ──────────────────────────────────────────────────────────────


def test_pool_sem_linhas_suficientes_LEVANTA():
    """Cortar o n em silêncio mudaria o protocolo e os números deixariam de ser
    comparáveis com os anteriores — sem nada avisando."""
    with pytest.raises(ValueError, match="desduplicar"):
        preparar_pool(_pares(n_docs=10, por_doc=5), 300)


def test_pool_sem_as_colunas_certas_LEVANTA():
    with pytest.raises(ValueError, match="ancora|positivo"):
        preparar_pool(pl.DataFrame({"texto": ["a", "b"]}), 2)


def test_o_sorteio_e_deterministico_e_a_semente_muda_o_pool():
    """Uma amostra que muda a cada execução tornaria o cache — e a comparação
    entre modelos medidos em execuções diferentes — sem sentido."""
    d = _pares()
    a = preparar_pool(d, 200)
    b = preparar_pool(d, 200)
    assert a["positivo"].to_list() == b["positivo"].to_list()
    c = preparar_pool(d, 200, semente=99)
    assert c["positivo"].to_list() != a["positivo"].to_list()
    assert SEMENTE_POOL == 17


def test_o_pool_nao_e_o_prefixo_do_arquivo():
    """O sorteio tem de ser observável: um pool que devolvesse a ordem do disco
    passaria em tudo acima e continuaria sendo `head`."""
    d = _pares()
    pool = preparar_pool(d, 200)
    # `head` sobre um arquivo agrupado pega os primeiros documentos citados;
    # o sorteio espalha.
    primeiros = {f"cit{i:04d}" for i in range(40)}
    do_sorteio = {x.split()[-1] for x in pool["positivo"].to_list()}
    dentro = sum(1 for x in pool["positivo"].to_list()
                 if f"cit{int(x.split()[-1]):04d}" in primeiros)
    assert dentro < 60, (
        f"{dentro} de 200 linhas vieram dos 40 primeiros documentos — isso é a "
        "ordem do disco, não um sorteio")
    assert len(do_sorteio) == 200


# ─── um só caminho de código ─────────────────────────────────────────────────


def test_o_digesto_do_cache_passa_pelo_MESMO_pool():
    """O digesto é a chave do cache: ele tem de descrever a amostra que foi de
    fato avaliada.

    Montar a amostra em dois lugares é o defeito que o alvo pré-comprometido da
    folha de revisão teve — escrito duas vezes, os dois divergem e nada avisa.
    Aqui a divergência seria pior: o cache diria "mesmo protocolo" para números
    medidos em pools diferentes.
    """
    # `so_codigo_de` remove comentários e docstrings: ver tests/conftest.py, onde
    # uma asserção de ausência já reprovou cinco vezes a própria explicação.
    fonte = so_codigo_de(RAIZ / "src/phifm/eval/encoders.py")
    assert "val.head(n)" not in fonte, (
        "voltou o `head` no protocolo do G1 — ver a docstring deste módulo")
    corpo = fonte.split("def _digesto")[1].split("\ndef ")[0]
    assert "preparar_pool" in corpo, (
        "o digesto deixou de usar `preparar_pool`; ele passaria a descrever uma "
        "amostra diferente da avaliada")

    # ⚠️ Em 2026-09-14 a métrica saiu de `avaliar_um` para `avaliar_carregado`, de
    # onde o ΦEnc também a consome — ele carrega de um `state_dict` cru e não por
    # `AutoModel.from_pretrained`. A propriedade que este teste protege não mudou;
    # o que mudou foi onde ela mora, e a asserção segue o caminho em vez de ser
    # afrouxada. Quem monta o pool agora é `avaliar_carregado`, e `avaliar_um`
    # tem de DELEGAR — se ele voltar a montar o seu próprio, passam a existir
    # dois pools e o cache diria "mesmo protocolo" para amostras diferentes.
    corpo_medida = fonte.split("def avaliar_carregado")[1].split("\ndef ")[0]
    assert "preparar_pool" in corpo_medida, (
        "`avaliar_carregado` deixou de usar `preparar_pool` — é ele que mede")

    corpo_aval = fonte.split("def avaliar_um")[1].split("\ndef ")[0]
    assert "avaliar_carregado" in corpo_aval, (
        "`avaliar_um` deixou de delegar a medição; se ele recalcular a métrica, "
        "passam a existir dois caminhos que divergem em silêncio")
    assert "preparar_pool" not in corpo_aval, (
        "`avaliar_um` voltou a montar o próprio pool em vez de delegar")


def test_a_avaliacao_do_PHIENC_usa_o_mesmo_caminho_de_medida():
    """O ΦEnc é o terceiro consumidor da métrica, e o mais fácil de duplicar.

    Ele carrega diferente de todo o resto — `state_dict` cru do laço de
    pré-treino, tokenizer que é JSON do `tokenizers` — e essa diferença é
    exatamente o convite a escrever "só um avaliadorzinho" ao lado. Se isso
    acontecer, o braço tratado e o PhysBERT passam a ser medidos por códigos
    diferentes, e a tabela não avisa.

    ⚠️ Os proibidos são CONSTRUÇÕES, não o nome da métrica. A primeira versão
    deste teste também proibia `"ndcg"`, e reprovou — no `RESSALVA`, que menciona
    "nDCG@10 0,207" justamente para explicar por que o número absoluto não vale.
    É a quinta ocorrência da armadilha que o `tests/conftest.py` cataloga: uma
    asserção de ausência encontrando o texto que explica a ausência. `so_codigo_de`
    remove comentários, mas uma constante de string é código — e tem de ser.
    """
    fonte = so_codigo_de(RAIZ / "src/phifm/eval/phienc.py")
    assert "avaliar_carregado" in fonte, (
        "`phienc.py` deixou de usar a medida compartilhada de `encoders.py`")
    for proibido in ("argsort", "log2"):
        assert proibido not in fonte.lower(), (
            f"`phienc.py` contém {proibido!r}: a métrica foi reimplementada aqui em "
            "vez de delegada, e duas cópias divergem sem nada apontar qual vale")


def test_o_avaliar_do_TREINO_usa_o_mesmo_pool():
    """É ele que elege o checkpoint `-melhor`.

    Ruído arbitrário na métrica é exatamente como um checkpoint pior vira
    campeão — já aconteceu uma vez neste repositório, por outro motivo (o
    `-gc-melhor` eleito por MRR quando o portão pede nDCG@10). Deixar o defeito
    aqui e consertar só o portão faria o portão medir bem um modelo escolhido
    mal.
    """
    fonte = so_codigo_de(RAIZ / "src/phifm/training/embedding.py")
    corpo = fonte.split("def avaliar")[1].split("\n    def ")[0]
    assert "val.head(n)" not in corpo, (
        "voltou o `head` no `avaliar` do treino")
    assert "preparar_pool" in corpo


def test_o_avaliar_do_treino_devolve_QUATRO_metricas():
    """A anotação dizia três e o corpo devolvia quatro.

    Não quebrava nada porque todos os chamadores desempacotam quatro — mas o
    nDCG@10, que é a métrica do PORTÃO, era o que a anotação omitia.
    """
    fonte = (RAIZ / "src/phifm/training/embedding.py").read_text(encoding="utf-8")
    assinatura = fonte.split("def avaliar(")[1].split(":\n")[0]
    assert "tuple[float, float, float, float]" in assinatura, assinatura


def test_o_resultado_versionado_carrega_o_teto():
    """Um nDCG@10 de 0,4657 não diz se o modelo é mediano ou se o protocolo é.

    Foi essa pergunta que ficou sem resposta enquanto o teto era 0,7562 e ninguém
    tinha medido.
    """
    fonte = so_codigo_de(RAIZ / "src/phifm/eval/encoders.py")
    corpo = fonte.split("def salvar")[1].split("\ndef ")[0]
    assert "teto" in corpo, (
        "o artefato versionado do G1 perdeu o teto do protocolo")
    script = so_codigo_de(RAIZ / "scripts/avaliar_encoders.py")
    assert "teto_do_pool" in script, (
        "o script não mede mais o teto; ele voltaria a ser suposto")
