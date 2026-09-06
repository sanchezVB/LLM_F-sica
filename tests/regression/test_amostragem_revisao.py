"""A amostra que decide se o corpus é confiável cobria 0,67% dele.

O filtro montava a amostra de revisão de passagem:

    if len(f.amostra) < N_AMOSTRA and f.vistos % 97 == 0:

Ele **para** de amostrar assim que junta 400. Medido nos parquets prontos do peS2o:
os 400 documentos de `_amostra_para_revisao.json` estão todos em `part-00000` e
`part-00001`, e o último ocupa a posição **37.087 de 5.526.331 aceitos**.

E o corpus não é homogêneo nessa ordem. Ele tem dois regimes, e os resumos vêm
primeiro:

    resumo (s2ag)        3.626.168 docs   65,6% dos docs    7,9% dos tokens
    texto pleno (s2orc)  1.900.163 docs   34,4% dos docs   92,1% dos tokens

A amostra antiga é 100% resumo. Ela mede a precisão do filtro nos 7,9% do corpus, e
nada nos 92,1% — e isso inverte a justificativa escrita em `apurar_revisao.py`, que
era exatamente "a referência veio de RESUMOS do arXiv, e o corpus agora é texto
pleno, então a taxa não transfere".

Sexta ocorrência de `head()` sobre dado ordenado neste repositório. Estes testes
fixam as duas metades do conserto: o reservatório dentro do filtro, e o sorteio
estratificado sobre os parquets prontos.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

pl = pytest.importorskip("polars")

from phifm.corpus.slices.hf_filtrado import N_AMOSTRA, Filtragem  # noqa: E402

SCRIPT = RAIZ / "scripts/amostrar_para_revisao.py"


# ─── o reservatório dentro do filtro ─────────────────────────────────────────


def _reservatorio(n_aceitos: int) -> Filtragem:
    f = Filtragem(fonte="teste")
    for i in range(n_aceitos):
        f.aceitos += 1
        f.considerar({"i": i})
    return f


def test_o_reservatorio_alcanca_o_fim_do_fluxo_e_nao_so_o_comeco():
    """⚠️ A asserção central. Ver a docstring do módulo para o que custou.

    Com 100.000 aceitos e reservatório de 400, a chance de nenhum sorteado cair
    no último décimo é 0,9**400 ≈ 1e-19. Se este teste falhar, o amostrador
    voltou a parar no começo.
    """
    f = _reservatorio(100_000)
    assert len(f.amostra) == N_AMOSTRA
    maior = max(x["i"] for x in f.amostra)
    assert maior > 90_000, (
        f"o maior índice sorteado foi {maior} de 100.000 — a amostra está "
        "concentrada no começo do fluxo, que é o defeito que este arquivo "
        "documenta")


def test_a_regra_ANTIGA_falha_este_teste():
    """A prova de que o defeito era real, executando a regra antiga.

    Sem isto, o teste acima poderia estar passando por outro motivo, e ninguém
    saberia se a versão anterior teria passado também.
    """
    amostra: list[dict] = []
    # `vistos` de 1 a N, como no original: `f.vistos` era incrementado ANTES do
    # teste, então o primeiro múltiplo de 97 caía no documento de índice 96.
    for vistos in range(1, 100_001):
        if len(amostra) < N_AMOSTRA and vistos % 97 == 0:
            amostra.append({"i": vistos - 1})
    assert len(amostra) == N_AMOSTRA
    maior = max(x["i"] for x in amostra)
    assert maior < 40_000, (
        "a regra antiga deixou de se concentrar no começo — então ou ela nunca "
        "foi o defeito, ou este teste parou de reproduzi-lo")


def test_o_reservatorio_e_aproximadamente_uniforme():
    """Alcançar o fim não basta: se os últimos dominassem, seria o mesmo erro
    ao contrário."""
    f = _reservatorio(100_000)
    quartis = Counter(x["i"] // 25_000 for x in f.amostra)
    assert sorted(quartis) == [0, 1, 2, 3]
    for q, n in quartis.items():
        assert 60 < n < 140, f"quartil {q} com {n} de ~100 sorteados: {dict(quartis)}"


def test_o_reservatorio_e_deterministico_na_semente():
    """Sem determinismo a amostra muda a cada execução, e o julgamento humano de
    400 documentos deixa de poder ser retomado."""
    a = [x["i"] for x in _reservatorio(50_000).amostra]
    b = [x["i"] for x in _reservatorio(50_000).amostra]
    assert a == b


def test_um_fluxo_menor_que_o_reservatorio_entra_inteiro():
    f = _reservatorio(37)
    assert [x["i"] for x in f.amostra] == list(range(37))


# ─── o sorteio estratificado sobre os parquets prontos ───────────────────────


def _corpus(tmp_path: Path) -> Path:
    """Um corpus com a MESMA patologia do real: os resumos primeiro, os papers
    depois, e os papers muito mais longos."""
    raiz = tmp_path / "pes2o"
    raiz.mkdir(parents=True, exist_ok=True)
    for i in range(6):  # resumos
        pl.DataFrame({
            "texto": [f"resumo {i}-{j}. " + "prosa curta. " * 20 for j in range(50)],
            "url": ["s2ag/train"] * 50,
            "score": [0.9 + 0.001 * (j % 90) for j in range(50)],
        }).write_parquet(raiz / f"part-{i:05d}.parquet")
    for i in range(6, 10):  # texto pleno, ~20x mais longo
        pl.DataFrame({
            "texto": [f"paper {i}-{j}. " + "secao com muito texto. " * 400
                      for j in range(50)],
            "url": ["s2orc/train"] * 50,
            "score": [0.9 + 0.001 * (j % 90) for j in range(50)],
        }).write_parquet(raiz / f"part-{i:05d}.parquet")
    return raiz


def _rodar(corpus: Path, tmp_path: Path, *extra: str):
    tmp_path.mkdir(parents=True, exist_ok=True)
    saida = tmp_path / "amostra.json"
    man = tmp_path / "manifesto.json"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--corpus", str(corpus), "--out", str(saida),
         "--manifesto", str(man), *extra],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=RAIZ, env={**os.environ, "PYTHONUTF8": "1",
                       "PYTHONPATH": str(RAIZ / "src")})
    assert r.returncode == 0, r.stdout + r.stderr
    return (json.loads(saida.read_text(encoding="utf-8")),
            json.loads(man.read_text(encoding="utf-8")))


def test_sorteia_dos_dois_estratos_e_espalha_pelos_arquivos(tmp_path):
    """⚠️ A asserção que a amostra antiga reprovaria.

    Ela vinha inteira dos dois primeiros arquivos e de um estrato só.
    """
    docs, _ = _rodar(_corpus(tmp_path), tmp_path, "--por-estrato", "40")
    estratos = Counter(d["estrato"] for d in docs)
    assert estratos == {"resumo": 40, "texto pleno": 40}
    arquivos = {d["arquivo"] for d in docs}
    assert len(arquivos) >= 8, (
        f"a amostra saiu de {len(arquivos)} arquivos de 10: {sorted(arquivos)}")


def test_o_peso_em_token_NAO_e_a_fracao_de_documentos(tmp_path):
    """É a razão inteira de estratificar.

    Os dois estratos têm o mesmo número de documentos neste corpus de teste, e
    pesos em token muito diferentes. Reportar um só número, ou ponderar por
    documento, responderia a uma pergunta que ninguém fez.
    """
    _, man = _rodar(_corpus(tmp_path), tmp_path, "--por-estrato", "20")
    e = man["estratos"]
    assert e["resumo"]["fracao_de_documentos"] == pytest.approx(0.6, abs=0.01)
    assert e["texto pleno"]["fracao_de_documentos"] == pytest.approx(0.4, abs=0.01)
    assert e["resumo"]["peso_em_tokens"] < 0.10, (
        f"peso do resumo {e['resumo']['peso_em_tokens']}: se ele for parecido com "
        "a fração de documentos, o peso não está medindo tamanho")
    assert sum(v["peso_em_tokens"] for v in e.values()) == pytest.approx(1, abs=0.01)


def test_os_estratos_vem_embaralhados(tmp_path):
    """Julgar 40 resumos e depois 40 papers confundiria o efeito do estrato com o
    cansaço de quem julga."""
    docs, _ = _rodar(_corpus(tmp_path), tmp_path, "--por-estrato", "40")
    rotulos = [d["estrato"] for d in docs]
    # `strict=False`: o pareamento de vizinhos tem um elemento a menos, de propósito
    trocas = sum(1 for a, b in zip(rotulos, rotulos[1:], strict=False) if a != b)
    assert trocas > 20, (
        f"só {trocas} trocas de estrato em 80 documentos — eles estão em blocos")


def test_o_sorteio_e_deterministico_e_a_semente_muda_a_amostra(tmp_path):
    corpus = _corpus(tmp_path)
    a, _ = _rodar(corpus, tmp_path / "a", "--por-estrato", "20")
    b, _ = _rodar(corpus, tmp_path / "b", "--por-estrato", "20")
    assert [(d["arquivo"], d["linha"]) for d in a] == \
           [(d["arquivo"], d["linha"]) for d in b]
    c, _ = _rodar(corpus, tmp_path / "c", "--por-estrato", "20", "--semente", "99")
    assert {(d["arquivo"], d["linha"]) for d in c} != \
           {(d["arquivo"], d["linha"]) for d in a}


def test_o_trecho_do_miolo_so_vem_para_documento_longo(tmp_path):
    """Num paper de 28 mil caracteres o início é a capa; a contaminação aparece
    no corpo. Num resumo de 1.200 não há miolo separado do início."""
    docs, _ = _rodar(_corpus(tmp_path), tmp_path, "--por-estrato", "20")
    for d in docs:
        if d["estrato"] == "texto pleno":
            assert "meio" in d and d["meio"]
        else:
            assert "meio" not in d


def test_a_amostra_tem_a_procedencia_de_cada_documento(tmp_path):
    """Sem (arquivo, linha) ninguém consegue reconferir um veredicto depois, nem
    dizer de onde veio um documento que virou notícia ruim."""
    docs, _ = _rodar(_corpus(tmp_path), tmp_path, "--por-estrato", "20")
    for d in docs:
        assert d["arquivo"].startswith("part-") and isinstance(d["linha"], int)
        assert d["n_chars"] > 0
    assert len({(d["arquivo"], d["linha"]) for d in docs}) == len(docs), \
        "há documento repetido: o sorteio é SEM reposição"


def test_subconjunto_desconhecido_LEVANTA(tmp_path):
    """Um subconjunto novo entrando calado no estrato errado desloca o peso em
    tokens, e o número final sairia com a cara certa."""
    raiz = tmp_path / "pes2o"
    raiz.mkdir(parents=True)
    pl.DataFrame({"texto": ["algo"], "url": ["s2xyz/train"], "score": [0.95]}) \
        .write_parquet(raiz / "part-00000.parquet")
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--corpus", str(raiz),
         "--out", str(tmp_path / "a.json"),
         "--manifesto", str(tmp_path / "m.json")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=RAIZ, env={**os.environ, "PYTHONUTF8": "1",
                       "PYTHONPATH": str(RAIZ / "src")})
    assert r.returncode != 0
    assert "s2xyz" in (r.stdout + r.stderr)


def test_o_manifesto_registra_como_a_amostra_ANTIGA_era(tmp_path):
    """O manifesto é o que sobra quando a conversa acabar. Se ele não disser que
    houve uma amostra anterior e por que ela foi trocada, alguém compara os dois
    resultados como se fossem a mesma medição."""
    _, man = _rodar(_corpus(tmp_path), tmp_path, "--por-estrato", "20")
    ant = man["amostra_anterior"]
    assert "0,67%" in ant["cobertura"]
    assert "100% resumo" in ant["estratos"]
