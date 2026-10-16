"""O teste de fora do AstroBench (DOC-13 §9.10): o conjunto, os braços, a leitura da letra,
o envio e a apuração — sem modelo, sem índice e sem o conjunto de verdade."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.eval import astrobench as ab  # noqa: E402
from phifm.rag.assistente import Fonte  # noqa: E402

N_LINHAS = ab.N_TESTE + ab.N_DEV + 30


def _tabela(n: int = N_LINHAS, gabarito: bool = False) -> pl.DataFrame:
    d = {"question": [f"What sets the luminosity of star {i}?" for i in range(n)],
         "A": [f"mass {i}" for i in range(n)], "B": ["age"] * n, "C": ["metallicity"] * n,
         "D": ["rotation"] * n}
    if gabarito:
        d["answer"] = ["A"] * n
    return pl.DataFrame(d)


def _parquet(tmp_path: Path, **kw) -> Path:
    arq = tmp_path / "astrobench.parquet"
    _tabela(**kw).write_parquet(arq)
    return arq


# ── o conjunto ───────────────────────────────────────────────────────────────


def test_carregar_le_as_cinco_colunas_e_o_id_e_por_conteudo(tmp_path):
    itens = ab.carregar(_parquet(tmp_path))
    assert len(itens) == N_LINHAS and itens[3].indice == 3
    assert itens[3].opcoes == ("mass 3", "age", "metallicity", "rotation")
    assert itens[3].gabarito is None
    assert itens[3].id == ab.id_do_item(itens[3].pergunta, itens[3].opcoes)
    assert ab.id_do_item("q", ("a", "b", "c", "d")) != ab.id_do_item("q", ("a", "b", "c", "e"))


def test_carregar_recusa_coluna_faltando_opcao_vazia_e_gabarito_fora_de_A_D(tmp_path):
    arq = tmp_path / "x.csv"
    _tabela(5).drop("D").write_csv(arq)
    with pytest.raises(ValueError, match="faltam as colunas"):
        ab.carregar(arq)
    _tabela(5).with_columns(pl.when(pl.int_range(5) == 2).then(pl.lit("")).otherwise(pl.col("B"))
                            .alias("B")).write_csv(arq)
    with pytest.raises(ValueError, match="vazios"):
        ab.carregar(arq)
    _tabela(5).with_columns(pl.lit("E").alias("answer")).write_csv(arq)
    with pytest.raises(ValueError, match="fora de A–D"):
        ab.carregar(arq)


def test_carregar_aceita_gabarito_proprio(tmp_path):
    itens = ab.carregar(_parquet(tmp_path, gabarito=True))
    assert {i.gabarito for i in itens} == {"A"}


def test_amostras_fixas_disjuntas_e_do_tamanho_da_regra():
    teste, dev = ab.amostras(N_LINHAS)
    assert (len(teste), len(dev)) == (ab.N_TESTE, ab.N_DEV)
    assert not set(teste) & set(dev) and len(set(teste)) == ab.N_TESTE
    assert ab.amostras(N_LINHAS) == (teste, dev)
    assert teste[:20] != sorted(teste[:20])        # a ordem é a da permutação
    with pytest.raises(ValueError, match="a regra pede"):
        ab.amostras(ab.N_TESTE)


def test_manifesto_congela_o_arquivo(tmp_path):
    arq = _parquet(tmp_path)
    itens = ab.carregar(arq)
    man = ab.manifesto(arq, itens, "teste")
    ab.conferir_manifesto(arq, itens, man)
    assert man["n_teste"] == ab.N_TESTE and man["tem_gabarito"] is False
    _tabela(N_LINHAS + 1).write_parquet(arq)
    with pytest.raises(ValueError, match="não é o registrado"):
        ab.conferir_manifesto(arq, ab.carregar(arq), man)


# ── os braços ────────────────────────────────────────────────────────────────


class Modelo:
    def __init__(self, texto="Answer: B"):
        self.texto, self.chamadas = texto, []

    def gerar(self, sistema, usuario, **kw):
        self.chamadas.append((sistema, usuario, kw))
        return self.texto


def _item():
    return ab.Item(7, "abc", "Which process powers the Sun?",
                   ("fusion", "fission", "gravity", "chemistry"))


def test_C_sem_fontes_e_A_com_o_bloco_do_produto_antes_da_pergunta():
    modelo, it = Modelo(), _item()
    ab.responder(modelo, it)
    sistema, usuario, kw = modelo.chamadas[-1]
    assert sistema == ab.SISTEMA_SEM_FONTES and "Sources" not in usuario
    assert usuario.endswith("A. fusion\nB. fission\nC. gravity\nD. chemistry")
    assert kw["max_tokens"] == ab.MAX_TOKENS_RESPOSTA and kw["semente"] == ab.SEMENTE
    assert "temperatura" not in kw                 # a do produto, o padrão de `gerar`
    f = Fonte(1, "2101.00001", "Solar neutrinos", 2021, "pp chain.", 0.5)
    ab.responder(modelo, it, [f])
    sistema, usuario, _ = modelo.chamadas[-1]
    assert sistema == ab.SISTEMA_COM_FONTES
    assert usuario.startswith("Sources:\n\n[1] Solar neutrinos (2021, arXiv:2101.00001)\npp chain.")
    assert usuario.index("Sources") < usuario.index("Question: Which process")


@pytest.mark.parametrize("texto,letra", [
    ("Answer: B", "B"), ("answer: c", None), ("The answer is (C).", "C"),
    ("Final answer: **D**", "D"), ('{"ANSWER": "A", "EXPLANATION": "..."}', "A"),
    ("B", "B"), ("B. Because of fusion", "B"), ("(D) chemistry", "D"), ("**C**", "C"),
    ("The correct option is D.", "D"), ("The answer is a planet", None),
    ("A planet orbits the star", None), ("Answer: All of the above", None), ("", None),
    ("Answer: A\nAnswer: B", "A"),
])
def test_extrair_letra(texto, letra):
    assert ab.extrair_letra(texto) == letra


# ── o envio ──────────────────────────────────────────────────────────────────


def test_exportar_recusa_abaixo_da_guarda_e_grava_em_branco_o_que_nao_leu(tmp_path):
    itens = [ab.Item(i, f"id{i}", "q", ("a", "b", "c", "d")) for i in range(100)]
    with pytest.raises(ValueError, match="mínimo 98%"):
        ab.exportar(itens, ["A"] * 97 + [None] * 3, tmp_path / "x.csv")
    sha = ab.exportar(itens, ["A"] * 98 + [None] * 2, tmp_path / "x.csv")
    with (tmp_path / "x.csv").open(encoding="utf-8") as f:
        linhas = list(csv.reader(f))
    assert linhas[0] == list(ab.CABECALHO_ENVIO) and linhas[1] == ["0", "id0", "A"]
    assert linhas[-1] == ["99", "id99", ""] and sha == ab.sha256_arquivo(tmp_path / "x.csv")


def test_o_email_pede_o_acerto_por_item_e_aceita_o_agregado():
    texto = ab.rascunho_de_email(1000, "f" * 64, ["astrobench_C.csv", "astrobench_A.csv"])
    assert texto.startswith(f"To: {ab.ENDERECO_DE_CORRECAO}")
    assert "per-question correctness" in texto and "accuracy of each file" in texto
    assert "not publish" in texto and "1000 questions" in texto


# ── a correção e a leitura ───────────────────────────────────────────────────


def test_ler_correcao_com_nomes_de_coluna_variados_e_cobertura(tmp_path):
    arq = tmp_path / "c.csv"
    arq.write_text("Row_Index,Correct\n0,1\n1,0\n2,true\n", encoding="utf-8")
    c = ab.ler_correcao(arq)
    assert c == {0: True, 1: False, 2: True}
    assert list(ab.acertos_de([2, 0], c)) == [1.0, 1.0]
    with pytest.raises(ValueError, match="não cobre"):
        ab.acertos_de([0, 5], c)
    arq.write_text("row_index,correct\n0,talvez\n", encoding="utf-8")
    with pytest.raises(ValueError, match="não é 1/0"):
        ab.ler_correcao(arq)


def test_corrigir_conta_letra_nao_lida_como_erro(tmp_path):
    arq = tmp_path / "g.csv"
    arq.write_text("row_index,answer\n0,A\n1,(B)\n2,C\n", encoding="utf-8")
    gab = ab.ler_gabarito(arq)
    assert gab == {0: "A", 1: "B", 2: "C"}
    assert list(ab.corrigir([0, 1, 2], ["A", None, "D"], gab)) == [1.0, 0.0, 0.0]


def test_a_leitura_pareada_ajuda_nao_distingue_e_invalida():
    rng = np.random.default_rng(1)
    c = (rng.random(1000) < 0.70).astype(float)
    a = c.copy()
    a[np.flatnonzero(c == 0)[:100]] = 1           # A acerta 100 a mais, nunca perde
    r = ab.apurar_por_item(c, a)
    assert r["leitura"] == "ajuda" and r["material"] and (r["ganha"], r["perde"]) == (100, 0)
    assert r["ic95"][0] > ab.LIMIAR_MATERIAL and r["mcnemar_exato_p (relatado)"] < 1e-6
    assert ab.apurar_por_item(c, c)["leitura"] == "nao_se_distingue"
    pior = c.copy()
    pior[np.flatnonzero(c == 1)[:100]] = 0
    assert ab.apurar_por_item(c, pior)["leitura"] == "atrapalha"
    baixo = (rng.random(1000) < 0.30).astype(float)
    assert ab.apurar_por_item(baixo, np.ones(1000))["leitura"] == "invalida"


def test_newcombe_e_mais_largo_que_o_pareado_quando_os_bracos_erram_juntos():
    rng = np.random.default_rng(2)
    c = (rng.random(1000) < 0.72).astype(float)
    a = c.copy()
    a[np.flatnonzero(c == 0)[:30]] = 1
    r_pareado = ab.apurar_por_item(c, a)
    pareado = r_pareado["ic95"]
    lo, hi = ab.newcombe(int(a.sum()), 1000, int(c.sum()), 1000)
    assert lo < pareado[0] and hi > pareado[1] and lo < 0.03 < hi
    # Os mesmos 30 ganhos: pareados, a busca ajuda; sem o pareamento, não se distingue.
    assert r_pareado["leitura"] == "ajuda"
    r = ab.apurar_agregado(int(c.sum()), int(a.sum()), 1000)
    assert r["pareado"] is False and r["ic95"] == (lo, hi) and r["leitura"] == "nao_se_distingue"
    with pytest.raises(ValueError):
        ab.apurar_agregado(1001, 10, 1000)


def test_secundarias_sem_gabarito():
    meta = {"x1": ("astro-ph.GA", 2020), "x2": ("hep-ph", 2010), "x3": ("astro-ph.SR", 2022)}
    s = ab.secundarias(["A", "B", None, "C"], ["A", "C", "B", None],
                       [["x1", "x2"], ["x2"], ["x3"], ["x2"]], meta)
    assert s["leitura_C"] == 0.75 and s["resposta_mudou (A ≠ C)"] == 0.5
    assert s["fontes_astro_ph"] == 0.4 and s["itens_com_fonte_astro_ph"] == 0.5
    assert s["ano_mediano_das_fontes"] == 2010.0 and s["letras_C"][None] == 1


# ── o script: a trava, o envio e a apuração ponta a ponta ────────────────────


def test_a_regra_esta_aceita_e_a_trava_ainda_recusa_sem_ela(monkeypatch):
    import medir_astrobench as mab

    assert ab.REGRA_9_10_ACEITA is True          # aceita pelo dono em 2026-10-16
    mab._recusa_sem_aceite(dev=False)
    monkeypatch.setattr(ab, "REGRA_9_10_ACEITA", False)
    with pytest.raises(SystemExit, match="PROPOSTA"):
        mab._recusa_sem_aceite(dev=False)
    mab._recusa_sem_aceite(dev=True)


@pytest.fixture
def script(tmp_path, monkeypatch):
    import medir_astrobench as mab

    pasta = tmp_path / "astrobench"
    monkeypatch.setattr(mab, "PASTA", pasta)
    monkeypatch.setattr(mab, "ENVIO", pasta / "envio")
    monkeypatch.setattr(mab, "MANIFESTO", tmp_path / "astrobench_manifesto.json")
    monkeypatch.setattr(mab, "AGREGADO", tmp_path / "astrobench.json")
    monkeypatch.setattr(mab, "RAIZ", tmp_path)
    monkeypatch.setattr(mab, "_assinatura", lambda: "f" * 64)
    indice = tmp_path / "indice"
    indice.mkdir()
    pl.DataFrame({"arxiv_id": ["x1", "x2"], "primary_category": ["astro-ph.CO", "gr-qc"],
                  "year": [2019, 2021]}).write_parquet(indice / "documentos.parquet")
    return mab, indice


def _fases_falsas(mab, itens, letra_a="A"):
    """O que as três fases gravariam: C responde B, A responde `letra_a`."""
    mab.PASTA.mkdir(parents=True, exist_ok=True)
    for fase, reg in (("consulta_C", lambda i: {"consulta": "t", "segundos_consulta": 4.0,
                                                 "bruto_C": "Answer: B", "segundos_C": 1.0}),
                      ("fontes", lambda i: {"fontes": ["x1", "x2"], "segundos": 13.0}),
                      ("A", lambda i: {"bruto_A": f"Answer: {letra_a}", "segundos": 3.0})):
        with mab._arq("teste", fase).open("w", encoding="utf-8") as f:
            for i in itens:
                f.write(json.dumps({"indice": i.indice, "id": i.id, **reg(i)}) + "\n")


def test_registrar_congela_e_recusa_outro_conjunto(tmp_path, script):
    mab, _ = script
    mab.registrar(_parquet(tmp_path), "teste")
    man = json.loads(mab.MANIFESTO.read_text(encoding="utf-8"))
    assert man["arquivo"] == "astrobench_publico.parquet" and man["linhas"] == N_LINHAS
    outro = tmp_path / "outro.parquet"
    _tabela(N_LINHAS + 5).write_parquet(outro)
    registrado = (mab.PASTA / "astrobench_publico.parquet").read_bytes()
    with pytest.raises(SystemExit, match="não troca de conjunto"):
        mab.registrar(outro, "teste")
    assert (mab.PASTA / "astrobench_publico.parquet").read_bytes() == registrado
    mab.registrar(mab.PASTA / "astrobench_publico.parquet", "de novo")   # o mesmo passa


def test_exportar_e_apurar_ponta_a_ponta(tmp_path, script):
    mab, indice = script
    mab.registrar(_parquet(tmp_path), "teste")
    _, itens = mab._conjunto(dev=False)
    _fases_falsas(mab, itens)
    a = type("A", (), {"indice": indice, "spine": None})()
    mab.exportar(a)
    ag = json.loads(mab.AGREGADO.read_text(encoding="utf-8"))
    sec = ag["secundarias (relatadas, não decidem)"]
    assert ag["primario"] is None and set(ag["envio"]) == {"C", "A"}
    assert sec["resposta_mudou (A ≠ C)"] == 1.0 and sec["fontes_astro_ph"] == 0.5
    assert (mab.ENVIO / "email.txt").exists()
    # A correção volta item a item: o gabarito é A em todos — A acerta tudo, C nada.
    for b, ok in (("C", 0), ("A", 1)):
        (tmp_path / f"corr_{b}.csv").write_text(
            "row_index,correct\n" + "".join(f"{i.indice},{ok}\n" for i in itens),
            encoding="utf-8")
    args = type("Args", (), {"correcao_C": tmp_path / "corr_C.csv",
                             "correcao_A": tmp_path / "corr_A.csv", "gabarito": None,
                             "agregado_C": None, "agregado_A": None})()
    mab.apurar(args)
    ag = json.loads(mab.AGREGADO.read_text(encoding="utf-8"))
    # C em 0 é a guarda de sanidade: a medida é inválida, não "a busca ajuda".
    assert ag["decisao"] == "invalida" and ag["primario"]["pareado"] is True
    assert _maior_lista(ag) <= 4                 # o agregado vai ao git: nada item a item
    with pytest.raises(SystemExit, match="já foi apurada"):
        mab.exportar(a)


def _maior_lista(x) -> int:
    if isinstance(x, dict):
        return max((_maior_lista(v) for v in x.values()), default=0)
    if isinstance(x, list):
        return max([len(x), *(_maior_lista(v) for v in x)])
    return 0


def test_exportar_recusa_reescrever_um_envio_diferente(tmp_path, script):
    mab, indice = script
    mab.registrar(_parquet(tmp_path), "teste")
    _, itens = mab._conjunto(dev=False)
    a = type("A", (), {"indice": indice, "spine": None})()
    _fases_falsas(mab, itens, "A")
    mab.exportar(a)
    mab.exportar(a)                                          # o mesmo envio passa
    enviado = (mab.ENVIO / "astrobench_A.csv").read_bytes()
    _fases_falsas(mab, itens, "C")
    with pytest.raises(SystemExit, match="não muda depois de enviado"):
        mab.exportar(a)
    assert (mab.ENVIO / "astrobench_A.csv").read_bytes() == enviado


def test_apurar_recusa_arquivo_de_envio_mexido(tmp_path, script):
    mab, indice = script
    mab.registrar(_parquet(tmp_path), "teste")
    _, itens = mab._conjunto(dev=False)
    _fases_falsas(mab, itens)
    mab.exportar(type("A", (), {"indice": indice, "spine": None})())
    with (mab.ENVIO / "astrobench_A.csv").open("a", encoding="utf-8") as f:
        f.write("9999,x,A\n")
    args = type("Args", (), {"correcao_C": None, "correcao_A": None, "gabarito": None,
                             "agregado_C": 700, "agregado_A": 720})()
    with pytest.raises(SystemExit, match="não é o arquivo enviado"):
        mab.apurar(args)
