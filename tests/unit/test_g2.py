"""A regra do portão G2 (DOC-12 §3.7) com notas SINTÉTICAS. O que se tranca:

1. as travas: sem aceite, sem congelamento ou com o g2.yaml mudado, nada toca o teste;
2. o livro de acessos: 1 medida e 1 repetição por falha operacional; a base re-medida só
   com outro hash de configuração;
3. os grupos por 13-grama e a partição do UG, sem dividir grupo;
4. o bootstrap por agrupamento e as três decisões;
5. o g2.yaml e as constantes do código dizem a mesma coisa.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2  # noqa: E402

N_BOOT = 2000


# ── as travas ────────────────────────────────────────────────────────────────


def test_trava_sem_aceite(tmp_path):
    arq = tmp_path / "g2.yaml"
    arq.write_text("estado: PROPOSTA\n", encoding="utf-8")
    with pytest.raises(g2.RegraNaoAceita):
        g2.exigir_aceite_e_congelamento(arq, aceita=False, sha_congelado="x")


def test_trava_sem_congelamento(tmp_path):
    arq = tmp_path / "g2.yaml"
    arq.write_text("estado: ACEITA\n", encoding="utf-8")
    with pytest.raises(g2.ProtocoloNaoCongelado):
        g2.exigir_aceite_e_congelamento(arq, aceita=True, sha_congelado="")


def test_trava_yaml_mudou_depois_do_congelamento(tmp_path):
    arq = tmp_path / "g2.yaml"
    arq.write_text("estado: ACEITA\n", encoding="utf-8")
    sha = g2.sha256_arquivo(arq)
    assert g2.exigir_aceite_e_congelamento(arq, aceita=True, sha_congelado=sha) == sha
    arq.write_text("estado: ACEITA\nk: 1\n", encoding="utf-8")
    with pytest.raises(g2.ProtocoloMudou):
        g2.exigir_aceite_e_congelamento(arq, aceita=True, sha_congelado=sha)


def test_as_constantes_trancam_o_teste_ate_o_congelamento():
    # Sem aceite, ou aceita e ainda não congelada, as constantes do código trancam o teste.
    if not g2.REGRA_G2_ACEITA:
        with pytest.raises(g2.RegraNaoAceita):
            g2.exigir_aceite_e_congelamento()
    elif not g2.PROTOCOLO_G2_SHA256:
        with pytest.raises(g2.ProtocoloNaoCongelado):
            g2.exigir_aceite_e_congelamento()


# ── o livro de acessos ───────────────────────────────────────────────────────


def test_livro_uma_medida_e_uma_repeticao(tmp_path):
    livro = tmp_path / "acessos.jsonl"
    g2.registrar_acesso("1.7B", "base", "medida", "h1", livro=livro)
    with pytest.raises(g2.AcessoRecusado):
        g2.registrar_acesso("1.7B", "base", "medida", "h1", livro=livro)
    g2.registrar_acesso("1.7B", "base", "falha_operacional", "h1", livro=livro)
    with pytest.raises(g2.AcessoRecusado):
        g2.registrar_acesso("1.7B", "base", "falha_operacional", "h1", livro=livro)
    linhas = [json.loads(x) for x in livro.read_text(encoding="utf-8").splitlines()]
    assert [r["motivo"] for r in linhas] == ["medida", "falha_operacional"]


def test_livro_base_re_medida_so_com_outro_hash(tmp_path):
    livro = tmp_path / "acessos.jsonl"
    g2.registrar_acesso("8B", "base", "medida", "h1", livro=livro)
    g2.registrar_acesso("8B", "base", "medida", "h2", livro=livro)   # a configuração mudou
    g2.registrar_acesso("8B", "phigen", "medida", "h2", livro=livro)
    with pytest.raises(g2.AcessoRecusado):                            # ΦGen: um acesso só
        g2.registrar_acesso("8B", "phigen", "medida", "h3", livro=livro)
    g2.registrar_acesso("1.7B", "phigen", "medida", "h9", livro=livro)  # outro tamanho


def test_livro_repeticao_sem_medida_e_motivo_desconhecido(tmp_path):
    livro = tmp_path / "acessos.jsonl"
    with pytest.raises(g2.AcessoRecusado):
        g2.registrar_acesso("8B", "phigen", "falha_operacional", "h1", livro=livro)
    with pytest.raises(g2.AcessoRecusado):
        g2.registrar_acesso("8B", "phigen", "curiosidade", "h1", livro=livro)
    assert not livro.exists()


# ── grupos e partição ────────────────────────────────────────────────────────


def _frase(semente: int, n: int = 20) -> str:
    rng = np.random.default_rng(semente)
    return " ".join(f"w{x}" for x in rng.integers(0, 10**6, n))


def test_grupos_por_13grama_transitivos_e_sem_frase_feita():
    a, b, c = _frase(1), _frase(2), _frase(3)
    textos = [
        a + " " + b,             # 0: compartilha a com 1
        a + " " + c,             # 1: compartilha c com 2
        c + " fim",              # 2
        _frase(4),               # 3: isolado
    ]
    grupo = g2.grupos_por_ngramas(textos)
    assert grupo[0] == grupo[1] == grupo[2] == 0
    assert grupo[3] == 3
    # um 13-grama em itens demais é frase-feita do benchmark e não agrupa
    comum = _frase(9)
    muitos = [comum + " " + _frase(100 + i) for i in range(5)]
    assert g2.grupos_por_ngramas(muitos, max_itens=4) == list(range(5))
    assert g2.grupos_por_ngramas(muitos, max_itens=50) == [0] * 5


def test_particao_nao_divide_grupo_e_so_tem_elegiveis():
    rng = np.random.default_rng(0)
    n = 3000
    materia = [f"m{x}" for x in rng.integers(0, 5, n)]
    grupo = [i if rng.random() < 0.7 else max(0, i - int(rng.integers(1, 4))) for i in range(n)]
    # o rótulo do grupo tem de ser o de um item do próprio grupo
    for i in range(n):
        while grupo[grupo[i]] != grupo[i]:
            grupo[i] = grupo[grupo[i]]
    elegivel = [bool(x) for x in rng.random(n) < 0.8]
    p = g2.particionar_por_grupos(materia, elegivel, grupo, n_teste=1000)
    conjuntos = {k: set(v) for k, v in p.items()}
    assert not (conjuntos["dev"] & conjuntos["teste"])
    assert not (conjuntos["teste"] & conjuntos["reserva"])
    assert sum(len(v) for v in conjuntos.values()) == sum(elegivel)
    assert all(elegivel[i] for v in conjuntos.values() for i in v)
    onde = {i: k for k, v in conjuntos.items() for i in v}
    por_grupo: dict[int, set] = {}
    for i, g in enumerate(grupo):
        if elegivel[i]:
            por_grupo.setdefault(g, set()).add(onde[i])
    assert all(len(s) == 1 for s in por_grupo.values())
    assert abs(len(p["teste"]) - 1000) <= 15
    assert p == g2.particionar_por_grupos(materia, elegivel, grupo, n_teste=1000)


# ── bootstrap e decisões ─────────────────────────────────────────────────────


def test_bootstrap_por_grupo_alarga_o_ic_quando_o_grupo_se_parece():
    rng = np.random.default_rng(1)
    efeito_do_grupo = np.repeat(rng.normal(0, 0.3, 50), 10)
    base = np.clip(0.4 + efeito_do_grupo * 0, 0, 1)
    phigen = np.clip(base + 0.05 + efeito_do_grupo, 0, 1)
    grupo = np.repeat(np.arange(50), 10)
    ponto, por_item = g2.reps_delta_agrupado(phigen, base, None, np.random.default_rng(2), N_BOOT)
    _, por_grupo = g2.reps_delta_agrupado(phigen, base, grupo, np.random.default_rng(2), N_BOOT)
    assert ponto == pytest.approx(100 * float(np.mean(phigen - base)))
    assert np.std(por_grupo) > 1.5 * np.std(por_item)


def _bench(ganho: dict[str, float], n: int = 400, semente: int = 0):
    rng = np.random.default_rng(semente)
    base, phigen, grupos = {}, {}, {}
    for b in g2.BENCH_G2_1:
        p = rng.uniform(0.05, 0.5, n)
        base[b] = rng.binomial(8, p) / 8
        phigen[b] = rng.binomial(8, np.clip(p + ganho[b], 0, 1)) / 8
        grupos[b] = np.arange(n) // 2 if b in ("ugphysics", "olympiadbench") else None
    return phigen, base, grupos


def test_g2_1_passa_com_ganho_grande():
    phigen, base, grupos = _bench(dict.fromkeys(g2.BENCH_G2_1, 0.20))
    nada = {b: np.zeros(400, dtype=bool) for b in g2.BENCH_G2_1}
    r = g2.decidir_g2_1(phigen, base, grupos, nada, n_bootstrap=N_BOOT)
    assert r["passa"] and r["leitura"] == "passa"
    assert r["todos"]["ic_macro"][0] >= 10


def test_g2_1_ganho_pequeno_e_demonstrado_abaixo():
    phigen, base, grupos = _bench(dict.fromkeys(g2.BENCH_G2_1, 0.03))
    nada = {b: np.zeros(400, dtype=bool) for b in g2.BENCH_G2_1}
    r = g2.decidir_g2_1(phigen, base, grupos, nada, n_bootstrap=N_BOOT)
    assert not r["passa"] and r["leitura"] == "demonstrado abaixo"


def test_g2_1_um_benchmark_piorando_reprova():
    ganho = dict.fromkeys(g2.BENCH_G2_1, 0.30)
    ganho["phybench"] = -0.15
    phigen, base, grupos = _bench(ganho, n=800)
    nada = {b: np.zeros(800, dtype=bool) for b in g2.BENCH_G2_1}
    r = g2.decidir_g2_1(phigen, base, grupos, nada, n_bootstrap=N_BOOT)
    assert r["condicoes"]["ic_macro_acima_do_limiar"]
    assert not r["condicoes"]["nenhum_benchmark_piora"] and not r["passa"]


def test_g2_1_ganho_so_nos_itens_memorizados_reprova():
    rng = np.random.default_rng(3)
    n = 400
    base, phigen, grupos, memo = {}, {}, {}, {}
    for b in g2.BENCH_G2_1:
        p = rng.uniform(0.05, 0.4, n)
        base[b] = rng.binomial(8, p) / 8
        m = rng.random(n) < 0.3
        phigen[b] = np.where(m, 1.0, rng.binomial(8, p) / 8)   # ganha só onde memorizou
        grupos[b], memo[b] = None, m
    r = g2.decidir_g2_1(phigen, base, grupos, memo, n_bootstrap=N_BOOT)
    assert r["condicoes"]["ic_macro_acima_do_limiar"]
    assert not r["condicoes"]["sem_memorizacao_nova"] and not r["passa"]


def test_g2_3_assimetrico():
    rng = np.random.default_rng(4)
    mmlu = rng.binomial(1, 0.6, 13402).astype(float)
    codigo = rng.binomial(10, 0.5, 542) / 10
    ifeval = rng.binomial(16, 0.35, 541) / 16
    # sem regressão: passa
    r = g2.decidir_g2_3({"mmlu_sem_fisica": (mmlu, mmlu), "evalplus": (codigo, codigo),
                         "ifeval_envelope": (ifeval, ifeval)}, n_bootstrap=N_BOOT)
    assert r["passa"]
    # código caindo ~6 pontos: desclassifica
    pior = np.clip(codigo - 0.06, 0, 1)
    r = g2.decidir_g2_3({"mmlu_sem_fisica": (mmlu, mmlu), "evalplus": (pior, codigo),
                         "ifeval_envelope": (ifeval, ifeval)}, n_bootstrap=N_BOOT)
    assert not r["passa"] and r["medidas"]["evalplus"]["falha"]
    # MMLU caindo 3 pontos: a não-inferioridade não se demonstra
    caiu = mmlu.copy()
    caiu[np.flatnonzero(mmlu == 1)[: int(0.03 * mmlu.size)]] = 0
    r = g2.decidir_g2_3({"mmlu_sem_fisica": (caiu, mmlu), "evalplus": (codigo, codigo),
                         "ifeval_envelope": (ifeval, ifeval)}, n_bootstrap=N_BOOT)
    assert not r["passa"] and r["medidas"]["mmlu_sem_fisica"]["falha"]


def test_g2_2_sem_gpqa_exige_3_de_4_e_margem_contra_quantizado():
    rng = np.random.default_rng(5)
    n = 600
    benches = ("olympiadbench", "phybench", "scibench", "ugphysics")
    comp = {"A": {}, "B": {}}
    phigen = {}
    for j, b in enumerate(benches):
        p = rng.uniform(0.1, 0.4, n)
        comp["A"][b] = rng.binomial(2, p) / 2
        comp["B"][b] = rng.binomial(2, p) / 2
        ganho = 0.25 if j < 3 else 0.0      # vence só em 3 dos 4
        phigen[b] = rng.binomial(8, np.clip(p + ganho, 0, 1)) / 8
    grupos = dict.fromkeys(benches)
    r = g2.decidir_g2_2(phigen, comp, grupos, n_bootstrap=N_BOOT)
    assert r["vitorias"] == 3 and r["passa"]
    assert not r["por_benchmark"]["gpqa_fisica"]["medido"]
    assert not r["por_benchmark"]["ugphysics"]["vence"]
    # sem um dos três: 2 de 5, não passa
    phigen2 = {b: (v if b != "olympiadbench" else comp["A"][b]) for b, v in phigen.items()}
    assert not g2.decidir_g2_2(phigen2, comp, grupos, n_bootstrap=N_BOOT)["passa"]


def test_g2_2_margem_de_um_ponto_contra_quantizado():
    rng = np.random.default_rng(6)
    n = 4000
    p = rng.uniform(0.2, 0.4, n)
    comp = rng.binomial(1, p).astype(float)
    phigen = np.clip(comp + (rng.random(n) < 0.012), 0, 1)   # ~+1,2 ponto, IC estreito
    b = "phybench"
    r_q8 = g2.decidir_g2_2({b: phigen}, {"C": {b: comp}}, {b: None}, n_bootstrap=N_BOOT)
    r_q5 = g2.decidir_g2_2({b: phigen}, {"C": {b: comp}}, {b: None},
                           abaixo_de_q8=frozenset({"C"}), n_bootstrap=N_BOOT)
    ic = r_q8["por_benchmark"][b]["contra"]["C"]["ic99"]
    assert ic[0] > 0 and r_q8["por_benchmark"][b]["vence"]
    assert (ic[0] > 1.0) == r_q5["por_benchmark"][b]["vence"]


# ── o g2.yaml e o código ─────────────────────────────────────────────────────


def test_yaml_e_codigo_concordam():
    yaml = pytest.importorskip("yaml")
    pre = yaml.safe_load(g2.PREREGISTRO.read_text(encoding="utf-8"))
    assert pre["estado"] == ("ACEITA" if g2.REGRA_G2_ACEITA else "PROPOSTA")
    assert pre["semente"] == g2.SEMENTE_G2
    assert pre["bootstrap"]["replicas"] == g2.N_BOOTSTRAP
    assert tuple(pre["g2_1"]["benchmarks"]) == g2.BENCH_G2_1
    assert pre["g2_1"]["limiar_pontos"] == g2.LIMIAR_G2_1
    assert pre["g2_3"]["margem_pontos"] == g2.MARGEM_G2_3
    assert tuple(pre["g2_2"]["benchmarks_do_charter"]) == g2.BENCH_G2_2
    assert pre["g2_2"]["vitorias_minimas"] == g2.VITORIAS_G2_2
    assert tuple(pre["g2_2"]["comparadores"]) == g2.COMPARADORES_G2_2
    assert pre["g2_2"]["margem_contra_quantizado_abaixo_de_q8"] == g2.MARGEM_QUANTIZADO_G2_2
    assert set(pre["g2_3"]) >= {*g2.NAO_INFERIORIDADE_G2_3, *g2.DESCLASSIFICA_G2_3}
