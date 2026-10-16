"""O juiz contra físicos (DOC-13 §9.11): a amostra, a folha às cegas, a leitura dos
veredictos e a regra — com dados de mentira, sem o cache das medidas."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.eval import julgamento_humano as jh  # noqa: E402


def _entradas() -> list[dict]:
    es = []
    for k in range(100):
        origem = "selado_r2" if k < 50 else ("A2" if k < 75 else "A5")
        p, r = f"pergunta {k}", f"resposta {k}"
        juiz = "certo" if k % 4 else "errado"
        es.append({"id": jh.id_opaco(p, r), "origem": origem,
                   "braco": "A" if origem == "selado_r2" else origem, "estrato": "primario",
                   "arxiv_id": f"x{k}", "pergunta": p, "gabarito": f"g{k}", "resposta": r,
                   "juiz": juiz, "claude": juiz if origem == "selado_r2" else None})
    return es


def _baixado(entradas, assinatura, veredicto=lambda e: e["juiz"], passo_ms=5000):
    return {"assinatura_amostra": assinatura, "n_amostra": len(entradas),
            "veredictos": [{"indice": n, "id": e["id"], "veredicto": veredicto(e),
                            "ms": 1_000_000 + n * passo_ms} for n, e in enumerate(entradas)]}


# ── a amostra ────────────────────────────────────────────────────────────────


def test_sortear_selados_fixo_e_do_tamanho_da_regra():
    s = jh.sortear_selados(100)
    assert len(set(s)) == jh.N_SELADOS and s == sorted(s) and max(s) < 100
    assert jh.sortear_selados(100) == s
    with pytest.raises(ValueError, match="a regra pede"):
        jh.sortear_selados(40)


def test_sortear_novos_so_do_primario_fora_das_rodadas_e_disjuntos():
    chaves = [("primario", f"p{k}") for k in range(80)] + [("pos_corte", f"q{k}") for k in range(50)]
    excluir = {("primario", f"p{k}") for k in range(20)}
    n = jh.sortear_novos(chaves, excluir)
    assert set(n) == set(jh.BRACOS_NOVOS)
    todos = [k for ks in n.values() for k in ks]
    assert len(todos) == len(set(todos)) == 2 * jh.N_NOVOS_POR_BRACO
    assert all(k[0] == "primario" and k not in excluir for k in todos)
    assert jh.sortear_novos(chaves, excluir) == n
    with pytest.raises(ValueError, match="elegíveis"):
        jh.sortear_novos(chaves[:60], excluir)


def test_conferir_amostra_composicao_ids_textos_e_claude_so_nas_seladas():
    es = _entradas()
    jh.conferir_amostra(es)
    with pytest.raises(ValueError, match="composição"):
        jh.conferir_amostra(es[1:])
    dup = [*es[:-1], {**es[-1], "pergunta": es[-2]["pergunta"], "resposta": es[-2]["resposta"],
                      "id": "outro"}]
    with pytest.raises(ValueError, match="duas vezes"):
        jh.conferir_amostra(dup)
    with pytest.raises(ValueError, match="só nas seladas"):
        jh.conferir_amostra([*es[:-1], {**es[-1], "claude": "certo"}])


def test_a_folha_nao_vaza_braco_juiz_nem_claude_e_cada_um_tem_sua_ordem():
    es = _entradas()
    f1, f2 = jh.itens_da_folha(es, "fisico1"), jh.itens_da_folha(es, "fisico2")
    assert [i["id"] for i in f1] != [i["id"] for i in f2]
    assert sorted(i["id"] for i in f1) == sorted(e["id"] for e in es)
    texto = json.dumps(f1, ensure_ascii=False)
    for proibido in ("selado", "A2", "A5", "certo", "errado", "x1", "origem", "juiz"):
        assert proibido not in texto
    assert [c["rotulo"] for c in f1[0]["campos"]] == ["pergunta", "gabarito",
                                                      "resposta do assistente"]


def test_ler_veredictos_confere_folha_completude_categorias_e_tempo():
    es = _entradas()
    ids = {e["id"] for e in es}
    assert len(jh.ler_veredictos(_baixado(es, "abc"), "abc", ids)) == 100
    with pytest.raises(ValueError, match="outra folha"):
        jh.ler_veredictos(_baixado(es, "abc"), "xyz", ids)
    with pytest.raises(ValueError, match="incompleta"):
        jh.ler_veredictos(_baixado(es[:99], "abc"), "abc", ids)
    with pytest.raises(ValueError, match="quatro categorias"):
        jh.ler_veredictos(_baixado(es, "abc", lambda e: "talvez"), "abc", ids)
    with pytest.raises(ValueError, match="não é leitura"):
        jh.ler_veredictos(_baixado(es, "abc", passo_ms=200), "abc", ids)


# ── a regra ──────────────────────────────────────────────────────────────────


def test_kappa_ic_concorda_com_o_kappa_e_e_1_na_concordancia_total():
    x = ["certo", "errado", "parcial", "certo"] * 25
    k, ic = jh.kappa_ic(x, x)
    assert k == 1.0 and ic == (1.0, 1.0)
    y = ["certo", "certo", "parcial", "errado"] * 25
    k, (lo, hi) = jh.kappa_ic(x, y)
    assert k == round(jh.kappa(jh._certo(x), jh._certo(y)), 4) and lo < k < hi


def test_passa_quando_os_fisicos_concordam_entre_si_e_com_o_juiz():
    es = _entradas()
    h = {e["id"]: e["juiz"] for e in es}
    r = jh.apurar(es, {"fisico1": h, "fisico2": h})
    assert r["decisao"] == "passa"
    assert r["juiz_x_fisico (decide)"]["fisico1"]["kappa_certo"] == 1.0
    assert r["claude_x_fisico nas seladas (relatado)"]["fisico2"]["n"] == jh.N_SELADOS


def test_nao_passa_quando_o_juiz_infla_e_inconclusivo_quando_os_fisicos_discordam():
    es = _entradas()
    # Os dois físicos negam o "certo" do juiz em metade dos itens: concordam entre si.
    h = {e["id"]: ("errado" if k % 2 else e["juiz"]) for k, e in enumerate(es)}
    r = jh.apurar(es, {"fisico1": h, "fisico2": h})
    assert r["decisao"] == "nao_passa"
    assert r["por_origem (relatado)"]["A5"]["juiz_certo_fisico1_nao"] > 0
    # Um segundo físico que discorda do primeiro: a referência humana não é estável.
    outro = {e["id"]: ("certo" if v != "certo" else "errado") for e, v in zip(es, h.values(),
                                                                           strict=True)}
    assert jh.apurar(es, {"fisico1": h, "fisico2": outro})["decisao"] == "inconclusivo"
    with pytest.raises(ValueError, match="a regra pede"):
        jh.apurar(es, {"fisico1": h})


# ── o script ─────────────────────────────────────────────────────────────────


@pytest.fixture
def script(tmp_path, monkeypatch):
    import julgamento_humano as js

    monkeypatch.setattr(js, "DIR_H", tmp_path / "humano")
    monkeypatch.setattr(js, "AMOSTRA", tmp_path / "humano" / "amostra_humano.json")
    monkeypatch.setattr(js, "SELO", tmp_path / "selo.json")
    monkeypatch.setattr(js, "AGREGADO", tmp_path / "agregado.json")
    monkeypatch.setattr(js, "RAIZ", tmp_path)
    monkeypatch.setattr(js, "montar_amostra", _entradas)
    return js


def test_a_regra_esta_aceita_e_a_trava_ainda_recusa_sem_ela(script, monkeypatch):
    assert jh.REGRA_9_11_ACEITA is True          # aceita pelo dono em 2026-10-16
    monkeypatch.setattr(jh, "REGRA_9_11_ACEITA", False)
    with pytest.raises(SystemExit, match="PROPOSTA"):
        script.amostrar()


def test_selo_folhas_e_apuracao_ponta_a_ponta(script, monkeypatch):
    monkeypatch.setattr(jh, "REGRA_9_11_ACEITA", True)
    script.amostrar()
    script.folha("fisico1")
    script.folha("fisico2")
    selo = json.loads(script.SELO.read_text(encoding="utf-8"))
    assert set(selo["folhas"]) == {"fisico1", "fisico2"} and selo["n"] == 100
    pagina = (script.DIR_H / "folha_fisico1.html").read_text(encoding="utf-8")
    assert "selado_r2" not in pagina and '"juiz"' not in pagina and "pergunta 0" in pagina
    es = _entradas()
    arqs = []
    for nome in ("fisico1", "fisico2"):
        arq = script.DIR_H / f"v_{nome}.json"
        arq.write_text(json.dumps(_baixado(es, selo["folhas"][nome])), encoding="utf-8")
        arqs.append(arq)
    script.apurar(arqs)
    ag = json.loads(script.AGREGADO.read_text(encoding="utf-8"))
    assert ag["decisao"] == "passa" and ag["julgadores"] == ["fisico1", "fisico2"]
    assert "pergunta 0" not in json.dumps(ag)            # o agregado vai ao git
    with pytest.raises(SystemExit, match="já entrou"):
        script.apurar([arqs[0], arqs[0]])


def test_a_amostra_nao_muda_depois_de_selada(script, monkeypatch):
    monkeypatch.setattr(jh, "REGRA_9_11_ACEITA", True)
    script.amostrar()
    monkeypatch.setattr(script, "montar_amostra", lambda: list(reversed(_entradas())))
    with pytest.raises(SystemExit, match="não muda depois de selada"):
        script.amostrar()
    with pytest.raises(SystemExit, match="pseudônimo"):
        script.folha("Fulano de Tal")
