"""A precisão de citação (DOC-13 §6.3): o corte em pares, o veredicto, a amostra do Claude,
o selo antes do juiz e a apuração — com respostas e modelo de mentira."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.eval import citacoes as ct  # noqa: E402

# ── o corte ──────────────────────────────────────────────────────────────────


def test_frases_citadas_pelo_corte_do_portao():
    texto = ("A temperatura é 650 K [1]. O filme cresce em Si [2][3]. "
             "Uma frase sem fonte nenhuma aqui. Valor de 1.5 eV [1-2].")
    assert ct.frases_citadas(texto, 3) == [
        ("A temperatura é 650 K.", [1]), ("O filme cresce em Si.", [2, 3]),
        ("Valor de 1.5 eV.", [1, 2])]


def test_citacao_no_comeco_da_frase_volta_para_a_anterior_e_invalida_sai():
    texto = "É 650 K. [2] Além disso, cresce em Si [9]. Por fim [1]."
    assert ct.frases_citadas(texto, 3) == [("É 650 K.", [2]), ("Por fim.", [1])]


def test_pares_um_por_fonte_citada_sem_repetir():
    ps = ct.pares("É 650 K [1][1][3]. Outra [2].", ["a", "b", "c"])
    assert [(p["frase_n"], p["n"], p["arxiv_id"]) for p in ps] == [(0, 1, "a"), (0, 3, "c"),
                                                                   (1, 2, "b")]
    assert ct.pares("Não sei responder com estas fontes.", ["a"]) == []


@pytest.mark.parametrize("bruto,v", [
    ('{"motivo": "ok", "veredicto": "sustenta"}', "sustenta"),
    ('texto {"veredicto": "não_sustenta", "motivo": "x"} fim', "nao_sustenta"),
    ("I think it is parcial", "parcial"), ("nada", "ilegivel"), ("", "ilegivel"),
])
def test_ler_veredicto(bruto, v):
    assert ct.ler_veredicto(bruto)[0] == v


class Modelo:
    def __init__(self, resposta='{"motivo": "m", "veredicto": "sustenta"}'):
        self.resposta, self.chamadas = resposta, []

    def gerar(self, sistema, usuario, **kw):
        self.chamadas.append((sistema, usuario, kw))
        return self.resposta


def test_julgar_com_a_instrucao_e_temperatura_zero():
    m = Modelo()
    assert ct.julgar(m, "Qual a T?", "É 650 K.", "Título (2020, arXiv:x)\nResumo")[0] == "sustenta"
    sistema, usuario, kw = m.chamadas[0]
    assert sistema == ct.SISTEMA_SUSTENTA and kw["temperatura"] == 0.0
    assert usuario.startswith("Question: Qual a T?") and "Sentence: É 650 K." in usuario


# ── a amostra, a guarda e a conta ───────────────────────────────────────────


def test_amostras_das_rodadas_sao_disjuntas_e_fixas():
    r1, r2 = ct.amostra_validacao(1000, 1), ct.amostra_validacao(1000, 2)
    assert len(r1) == ct.N_VALIDACAO and not set(r1) & set(r2)
    assert ct.amostra_validacao(1000, 1) == r1
    with pytest.raises(ValueError):
        ct.amostra_validacao(200, 2)
    with pytest.raises(ValueError):
        ct.amostra_validacao(1000, ct.MAX_RODADAS + 1)


def test_validade_mostra_a_direcao_do_desacordo():
    claude = ["sustenta"] * 80 + ["nao_sustenta"] * 20
    juiz = ["sustenta"] * 80 + ["sustenta"] * 10 + ["nao_sustenta"] * 10
    v = ct.validade(claude, juiz)
    assert v["juiz_sustenta_claude_nao"] == 10 and v["claude_sustenta_juiz_nao"] == 0
    assert v["precisao_pelo_claude"] == 0.8 and v["precisao_pelo_juiz"] == 0.9
    assert ct.validade(claude, claude)["passa"]


def test_precisao_por_resposta_e_a_decisao_de_um_lado_da_meta():
    tudo = [["sustenta"] * 4 for _ in range(300)]
    p, ic = ct.precisao(tudo)
    assert p == 1.0 and ic == (1.0, 1.0) and ct.decidir(ic, True) == "passa"
    metade = [["sustenta", "nao_sustenta"] for _ in range(300)]
    p, ic = ct.precisao(metade)
    assert p == 0.5 and ct.decidir(ic, True) == "nao_passa"
    assert ct.decidir((0.94, 0.96), True) == "fronteira"
    assert ct.decidir((0.99, 1.0), False) == "juiz_invalido"


# ── o script: selo antes do juiz, e a apuração ──────────────────────────────


def _fonte(arxiv_id):
    return SimpleNamespace(arxiv_id=arxiv_id, titulo=f"T {arxiv_id}", ano=2020,
                           resumo=f"resumo de {arxiv_id}")


@pytest.fixture
def montado(tmp_path, monkeypatch):
    """200 respostas do primário (cada uma com duas frases citadas) e 10 do pós-corte."""
    import medir_citacoes as script

    itens, respostas = [], []
    for k in range(210):
        estrato = "primario" if k < 200 else "pos_corte"
        itens.append({"estrato": estrato, "arxiv_id": f"p{k}", "pergunta": f"pergunta {k}"})
        respostas.append(SimpleNamespace(
            estrato=estrato, arxiv_id=f"p{k}", fontes=[f"p{k}", f"o{k}"],
            texto=f"Fato {k} [1]. Outro fato {k} [2]."))
    ps = script.montar_pares(itens, respostas, lambda ids: [_fonte(i) for i in ids])
    pasta, aval = tmp_path / "citacoes", tmp_path / "avaliacao"
    pasta.mkdir()
    aval.mkdir()
    bruto = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in ps)
    (pasta / "pares.jsonl").write_text(bruto, encoding="utf-8")
    (aval / "citacoes_pares.json").write_text(json.dumps(
        {"sha256_pares": script.mp._sha(bruto)}), encoding="utf-8")
    return script, pasta, aval, ps


def test_montar_pares_traz_a_fonte_como_o_modelo_leu(montado):
    _, _, _, ps = montado
    assert len(ps) == 420 and ps[0]["fonte"] == "T p0 (2020, arXiv:p0)\nresumo de p0"
    assert ps[0]["e_o_artigo_da_pergunta"] and not ps[1]["e_o_artigo_da_pergunta"]


def test_o_juiz_nao_roda_sem_o_selo_e_o_selo_nao_muda(montado, tmp_path):
    script, pasta, aval, _ = montado
    with pytest.raises(SystemExit, match="não foi selada"):
        script._selados(1, pasta, aval)
    script.exportar_validacao(1, pasta, aval)
    vered = tmp_path / "v.json"
    vered.write_text(json.dumps([{"indice": k, "veredicto": "sustenta"}
                                 for k in range(ct.N_VALIDACAO)]), encoding="utf-8")
    script.selar(vered, 1, pasta, aval)
    assert len(script._selados(1, pasta, aval)) == ct.N_VALIDACAO
    with pytest.raises(SystemExit, match="já foi selada"):
        script.selar(vered, 1, pasta, aval)
    amostra = json.loads((pasta / "validacao_r1.json").read_text(encoding="utf-8"))
    assert set(amostra[0]) == {"indice", "pergunta", "frase", "fonte"}   # nada além do texto


def test_apurar_ponta_a_ponta(montado, tmp_path):
    script, pasta, aval, ps = montado
    # O juiz: sustenta tudo o que cita o artigo da pergunta; das outras, 1 em 5 não.
    vered_de = {p["chave"]: ("nao_sustenta" if not p["e_o_artigo_da_pergunta"] and (k // 2) % 5 == 0
                             else "sustenta") for k, p in enumerate(ps)}
    with script._cache_do_juiz(pasta, "a" * 64).open("w", encoding="utf-8") as f:
        for p in ps:
            f.write(json.dumps({"chave": p["chave"], "veredicto": vered_de[p["chave"]]}) + "\n")
    # O Claude concorda com ele na amostra: a guarda passa.
    script.exportar_validacao(1, pasta, aval)
    prim = [p for p in ps if p["estrato"] == "primario"]
    amostra = [prim[j] for j in ct.amostra_validacao(len(prim), 1)]
    vered = tmp_path / "v.json"
    vered.write_text(json.dumps([{"indice": k, "veredicto": vered_de[p["chave"]]}
                                 for k, p in enumerate(amostra)]), encoding="utf-8")
    script.selar(vered, 1, pasta, aval)
    r = script.apurar(1, pasta, aval, "a" * 64)
    assert r["primario (decide)"]["pares"] == 400 and r["guarda (o juiz contra o Claude)"]["passa"]
    assert r["relatado"]["citando_o_artigo_da_pergunta"]["precisao"] == 1.0
    assert r["primario (decide)"]["precisao"] == 0.9 and r["decisao"] == "nao_passa"
    assert r["relatado"]["frases_com_ao_menos_uma_citacao_que_sustenta"] == 0.9
    assert json.loads((aval / "citacoes.json").read_text(encoding="utf-8"))["rodada"] == 1


def test_a_regra_e_proposta_e_o_juiz_recusa():
    import medir_citacoes as script

    assert ct.REGRA_6_3_ACEITA is False
    with pytest.raises(SystemExit, match="PROPOSTA"):
        script.julgar_fase(1)
