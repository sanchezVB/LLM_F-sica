"""Os itens da medida do assistente (DOC-13 §9.1), com o modelo FALSO.

Tranca o que pode viciar o conjunto sem dar erro:

1. os dois estratos são DISJUNTOS e a ordem sai igual da mesma semente;
2. o modelo não vê o título, e a guarda derruba a pergunta que o copia;
3. toda tentativa fica registrada, e a retomada não refaz nem pula nenhuma;
4. I1 só passa com a revisão COMPLETA e o mínimo de válidas.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import assistente as m  # noqa: E402


class ModeloFalso:
    """Responde conforme uma marca no resumo; guarda o que recebeu para GERAR (o crítico
    aprova tudo e não entra na conta)."""

    def __init__(self):
        self.recebidos = []

    def gerar(self, sistema, usuario, **kw):
        if sistema == m.SISTEMA_CRITICO:
            return '{"autonoma": true, "responde": true, "sustentada": true}'
        self.recebidos.append(usuario)
        if "SEMFATO" in usuario:
            return "NENHUMA"
        if "QUEBRADO" in usuario:
            return "desculpe, não sei"
        if "COPIA" in usuario:
            return '{"pergunta": "Graphene superconductivity twisted bilayer?", "gabarito": "x"}'
        return ('```json\n{"pergunta": "Qual fase aparece em redes de átomos frios?", '
                '"gabarito": "Uma fase isolante."}\n```')


def _dados(tmp_path, marcas):
    n = len(marcas)
    ids = [f"2{i:03d}.{i:05d}" for i in range(n)]
    datas = ["2024-01-01"] * (n - 3) + ["2025-07-01"] * 3
    pl.DataFrame({"arxiv_id": ids, "created": datas,
                  "title": ["Graphene superconductivity in twisted bilayer"] * n,
                  "abstract": [f"Resumo {mk} do artigo {i}." for i, mk in enumerate(marcas)]}
                 ).write_parquet(tmp_path / "spine.parquet")
    pl.DataFrame({"arxiv_id": ids + ids, "arxiv_citado": ["x"] * (2 * n),
                  "ancora": ["a"] * (2 * n), "positivo": ["p"] * (2 * n)}
                 ).write_parquet(tmp_path / "val.parquet")
    return tmp_path / "val.parquet", tmp_path / "spine.parquet"


def test_estratos_DISJUNTOS_e_ordem_reprodutivel(tmp_path):
    val, spine = _dados(tmp_path, ["ok"] * 12)
    a, b = m.sortear(val, spine), m.sortear(val, spine)
    assert a == b
    assert len(a["primario"]) == 9 and len(a["pos_corte"]) == 3
    assert not set(a["primario"]) & set(a["pos_corte"])
    assert m.sortear(val, spine, semente=1) != a


def test_cotas_de_revisao_na_proporcao_dos_estratos():
    assert m.cotas_de_revisao() == {"primario": 31, "pos_corte": 9}


def test_ler_pergunta_aceita_json_cercado_e_recusa_o_resto():
    assert m.ler_pergunta('```json\n{"pergunta": "P?", "gabarito": "G"}\n```') == ("P?", "G")
    assert m.ler_pergunta("NENHUMA") is None
    assert m.ler_pergunta("NENHUMA.") is None
    assert m.ler_pergunta('{"pergunta": "P?"}') is None
    assert m.ler_pergunta("texto sem json") is None


def test_sobreposicao_ve_termos_tecnicos_e_ignora_palavras_vazias():
    titulo = "Superconductivity in twisted bilayer graphene"
    assert m.sobreposicao_com_titulo("Qual é a supercondutividade em grafeno?", titulo) == 0.0
    assert m.sobreposicao_com_titulo("Twisted bilayer graphene: qual fase?", titulo) == 0.75


def test_gerar_registra_TODA_tentativa_e_nao_mostra_o_titulo(tmp_path):
    marcas = ["SEMFATO", "ok", "QUEBRADO", "COPIA", "ok", "ok", "ok", "ok", "ok", "ok",
              "ok", "ok"]
    val, spine = _dados(tmp_path, marcas)
    ordem = m.sortear(val, spine)
    modelo = ModeloFalso()
    tent = m.gerar(modelo, ordem, spine, {"primario": 4, "pos_corte": 2}, tmp_path / "c.jsonl")

    assert all("Graphene" not in r for r in modelo.recebidos), "o título vazou para o modelo"
    aceitas = {e: sum(t.situacao == "aceita" for t in tent if t.estrato == e)
               for e in ("primario", "pos_corte")}
    assert aceitas == {"primario": 4, "pos_corte": 2}
    linhas = (tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(linhas) == len(tent) == len(modelo.recebidos)
    por_id = {t.arxiv_id: t.situacao for t in tent}
    ids = sorted(pl.read_parquet(spine)["arxiv_id"].to_list())
    for i, esperado in ((0, "nenhuma"), (2, "formato"), (3, "copia_titulo")):
        if ids[i] in por_id:
            assert por_id[ids[i]] == esperado


def test_a_RETOMADA_nao_refaz_e_estende_a_cota(tmp_path):
    val, spine = _dados(tmp_path, ["ok"] * 12)
    ordem = m.sortear(val, spine)
    cache = tmp_path / "c.jsonl"
    m.gerar(ModeloFalso(), ordem, spine, {"primario": 2, "pos_corte": 1}, cache)
    segundo = ModeloFalso()
    tent = m.gerar(segundo, ordem, spine, {"primario": 5, "pos_corte": 1}, cache)
    assert len(segundo.recebidos) == 3, "refez tentativas que já estavam no cache"
    assert [t.arxiv_id for t in tent if t.estrato == "primario"] == ordem["primario"][:5]


def _folha(n_validas, n=40, passo_ms=20_000):
    vs = [{"indice": k, "veredicto": "valida" if k < n_validas else "confusa",
           "ms": 1_000_000 + k * passo_ms} for k in range(n)]
    return {"n_amostra": 40, "veredictos": vs}


def test_I1_so_passa_COMPLETA_e_com_o_minimo():
    assert m.apurar_i1(_folha(32))["passa"]
    r = m.apurar_i1(_folha(31))
    assert not r["passa"] and r["invalidas_por_motivo"] == {"confusa": 9}
    assert not m.apurar_i1(_folha(32, n=35))["passa"], "passou com revisão incompleta"


def test_I1_RECUSA_folha_julgada_rapido_demais_ou_sem_horarios():
    """O acidente de 2026-09-24: um script marcou as 40 como válidas em um segundo."""
    import pytest

    with pytest.raises(ValueError, match="clique automático"):
        m.apurar_i1(_folha(40, passo_ms=25))
    sem_horario = _folha(40)
    for v in sem_horario["veredictos"]:
        v.pop("ms")
    with pytest.raises(ValueError, match="versão antiga"):
        m.apurar_i1(sem_horario)
    humana = _folha(40)   # umas poucas teclas rápidas acontecem, e passam
    for k in (5, 6, 20):
        humana["veredictos"][k]["ms"] = humana["veredictos"][k - 1]["ms"] + 300
    assert m.apurar_i1(humana)["passa"]


def test_o_formal_EXCLUI_os_artigos_que_o_desenvolvimento_viu(tmp_path, monkeypatch):
    val, spine = _dados(tmp_path, ["ok"] * 12)
    monkeypatch.setattr(m, "N_DEV_PRIMARIO", 4)
    monkeypatch.setattr(m, "N_DEV_POS_CORTE", 1)
    dev = m.sortear(val, spine, m.SEMENTE_DEV)
    formal = m.ordem_formal(val, spine)
    vistos = set(dev["primario"][:4]) | set(dev["pos_corte"][:1])
    assert not vistos & (set(formal["primario"]) | set(formal["pos_corte"]))
    assert len(formal["primario"]) == 5 and len(formal["pos_corte"]) == 2


def test_pergunta_que_DEPENDE_do_artigo_cai(tmp_path):
    class Dependente:
        def gerar(self, sistema, usuario, **kw):
            return '{"pergunta": "Qual é o cenário previsto pelo modelo?", "gabarito": "x"}'

    t = m.tentar(Dependente(), "primario", 0, "1", "Titulo qualquer", "resumo")
    assert t.situacao == "depende_do_artigo"
    for ruim in ("Qual o sistema proposto no estudo?", "Segundo o enfoque, qual é a natureza?",
                 "Qual material foi estudado?", "O que os autores concluem?",
                 "Qual cenário é previsto pelo modelo?"):
        assert m._DEPENDE_DO_ARTIGO.search(ruim), ruim
    for boa in ("Em filmes finos de hélio, o que acontece com a temperatura de rigidez?",
                "Qual fase é prevista pelo modelo de Hubbard em meio preenchimento?",
                "Segundo o formalismo de Keldysh, qual é a corrente estacionária?"):
        assert not m._DEPENDE_DO_ARTIGO.search(boa), boa


def test_o_CRITICO_derruba_o_item_e_diz_por_que(tmp_path):
    class ComCritico:
        def gerar(self, sistema, usuario, **kw):
            if sistema == m.SISTEMA_CRITICO:
                return 'pensei. {"autonoma": true, "responde": false, "sustentada": true}'
            return '{"pergunta": "Qual é a massa do bóson W em redes?", "gabarito": "É grande."}'

    t = m.tentar(ComCritico(), "primario", 0, "1", "Titulo", "resumo")
    assert t.situacao == "reprovada_critico" and t.critica == "responde"


def test_importar_so_aceita_a_CONTINUACAO_da_ordem_e_aplica_as_guardas(tmp_path):
    import pytest

    val, spine = _dados(tmp_path, ["ok"] * 12)
    ordem = m.sortear(val, spine)
    cache = tmp_path / "c.jsonl"
    prim = ordem["primario"]
    lote = [{"estrato": "primario", "arxiv_id": prim[0], "nenhuma": True},
            {"estrato": "primario", "arxiv_id": prim[1],
             "pergunta": "Qual fase aparece em redes ópticas de átomos frios?", "gabarito": "g"},
            {"estrato": "primario", "arxiv_id": prim[2],
             "pergunta": "Qual o sistema proposto no estudo?", "gabarito": "g"}]
    novas = m.importar(lote, ordem, spine, cache)
    assert [t.situacao for t in novas] == ["nenhuma", "aceita", "depende_do_artigo"]
    with pytest.raises(ValueError, match="não continua"):   # pulou prim[3]
        m.importar([{"estrato": "primario", "arxiv_id": prim[4], "nenhuma": True}],
                   ordem, spine, cache)
    with pytest.raises(RuntimeError, match="escreva mais"):
        m.aceitas_em_ordem(ordem, m.ler_cache(cache), {"primario": 2})
    assert [t.arxiv_id for t in m.aceitas_em_ordem(ordem, m.ler_cache(cache),
                                                   {"primario": 1})] == [prim[1]]
    assert m.pendentes(ordem, m.ler_cache(cache), "primario", 2)[0]["arxiv_id"] == prim[3]


def test_critico_fora_do_formato_reprova_os_tres():
    class Mudo:
        def gerar(self, sistema, usuario, **kw):
            return "não sei"

    assert m.criticar(Mudo(), "r", "p", "g") == ["autonoma", "responde", "sustentada"]


def test_copia_de_EXEMPLO_do_prompt_cai():
    class Copiador:
        def gerar(self, sistema, usuario, **kw):
            return ('{"pergunta": "Em filmes finos de hélio adsorvido, o que acontece com a '
                    'temperatura em que surge a rigidez quando a cobertura se aproxima da '
                    'cobertura crítica?", "gabarito": "Ela vai a zero kelvin."}')

    assert m.tentar(Copiador(), "pos_corte", 0, "1", "Co NbS2", "r").situacao == "copia_exemplo"
