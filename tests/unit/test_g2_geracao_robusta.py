"""O cliente de geração do G2, sem servidor: a guarda de repetição, a leitura do stream,
a semente por amostra e a amostragem com tudo desligado além de T e top-p."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2_geracao_robusta as ger  # noqa: E402


def _linha(**d) -> bytes:
    return ("data: " + json.dumps(d) + "\n").encode()


def test_guarda_corta_na_quarta_repeticao_da_janela_em_laco():
    g = ger.GuardaDeRepeticao(janela=3, vezes=4)
    assert not g.alimentar([1, 2, 3, 9])
    assert not g.alimentar([1, 2, 3, 8, 1, 2, 3])   # (1,2,3) três vezes, a cada 4 fichas
    assert g.alimentar([7, 1, 2, 3])                 # a quarta, no mesmo intervalo, corta


def test_guarda_nao_corta_expressao_retomada_ao_longo_da_derivacao():
    """A expressão longa derivada, simplificada, retomada e posta na caixa: 4 vezes, mas em
    intervalos desiguais. A guarda antiga cortava no meio da caixa final (iteração 1)."""
    g = ger.GuardaDeRepeticao(janela=3, vezes=4)
    novas = iter(range(100, 10_000))              # texto entre as retomadas, sem repetição
    texto = []
    for n in (10, 3, 25, 6):
        texto += [1, 2, 3] + [next(novas) for _ in range(n)]
    assert not g.alimentar(texto + [1, 2, 3])


def test_guarda_corta_na_oitava_vez_mesmo_sem_intervalos_iguais():
    g = ger.GuardaDeRepeticao(janela=3, vezes=4, teto=8)
    novas = iter(range(100, 10_000))
    texto = []
    for k in range(7):
        texto += [1, 2, 3] + [next(novas) for _ in range(1 + 2 * k)]
    assert not g.alimentar(texto)
    assert g.alimentar([1, 2, 3])


def test_guarda_nao_corta_texto_que_so_parece():
    g = ger.GuardaDeRepeticao(janela=32, vezes=4)
    assert not g.alimentar(list(range(5000)))


def test_ler_fluxo_parada_normal_e_limite():
    linhas = [_linha(content="The answer ", tokens=[1, 2]), b"\n",
              _linha(content="is 5.", tokens=[3], stop=False),
              _linha(content="", tokens=[], stop=True, stop_type="word", tokens_predicted=3)]
    assert ger.ler_fluxo(linhas, ger.GuardaDeRepeticao()) == ("The answer is 5.", 3, "word")
    fim = [_linha(content="x", tokens=[1], stop=True, stop_type="limit", tokens_predicted=4096)]
    texto, n, parada = ger.ler_fluxo(fim, ger.GuardaDeRepeticao())
    assert (texto, n, parada) == ("x", 4096, "limit")


def test_ler_fluxo_corta_por_repeticao_e_conta_como_truncada():
    linhas = [_linha(content="ab", tokens=[1, 2, 3]) for _ in range(10)]
    texto, _, parada = ger.ler_fluxo(linhas, ger.GuardaDeRepeticao(janela=3, vezes=4))
    assert parada == "repeticao"
    a = ger.Amostra("b", "i", 0, 1, texto, 12, parada, 0.1)
    assert a.truncada
    assert not ger.Amostra("b", "i", 0, 1, "", 1, "word", 0.1).truncada


def test_semente_e_estavel_e_distinta_por_amostra():
    assert ger.semente("ugphysics", "X/1", 0) == ger.semente("ugphysics", "X/1", 0)
    assert len({ger.semente("ugphysics", "X/1", j) for j in range(16)}) == 16
    assert 0 <= ger.semente("a", "b", 3) < 2 ** 32


def test_amostragem_desliga_os_padroes_do_servidor():
    p = ger.amostragem()
    assert p["temperature"] == 0.6 and p["top_p"] == 0.95
    assert p["top_k"] == 0 and p["min_p"] == 0.0 and p["repeat_penalty"] == 1.0
    assert p["dry_multiplier"] == 0.0 and p["xtc_probability"] == 0.0 and p["mirostat"] == 0


def test_flags_do_servidor_dao_10240_por_slot():
    s = ger.ServidorG2(Path("x.gguf"))
    f = s.flags()
    assert f[f.index("-c") + 1] == "40960" and f[f.index("-np") + 1] == "4"
    assert "--no-kv-unified" in f and f[f.index("-ctk") + 1] == "q8_0"
    assert s.porta != 8080


def test_retomada_recusa_outra_configuracao_sem_mudar_o_arquivo(tmp_path, monkeypatch):
    saida = tmp_path / "amostras.jsonl"
    antiga = {"bench": "ugphysics", "item_id": "x", "j": 0,
              "hash_config": "OLD", "parada": "eos"}
    saida.write_text(json.dumps(antiga) + "\n", encoding="utf-8")
    original = saida.read_bytes()
    chamadas = []
    monkeypatch.setattr(ger, "gerar_uma", lambda *a: chamadas.append(a))
    with pytest.raises(ValueError, match="configuração"):
        list(ger.gerar_lote("unused", [("ugphysics", "x", 0, "prompt")], [], {},
                            saida, "NEW"))
    assert chamadas == []
    assert saida.read_bytes() == original


@pytest.mark.parametrize("mudanca", ["prompt", "parada", "amostragem", "limite", "artefato"])
def test_retomada_recusa_mudanca_no_mesmo_hash_do_motor(tmp_path, monkeypatch, mudanca):
    saida = tmp_path / "amostras.jsonl"
    monkeypatch.setattr(ger, "gerar_uma", lambda *a: ("resposta", 3, "eos"))
    pedidos = [("ugphysics", "x", 0, "prompt")]
    paradas, pedido, limite, artefatos = ["STOP"], {"temperature": 0.6}, 4096, {"codigo": "a"}
    list(ger.gerar_lote("unused", pedidos, paradas, pedido, saida, "motor",
                        fichas_max=limite, artefatos=artefatos))
    original = saida.read_bytes()
    if mudanca == "prompt":
        pedidos = [("ugphysics", "x", 0, "outro prompt")]
    elif mudanca == "parada":
        paradas = ["OUTRA"]
    elif mudanca == "amostragem":
        pedido = {"temperature": 0.7}
    elif mudanca == "limite":
        limite = 100
    else:
        artefatos = {"codigo": "b"}
    with pytest.raises(ValueError, match="execução"):
        list(ger.gerar_lote("unused", pedidos, paradas, pedido, saida, "motor",
                            fichas_max=limite, artefatos=artefatos))
    assert saida.read_bytes() == original


def test_retomada_repete_so_o_erro_e_o_agregado_tem_uma_amostra(tmp_path, monkeypatch):
    saida = tmp_path / "amostras.jsonl"
    pedidos = [("ugphysics", "x", 0, "prompt x"), ("ugphysics", "y", 0, "prompt y")]
    chamadas = []

    def gerar(_url, prompt, *args):
        chamadas.append(prompt)
        if prompt == "prompt x" and chamadas.count(prompt) == 1:
            raise OSError("interrupção")
        return prompt, 2, "eos"

    monkeypatch.setattr(ger, "gerar_uma", gerar)
    primeira = list(ger.gerar_lote("unused", pedidos, [], {}, saida, "motor", paralelos=1))
    segunda = list(ger.gerar_lote("unused", pedidos, [], {}, saida, "motor", paralelos=1))
    terceira = list(ger.gerar_lote("unused", pedidos, [], {}, saida, "motor", paralelos=1))
    assert len(primeira) == 2 and len(segunda) == 1 and terceira == []
    assert chamadas == ["prompt x", "prompt y", "prompt x"]
    he = ger.hash_da_execucao("motor", pedidos, [], {})
    amostras = ger.carregar_amostras(saida, "motor", he)
    assert len(amostras) == 2
    assert all(a["parada"] == "eos" for a in amostras)
    # O histórico bruto continua íntegro, inclusive a tentativa que falhou.
    assert len(saida.read_text(encoding="utf-8").splitlines()) == 3


def test_hash_da_execucao_independe_da_ordem_dos_pedidos():
    pedidos = [("b", "x", 0, "p"), ("b", "y", 1, "q")]
    assert ger.hash_da_execucao("h", pedidos, [], {}) == ger.hash_da_execucao(
        "h", list(reversed(pedidos)), [], {})


def test_pedidos_duplicados_sao_recusados_antes_de_gerar(tmp_path, monkeypatch):
    monkeypatch.setattr(ger, "gerar_uma", lambda *a: pytest.fail("não deve gerar"))
    with pytest.raises(ValueError, match="duplicad"):
        list(ger.gerar_lote("unused", [("b", "x", 0, "p")] * 2, [], {},
                            tmp_path / "amostras.jsonl", "h"))


def test_linha_interrompida_e_recusada_sem_apagar_a_evidencia(tmp_path):
    saida = tmp_path / "amostras.jsonl"
    saida.write_text('{"bench":', encoding="utf-8")
    with pytest.raises(ValueError, match="linha 1"):
        ger.carregar_amostras(saida, "h", "e")
    assert saida.read_text(encoding="utf-8") == '{"bench":'


def test_texto_adulterado_e_recusado(tmp_path, monkeypatch):
    saida = tmp_path / "amostras.jsonl"
    pedidos = [("b", "x", 0, "p")]
    monkeypatch.setattr(ger, "gerar_uma", lambda *a: ("original", 1, "eos"))
    list(ger.gerar_lote("unused", pedidos, [], {}, saida, "h"))
    linha = json.loads(saida.read_text(encoding="utf-8"))
    linha["texto"] = "alterado"
    saida.write_text(json.dumps(linha) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="texto"):
        ger.carregar_amostras(saida, "h", ger.hash_da_execucao("h", pedidos, [], {}))
