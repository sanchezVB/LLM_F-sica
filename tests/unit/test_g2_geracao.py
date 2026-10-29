"""O cliente de geração do G2, sem servidor: a guarda de repetição, a leitura do stream,
a semente por amostra e a amostragem com tudo desligado além de T e top-p."""
from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2_geracao as ger  # noqa: E402


def _linha(**d) -> bytes:
    return ("data: " + json.dumps(d) + "\n").encode()


def test_jsonl_com_separador_cru_nao_parte_a_linha(tmp_path):
    """Emenda 1: uma geração com U+0085 cru travou a retomada da medida da base, porque o
    leitor usava splitlines. O gravador escapa e o leitor separa só pelo "\\n", lendo também
    o arquivo antigo, de linha crua."""
    reg = {"bench": "b", "item_id": "i", "j": 0, "parada": "word",
           "texto": "a\u0085b c d", "hash_config": "h"}
    linha = ger.linha_jsonl(reg)
    assert len(linha.splitlines()) == 1 and json.loads(linha) == reg
    arq = tmp_path / "g.jsonl"
    cru = json.dumps(reg | {"j": 1}, ensure_ascii=False)          # como o gravador antigo
    arq.write_text(linha + "\n" + cru + "\n", encoding="utf-8")
    lidos = list(ger.ler_jsonl(arq))
    assert [d["j"] for d in lidos] == [0, 1] and all(d["texto"] == reg["texto"] for d in lidos)
    assert ger.ja_gerados(arq) == {("b", "i", 0), ("b", "i", 1)}


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


def test_gpu_e_executavel_do_ambiente():
    """Emenda 3: no Kaggle, o executor da 8B põe o llama-server e o rótulo da GPU no
    ambiente; sem as variáveis, vale a RX 7600 local (o hash da 1,7B não muda). O
    executável sem o rótulo é recusado: o hash levaria o rótulo errado."""
    import os
    import subprocess

    codigo = ("import sys; sys.path.insert(0, 'src'); from phifm.eval import g2_geracao as g; "
              "print(g.GPU); print(g.EXECUTAVEL.name)")
    limpo = {k: v for k, v in os.environ.items() if not k.startswith("G2_")}

    def rodar(**extra):
        return subprocess.run([sys.executable, "-X", "utf8", "-c", codigo], cwd=RAIZ, capture_output=True,
                              text=True, encoding="utf-8", env=limpo | extra)

    r = rodar()
    assert r.stdout.split("\n")[:2] == ["RX 7600 (Vulkan)", "llama-server.exe"]
    r = rodar(G2_LLAMA_SERVER="/kaggle/tmp/llama-server", G2_GPU="2×T4 (CUDA)")
    assert r.stdout.split("\n")[:2] == ["2×T4 (CUDA)", "llama-server"]
    r = rodar(G2_LLAMA_SERVER="/kaggle/tmp/llama-server")
    assert r.returncode != 0 and "G2_GPU" in r.stderr
