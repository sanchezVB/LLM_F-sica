"""A folha de revisão só vale se o julgamento humano não for contaminado.

Um julgamento de 400 itens é caro em atenção, e é gasto uma vez. Se a folha ancorar a
resposta mostrando o escore do classificador antes do clique, o número que sai não
mede a precisão do filtro — mede a concordância da pessoa com o modelo, que é alta
por construção e não diz nada.

Estes testes fixam o que faz o número valer:

  1. o escore fica escondido até depois do julgamento;
  2. a assinatura da amostra entra na chave do armazenamento, para julgamentos não
     migrarem para os documentos errados;
  3. a página não faz rede nenhuma — os dados vão embutidos, e o arquivo fica no HD.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts"))

import folha_de_revisao  # noqa: E402

PAGINA = folha_de_revisao.PAGINA

AMOSTRA_FALSA = [
    {"score": "0.912", "url": "s2ag/train", "inicio": "Sobre o efeito Hall quântico."},
    {"score": "0.998", "url": "s2orc/train", "inicio": "Uma revisão de mercado."},
]


def _gerar(tmp_path: Path, docs=None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    amostra = tmp_path / "_amostra_para_revisao.json"
    amostra.write_text(json.dumps(docs if docs is not None else AMOSTRA_FALSA,
                                  ensure_ascii=False), encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(RAIZ / "scripts/folha_de_revisao.py"),
         "--amostra", str(amostra)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**__import__("os").environ, "PYTHONUTF8": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    saida = amostra.parent / "revisao.html"
    assert saida.exists()
    return saida


def test_o_escore_nao_e_renderizado_antes_do_julgamento():
    """A asserção central deste arquivo.

    O cartão do documento em julgamento não pode conter o escore. Ele existe nos
    dados embutidos — o JavaScript precisa dele para revelar depois —, mas o trecho
    que monta o cartão não pode tocá-lo.
    """
    cartao = PAGINA.split('document.getElementById("area").innerHTML =')[1]
    cartao = cartao.split("document.querySelector")[0]
    assert "d.score" not in cartao, (
        "o escore voltou para o cartão do documento em julgamento — isto ancora a "
        "resposta e o número medido deixa de significar precisão do filtro")


def test_o_escore_e_revelado_depois_do_julgamento():
    """Esconder para sempre também seria ruim: a pessoa não veria onde discordou.

    A revelação é da decisão ANTERIOR, e o bug que este teste trava é real: a
    primeira versão lia `v[String(i)]` depois de `i++`, então era sempre `undefined`
    e o escore nunca aparecia.
    """
    assert 'document.getElementById("ultima")' in PAGINA
    revelacao = PAGINA.split('document.getElementById("ultima").innerHTML =')[1]
    revelacao = revelacao.split(";")[0]
    assert "DOCS[ant].score" in revelacao
    assert "i-1" in PAGINA, "a revelação tem de ser do documento anterior, não do atual"


def test_a_assinatura_da_amostra_entra_na_chave_de_armazenamento(tmp_path):
    """Duas amostras diferentes não podem compartilhar os julgamentos.

    Sem isto, regenerar a folha sobre outra amostra herdaria 400 veredictos colados
    nos documentos errados — e nada avisaria, porque a contagem continuaria certa.
    """
    a = _gerar(tmp_path / "a", AMOSTRA_FALSA)
    b = _gerar(tmp_path / "b", list(reversed(AMOSTRA_FALSA)))

    def chave(p: Path) -> str:
        linha = next(x for x in p.read_text(encoding="utf-8").splitlines()
                     if "const CHAVE" in x)
        return linha

    assert chave(a) != chave(b), (
        "a chave do localStorage não distingue as duas amostras")
    assert "revisao_pes2o_" in chave(a)


def test_a_pagina_nao_faz_rede(tmp_path):
    """Um `fetch` do JSON não funcionaria em `file://` e, pior, tentaria sair.

    Os dados do usuário ficam no HD. A folha é um arquivo único.
    """
    html = _gerar(tmp_path).read_text(encoding="utf-8")
    for proibido in ("fetch(", "XMLHttpRequest", "<script src", "<link ", "@import",
                     "https://", "http://"):
        assert proibido not in html, f"a página tenta rede: {proibido!r}"


def test_os_dados_vao_embutidos_e_completos(tmp_path):
    html = _gerar(tmp_path).read_text(encoding="utf-8")
    assert "__DADOS__" not in html, "o marcador não foi substituído"
    assert "__ASSINATURA__" not in html
    assert "efeito Hall" in html and "revisão de mercado" in html


def test_amostra_sem_as_chaves_certas_quebra_alto(tmp_path):
    """Gerar uma folha que não mostra texto custaria a sessão de julgamento inteira
    para ser descoberto."""
    amostra = tmp_path / "_amostra_para_revisao.json"
    amostra.write_text(json.dumps([{"score": "0.9"}]), encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(RAIZ / "scripts/folha_de_revisao.py"),
         "--amostra", str(amostra)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**__import__("os").environ, "PYTHONUTF8": "1"})
    assert r.returncode != 0
    assert "url" in (r.stdout + r.stderr) and "inicio" in (r.stdout + r.stderr)


def test_amostra_ausente_quebra_alto(tmp_path):
    r = subprocess.run(
        [sys.executable, str(RAIZ / "scripts/folha_de_revisao.py"),
         "--amostra", str(tmp_path / "nao_existe.json")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**__import__("os").environ, "PYTHONUTF8": "1"})
    assert r.returncode != 0
    assert "não existe" in (r.stdout + r.stderr)


def test_o_alvo_e_pre_comprometido_e_o_vies_de_parada_esta_dito():
    """Mostrar o intervalo ao vivo convida a parar quando ele fica bonito.

    O convite não pode ser retirado — o intervalo é útil —, então o aviso tem de
    estar na página, não só na docstring que o usuário não lê.
    """
    assert "parada opcional" in PAGINA or "vies de parada" in PAGINA
    assert "alvo_pre_comprometido: 200" in PAGINA


# ─── a apuração ──────────────────────────────────────────────────────────────


def _apurar(tmp_path: Path, veredictos: list[dict]) -> dict:
    entrada = tmp_path / "v.json"
    entrada.write_text(json.dumps({"assinatura_amostra": "abc",
                                   "n_amostra": len(veredictos),
                                   "alvo_pre_comprometido": 200,
                                   "veredictos": veredictos}), encoding="utf-8")
    saida = tmp_path / "apurado.json"
    r = subprocess.run(
        [sys.executable, str(RAIZ / "scripts/apurar_revisao.py"),
         "--veredictos", str(entrada), "--out", str(saida)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=RAIZ,
        env={**__import__("os").environ, "PYTHONUTF8": "1",
             "PYTHONPATH": str(RAIZ / "src")})
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(saida.read_text(encoding="utf-8"))


def test_as_duvidas_saem_do_denominador_e_sao_reportadas(tmp_path):
    """Contá-las como Física baixaria a taxa; como não-Física, subiria.

    Tirá-las mede a taxa entre os decidíveis — e a contagem tem de aparecer, porque
    muitas dúvidas tornam a medição frágil e isso precisa ser visível.
    """
    v = ([{"veredicto": "fisica", "score": "0.95", "url": "s"}] * 9
         + [{"veredicto": "nao", "score": "0.91", "url": "s"}]
         + [{"veredicto": "duvida", "score": "0.99", "url": "s"}] * 5)
    for i, x in enumerate(v):
        x["indice"] = i
    d = _apurar(tmp_path, v)
    assert d["n_decidiveis"] == 10
    assert d["duvidas"] == 5
    assert d["taxa"] == pytest.approx(0.1)


def test_o_veredito_diz_quando_o_limiar_nao_transfere(tmp_path):
    """Se a taxa em texto pleno estourar o teto medido em resumos, o limiar 0,9 foi
    calibrado no lugar errado — e a saída tem de dizer isso, não só imprimir o número.
    """
    v = [{"indice": i, "veredicto": "nao" if i < 100 else "fisica",
          "score": "0.95", "url": "s"} for i in range(200)]
    d = _apurar(tmp_path, v)
    assert d["taxa"] == pytest.approx(0.5)
    assert "NÃO transfere" in d["veredito"]

    v2 = [{"indice": i, "veredicto": "nao" if i < 4 else "fisica",
           "score": "0.95", "url": "s"} for i in range(200)]
    d2 = _apurar(tmp_path, v2)
    assert "TRANSFERE" in d2["veredito"]


def test_a_apuracao_estratifica_por_faixa_de_escore(tmp_path):
    """Se os falsos positivos se concentram perto de 0,9, subir o limiar resolve
    barato — e essa é a informação que decide o que fazer com o resultado."""
    v = ([{"indice": i, "veredicto": "nao", "score": "0.91", "url": "s"}
          for i in range(10)]
         + [{"indice": 10 + i, "veredicto": "fisica", "score": "0.999", "url": "s"}
            for i in range(90)])
    d = _apurar(tmp_path, v)
    faixas = d["por_faixa_de_escore"]
    assert faixas["0,90–0,95"]["ruins"] == 10
    assert faixas["0,99–1,00"]["ruins"] == 0


def test_veredictos_vazios_quebram_alto(tmp_path):
    entrada = tmp_path / "v.json"
    entrada.write_text(json.dumps({"veredictos": []}), encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(RAIZ / "scripts/apurar_revisao.py"),
         "--veredictos", str(entrada), "--out", str(tmp_path / "o.json")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=RAIZ,
        env={**__import__("os").environ, "PYTHONUTF8": "1",
             "PYTHONPATH": str(RAIZ / "src")})
    assert r.returncode != 0


def test_a_apuracao_diz_que_mede_precisao_e_nao_recall(tmp_path):
    """Confundir as duas é o erro mais comum ao ler um número de filtro.

    Esta amostra é dos ACEITOS; quanta Física o filtro jogou fora exige uma amostra
    dos rejeitados, e é outra medição.
    """
    v = [{"indice": 0, "veredicto": "fisica", "score": "0.95", "url": "s"}]
    d = _apurar(tmp_path, v)
    assert "PRECISÃO" in d["nota"] and "recall" in d["nota"]
