"""A preparação do corpus do ΦEnc, e o viés que ela quase gravou.

A primeira execução preparou 2 B tokens das **8 primeiras** de 44 partes. Medido
depois:

    partes 0-7  (usadas)      math 32,4%   display 18,8%
    partes 8-43 (NÃO usadas)  math 42,7%   display 29,5%

As partes não são intercambiáveis — as que ficaram de fora têm 57% mais equação em
display. Pegar as primeiras é `head()` no nível de arquivo, e é a quarta vez que
este repositório tropeça nisso.

O viés ia **contra** a hipótese do DOC-07 §2.3 (menos equação enfraquece o
tratamento), o que é o lado seguro — e um viés que ajuda não deixa de ser viés.

Estes testes fixam as três coisas que o conserto precisa ter para não virar outro
problema: sorteio determinístico, a lista no manifesto, e recusa de retomar com
parâmetros diferentes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

pytest.importorskip("tokenizers", reason="requer a venv de treino (.venv-treino)")
pl = pytest.importorskip("polars")

from phifm.training.pretrain.dados import NOME_MANIFESTO  # noqa: E402

SCRIPT = RAIZ / "scripts/preparar_dados_phienc.py"
TOKENIZER = RAIZ / "data/processed/tokenizer/variante_A.json"


def _corpus(tmp_path: Path, n_partes: int = 6) -> Path:
    """Partes com conteúdo DISTINGUÍVEL, para o sorteio ser observável."""
    raiz = tmp_path / "corpus"
    raiz.mkdir(parents=True, exist_ok=True)
    for i in range(n_partes):
        # Cada parte tem um marcador próprio no texto e uma equação em display,
        # para a marcação ter o que marcar.
        textos = [f"parte {i} documento {j}. "
                  + r"\begin{equation} E_{" + str(i) + r"} = m c^2 \end{equation}"
                  + " prosa depois com bastante texto para o documento nao ser "
                  "degenerado e o tokenizer ter o que fazer. " * 3
                  for j in range(12)]
        pl.DataFrame({"arxiv_id": [f"{i}.{j}" for j in range(12)],
                      "texto": textos}).write_parquet(raiz / f"part-{i:05d}.parquet")
    return raiz


def _rodar(corpus: Path, out: Path, *extra: str):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--corpus", str(corpus), "--out", str(out),
         "--tokenizer", str(TOKENIZER), *extra],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=RAIZ, env={**os.environ, "PYTHONUTF8": "1",
                       "PYTHONPATH": str(RAIZ / "src")})


@pytest.fixture(autouse=True)
def _exige_tokenizer():
    if not TOKENIZER.exists():
        pytest.skip("o tokenizer treinado é gitignored; ver bakeoff_tokenizer.py")


def test_as_partes_sao_sorteadas_e_nao_as_primeiras(tmp_path):
    """⚠️ A asserção central. Ver a docstring do módulo para o que custou.

    Com um teto que só cabe algumas partes, ler na ordem do disco significa ler
    sempre as mesmas — e se o corpus tiver qualquer ordenação, o subconjunto não
    representa o conjunto.
    """
    import random

    corpus = _corpus(tmp_path, n_partes=8)
    r = _rodar(corpus, tmp_path / "s", "--max-tokens", "1500", "--recomecar")
    assert r.returncode == 0, r.stdout + r.stderr
    man = json.loads((tmp_path / "s" / NOME_MANIFESTO).read_text(encoding="utf-8"))
    usadas = man["partes_usadas"]
    assert 0 < len(usadas) < 8, f"o teto não limitou nada: {usadas}"

    # ⚠️ Afirmar o sorteio DIRETAMENTE, e nao por "nao e a ordem do disco".
    # A primeira versao deste teste comparava com as primeiras partes, e com um teto
    # que so cabe UMA parte isso era cara-ou-coroa: o sorteio pos a part-00000 em
    # primeiro e o teste reprovou um comportamento correto.
    esperado = [f"part-{i:05d}.parquet" for i in range(8)]
    random.Random(17).shuffle(esperado)
    assert usadas == esperado[:len(usadas)], (
        f"a ordem nao e a do sorteio com semente 17: {usadas} contra "
        f"{esperado[:len(usadas)]}")
    assert usadas != [f"part-{i:05d}.parquet" for i in range(len(usadas))], (
        "o sorteio com esta semente devolveu a ordem do disco; escolha outra "
        "semente para o teste, porque assim ele nao distingue nada")


def test_o_sorteio_e_deterministico_na_semente(tmp_path):
    """Sem determinismo a preparação deixa de ser reproduzível, e o `(semente,
    passo)` do DOC-08 §7.2 passa a apontar para tokens diferentes."""
    corpus = _corpus(tmp_path)
    saidas = []
    for i in range(2):
        r = _rodar(corpus, tmp_path / f"s{i}", "--max-tokens", "600",
                   "--recomecar", "--semente", "17")
        assert r.returncode == 0, r.stdout + r.stderr
        saidas.append(json.loads(
            (tmp_path / f"s{i}" / NOME_MANIFESTO).read_text(encoding="utf-8")))
    assert saidas[0]["partes_usadas"] == saidas[1]["partes_usadas"]
    assert saidas[0]["tokens"] == saidas[1]["tokens"]

    # E outra semente dá outra ordem.
    r = _rodar(corpus, tmp_path / "s9", "--max-tokens", "600", "--recomecar",
               "--semente", "99")
    outra = json.loads((tmp_path / "s9" / NOME_MANIFESTO).read_text(encoding="utf-8"))
    assert outra["partes_usadas"] != saidas[0]["partes_usadas"]


def test_o_manifesto_guarda_a_LISTA_e_nao_so_a_contagem(tmp_path):
    """⚠️ Com sorteio, saber que 8 partes entraram não permite reconstruir QUAIS.

    Sem a lista, uma execução não é reproduzível nem auditável: ninguém consegue
    dizer depois se um resultado veio de um subconjunto enviesado.
    """
    corpus = _corpus(tmp_path)
    _rodar(corpus, tmp_path / "s", "--max-tokens", "600", "--recomecar")
    man = json.loads((tmp_path / "s" / NOME_MANIFESTO).read_text(encoding="utf-8"))
    assert isinstance(man["partes_usadas"], list)
    assert len(man["partes_usadas"]) == man["partes_feitas"]
    assert man["semente_do_sorteio"] == 17
    assert man["em_ordem"] is False
    assert "SORTEADAS" in man["nota"] and "32,4%" in man["nota"]


def test_retomar_com_outra_semente_LEVANTA(tmp_path):
    """Retomar com outra semente leria partes diferentes achando que continua a
    mesma preparação — e o binário teria dois subconjuntos concatenados, com a
    contagem certa."""
    corpus = _corpus(tmp_path)
    out = tmp_path / "s"
    r = _rodar(corpus, out, "--max-tokens", "600", "--recomecar", "--semente", "17")
    assert r.returncode == 0, r.stdout + r.stderr
    r2 = _rodar(corpus, out, "--max-tokens", "1200", "--semente", "99")
    assert r2.returncode != 0
    assert "semente" in (r2.stdout + r2.stderr)


def test_retomar_com_outro_tokenizer_LEVANTA(tmp_path):
    """Concatenar tokens de dois vocabulários produz um corpus que nenhum modelo
    lê, com a contagem certa."""
    corpus = _corpus(tmp_path)
    out = tmp_path / "s"
    _rodar(corpus, out, "--max-tokens", "600", "--recomecar")
    prog = json.loads((out / "_progresso.json").read_text(encoding="utf-8"))
    prog["tokenizer_sha"] = "0" * 16
    (out / "_progresso.json").write_text(json.dumps(prog), encoding="utf-8")
    r = _rodar(corpus, out, "--max-tokens", "1200")
    assert r.returncode != 0
    assert "tokenizer" in (r.stdout + r.stderr)


def test_em_ordem_existe_para_reproduzir_e_diz_que_enviesa(tmp_path):
    """A rota antiga continua acessível — uma preparação anterior tem de poder ser
    reproduzida —, mas rotulada."""
    corpus = _corpus(tmp_path)
    r = _rodar(corpus, tmp_path / "s", "--max-tokens", "600", "--recomecar",
               "--em-ordem")
    assert r.returncode == 0, r.stdout + r.stderr
    man = json.loads((tmp_path / "s" / NOME_MANIFESTO).read_text(encoding="utf-8"))
    assert man["em_ordem"] is True
    n = len(man["partes_usadas"])
    assert man["partes_usadas"] == [f"part-{i:05d}.parquet" for i in range(n)]

    fonte = SCRIPT.read_text(encoding="utf-8")
    assert "enviesa" in fonte, "a bandeira perdeu o aviso de que enviesa"


def test_os_binarios_tem_o_mesmo_numero_de_tokens_e_marcas(tmp_path):
    """Um desencontro alinharia marcas com os tokens errados, e o mascaramento
    trataria prosa como equação."""
    out = tmp_path / "s"
    _rodar(_corpus(tmp_path), out, "--max-tokens", "600", "--recomecar")
    n_tok = (out / "tokens.u16.bin").stat().st_size // 2
    n_mar = (out / "marcas.u8.bin").stat().st_size
    assert n_tok == n_mar
    man = json.loads((out / NOME_MANIFESTO).read_text(encoding="utf-8"))
    assert man["tokens"] == n_tok
