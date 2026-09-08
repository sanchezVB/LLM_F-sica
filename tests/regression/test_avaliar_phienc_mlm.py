"""A passagem pelo modelo do MLM por região, e as recusas que a fazem valer.

A parte pura está em `phifm.eval.mlm_regiao` e é testada em
`tests/unit/test_mlm_regiao.py` — máscara uniforme, determinística em
`(semente, sequência)`, contagem separada por região.

Aqui o que se confere é o que só aparece com o modelo e o binário na mão:

  1. a fatia de avaliação tem de ser **disjunta do treino**, senão o número mede
     memorização com a cara de generalização;
  2. o tokenizer da fatia tem de ser o da variante — o §11.2 tem três tamanhos de
     vocabulário e ids da variante errada significam outra coisa;
  3. `certo` e `em_equacao` saem alinhados, ou o acerto vai para a região errada;
  4. o protocolo (contexto, fração, semente) vai gravado, para o comparador poder
     RECUSAR em vez de subtrair duas tarefas diferentes.

Precisa de torch, então pula na suíte rápida (roda na `.venv-treino`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

torch = pytest.importorskip("torch", reason="requer a venv de treino (.venv-treino)")
pytest.importorskip("transformers")

from tokenizers import Tokenizer, models  # noqa: E402

from phifm.models.encoder.config import ConfigEnc  # noqa: E402
from phifm.models.encoder.modelo import construir  # noqa: E402
from phifm.training.pretrain.dados import (  # noqa: E402
    BIT_MATH,
    NOME_MANIFESTO,
    NOME_MARCAS,
    NOME_TOKENS,
)
from phifm.training.pretrain.laco import NOME_ESTADO, NOME_METRICAS  # noqa: E402

VOCAB = 512
CONTEXTO_AVAL = 128
MINI = ConfigEnc(nome="mini-mlm", camadas=2, d_model=64, cabecas=4, ffn=96,
                 vocab=VOCAB, contexto=CONTEXTO_AVAL, janela_local=64)

EXPORTADOR = RAIZ / "scripts" / "exportar_phienc.py"
AVALIADOR = RAIZ / "scripts" / "avaliar_phienc_mlm.py"

SHA_TOK = "0" * 16


def _modelo(tmp: Path) -> Path:
    """Um ΦEnc minúsculo, exportado pelo script de verdade."""
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    vocab.update({f"t{i}": i for i in range(5, VOCAB)})
    caminho_tok = tmp / "tok.json"
    Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]")).save(
        str(caminho_tok))

    run = tmp / "run"
    run.mkdir(parents=True)
    modelo = construir(MINI, torch.device("cpu"), atencao="eager")
    torch.save({"passo": 3, "modelo": modelo.state_dict()}, run / NOME_ESTADO)
    (run / NOME_METRICAS).write_text(json.dumps({
        "modelo": MINI.como_dict(),
        "dados": {"tokenizer": str(caminho_tok)},
        "metricas": {"passo": 3, "perda": 6.2, "tokens": 100},
        "concluido": True,
    }, ensure_ascii=False), encoding="utf-8")

    para = tmp / "phienc-mini"
    r = subprocess.run(
        [sys.executable, str(EXPORTADOR), "--run", str(run), "--para", str(para)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(RAIZ))
    assert r.returncode == 0, r.stdout + r.stderr
    # O exportador grava o sha real do tokenizer; para o teste de casamento de
    # tokenizer o valor exato não importa, só que os dois lados batam.
    prov = para / "phienc_exportado.json"
    d = json.loads(prov.read_text(encoding="utf-8"))
    d["tokenizer_sha"] = SHA_TOK
    prov.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return para


def _fatia(tmp: Path, *, disjunto: bool = True, com_equacao: bool = True,
           sha_tok: str = SHA_TOK, n_seq: int = 6) -> Path:
    """Um binário de tokens/marcas com uma fração de equação conhecida."""
    d = tmp / "fatia"
    d.mkdir(parents=True, exist_ok=True)
    n = CONTEXTO_AVAL * n_seq
    rng = np.random.default_rng(7)
    # Todo id >= 5 para nada cair em posição proibida por ser especial.
    ids = rng.integers(5, VOCAB, size=n, dtype=np.uint16)
    marcas = np.zeros(n, dtype=np.uint8)
    if com_equacao:
        # 40% de cada sequência é equação, em bloco — como uma equação de verdade.
        for i in range(n_seq):
            a = i * CONTEXTO_AVAL
            marcas[a + 10:a + 10 + int(CONTEXTO_AVAL * 0.4)] |= BIT_MATH
    (d / NOME_TOKENS).write_bytes(ids.tobytes())
    (d / NOME_MARCAS).write_bytes(marcas.tobytes())
    (d / NOME_MANIFESTO).write_text(json.dumps({
        "etapa": "phienc_dados", "tokens": n, "tokenizer_sha": sha_tok,
        "disjunto_do_treino": disjunto,
        "partes_excluidas": ["part-00000.parquet"] if disjunto else [],
        "excluido_de": "data/processed/phienc_dados" if disjunto else None,
    }, ensure_ascii=False), encoding="utf-8")
    return d


def _rodar(modelo: Path, fatia: Path, out: Path,
           *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(AVALIADOR), "--modelo", str(modelo),
         "--dados", str(fatia), "--out", str(out),
         "--contexto", str(CONTEXTO_AVAL), "--n-sequencias", "4",
         "--dispositivo", "cpu", *extra],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(RAIZ))


def test_mede_as_DUAS_regioes_e_grava_os_vetores_do_pareado(tmp_path):
    """O caminho feliz: dois números por região e o vetor por token.

    Sem `acertos_por_token`, comparar dois braços vira comparar duas proporções
    soltas — o argumento que o resto do projeto já usa no McNemar pareado.
    """
    modelo, fatia = _modelo(tmp_path), _fatia(tmp_path)
    out = tmp_path / "mlm.json"
    r = _rodar(modelo, fatia, out)
    assert r.returncode == 0, r.stdout + r.stderr

    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["tokens_equacao"] > 0 and d["tokens_prosa"] > 0
    assert d["vantagem_em_equacao"] is not None
    assert len(d["acertos_por_token"]) == len(d["e_equacao_por_token"])
    assert len(d["acertos_por_token"]) == d["tokens_equacao"] + d["tokens_prosa"]
    # A soma dos que estão em equação tem de bater com o total da região.
    assert sum(d["e_equacao_por_token"]) == d["tokens_equacao"]
    # E o rótulo do modelo aleatório: acerta pouco, mas o número é válido.
    assert 0.0 <= d["acuracia_total"] <= 1.0


def test_RECUSA_fatia_que_nao_e_disjunta_do_treino(tmp_path):
    """⚠️ A recusa central. Medir MLM no texto de treino mede memorização, e
    entre seis variantes isso ordenaria capacidade de memorizar — não a
    capacidade de que a hipótese do DOC-07 §2.3 fala."""
    modelo = _modelo(tmp_path)
    fatia = _fatia(tmp_path, disjunto=False)
    r = _rodar(modelo, fatia, tmp_path / "mlm.json")
    assert r.returncode != 0
    saida = r.stdout + r.stderr
    assert "disjunto_do_treino" in saida
    assert "memoriza" in saida
    assert not (tmp_path / "mlm.json").exists()


def test_RECUSA_tokenizer_diferente_do_da_variante(tmp_path):
    """O §11.2 tem vocabulários de 32.768, 40.960 e 65.536. Avaliar a variante A
    sobre tokens da C dá ids que significam outra coisa — e se a C for menor, sem
    levantar em lugar nenhum."""
    modelo = _modelo(tmp_path)
    fatia = _fatia(tmp_path, sha_tok="ffffffffffffffff")
    r = _rodar(modelo, fatia, tmp_path / "mlm.json")
    assert r.returncode != 0
    assert "tokenizer" in (r.stdout + r.stderr)


def test_RECUSA_fatia_sem_nenhum_token_de_equacao(tmp_path):
    """Sem equação a diferença entre regiões não pode aparecer, por mais correta
    que a hipótese esteja — e um zero silencioso seria lido como refutação."""
    modelo = _modelo(tmp_path)
    fatia = _fatia(tmp_path, com_equacao=False)
    r = _rodar(modelo, fatia, tmp_path / "mlm.json")
    assert r.returncode != 0
    assert "equação" in (r.stdout + r.stderr)


def test_o_PROTOCOLO_vai_gravado(tmp_path):
    """⚠️ Contexto, fração e semente diferentes fazem duas tarefas diferentes.

    Gravados, o comparador pode RECUSAR; ausentes, ele subtrai dois números que
    não são comparáveis e o resultado sai com a cara certo.
    """
    modelo, fatia = _modelo(tmp_path), _fatia(tmp_path)
    out = tmp_path / "mlm.json"
    assert _rodar(modelo, fatia, out, "--semente", "23").returncode == 0
    p = json.loads(out.read_text(encoding="utf-8"))["protocolo"]
    assert p == {"contexto": CONTEXTO_AVAL, "fracao_mascara": 0.15,
                 "semente": 23, "n_sequencias_pedidas": 4}


def test_a_MESMA_semente_da_as_MESMAS_posicoes_entre_execucoes(tmp_path):
    """É o que permite o pareado: dois braços com a mesma semente vêem as mesmas
    máscaras, e o McNemar sobre "acertou este token" enxerga o que duas taxas
    não enxergam."""
    modelo, fatia = _modelo(tmp_path), _fatia(tmp_path)
    a, b, c = (tmp_path / f"{x}.json" for x in "abc")
    assert _rodar(modelo, fatia, a, "--semente", "17").returncode == 0
    assert _rodar(modelo, fatia, b, "--semente", "17").returncode == 0
    assert _rodar(modelo, fatia, c, "--semente", "99").returncode == 0

    da, db, dc = (json.loads(x.read_text(encoding="utf-8")) for x in (a, b, c))
    assert da["e_equacao_por_token"] == db["e_equacao_por_token"], (
        "a mesma semente deu posições diferentes; o pareado seria inválido")
    assert da["acertos_por_token"] == db["acertos_por_token"]
    assert da["e_equacao_por_token"] != dc["e_equacao_por_token"], (
        "sementes diferentes deram as mesmas posições")
