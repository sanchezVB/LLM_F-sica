"""`--sem-fusao`: o avaliador medindo a composição que a regra decidiu.

Em 2026-09-08 a regra pré-registrada tirou o BM25: a cadeia é `ΦEmb → ΦRank`. Sem
a chave, o avaliador ainda indexa BM25, funde por RRF e reordena **duas** vezes por
consulta — e reordenar é 96% do custo, então medir a composição que não existe mais
dobra o preço de cada braço.

O que este arquivo confere, rodando o script de verdade sobre um modelo minúsculo:

  1. sem fusão a tabela não tem BM25 nem RRF, e tem a cadeia;
  2. ⚠️ a **referência do pareado** vira o ΦEmb. Manter `pos_rrf` como referência
     compararia contra uma lista VAZIA, e o artefato sairia com `erro` no lugar
     dos vereditos — depois de três horas de T4;
  3. o teto passa a ser o do ΦEmb, e não sai duplicado;
  4. com a chave, **uma** passagem de reordenação por consulta em vez de duas.

Precisa de torch, então pula na suíte rápida (roda na `.venv-treino`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

torch = pytest.importorskip("torch", reason="requer a venv de treino (.venv-treino)")
pytest.importorskip("transformers")
pl = pytest.importorskip("polars")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    ModernBertConfig,
    ModernBertForSequenceClassification,
    PreTrainedTokenizerFast,
)

from phifm.models.encoder.config import ConfigEnc  # noqa: E402
from phifm.models.encoder.modelo import construir  # noqa: E402

AVALIADOR = RAIZ / "scripts" / "avaliar_t1b.py"
VOCAB = 512
MINI = ConfigEnc(nome="mini-t1b", camadas=2, d_model=64, cabecas=4, ffn=96,
                 vocab=VOCAB, contexto=128, janela_local=64)


def _tokenizer() -> PreTrainedTokenizerFast:
    """WordLevel sobre palavras, para textos de verdade virarem ids."""
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    vocab.update({f"w{i}": i for i in range(5, VOCAB)})
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        model_input_names=["input_ids", "attention_mask"],
        pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]")


def _emb(tmp: Path) -> Path:
    d = tmp / "emb"
    construir(MINI, torch.device("cpu"), atencao="eager").save_pretrained(d)
    _tokenizer().save_pretrained(d)
    return d


def _rank(tmp: Path) -> Path:
    d = tmp / "rank"
    hf = ModernBertConfig(**MINI.para_transformers(), reference_compile=False,
                          num_labels=1)
    ModernBertForSequenceClassification(hf).save_pretrained(d)
    _tokenizer().save_pretrained(d)
    # Confere que carrega pelo caminho que o avaliador usa.
    AutoModelForSequenceClassification.from_pretrained(d, num_labels=1)
    return d


def _pares(tmp: Path, n: int = 30) -> Path:
    d = tmp / "pares"
    d.mkdir(parents=True)
    pl.DataFrame({
        "arxiv_id": [f"a{i}" for i in range(n)],
        "arxiv_citado": [f"c{i}" for i in range(n)],
        "ancora": [f"w{10 + i} w{40 + i} w{70 + i}" for i in range(n)],
        "positivo": [f"w{100 + i} w{130 + i} w{160 + i}" for i in range(n)],
    }).write_parquet(d / "pares_validacao.parquet")
    return d


def _rodar(tmp: Path, *extra: str, com_rank: bool = True):
    saida = tmp / f"t1b{'-sf' if '--sem-fusao' in extra else ''}.json"
    cmd = [sys.executable, str(AVALIADOR),
           "--pares", str(_pares(tmp)), "--emb", str(_emb(tmp)),
           "--n-consultas", "10", "--profundidade", "5",
           "--max-tokens", "16", "--lote", "4", "--lote-rank", "4",
           "--out", str(saida), "--dispositivo", "cpu", *extra]
    if com_rank:
        cmd += ["--rank", str(_rank(tmp))]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(RAIZ))
    return r, saida


def test_sem_fusao_a_tabela_e_a_composicao_DECIDIDA(tmp_path):
    r, saida = _rodar(tmp_path, "--sem-fusao")
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(saida.read_text(encoding="utf-8"))
    nomes = [s["sistema"] for s in d["sistemas"]]
    assert nomes == ["ΦEmb", "ΦEmb+ΦRank"], nomes
    assert d["sem_fusao"] is True
    # E o BM25 não foi nem indexado.
    assert "bm25_indexar_s" not in d["custo_segundos"]
    assert "BM25 NAO indexado" in r.stdout


def test_sem_fusao_a_REFERENCIA_do_pareado_vira_o_recuperador(tmp_path):
    """⚠️ A asserção que protege três horas de T4.

    Se a referência continuasse sendo `pos_rrf`, ela seria uma lista VAZIA e cada
    pareado voltaria `{"erro": "tamanhos diferentes"}` — o artefato chegaria sem
    veredito nenhum, depois de a GPU ter sido gasta.
    """
    r, saida = _rodar(tmp_path, "--sem-fusao")
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(saida.read_text(encoding="utf-8"))
    assert d["pareado_contra_a_fusao"], "nenhum pareado saiu"
    for x in d["pareado_contra_a_fusao"]:
        assert "erro" not in x, x
        assert x["a"] == "ΦEmb", x
        assert x["b"] == "ΦEmb+ΦRank", x
    assert "ΦEmb" in d["nota_pareado"]
    assert "RECUPERADOR" in d["nota_pareado"]
    # E o confronto das cadeias não se aplica: a regra já decidiu.
    assert d["confronto_das_cadeias"] == []


def test_sem_fusao_o_teto_e_o_do_RECUPERADOR_e_nao_sai_duplicado(tmp_path):
    """Dois rótulos sobre o mesmo número convidam a ler dois tetos onde há um."""
    r, saida = _rodar(tmp_path, "--sem-fusao")
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(saida.read_text(encoding="utf-8"))
    emb = next(s for s in d["sistemas"] if s["sistema"] == "ΦEmb")
    assert d["teto_do_reranker"] == emb["recall_5"]
    assert "do ΦEmb" in r.stdout
    assert "TETO sem a fusão" not in r.stdout


def test_sem_fusao_reordena_UMA_vez_por_consulta(tmp_path):
    """A economia que faz dois braços de reranqueador caberem na mesma sessão.

    Reordenar é 96% do custo. Duas passagens por consulta dobrariam o preço de
    medir uma composição que a regra já descartou.
    """
    com, _ = _rodar(tmp_path / "com", com_rank=True)
    sem, _ = _rodar(tmp_path / "sem", "--sem-fusao", com_rank=True)
    assert com.returncode == 0 and sem.returncode == 0

    d_com = json.loads((tmp_path / "com" / "t1b.json").read_text(encoding="utf-8"))
    d_sem = json.loads(
        (tmp_path / "sem" / "t1b-sf.json").read_text(encoding="utf-8"))
    # Cinco linhas contra duas: é a assinatura das duas passagens contra uma.
    assert len(d_com["sistemas"]) == 5
    assert len(d_sem["sistemas"]) == 2
    assert len(d_com["pareado_contra_a_fusao"]) == 8
    assert len(d_sem["pareado_contra_a_fusao"]) == 2


def test_COM_fusao_nada_mudou(tmp_path):
    """O artefato de 2026-09-08 tem de seguir reproduzível: a chave é adição."""
    r, saida = _rodar(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(saida.read_text(encoding="utf-8"))
    assert [s["sistema"] for s in d["sistemas"]] == [
        "BM25", "ΦEmb", "ΦEmb+BM25 (RRF)", "ΦEmb+BM25+ΦRank",
        "ΦEmb+ΦRank (sem fusão)"]
    assert d["sem_fusao"] is False
    assert "bm25_indexar_s" in d["custo_segundos"]
    assert len(d["confronto_das_cadeias"]) == 2
    for x in d["pareado_contra_a_fusao"]:
        assert x["a"] == "ΦEmb+BM25 (RRF)", x
