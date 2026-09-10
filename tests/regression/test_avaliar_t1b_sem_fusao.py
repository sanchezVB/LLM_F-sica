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


def test_as_POSICOES_por_consulta_vao_no_artefato(tmp_path):
    """⚠️ O que permite parear DUAS EXECUÇÕES, e o que faltou em 2026-09-08.

    A regra pré-registrada do T1b2 pedia um confronto entre duas cadeias; o run
    mediu as duas e o teste não podia mais ser calculado, porque só as métricas
    agregadas eram gravadas. Ali a aritmética decidiu, por sorte do tamanho do
    efeito. O retreino do ΦRank compara dois reranqueadores em invocações
    separadas — sem isto, o confronto seria impossível de novo.
    """
    import sys as _sys

    r, saida = _rodar(tmp_path, "--sem-fusao")
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(saida.read_text(encoding="utf-8"))
    for s in d["sistemas"]:
        assert "posicoes" in s, s["sistema"]
        assert len(s["posicoes"]) == d["n_consultas"]
        assert all(x is None or isinstance(x, int) for x in s["posicoes"])
    # E as posições reproduzem as métricas agregadas — se divergissem, uma das
    # duas estaria errada e não haveria como saber qual.
    _sys.path.insert(0, str(RAIZ / "src"))
    from phifm.eval.hibrido import recall_em

    for s in d["sistemas"]:
        assert round(recall_em(s["posicoes"], 1), 4) == s["recall_1"], s["sistema"]
        assert round(recall_em(s["posicoes"], 10), 4) == s["recall_10"]
    assert "mcnemar_em" in d["nota_posicoes"]


# ── sub-profundidades: a curva de profundidade DE GRAÇA ─────────────────────


def test_as_sub_profundidades_saem_da_MESMA_passagem(tmp_path):
    """⚠️ O escore do cross-encoder é do par (consulta, documento) e não depende
    de quem mais está no conjunto.

    Pontuados os `--profundidade` candidatos, a cadeia a uma profundidade MENOR é
    só reordenar os N primeiros da ordem densa pelos mesmos escores. Medir @2 num
    segundo run custaria outra passagem inteira e cairia na armadilha do T1b2:
    comparar sessões diferentes.
    """
    r, saida = _rodar(tmp_path, "--sem-fusao", "--sub-profundidades", "2,3")
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(saida.read_text(encoding="utf-8"))
    nomes = [s["sistema"] for s in d["sistemas"]]
    assert nomes == ["ΦEmb", "ΦEmb+ΦRank", "ΦEmb+ΦRank @2", "ΦEmb+ΦRank @3"], nomes
    # Cada sub-profundidade tem as suas posições, do mesmo tamanho.
    for s in d["sistemas"]:
        assert len(s["posicoes"]) == d["n_consultas"], s["sistema"]


def test_a_sub_profundidade_NAO_pode_exceder_a_profundidade(tmp_path):
    """Não há escore para candidato que a execução nunca pontuou; deixar passar
    daria uma linha silenciosamente idêntica à profundidade cheia."""
    r, _ = _rodar(tmp_path, "--sem-fusao", "--sub-profundidades", "999")
    assert r.returncode != 0
    assert "maior que --profundidade" in (r.stdout + r.stderr)


def test_o_confronto_entre_PROFUNDIDADES_e_calculado(tmp_path):
    """A pergunta de 2026-09-10: o reranqueador ganha com mais candidatos, ou
    eles só trazem distratores? Pareado exato, porque os recortes saem da mesma
    passagem."""
    r, saida = _rodar(tmp_path, "--sem-fusao", "--sub-profundidades", "2")
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(saida.read_text(encoding="utf-8"))
    c = d["confronto_das_cadeias"]
    assert len(c) == 2, c
    for x in c:
        assert x["a"] == "ΦEmb+ΦRank @2", x
        assert x["b"] == "ΦEmb+ΦRank @5", x
    assert {x["k"] for x in c} == {1, 10}


def test_a_sub_profundidade_igual_a_profundidade_reproduz_a_cadeia(tmp_path):
    """A conferência de sanidade do recorte: restringir aos `profundidade`
    primeiros é não restringir, e tem de dar exatamente a mesma coisa.

    Se der diferente, o recorte está desalinhado com os escores — e um
    desalinhamento produziria uma curva de profundidade inteira errada, com a
    cara certa.
    """
    r, saida = _rodar(tmp_path, "--sem-fusao", "--sub-profundidades", "5")
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(saida.read_text(encoding="utf-8"))
    cheia = next(s for s in d["sistemas"] if s["sistema"] == "ΦEmb+ΦRank")
    recorte = next(s for s in d["sistemas"] if s["sistema"] == "ΦEmb+ΦRank @5")
    assert recorte["posicoes"] == cheia["posicoes"]
    assert recorte["ndcg_10"] == cheia["ndcg_10"]
