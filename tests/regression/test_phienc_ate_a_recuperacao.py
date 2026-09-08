"""A corrente inteira: laço → exportador → avaliador de recuperação do §11.2.

## ⚠️ O avaliador de recuperação do §11.2 NÃO é código novo

A primeira leitura do DOC-05 §11.2 ("avaliar em recuperação de Física, MLM em
texto denso em equações, e uma sonda de estrutura tensorial") sugere três
avaliadores a construir. Mas o primeiro já existe: `scripts/avaliar_encoders.py`
carrega um diretório com `AutoModel.from_pretrained`, faz média mascarada,
normaliza, e roda o protocolo do G1 com teto declarado e McNemar pareado.

Escrever um segundo daria duas réguas com o mesmo nome — é a objeção que o
`eval/hibrido.py` já registra sobre o nDCG. Então o que faltava era só o
**exportador**, e o que este arquivo confere é que a corrente fecha: um checkpoint
do laço vira um diretório que o avaliador do G1 aceita como qualquer outro ponto
da curva.

## E a medida discrimina modelos sem treino contrastivo

A dúvida legítima era se um encoder só de MLM, sem estágio contrastivo, produziria
embeddings tão fracos que as seis variantes cairiam todas no ruído — e a medida não
decidiria nada. Os números do próprio `g1_resultado.json` respondem: o SciBERT dá
nDCG@10 **0,2537** e o PhysBERT **0,3507**, ambos MLM puro, média mascarada, zero
treino contrastivo. **0,097 de separação** entre dois encoders que diferem no
domínio do pré-treino.

É a comparação análoga à do §11.2, e ela tem faixa dinâmica de sobra. Nenhum
estágio de adaptação é necessário para o bake-off decidir.
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

from tokenizers import Tokenizer, models  # noqa: E402

from phifm.models.encoder.config import ConfigEnc  # noqa: E402
from phifm.models.encoder.modelo import construir  # noqa: E402
from phifm.training.pretrain.laco import NOME_ESTADO, NOME_METRICAS  # noqa: E402

VOCAB = 512
MINI = ConfigEnc(nome="mini-corrente", camadas=2, d_model=64, cabecas=4, ffn=96,
                 vocab=VOCAB, contexto=128, janela_local=64)

EXPORTADOR = RAIZ / "scripts" / "exportar_phienc.py"
AVALIADOR = RAIZ / "scripts" / "avaliar_encoders.py"


def _exportar_mini(tmp: Path) -> Path:
    """Um ΦEnc minúsculo, salvo como o laço salva e exportado pelo script."""
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    vocab.update({f"t{i}": i for i in range(5, VOCAB)})
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    caminho_tok = tmp / "tok.json"
    tok.save(str(caminho_tok))

    run = tmp / "run"
    run.mkdir(parents=True)
    modelo = construir(MINI, torch.device("cpu"), atencao="eager")
    torch.save({"passo": 7, "modelo": modelo.state_dict()}, run / NOME_ESTADO)
    (run / NOME_METRICAS).write_text(json.dumps({
        "modelo": MINI.como_dict(),
        "dados": {"tokenizer": str(caminho_tok)},
        "metricas": {"passo": 7, "perda": 6.2, "tokens": 1000},
        "concluido": True,
    }, ensure_ascii=False), encoding="utf-8")

    para = tmp / "phienc-mini"
    r = subprocess.run(
        [sys.executable, str(EXPORTADOR), "--run", str(run), "--para", str(para)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(RAIZ))
    assert r.returncode == 0, r.stdout + r.stderr
    return para


def _pares(tmp: Path, n: int = 40) -> Path:
    """Um `pares_validacao.parquet` de brinquedo com âncoras e alvos únicos."""
    d = tmp / "pares"
    d.mkdir(parents=True)
    pl.DataFrame({
        "ancora": [f"consulta sobre o fenômeno número {i} em física" for i in range(n)],
        "positivo": [f"artigo que descreve o fenômeno {i} com equações" for i in range(n)],
    }).write_parquet(d / "pares_validacao.parquet")
    return d


def test_o_exportado_e_aceito_pelo_avaliador_do_G1(tmp_path):
    """A asserção central: o diretório exportado entra como ponto da curva.

    Se isto falhar, o §11.2 continua bloqueado mesmo com o exportador escrito —
    e o sintoma seria descobrir isso depois de 44 h de T4 por variante.
    """
    modelo = _exportar_mini(tmp_path)
    pares = _pares(tmp_path)
    saida = tmp_path / "g1.json"

    r = subprocess.run(
        [sys.executable, str(AVALIADOR),
         "--pares", str(pares), "--n", "20", "--max-tokens", "32", "--lote", "4",
         "--modelo", f"variante mini={modelo}",
         "--out", str(saida), "--cache", str(tmp_path / "cache")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(RAIZ))
    assert r.returncode == 0, r.stdout + r.stderr
    assert saida.exists(), "o avaliador não gravou artefato"

    d = json.loads(saida.read_text(encoding="utf-8"))
    nomes = [m["nome"] for m in d["modelos"]]
    assert "variante mini" in nomes, f"a variante não entrou na tabela: {nomes}"
    m = next(m for m in d["modelos"] if m["nome"] == "variante mini")
    # Um modelo de pesos aleatórios não tem de acertar nada; o que se confere é
    # que a métrica SAIU, com valor no intervalo válido.
    assert 0.0 <= m["ndcg_10"] <= 1.0
    assert 0.0 <= m["recall_1"] <= 1.0


def test_o_teto_do_protocolo_vem_junto(tmp_path):
    """⚠️ Sem o teto, um nDCG do bake-off não diz se o limite é o tokenizer ou a
    régua — foi o que fez o G1.2 "passar" por ruído de desempate em 2026-09-06.
    """
    modelo = _exportar_mini(tmp_path)
    pares = _pares(tmp_path)
    saida = tmp_path / "g1.json"
    r = subprocess.run(
        [sys.executable, str(AVALIADOR),
         "--pares", str(pares), "--n", "20", "--max-tokens", "32", "--lote", "4",
         "--modelo", f"variante mini={modelo}",
         "--out", str(saida), "--cache", str(tmp_path / "cache")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(RAIZ))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "teto do protocolo" in r.stdout
    d = json.loads(saida.read_text(encoding="utf-8"))
    assert "teto_do_protocolo" in d


def test_a_medida_zero_shot_SEPARA_encoders_de_MLM_puro():
    """A faixa dinâmica que autoriza o §11.2 a dispensar estágio contrastivo.

    SciBERT e PhysBERT são MLM puro, média mascarada, sem treino contrastivo, e o
    protocolo do G1 os separa por 0,097 de nDCG@10. Se um dia essa separação
    encolher para o nível do ruído, o bake-off perde o instrumento e este teste é
    o aviso.
    """
    p = RAIZ / "data" / "processed" / "avaliacao" / "g1_resultado.json"
    if not p.exists():
        pytest.skip("g1_resultado.json ausente (artefato de execução)")
    d = json.loads(p.read_text(encoding="utf-8"))
    por_nome = {m["nome"]: m for m in d["modelos"]}
    sci = next(v for k, v in por_nome.items() if k.startswith("SciBERT"))
    phys = next(v for k, v in por_nome.items() if k.startswith("PhysBERT"))
    separacao = phys["ndcg_10"] - sci["ndcg_10"]
    assert separacao > 0.05, (
        f"a separação entre dois encoders de MLM puro caiu para {separacao:.4f}; "
        "sem faixa dinâmica o §11.2 não decide nada com esta medida")
