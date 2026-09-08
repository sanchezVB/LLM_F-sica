"""O exportador do ΦEnc, num modelo minúsculo, em CPU.

O `laco.py` grava um `state_dict` cru; nenhum dos três avaliadores do DOC-05 §11.2
abre isso. Este arquivo confere que o exportador fecha o buraco **e** que ele
recusa os quatro jeitos de exportar um artefato que parece certo e não é:

  1. configuração vinda de um nome em vez do run — pesos certos, comportamento outro;
  2. tokenizer com vocabulário de outro tamanho — o §11.2 tem três tamanhos, e o
     sintoma disso é o tokenizer parecer ruim;
  3. `strict=False` no load — uma camada com pesos aleatórios e sucesso no fim;
  4. round-trip não conferido — dtype ou tensor amarrado perdido na gravação.

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
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer, models  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

from phifm.models.encoder.config import ConfigEnc  # noqa: E402
from phifm.models.encoder.modelo import construir  # noqa: E402
from phifm.training.pretrain.laco import NOME_ESTADO, NOME_METRICAS  # noqa: E402

VOCAB = 512
MINI = ConfigEnc(nome="mini-export", camadas=2, d_model=64, cabecas=4, ffn=96,
                 vocab=VOCAB, contexto=128, janela_local=64)

EXPORTADOR = RAIZ / "scripts" / "exportar_phienc.py"


def _tokenizer(destino: Path, n: int = VOCAB) -> Path:
    """Um tokenizer de brinquedo com os ids especiais nos lugares do `ESPECIAIS`."""
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    vocab.update({f"t{i}": i for i in range(5, n)})
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    assert tok.get_vocab_size() == n
    tok.save(str(destino))
    return destino


def _run(tmp: Path, *, concluido: bool = True, n_spikes: int = 0,
         tokenizer: Path | None = None, cfg: ConfigEnc = MINI) -> Path:
    """Um diretório de run com checkpoint e métricas, como o laço os grava."""
    d = tmp / "run"
    d.mkdir(parents=True, exist_ok=True)
    modelo = construir(cfg, torch.device("cpu"), atencao="eager")
    torch.save({"passo": 1234, "modelo": modelo.state_dict(), "opt": {},
                "escala": None, "passos_warmup": 3, "passos_decay": 10,
                "detector": {}}, d / NOME_ESTADO)
    (d / NOME_METRICAS).write_text(json.dumps({
        "modelo": cfg.como_dict(),
        "dados": {"tokenizer": str(tokenizer or _tokenizer(tmp / "tok.json"))},
        "spike": {"n_spikes": n_spikes},
        "mascaramento": {"fracao_tratada": 0.9},
        "metricas": {"passo": 1234, "perda": 4.2, "tokens": 10_000},
        "concluido": concluido,
        "motivo_da_parada": "" if concluido else "cota da sessão",
    }, ensure_ascii=False), encoding="utf-8")
    return d


def _exportar(run: Path, para: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(EXPORTADOR), "--run", str(run), "--para", str(para),
         *extra],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(RAIZ))


# ── o caminho feliz ─────────────────────────────────────────────────────────


def test_o_exportado_CARREGA_com_from_pretrained(tmp_path):
    """A razão de o script existir: `AutoModel.from_pretrained` abrir o resultado.

    Antes disto, um ΦEnc treinado era um `.pt` que só o próprio laço entendia —
    e as três medidas do §11.2 precisam de um diretório do `transformers`.
    """
    run = _run(tmp_path)
    para = tmp_path / "saida"
    r = _exportar(run, para)
    assert r.returncode == 0, r.stdout + r.stderr

    modelo = AutoModelForMaskedLM.from_pretrained(para)
    assert modelo.config.vocab_size == VOCAB
    assert modelo.config.num_hidden_layers == MINI.camadas
    tok = AutoTokenizer.from_pretrained(para)
    assert tok.mask_token == "[MASK]" and tok.mask_token_id == 4
    assert tok.pad_token_id == 0


def test_os_PESOS_sobrevivem_a_exportacao(tmp_path):
    """Um exportador que grava a arquitetura certa com pesos errados passaria em
    todos os testes de forma e mediria ruído."""
    run = _run(tmp_path)
    para = tmp_path / "saida"
    assert _exportar(run, para).returncode == 0

    original = torch.load(run / NOME_ESTADO, map_location="cpu",
                          weights_only=True)["modelo"]
    saido = AutoModelForMaskedLM.from_pretrained(para).state_dict()
    for k, v in original.items():
        assert k in saido, f"{k} não chegou ao artefato"
        assert torch.equal(v, saido[k]), f"{k} mudou de valor"


def test_a_proveniencia_vai_junto(tmp_path):
    """Um diretório de pesos sem proveniência é um modelo que ninguém sabe de
    onde veio — e num bake-off de seis variantes isso é fatal."""
    run = _run(tmp_path)
    para = tmp_path / "saida"
    assert _exportar(run, para, "--nota", "variante de teste").returncode == 0

    p = json.loads((para / "phienc_exportado.json").read_text(encoding="utf-8"))
    assert p["passo"] == 1234
    assert p["config_do_encoder"]["camadas"] == MINI.camadas
    assert p["tokenizer_sha"] and len(p["tokenizer_sha"]) > 8
    assert p["ida_e_volta_delta_logits"] == 0.0
    assert p["nota"] == "variante de teste"
    # ⚠️ O aviso de que o estado do otimizador NÃO vem: quem pegar este
    # diretório para retomar o treino perderia momento e escala do fp16.
    assert "otimizador" in p["aviso"].lower()
    assert (para / "_manifesto_etapa.json").exists()


# ── as quatro recusas ───────────────────────────────────────────────────────


def test_tokenizer_de_OUTRO_TAMANHO_e_recusado(tmp_path):
    """⚠️ O §11.2 tem variantes de 32.768, 40.960 e 65.536 tokens.

    Exportar uma com o tokenizer de outra dá ids que significam coisas
    diferentes, o MLM despenca, e a tabela final registra "esse tokenizer é
    pior". É o erro mais caro que este script pode deixar passar, porque o
    resultado sai com a cara de uma medição.
    """
    run = _run(tmp_path)
    outro = _tokenizer(tmp_path / "tok_pequeno.json", n=256)
    r = _exportar(run, tmp_path / "saida", "--tokenizer", str(outro))
    assert r.returncode != 0
    saida = r.stdout + r.stderr
    assert "256" in saida and "512" in saida
    assert not (tmp_path / "saida" / "config.json").exists(), (
        "recusou mas deixou um artefato meio gravado, que alguém vai usar")


def test_a_config_vem_do_RUN_e_nao_de_um_nome(tmp_path):
    """A interface não aceita um nome de configuração, e isso é deliberado.

    `--config PHIENC_150M` deixaria uma constante mudada depois do treino
    produzir um modelo que carrega sem reclamar e não é o modelo treinado.
    """
    # ⚠️ A pergunta é sobre a INTERFACE, não sobre o texto do arquivo. A versão
    # anterior deste teste fazia `"--config" not in fonte` e quebrou no próprio
    # parágrafo da docstring que explica por que a opção não existe — a mesma
    # armadilha que já derrubou três testes deste repo.
    ajuda = subprocess.run([sys.executable, str(EXPORTADOR), "--help"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=str(RAIZ)).stdout
    opcoes = {p for p in ajuda.split() if p.startswith("--")}
    assert "--config" not in opcoes, (
        "voltou a existir uma forma de nomear a configuração na linha de "
        "comando; ver o § da docstring do módulo")
    assert "--run" in opcoes, "a ajuda não saiu como esperado; teste inválido"
    # E o run sem a chave da configuração não é adivinhado.
    run = _run(tmp_path)
    m = json.loads((run / NOME_METRICAS).read_text(encoding="utf-8"))
    del m["modelo"]
    (run / NOME_METRICAS).write_text(json.dumps(m), encoding="utf-8")
    r = _exportar(run, tmp_path / "saida")
    assert r.returncode != 0
    assert "modelo" in (r.stdout + r.stderr)


def test_state_dict_INCOMPLETO_derruba_a_exportacao(tmp_path):
    """`strict=False` deixaria uma camada com pesos aleatórios e a exportação
    terminaria com sucesso — o modelo rodaria e mediria mal."""
    run = _run(tmp_path)
    ck = torch.load(run / NOME_ESTADO, map_location="cpu", weights_only=True)
    alvo = next(k for k in ck["modelo"] if "layers.1" in k)
    del ck["modelo"][alvo]
    torch.save(ck, run / NOME_ESTADO)

    r = _exportar(run, tmp_path / "saida")
    assert r.returncode != 0
    assert "issing" in (r.stdout + r.stderr) or "strict" in (r.stdout + r.stderr)


def test_run_INACABADO_ou_com_SPIKE_exige_a_chave(tmp_path):
    """Num bake-off, uma variante que divergiu e foi exportada em silêncio é
    lida como "esse tokenizer é pior". A recusa preserva a distinção."""
    tok = _tokenizer(tmp_path / "tok.json")

    inacabado = _run(tmp_path / "a", concluido=False, tokenizer=tok)
    r = _exportar(inacabado, tmp_path / "a" / "saida")
    assert r.returncode != 0
    assert "concluído" in (r.stdout + r.stderr)

    com_spike = _run(tmp_path / "b", n_spikes=3, tokenizer=tok)
    r = _exportar(com_spike, tmp_path / "b" / "saida")
    assert r.returncode != 0
    assert "spike" in (r.stdout + r.stderr).lower()

    # E com a chave sai, com a ressalva GRAVADA no artefato.
    para = tmp_path / "b" / "saida"
    r = _exportar(com_spike, para, "--mesmo-assim")
    assert r.returncode == 0, r.stdout + r.stderr
    p = json.loads((para / "phienc_exportado.json").read_text(encoding="utf-8"))
    assert p["ressalvas"] and any("spike" in x.lower() for x in p["ressalvas"])


def test_o_round_trip_e_CONFERIDO_e_nao_presumido(tmp_path):
    """Se a conferência sair, o exportador volta a ser uma suposição.

    `save_pretrained` pode perder um tensor amarrado, mudar dtype ou reconstruir
    com outra atenção sem dizer nada, e o sintoma é uma métrica um pouco pior —
    indistinguível de um resultado.
    """
    run = _run(tmp_path)
    para = tmp_path / "saida"
    r = _exportar(run, para)
    assert r.returncode == 0, r.stdout + r.stderr
    # A linha só sai se `conferir_ida_e_volta` rodou e passou.
    assert "diferença máxima de logits = 0" in r.stdout, (
        "a conferência não é reportada; sem isso ela pode ter sido removida sem "
        "que nada acuse")


def test_a_conferencia_de_ida_e_volta_TEM_DENTES(tmp_path):
    """Uma conferência que nunca reprova não é uma conferência.

    Aqui os pesos em memória são mexidos DEPOIS da gravação: o disco tem uma
    coisa e a memória outra, que é exatamente o estado que a conferência existe
    para detectar. Se ela passar, ela não está olhando os pesos.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_exp", EXPORTADOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    modelo = construir(MINI, torch.device("cpu"), atencao="eager")
    destino = tmp_path / "saida"
    destino.mkdir()
    modelo.save_pretrained(destino)
    # Passa antes de mexer...
    assert mod.conferir_ida_e_volta(modelo, destino, MINI.contexto, "eager") == 0.0
    # ...e reprova depois.
    with torch.no_grad():
        next(iter(modelo.parameters())).add_(1.0)
    with pytest.raises(SystemExit, match="mudou de valor|não é o checkpoint"):
        mod.conferir_ida_e_volta(modelo, destino, MINI.contexto, "eager")
