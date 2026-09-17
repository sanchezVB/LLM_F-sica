"""Dois processos treinam o MESMO modelo que um. Prova por equivalência.

A sessão "T4 x2" do Kaggle quase dobra os tokens por hora de cota, e um braço treinado
em duas GPUs só pode ser comparado com um treinado em uma se o treino for o mesmo:
mesmos dados, mesmas máscaras, mesmo gradiente. "Parece funcionar" não serve — um
DDP que duplica ou pula micro-passos, ou que sorteia a mesma máscara nos dois
processos, treina normalmente e a perda desce.

Então: o mesmo treino minúsculo em um processo e em dois (`gloo`, CPU, `FileStore` —
no Windows o `TCPStore` do torch 2.4 pede libuv), e os pesos finais têm de coincidir.
Os contadores do mascaramento, somados entre processos, têm de coincidir EXATAMENTE:
é a prova de que os dois processos viram, juntos, as mesmas máscaras que um.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

torch = pytest.importorskip("torch", reason="requer a venv de treino (.venv-treino)")
pytest.importorskip("transformers")

from phifm.models.encoder.config import ConfigEnc  # noqa: E402
from phifm.training.pretrain.dados import (  # noqa: E402
    NOME_MANIFESTO,
    NOME_MARCAS,
    NOME_TOKENS,
    ConfigDados,
    Fluxo,
    marcas_de,
)
from phifm.training.pretrain.laco import (  # noqa: E402
    NOME_ESTADO,
    NOME_METRICAS,
    ConfigTreino,
    Distribuicao,
    Treinador,
)
from phifm.training.pretrain.mascaramento import ConfigMascara  # noqa: E402

VOCAB = 512
MINI = ConfigEnc(nome="mini", camadas=2, d_model=64, cabecas=4, ffn=96,
                 vocab=VOCAB, contexto=128, janela_local=64)


def _dados(raiz: Path, n: int = 128 * 40) -> Path:
    raiz.mkdir(parents=True, exist_ok=True)
    if (raiz / NOME_MANIFESTO).exists():
        return raiz
    rng = np.random.default_rng(3)
    ids = rng.integers(5, VOCAB, size=n, dtype=np.uint16)
    ide = np.full(n, -1, dtype=np.int32)
    disp = np.zeros(n, dtype=bool)
    for k, i in enumerate(range(20, n - 40, 100)):
        ide[i:i + 30] = k
        disp[i:i + 30] = True
    (raiz / NOME_TOKENS).write_bytes(ids.tobytes())
    (raiz / NOME_MARCAS).write_bytes(marcas_de(ide, disp).tobytes())
    (raiz / NOME_MANIFESTO).write_text(
        json.dumps({"tokens": int(n), "git_sha": "ddp", "tokenizer": "sintetico"}),
        encoding="utf-8")
    return raiz


def _treinador(raiz: Path, *, total: int, acumulacao: int,
               distribuicao: Distribuicao | None = None) -> Treinador:
    fluxo = Fluxo(ConfigDados(raiz=raiz, contexto=MINI.contexto, sequencias=1,
                              semente=17))
    return Treinador(
        MINI, ConfigTreino(total_passos=total, acumulacao=acumulacao, passos_log=1,
                           passos_estado=2, amp=False),
        ConfigMascara(p_equacao=0.6, semente=17), fluxo,
        dev=torch.device("cpu"), distribuicao=distribuicao)


def _trabalhador(rank: int, mundo: int, raiz: str, store: str, saida: str,
                 total: int, acumulacao: int) -> None:
    """Um processo do treino distribuído. No nível do módulo: o `spawn` o importa."""
    import os

    import torch.distributed as dist

    if sys.platform == "win32":
        # ⚠️ Sem isto o `init_process_group` TRAVA nesta máquina, sem erro: o gloo
        # escolhe a interface pelo nome do host, e ele resolve primeiro para um IPv6
        # link-local e para o IP do adaptador "Topaz Loopback" (módulo de segurança
        # de banco). Os dois processos nunca se encontram. A loopback do Windows
        # resolve, e não mexe em nada fora deste processo.
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "Loopback Pseudo-Interface 1")
    dist.init_process_group("gloo", init_method=Path(store).as_uri(), rank=rank,
                            world_size=mundo)
    try:
        _treinador(Path(raiz), total=total, acumulacao=acumulacao,
                   distribuicao=Distribuicao(rank, mundo)).treinar(Path(saida))
    finally:
        dist.destroy_process_group()


def test_a_mascara_nao_depende_de_quantas_vieram_antes(tmp_path):
    """⚠️ Até 2026-09-17 a máscara saía de um gerador único que avançava a cada
    chamada: o micro-passo 5 mascarava diferente conforme a história, e uma
    retomada de checkpoint já não reproduzia a execução contínua."""
    raiz = _dados(tmp_path / "dados")
    fresco = _treinador(raiz, total=10, acumulacao=2)
    com_historia = _treinador(raiz, total=10, acumulacao=2)
    com_historia._mascarar_lote(3)
    com_historia._mascarar_lote(0)
    e1, a1 = fresco._mascarar_lote(5)
    e2, a2 = com_historia._mascarar_lote(5)
    assert torch.equal(e1, e2) and torch.equal(a1, a2)


def test_acumulacao_que_nao_se_divide_pelos_processos_e_recusada(tmp_path):
    raiz = _dados(tmp_path / "dados")
    with pytest.raises(ValueError, match="não se divide"):
        _treinador(raiz, total=10, acumulacao=3, distribuicao=Distribuicao(0, 2))


def test_dois_processos_terminam_com_os_mesmos_pesos_que_um(tmp_path):
    raiz = _dados(tmp_path / "dados")
    total, acumulacao = 4, 2

    um = tmp_path / "um"
    _treinador(raiz, total=total, acumulacao=acumulacao).treinar(um)

    dois = tmp_path / "dois"
    # ⚠️ O `FileStore` não decodifica a URI: um caminho com "í" (`LLMFísica`) vira
    # `%C3%AD` e dá "No such file or directory". O store vai para um diretório ASCII.
    store = Path(tempfile.mkdtemp(prefix="phifm_ddp_")) / "store"
    torch.multiprocessing.spawn(
        _trabalhador,
        args=(2, str(raiz), str(store), str(dois), total, acumulacao),
        nprocs=2, join=True)

    p1 = torch.load(um / NOME_ESTADO, weights_only=False)
    p2 = torch.load(dois / NOME_ESTADO, weights_only=False)
    assert p1["passo"] == p2["passo"] == total
    assert p1["modelo"].keys() == p2["modelo"].keys(), (
        "as chaves do checkpoint mudaram: o DDP gravou com o prefixo `module.`")
    diferenca = max(float((p1["modelo"][k] - p2["modelo"][k]).abs().max())
                    for k in p1["modelo"])
    assert diferenca < 1e-5, f"pesos divergem por {diferenca:.2e}"

    j1 = json.loads((um / NOME_METRICAS).read_text(encoding="utf-8"))
    j2 = json.loads((dois / NOME_METRICAS).read_text(encoding="utf-8"))
    # Iguais EXATAMENTE: os dois processos juntos viram as mesmas máscaras que um.
    assert j1["mascaramento"] == j2["mascaramento"]
    assert j2["distribuicao"]["processos"] == 2
    assert j2["distribuicao"]["acumulacao_por_processo"] == 1
    perdas1 = [h["perda"] for h in j1["historico"]]
    perdas2 = [h["perda"] for h in j2["historico"]]
    assert perdas1 == pytest.approx(perdas2, abs=1e-5)
