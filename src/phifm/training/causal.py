"""Piloto causal em blocos fixos, com avaliação de desenvolvimento e retomada exata.

O ensaio usa o objetivo de próximo token do modelo Hugging Face. Não é o laço de
MLM do ΦEnc nem uma implementação de SFT/RLVR. Dados e modelo são fornecidos pelo
chamador; este módulo não baixa artefatos e não lê benchmarks.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ConfigCausal:
    passos: int = 20
    lote: int = 1
    acumulacao: int = 1
    lr: float = 1e-5
    weight_decay: float = 0.01
    clipping: float = 1.0
    semente: int = 17
    salvar_a_cada: int = 5

    def __post_init__(self):
        for nome in ("passos", "lote", "acumulacao", "salvar_a_cada"):
            if type(getattr(self, nome)) is not int or getattr(self, nome) < 1:
                raise ValueError(f"{nome} deve ser inteiro positivo")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr deve ser finita e positiva")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay deve ser finito e não negativo")
        if not math.isfinite(self.clipping) or self.clipping <= 0:
            raise ValueError("clipping deve ser finito e positivo")


def _hash_tokens(tokens: torch.Tensor) -> str:
    if tokens.dtype != torch.long or tokens.ndim != 2 or min(tokens.shape) < 1:
        raise ValueError("dados devem ser blocos long [exemplos, contexto]")
    if tokens.shape[1] < 2 or tokens.min().item() < 0:
        raise ValueError("blocos precisam de ao menos dois tokens não negativos")
    return hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest()


def _gravar_atomico(caminho: Path, estado: dict) -> None:
    temporario = caminho.with_suffix(".tmp")
    try:
        with temporario.open("wb") as f:
            torch.save(estado, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporario, caminho)
    finally:
        temporario.unlink(missing_ok=True)


def perda_de_dev(modelo, dados: torch.Tensor, lote: int) -> float:
    """Média por token previsto, sem alterar o modo treino do chamador."""
    dev = next(modelo.parameters()).device
    modo = modelo.training
    modelo.eval()
    soma, n = 0.0, 0
    try:
        with torch.no_grad():
            for bloco in dados.split(lote):
                x = bloco.to(dev)
                perda = modelo(input_ids=x, labels=x, use_cache=False).loss
                if not torch.isfinite(perda):
                    raise ValueError("perda de desenvolvimento não finita")
                alvos = x.shape[0] * (x.shape[1] - 1)
                soma += perda.item() * alvos
                n += alvos
    finally:
        modelo.train(modo)
    if not n:
        raise ValueError("desenvolvimento vazio")
    return soma / n


def treinar_causal(modelo, treino: torch.Tensor, dev: torch.Tensor, saida: Path,
                    config: ConfigCausal, identidade: dict, *,
                    passos_nesta_sessao: int | None = None) -> dict:
    """Treina/retoma; alterar dados/configuração/base recusa o checkpoint existente.

    `passos_nesta_sessao` só limita esta sessão, mantendo o orçamento total e o
    experimento. O checkpoint é local e confiável, nunca carregado de uma URL.
    """
    if not identidade:
        raise ValueError("identidade da base e da auditoria é obrigatória")
    if passos_nesta_sessao is not None and (
            type(passos_nesta_sessao) is not int or passos_nesta_sessao < 1):
        raise ValueError("passos_nesta_sessao deve ser inteiro positivo")
    contrato = {"schema": "phifm.causal-pilot.v1", "config": asdict(config),
                "identidade": identidade, "treino_sha256": _hash_tokens(treino),
                "dev_sha256": _hash_tokens(dev), "formas": [list(treino.shape), list(dev.shape)],
                "modelo_config": modelo.config.to_dict(), "torch": str(torch.__version__)}
    # Configs HF podem conter tuplas; canonizar evita diferença após salvar em JSON.
    contrato = json.loads(json.dumps(contrato, sort_keys=True, default=str))
    dispositivo = next(modelo.parameters()).device
    vocab = modelo.config.vocab_size
    if treino.max().item() >= vocab or dev.max().item() >= vocab:
        raise ValueError("token fora do vocabulário da base")
    if treino.shape[1] != dev.shape[1]:
        raise ValueError("treino e desenvolvimento devem usar o mesmo contexto")
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=config.lr,
                                   weight_decay=config.weight_decay)
    saida = Path(saida)
    checkpoint = saida / "estado_causal.pt"
    if checkpoint.exists():
        estado = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if estado.get("contrato") != contrato:
            raise ValueError("checkpoint pertence a outra base, configuração ou dados")
        modelo.load_state_dict(estado["modelo"])
        otimizador.load_state_dict(estado["otimizador"])
        torch.set_rng_state(estado["rng_torch"])
        if dispositivo.type == "cuda":
            if estado["rng_cuda"] is None:
                raise ValueError("retomada CUDA sem estado aleatório CUDA")
            torch.cuda.set_rng_state_all(estado["rng_cuda"])
        random.setstate(estado["rng_python"])
        np.random.set_state(estado["rng_numpy"])
        passo, historico = estado["passo"], estado["historico"]
        inicial = estado["perda_dev_inicial"]
    else:
        if saida.exists() and any(saida.iterdir()):
            raise ValueError("saída não vazia sem checkpoint causal; escolha outra pasta")
        saida.mkdir(parents=True, exist_ok=True)
        random.seed(config.semente)
        np.random.seed(config.semente)
        torch.manual_seed(config.semente)
        passo, historico = 0, []
        inicial = perda_de_dev(modelo, dev, config.lote)

    def salvar():
        _gravar_atomico(checkpoint, {
            "contrato": contrato, "passo": passo, "historico": historico,
            "perda_dev_inicial": inicial, "modelo": modelo.state_dict(),
            "otimizador": otimizador.state_dict(), "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if dispositivo.type == "cuda" else None,
            "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
        })

    fim = min(config.passos, passo + (passos_nesta_sessao or config.passos))
    modelo.train()
    while passo < fim:
        otimizador.zero_grad(set_to_none=True)
        gerador = torch.Generator().manual_seed(config.semente + passo)
        perda_passo = 0.0
        for _ in range(config.acumulacao):
            indices = torch.randint(len(treino), (config.lote,), generator=gerador)
            x = treino[indices].to(dispositivo)
            perda = modelo(input_ids=x, labels=x, use_cache=False).loss
            if not torch.isfinite(perda):
                raise ValueError(f"perda não finita no passo {passo}; último checkpoint preservado")
            (perda / config.acumulacao).backward()
            perda_passo += perda.item() / config.acumulacao
        norma = torch.nn.utils.clip_grad_norm_(modelo.parameters(), config.clipping,
                                                error_if_nonfinite=True)
        otimizador.step()
        passo += 1
        historico.append({"passo": passo, "perda": perda_passo, "norma": float(norma)})
        if passo % config.salvar_a_cada == 0:
            salvar()
    salvar()
    resultado = {"schema": contrato["schema"], "concluido": passo == config.passos,
                 "passo": passo, "perda_dev_inicial": inicial,
                 "perda_dev_final": perda_de_dev(modelo, dev, config.lote),
                 "tokens_previstos": passo * config.lote * config.acumulacao * (treino.shape[1] - 1),
                 "historico": historico, "contrato": contrato,
                 "escopo": "piloto de desenvolvimento; não é resultado G2"}
    temporario = saida / "resultado_causal.json.tmp"
    temporario.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporario.replace(saida / "resultado_causal.json")
    return resultado
