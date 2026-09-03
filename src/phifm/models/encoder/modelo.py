"""Constrói o ΦEnc a partir de uma `ConfigEnc`. É a única parte que importa torch.

Separado de `config.py` de propósito: a configuração e a contagem de parâmetros são
puras e rodam na suíte rápida; só a materialização dos pesos precisa da venv de
treino. Foi por não ter feito essa separação em `rerank.py` que `amostragem.py` teve
de ser extraído depois.

## ⚠️ O que este projeto NÃO tem para rodar o ΦEnc como especificado

O DOC-07 §2.1 pede **FlashAttention-2**, e ela exige SM 8.0+ (Ampere). As duas GPUs
disponíveis não servem:

| | arquitetura | SM | FA-2 | bf16 |
|---|---|---|---|---|
| RX 7600 (DirectML) | RDNA 3 | — | não | não |
| T4 do Kaggle | Turing | 7.5 | **não** | **não** |

Consequências, e nenhuma delas é opcional:

1. **`attn_implementation="sdpa"`**, não `flash_attention_2`. O ganho de 2–3× de
   vazão que o §2.1 credita ao unpadding + FA-2 **não** está disponível aqui, e a
   estimativa de horas do §2.4 tem de ser lida com isso em mente.
2. **fp16 com `GradScaler`, não bf16.** O DOC-08 §4 pede "bf16, mestre fp32"; a T4
   não tem bf16. fp16 tem faixa dinâmica menor e é justamente o regime em que os
   *spikes* de perda do DOC-08 §6.1 aparecem — a detecção deles deixa de ser luxo.
3. **`reference_compile=False`.** O ModernBERT do `transformers` compila trechos com
   `torch.compile` por padrão quando acha que vale; no DirectML isso falha, e num
   ambiente sem CUDA a compilação custa minutos e não devolve nada.

Nada disso é razão para não escrever o código: a receita é a mesma e o que muda é a
vazão. Mas registrar aqui evita que alguém leia "17 h numa H100" e planeje o
cronograma com esse número numa T4 — são **~285 h de FLOPs**, e mais na prática.
"""

from __future__ import annotations

import logging

import torch
from transformers import ModernBertConfig, ModernBertForMaskedLM

from phifm.models.encoder.config import ConfigEnc

log = logging.getLogger(__name__)


def escolher_atencao(dev: torch.device) -> str:
    """`sdpa` em tudo que temos; `eager` fora do CUDA.

    ⚠️ Nunca `flash_attention_2` aqui. Pedi-la numa T4 levanta na construção do
    modelo, e o DirectML não a tem de forma nenhuma — descobrir isso depois de
    montar o dataloader é perder a sessão.
    """
    return "sdpa" if dev.type == "cuda" else "eager"


def construir(cfg: ConfigEnc, dev: torch.device | None = None,
              atencao: str | None = None) -> ModernBertForMaskedLM:
    """Materializa o modelo, e CONFERE a contagem contra a analítica.

    ⚠️ A conferência não é decoração. `config.py` calcula o total analiticamente e o
    orçamento de computação do DOC-07 §2.4 sai desse número. Se uma versão nova do
    `transformers` mudar a estrutura do ModernBERT — uma norm a mais, um bias que
    volta —, o modelo passaria a ter outro tamanho e o orçamento estaria errado sem
    nada reclamar. Aqui reclama.
    """
    cfg.conferir_declarado()
    dev = dev or torch.device("cpu")
    hf = ModernBertConfig(
        **cfg.para_transformers(),
        # Ver a ressalva 3 na docstring do módulo.
        reference_compile=False,
    )
    modelo = ModernBertForMaskedLM(hf)

    esperado = cfg.parametros()
    real = sum(p.numel() for p in modelo.parameters())
    if real != esperado["total"]:
        raise ValueError(
            f"{cfg.nome}: a contagem analítica diz {esperado['total']:,} parâmetros "
            f"e o modelo tem {real:,} (diferença de {real - esperado['total']:+,}).\n"
            "A estrutura do ModernBERT no `transformers` mudou, ou a fórmula de "
            "`ConfigEnc.parametros()` está errada. Os dois casos invalidam o "
            "orçamento do DOC-07 §2.4, que é linear em N — conserte antes de treinar.")

    atencao = atencao or escolher_atencao(dev)
    if atencao != "sdpa":
        # Não é erro, é perda de vazão, e ela tem de aparecer no log de quem paga.
        log.warning("atenção %r (não sdpa): sem FA-2 e sem sdpa a vazão cai muito; "
                    "ver a docstring de phifm.models.encoder.modelo", atencao)
    modelo.config._attn_implementation = atencao
    modelo = modelo.to(dev)
    log.info("%s · %s parâmetros (%.1f M, %.1f%% embedding) · atenção %s · %s",
             cfg.nome, f"{real:,}", real / 1e6,
             100 * cfg.fracao_de_embedding(), atencao, dev)
    return modelo
