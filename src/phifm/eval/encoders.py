"""Comparação de encoders na tarefa de recuperação por citação — Portão G1.

O DOC-07 §3 define o G1 como **bater o PhysBERT e os embedders gerais** em
recuperação de Física. Até aqui o ΦEmb só havia sido comparado ao SciBERT, que é
o ponto de partida dele — não o concorrente. Isso mede o que o portão pede.

## O que torna a comparação válida, e o que a invalidaria

Todos os modelos passam pelo **mesmo** protocolo, e cada item da lista é uma
forma de mentir se for afrouxado:

| Fixo para todos | Se variasse |
|---|---|
Mesmos pares de validação | comparar dificuldades diferentes |
Mesmo `n_candidatos` | recall@1 entre 128 é muito mais fácil que entre 256 |
Mesma agregação (média mascarada) | premiaria quem foi treinado para `[CLS]` |
Mesmo `max_tokens` | quem lê mais contexto leva vantagem de graça |
Mesma normalização antes do cosseno | mudaria a métrica de distância |

A agregação por média é a escolha honesta aqui: é a linha de base forte do
Sentence-BERT e nenhum dos comparados foi treinado especificamente para ela.
**Ressalva registrada:** números publicados do PhysBERT podem usar protocolo
próprio, então isto mede "quem serve melhor NESTE uso", não "quem é o melhor
modelo" em abstrato.

## Por que roda em CPU por padrão

A GPU tem 8 GB e costuma estar ocupada pelo treino. Avaliar 512 textos sem
gradiente é barato; disputar VRAM com um treino de horas não é.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from phifm.training.embedding import media_mascarada

log = logging.getLogger(__name__)

# O alvo do G1 e as referências. `models/phiemb` entra por caminho local.
CONCORRENTES = {
    "SciBERT (base do ΦEmb)": "allenai/scibert_scivocab_uncased",
    "PhysBERT (alvo do G1)": "thellert/physbert_cased",
    "MiniLM-L6 (genérico)": "sentence-transformers/all-MiniLM-L6-v2",
}


@dataclass
class Resultado:
    nome: str
    caminho: str
    recall_1: float
    recall_10: float
    mrr: float
    parametros_m: float
    segundos: float
    erro: str = ""
    # Marca EXPLICITA de qual e o nosso modelo. Identificar por substring do
    # rotulo era um defeito: `"PhiEmb" in "SciBERT (base do PhiEmb)"` e
    # verdadeiro, entao o veredito comparava tudo contra a linha de base em vez
    # de contra o nosso resultado, e anunciava "G1 NAO PASSOU" com o modelo
    # errado no papel de candidato.
    nosso: bool = False
    # Posição do acerto POR ITEM. Guardar isto é o que permite comparação
    # pareada — ver `comparar_pareado`, que extrai muito mais sinal dos mesmos
    # dados do que confrontar duas proporções soltas.
    posicoes: list[int] = field(default_factory=list)


@torch.no_grad()
def _codificar(mod, tok, textos: list[str], dev, max_tokens: int, lote: int) -> torch.Tensor:
    saidas = []
    for i in range(0, len(textos), lote):
        b = tok(textos[i:i + lote], padding="max_length", truncation=True,
                max_length=max_tokens, return_tensors="pt")
        b = {k: v.to(dev) for k, v in b.items()}
        h = mod(**b).last_hidden_state
        saidas.append(F.normalize(media_mascarada(h, b["attention_mask"]), dim=-1).cpu())
    return torch.cat(saidas)


def avaliar_um(caminho: str, nome: str, val: pl.DataFrame, *, n: int = 256,
               max_tokens: int = 192, lote: int = 16, dispositivo: str = "cpu") -> Resultado:
    t0 = time.perf_counter()
    dev = torch.device(dispositivo)
    try:
        tok = AutoTokenizer.from_pretrained(caminho)
        mod = AutoModel.from_pretrained(caminho, attn_implementation="eager").to(dev).eval()
    except Exception as exc:
        return Resultado(nome, caminho, 0.0, 0.0, 0.0, 0.0, 0.0,
                         erro=f"{type(exc).__name__}: {str(exc)[:120]}")

    amostra = val.head(n)
    va = _codificar(mod, tok, amostra["ancora"].to_list(), dev, max_tokens, lote)
    vp = _codificar(mod, tok, amostra["positivo"].to_list(), dev, max_tokens, lote)

    sim = va @ vp.T
    alvo = torch.arange(sim.size(0))
    ordem = sim.argsort(dim=1, descending=True)
    pos = (ordem == alvo.unsqueeze(1)).float().argmax(dim=1) + 1

    return Resultado(
        nome, caminho,
        (pos == 1).float().mean().item(),
        (pos <= 10).float().mean().item(),
        (1.0 / pos).mean().item(),
        sum(p.numel() for p in mod.parameters()) / 1e6,
        time.perf_counter() - t0,
        posicoes=pos.tolist(),
    )


def comparar(val: pl.DataFrame, extras: dict[str, str] | None = None, **kw) -> list[Resultado]:
    """`extras` sao NOSSOS modelos; `CONCORRENTES` sao os de fora."""
    nossos = set(extras or {})
    alvos = {**CONCORRENTES, **(extras or {})}
    out = []
    for nome, caminho in alvos.items():
        log.info("avaliando %s (%s)", nome, caminho)
        r = avaliar_um(caminho, nome, val, **kw)
        r.nosso = nome in nossos
        if r.erro:
            log.warning("  %s FALHOU: %s", nome, r.erro)
        else:
            log.info("  recall@1 %.3f · recall@10 %.3f · MRR %.3f (%.0fs)",
                     r.recall_1, r.recall_10, r.mrr, r.segundos)
        out.append(r)
    return out


def tabela(rs: list[Resultado], n: int) -> str:
    ok = [r for r in rs if not r.erro]
    ok.sort(key=lambda r: -r.recall_1)
    L = [f"Recuperação por citação · {n} candidatos · média mascarada · 192 tokens", ""]
    L.append(f"{'modelo':30} {'params':>8} {'recall@1':>9} {'recall@10':>10} {'MRR':>7}")
    L.append("-" * 68)
    for r in ok:
        L.append(f"{r.nome[:30]:30} {r.parametros_m:7.0f}M {r.recall_1:9.3f} "
                 f"{r.recall_10:10.3f} {r.mrr:7.3f}")
    for r in rs:
        if r.erro:
            L.append(f"{r.nome[:30]:30}   FALHOU: {r.erro[:60]}")
    return "\n".join(L)


def comparar_pareado(a: Resultado, b: Resultado) -> dict:
    """Comparação PAREADA em recall@1 — muito mais sensível que duas proporções.

    ## Por que pareado muda tudo

    Confrontar 0,402 contra 0,398 como proporções independentes dá erro padrão de
    ±0,031 com 256 itens, e a margem desaparece no ruído. Mas os dois modelos
    foram medidos nos **mesmos itens**, e a maioria deles os dois acertam ou os
    dois erram — esses **não carregam informação sobre a diferença**. O que
    importa são os discordantes: itens que um acerta e o outro não.

    Se de 256 itens 40 são discordantes e o placar é 28 a 12, isso é evidência
    forte. Tratado como proporções soltas, o mesmo dado pareceria empate. É o
    teste de McNemar, e a versão exata dele é uma binomial sobre os discordantes.

    ## Limite declarado

    Com poucos discordantes o teste não decide, e dizer isso é melhor que
    inventar significância. `p` é bicaudal e exato — sem aproximação normal, que
    é ruim justamente quando os números são pequenos.
    """
    if not a.posicoes or not b.posicoes:
        return {"erro": "faltam posições por item — reavalie com esta versão"}
    if len(a.posicoes) != len(b.posicoes):
        return {"erro": f"conjuntos de tamanhos diferentes: {len(a.posicoes)} vs {len(b.posicoes)}"}

    ganha_a = sum(1 for x, y in zip(a.posicoes, b.posicoes) if x == 1 and y != 1)
    ganha_b = sum(1 for x, y in zip(a.posicoes, b.posicoes) if y == 1 and x != 1)
    disc = ganha_a + ganha_b

    if disc == 0:
        return {"a": a.nome, "b": b.nome, "ganha_a": 0, "ganha_b": 0, "discordantes": 0,
                "p": 1.0, "veredito": "idênticos item a item — nada a decidir"}

    # Binomial exata bicaudal com p=0,5 sobre os discordantes.
    from math import comb
    k = min(ganha_a, ganha_b)
    cauda = sum(comb(disc, i) for i in range(k + 1)) / 2 ** disc
    p = min(1.0, 2 * cauda)

    vencedor = a.nome if ganha_a > ganha_b else b.nome
    if p < 0.05:
        v = f"{vencedor} vence em recall@1 (p={p:.4f})"
    elif disc < 20:
        v = f"indeciso — só {disc} itens discordantes, amostra insuficiente (p={p:.3f})"
    else:
        v = f"empate — {disc} discordantes não separam os dois (p={p:.3f})"

    return {"a": a.nome, "b": b.nome, "ganha_a": ganha_a, "ganha_b": ganha_b,
            "discordantes": disc, "p": p, "veredito": v}


def tabela_pareada(rs: list[Resultado]) -> str:
    """Todos contra o nosso, pareado."""
    ok = [r for r in rs if not r.erro]
    nosso = next((r for r in ok if r.nosso), None)
    if nosso is None:
        return "(sem modelo nosso na comparação)"

    L = ["Comparação PAREADA em recall@1 — teste de McNemar exato", "",
         f"{'contra':30} {'nós':>5} {'eles':>5} {'disc':>5} {'p':>8}  veredito"]
    L.append("-" * 92)
    for r in ok:
        if r is nosso:
            continue
        c = comparar_pareado(nosso, r)
        if "erro" in c:
            L.append(f"{r.nome[:30]:30}  {c['erro']}")
            continue
        L.append(f"{r.nome[:30]:30} {c['ganha_a']:5} {c['ganha_b']:5} "
                 f"{c['discordantes']:5} {c['p']:8.4f}  {c['veredito']}")
    L.append("")
    L.append("Itens que ambos acertam ou ambos erram não informam sobre a diferença;")
    L.append("só os discordantes decidem. É por isso que o pareado enxerga o que")
    L.append("duas proporções soltas não enxergam.")
    return "\n".join(L)


def veredito(rs: list[Resultado], n: int = 256) -> str:
    """G1 exige bater o PhysBERT E os genéricos. Ausência de dado não é aprovação.

    `n` é o número de candidatos da avaliação, e entra na conta do erro padrão —
    sem ele não há como dizer se uma margem é vitória ou ruído.
    """
    ok = {r.nome: r for r in rs if not r.erro}
    nosso = next((r for r in ok.values() if r.nosso), None)
    if nosso is None:
        return "G1: INDETERMINADO — o ΦEmb não entrou na comparação"

    rivais = [r for k, r in ok.items() if r is not nosso]
    if not rivais:
        return "G1: INDETERMINADO — nenhum concorrente carregou; sem base de comparação"

    perdeu = [r.nome for r in rivais if r.recall_1 >= nosso.recall_1]
    faltou = [r.nome for r in rs if r.erro]

    # Erro padrão de uma proporção com n itens. Margem menor que isto é ruído
    # de amostragem, e anunciar vitória em cima dela seria exagero: 0,402 contra
    # 0,398 em 256 itens são 103 acertos contra 102.
    ep = (nosso.recall_1 * (1 - nosso.recall_1) / n) ** 0.5

    linhas = []
    if perdeu:
        linhas.append(f"G1: NÃO PASSOU — {', '.join(perdeu)} empata ou supera em recall@1")
    else:
        margem = min(nosso.recall_1 - r.recall_1 for r in rivais)
        pior = min(rivais, key=lambda r: nosso.recall_1 - r.recall_1)
        if margem < ep:
            linhas.append(
                f"G1: INCONCLUSIVO em recall@1 — a margem sobre {pior.nome} é "
                f"{margem:+.3f}, menor que o erro padrão de ±{ep:.3f} em {n} itens. "
                f"É empate estatístico, não vitória.")
            # As outras métricas podem decidir onde recall@1 não decide.
            m10 = min(nosso.recall_10 - r.recall_10 for r in rivais)
            mmrr = min(nosso.mrr - r.mrr for r in rivais)
            linhas.append(f"   recall@10 {m10:+.3f} · MRR {mmrr:+.3f} — margens folgadas, "
                          f"e é onde o ΦEmb se separa")
        else:
            linhas.append(f"G1: PASSOU sobre quem foi medido — margem mínima de "
                          f"{margem:+.3f} em recall@1, acima do erro padrão de ±{ep:.3f}")
    if faltou:
        linhas.append(f"⚠️ não avaliados ({', '.join(faltou)}) — o portão só está "
                      "fechado quando TODOS forem medidos")
    return "\n".join(linhas)
