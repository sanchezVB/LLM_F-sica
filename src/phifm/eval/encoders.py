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

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from phifm.training.embedding import media_mascarada

log = logging.getLogger(__name__)

# O alvo do G1 e as referências. `models/phiemb` entra por caminho local.
#
# ⚠️ O G1.2 exige superar «o melhor embedder GERAL com ≤ 1/10 dos parâmetros».
# A cláusula de tamanho é relativa ao RIVAL, então bater o MiniLM-L6 de 23M com
# um modelo de 23M não a satisfaz — ela só fecha contra um rival de ≥ 230M. Por
# isso entra um genérico FORTE de 335M: é ele quem torna o critério verificável.
#
# `gte-large` e não `e5-large`/`bge-large` porque estes dois pedem prefixo
# («query: » / «Represent this sentence…») para render o que rendem. Usá-los sem
# prefixo os handicapa e inflaria o nosso resultado; usá-los com prefixo quebra
# o «mesma entrada para todos». O gte não precisa de nenhum.
CONCORRENTES = {
    "SciBERT (base do ΦEmb)": "allenai/scibert_scivocab_uncased",
    "PhysBERT (alvo do G1.1)": "thellert/physbert_cased",
    "MiniLM-L6 (genérico 23M)": "sentence-transformers/all-MiniLM-L6-v2",
    "GTE-large (genérico 335M)": "thenlper/gte-large",
}

# Parâmetros do rival ≥ 10× os nossos é o que o G1.2 pede.
RAZAO_G1_2 = 10.0
# 5 pontos de nDCG@10 é o limiar do G1.1 (DOC-00 §5).
MARGEM_G1_1 = 0.05
# Quem é quem nos critérios. Sem isto o veredito teria de adivinhar pelo rótulo,
# que foi exatamente o defeito que fez «SciBERT (base do ΦEmb)» ser tomado pelo
# nosso modelo.
ALVO_DOMINIO = "thellert/physbert_cased"
GENERICOS = frozenset({
    "sentence-transformers/all-MiniLM-L6-v2",
    "thenlper/gte-large",
})


@dataclass
class Resultado:
    nome: str
    caminho: str
    recall_1: float
    recall_10: float
    mrr: float
    # A métrica que o G1.1 nomeia. Com UM documento relevante por consulta o
    # nDCG@10 se reduz a 1/log2(1+posição) quando a posição é ≤ 10 e a 0 depois:
    # o ganho ideal é um único acerto, então o denominador (IDCG) vale 1. Sai de
    # graça das posições — e sem ela o portão estava sendo julgado por proxy.
    ndcg_10: float
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
        return Resultado(nome, caminho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         erro=f"{type(exc).__name__}: {str(exc)[:120]}")

    amostra = val.head(n)
    va = _codificar(mod, tok, amostra["ancora"].to_list(), dev, max_tokens, lote)
    vp = _codificar(mod, tok, amostra["positivo"].to_list(), dev, max_tokens, lote)

    sim = va @ vp.T
    alvo = torch.arange(sim.size(0))
    ordem = sim.argsort(dim=1, descending=True)
    pos = (ordem == alvo.unsqueeze(1)).float().argmax(dim=1) + 1

    # nDCG@10 com um único relevante: ganho descontado só se caiu no top-10.
    dcg = torch.where(pos <= 10, 1.0 / torch.log2(pos.float() + 1.0),
                      torch.zeros_like(pos, dtype=torch.float))

    return Resultado(
        nome, caminho,
        (pos == 1).float().mean().item(),
        (pos <= 10).float().mean().item(),
        (1.0 / pos).mean().item(),
        dcg.mean().item(),
        sum(p.numel() for p in mod.parameters()) / 1e6,
        time.perf_counter() - t0,
        posicoes=pos.tolist(),
    )


def _digesto(val: pl.DataFrame, n: int) -> str:
    """Impressão digital da amostra avaliada.

    O cache só vale se a amostra for a MESMA. Comparar um modelo medido nos
    pares de ontem com outro medido nos de hoje seria o pior tipo de erro: a
    tabela pareceria válida e não seria.
    """
    import hashlib
    h = hashlib.sha256()
    for c in ("ancora", "positivo"):
        for s in val.head(n)[c].to_list():
            h.update(s.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def comparar(val: pl.DataFrame, extras: dict[str, str] | None = None,
             cache: Path | None = None, **kw) -> list[Resultado]:
    """`extras` sao NOSSOS modelos; `CONCORRENTES` sao os de fora.

    ## Por que existe cache

    Uma passada a 2.000 candidatos custa ~57 min de CPU, e a maior parte é gasta
    remedindo modelos que não mudaram — o PhysBERT sozinho leva 22 min. Somar UM
    modelo à comparação não deve custar a soma de todos.

    O cache é invalidado inteiro se `n`, `max_tokens` ou o **digesto da amostra**
    mudarem, porque nesse caso os números deixam de ser comparáveis entre si. Um
    cache que mistura protocolos é pior que nenhum: ele produz uma tabela que
    parece válida.
    """
    nossos = set(extras or {})
    alvos = {**CONCORRENTES, **(extras or {})}
    n, mt = kw.get("n", 256), kw.get("max_tokens", 192)
    dig = _digesto(val, n)

    guardado: dict[str, dict] = {}
    if cache and cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        if (d.get("digesto"), d.get("n"), d.get("max_tokens")) == (dig, n, mt):
            guardado = d.get("resultados", {})
            log.info("cache: %d modelo(s) já medidos no mesmo protocolo", len(guardado))
        else:
            log.info("cache descartado — protocolo diferente (n/max_tokens/amostra)")

    out = []
    for nome, caminho in alvos.items():
        g = guardado.get(str(caminho))
        if g and not g.get("erro"):
            r = Resultado(**{**g, "nome": nome})
            log.info("%s · do cache: recall@1 %.3f · MRR %.3f", nome, r.recall_1, r.mrr)
        else:
            log.info("avaliando %s (%s)", nome, caminho)
            r = avaliar_um(caminho, nome, val, **kw)
            if r.erro:
                log.warning("  %s FALHOU: %s", nome, r.erro)
            else:
                log.info("  recall@1 %.3f · recall@10 %.3f · MRR %.3f (%.0fs)",
                         r.recall_1, r.recall_10, r.mrr, r.segundos)
        r.nosso = nome in nossos
        out.append(r)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(
            {"digesto": dig, "n": n, "max_tokens": mt,
             "resultados": {r.caminho: asdict(r) for r in out if not r.erro}},
            ensure_ascii=False), encoding="utf-8")
    return out


def salvar(rs: list[Resultado], destino: Path, n: int) -> dict:
    """Grava o RESULTADO, que é coisa diferente do cache.

    O cache guarda posições por item e é chaveado pelo protocolo: existe para não
    remedir, e é descartado quando o protocolo muda. Isto é a evidência das
    afirmações do ESTADO.md, e por isso é versionado.

    Existe porque o resultado morava só no log de quem lançou o script — e o log
    do lançador vive em `data/raw/`, que é ignorado. Script que produz afirmação
    tem de ser dono do artefato que a sustenta.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    d = {
        "n_candidatos": n,
        "modelos": [{k: v for k, v in asdict(r).items() if k != "posicoes"} for r in rs],
        "pareado": [comparar_pareado(campeao(rs), r)
                    for r in rs if not r.erro and r is not campeao(rs)]
        if campeao(rs) else [],
        "veredito": veredito(rs, n),
    }
    destino.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("resultado → %s", destino)
    return d


def campeao(rs: list[Resultado]) -> Resultado | None:
    """O MELHOR dos nossos, não o primeiro.

    Com um só modelo nosso `next(...)` bastava. Com dois — o ΦEmb sobre SciBERT e
    o sobre MiniLM — ele passa a devolver o que a ordem do dicionário entregar,
    e o portão seria julgado por um modelo escolhido ao acaso.

    ## ⚠️ O critério é nDCG@10 porque é o do PORTÃO

    A primeira correção deste defeito trocou "o primeiro" por "o melhor" e deixou
    o critério em `recall@1` — que o G1 não menciona. Meia correção.

    Medido em 2026-08-16, com três modelos nossos:

        modelo                          nDCG@10   recall@1
        ΦEmb/MiniLM (127 neg)             0,458      0,254
        ΦEmb/MiniLM+GC (511 neg)          0,449      0,257   <- eleito por recall@1

    O modelo do GradCache ganhava em recall@1 e perdia em nDCG@10, então virou
    campeão e o portão foi julgado pelo PIOR dos nossos na métrica que decide. O
    veredito saiu com G1.2 a −0,014 quando o nosso melhor dá −0,005.

    Escolher campeão por uma métrica e julgar por outra é o mesmo erro que julgar
    o portão por recall@1 quando ele pede nDCG@10 — e eu já tinha corrigido esse.
    """
    nossos = [r for r in rs if r.nosso and not r.erro]
    return max(nossos, key=lambda r: r.ndcg_10) if nossos else None


def tabela(rs: list[Resultado], n: int) -> str:
    ok = [r for r in rs if not r.erro]
    # Ordena por nDCG@10, que é a métrica do portão.
    ok.sort(key=lambda r: -r.ndcg_10)
    L = [f"Recuperação por citação · {n} candidatos · média mascarada · 192 tokens", ""]
    L.append(f"{'modelo':30} {'params':>8} {'nDCG@10':>8} {'recall@1':>9} "
             f"{'recall@10':>10} {'MRR':>7}")
    L.append("-" * 78)
    for r in ok:
        L.append(f"{r.nome[:30]:30} {r.parametros_m:7.0f}M {r.ndcg_10:8.3f} "
                 f"{r.recall_1:9.3f} {r.recall_10:10.3f} {r.mrr:7.3f}")
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

    ganha_a = sum(1 for x, y in zip(a.posicoes, b.posicoes, strict=True) if x == 1 and y != 1)
    ganha_b = sum(1 for x, y in zip(a.posicoes, b.posicoes, strict=True) if y == 1 and x != 1)
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
    """Todos contra o MELHOR dos nossos, pareado."""
    ok = [r for r in rs if not r.erro]
    nosso = campeao(rs)
    if nosso is None:
        return "(sem modelo nosso na comparação)"

    L = [f"Comparação PAREADA em recall@1 — McNemar exato · nosso = {nosso.nome}", "",
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
    """Julga G1.1 e G1.2 pela redação do DOC-00 §5, não por proxy.

    | | Exigência | Métrica |
    |---|---|---|
    G1.1 | superar o PhysBERT em ≥ 5 pontos | **nDCG@10** |
    G1.2 | superar o melhor embedder GERAL com ≤ 1/10 dos parâmetros | nDCG@10 + params |

    Duas armadilhas que este código evita de propósito:

    **A métrica.** Antes eu julgava por recall@1, que o portão não menciona. Dá
    para acertar o veredito por sorte e errar o critério — e num portão o critério
    É o resultado.

    **A cláusula de tamanho é relativa ao RIVAL.** Superar um genérico de 23M com
    um modelo de 23M não fecha o G1.2: a razão precisa ser ≤ 1/10, o que exige um
    rival de ≥ 230M. Tratar «bati o MiniLM» como G1.2 fechado seria satisfazer o
    critério mais fácil e declarar o mais difícil.
    """
    ok = [r for r in rs if not r.erro]
    nosso = campeao(rs)
    if nosso is None:
        return "G1: INDETERMINADO — nenhum modelo nosso entrou na comparação"

    L = [f"nosso candidato: {nosso.nome} · {nosso.parametros_m:.0f}M params · "
         f"nDCG@10 {nosso.ndcg_10:.3f}", ""]

    # ── G1.1 · contra o competidor de mesmo domínio ────────────────────────
    alvo = next((r for r in ok if r.caminho == ALVO_DOMINIO), None)
    if alvo is None:
        L.append("G1.1: INDETERMINADO — o PhysBERT não carregou; sem dado não há aprovação")
    else:
        # Arredonda para as 3 casas EXIBIDAS antes de comparar. Sem isto o
        # relatório se contradiz na fronteira: `0.30 + 0.05` em float é
        # 0.34999999999999998, a diferença sai 0.049999999999999996, e o texto
        # imprime «+0.050 … NÃO PASSOU». Quem lê não tem como saber que a causa
        # é 1e-17. Julgar na precisão que se mostra elimina a contradição.
        d = round(nosso.ndcg_10 - alvo.ndcg_10, 3)
        pareado = comparar_pareado(nosso, alvo)
        st = "PASSOU" if d >= MARGEM_G1_1 else "NÃO PASSOU"
        L.append(f"G1.1: {st} — nDCG@10 {d:+.3f} sobre o PhysBERT "
                 f"(limiar +{MARGEM_G1_1:.2f}) · pareado em recall@1: {pareado['veredito']}")

    # ── G1.2 · contra o melhor genérico, com a cláusula de tamanho ─────────
    gs = [r for r in ok if r.caminho in GENERICOS]
    if not gs:
        L.append("G1.2: INDETERMINADO — nenhum embedder geral carregou")
    else:
        melhor = max(gs, key=lambda r: r.ndcg_10)
        razao = melhor.parametros_m / max(nosso.parametros_m, 1e-9)
        bate = nosso.ndcg_10 > melhor.ndcg_10
        cabe = razao >= RAZAO_G1_2
        if bate and cabe:
            st = "PASSOU"
        elif bate:
            st = "PARCIAL (vence, mas a razão de tamanho não fecha)"
        else:
            st = "NÃO PASSOU"
        L.append(f"G1.2: {st} — melhor genérico é {melhor.nome} "
                 f"({melhor.parametros_m:.0f}M, nDCG@10 {melhor.ndcg_10:.3f}); "
                 f"nDCG@10 {nosso.ndcg_10 - melhor.ndcg_10:+.3f}, "
                 f"razão de params 1/{razao:.1f} "
                 f"({'≤' if cabe else '>'} 1/{RAZAO_G1_2:.0f} exigido)")
        if bate and not cabe:
            L.append(f"     ⚠️ vencer um genérico de {melhor.parametros_m:.0f}M com "
                     f"{nosso.parametros_m:.0f}M não satisfaz a cláusula: ela pede rival "
                     f"de ≥ {RAZAO_G1_2 * nosso.parametros_m:.0f}M.")

    # ── ressalvas que impedem tratar isto como o portão fechado ────────────
    faltou = [r.nome for r in rs if r.erro]
    if faltou:
        L.append(f"⚠️ não avaliados: {', '.join(faltou)}")
    L += ["",
          "RESSALVAS que mantêm o G1 aberto mesmo com G1.1 e G1.2 verdes:",
          "  · benchmark PRÓPRIO (pares de citação), não um reservado e publicado",
          "  · G1.3 (ΦEnc em classificação/NER), G1.4 (ΦOCR) e G1.5 (reprodutibilidade)",
          "    não são tocados por esta medição",
          "  · nDCG@10 aqui tem UM relevante por consulta, então é 1/log2(1+pos);",
          "    um benchmark com julgamentos graduados daria outro número"]
    return "\n".join(L)
