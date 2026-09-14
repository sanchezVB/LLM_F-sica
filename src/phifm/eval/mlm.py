"""MLM em texto denso em equações — a segunda das três do DOC-05 §11.2.

    PYTHONPATH=src python scripts/avaliar_mlm_phienc.py \\
        --controle models/phienc-controle --tratado models/phienc-tratado \\
        --dados data/processed/phienc_holdout

Esta avaliação é mais sensível ao tratamento que a de recuperação, porque mede
diretamente o que o objetivo de pré-treino otimiza — a probabilidade atribuída aos
tokens mascarados — em vez de uma capacidade emergente medida de viés.

## ⚠️ O desenho 2×2, e por que NENHUM regime sozinho decide

Cada regime de mascaramento favorece um braço **por construção**:

|  | regime ALEATÓRIO (p_equacao=0) | regime EQUAÇÃO (p_equacao=1) |
|---|---|---|
| **controle** | é o objetivo em que ele treinou | tarefa que ele nunca viu |
| **tratado** | metade do orçamento foi para outra coisa | é o objetivo em que ele treinou |

Reportar só o regime de equação seria circular: o braço tratado passou o treino
reconstruindo equações inteiras, então ganhar nisso não é evidência de nada além
de que o treino aconteceu. Reportar só o aleatório penalizaria o tratado pelo
orçamento que ele gastou noutro lugar.

Os dois juntos mostram a **troca**, que é o que a decisão precisa:

- tratado não perde no aleatório **e** ganha no de equação → o tratamento é grátis
  e entrega o que promete;
- tratado perde no aleatório e ganha no de equação → há custo, e ele tem de ser
  comparado ao ganho;
- tratado não ganha no de equação → a hipótese do DOC-07 §2.3 está refutada, e o
  negativo é publicado como o documento promete.

## A célula que NÃO é tautológica

A decomposição por classe de token é onde mora a evidência de mecanismo. Sob o
regime **aleatório** — a tarefa do controle — parte dos tokens sorteados cai
dentro de equações. Se o mascaramento consciente de equações ensina de fato a
relação entre prosa e forma, o braço tratado deve ir melhor **nos tokens de
equação sob mascaramento aleatório**: mesma tarefa, mesmos alvos, e uma classe de
token em que ele não foi treinado de forma privilegiada.

É a única célula do 2×2 em que um ganho não é explicado por "foi nisso que ele
treinou", e é por isso que ela sai separada no artefato.

## Máscaras IDÊNTICAS, por construção e não por disciplina

A máscara é calculada **uma vez** por (sequência, regime) e aplicada aos dois
modelos. Gerar duas vezes com a mesma semente também funcionaria, mas depende de
`mascarar` consumir o gerador na mesma ordem — e uma mudança futura ali quebraria
a igualdade sem nada avisar. Calculada uma vez, os dois braços veem os mesmos
alvos nas mesmas posições porque é o mesmo array.

## ⚠️ O conjunto de avaliação tem de ser DISJUNTO do de treino

O `Fluxo` do pré-treino permuta **todas** as sequências do binário: não existe
divisão de validação. Avaliar sobre `data/processed/phienc_dados` mediria
perplexidade em dado visto, o que não é avaliação nenhuma.

`conferir_disjuncao` compara as listas `partes_usadas` dos dois manifestos e
**levanta** se houver interseção. A disjunção é ao nível de PARTIÇÃO, e portanto
de documento: nenhuma sequência atravessa a fronteira, o que uma divisão por
índice de sequência não garantiria.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from phifm.eval.statistics.proporcao import binomial_exata_bicaudal
from phifm.training.pretrain.dados import NOME_MANIFESTO, ConfigDados, Fluxo
from phifm.training.pretrain.mascaramento import (
    MIN_TOKENS_TRATAMENTO,
    ConfigMascara,
    Contadores,
    mascarar,
)

log = logging.getLogger(__name__)

# Os dois regimes do 2×2. O nome é do que a máscara faz, não de quem ele favorece.
REGIMES = {"aleatorio": 0.0, "equacao": 1.0}

ID_MASK = 4
IDS_ESPECIAIS = frozenset((0, 1, 2, 3, 4))

# Abaixo disto o regime de equação recaiu em aleatório na maior parte das
# sequências, e comparar os braços nele é comparar dois mascaramentos aleatórios.
# Mesma disciplina do piso de `phifm.eval.phienc`, e pela mesma razão: o empate
# que sairia daí não diz nada sobre a hipótese.
MIN_FRACAO_TRATADA = 0.50

RESSALVA = (
    "Cada regime favorece um braço POR CONSTRUÇÃO: o aleatório é o objetivo em que o "
    "controle treinou, o de equação é o do tratado. Nenhum dos dois decide sozinho. A "
    "célula não tautológica é «tokens de equação sob mascaramento ALEATÓRIO», onde a "
    "tarefa é a do controle e a classe de token é a que o tratamento deveria ensinar.")


@dataclass
class Medida:
    """Perda e acerto de um braço num regime, decompostos por classe de token."""

    nome: str
    regime: str
    # Entropia cruzada média nas posições mascaradas. `exp` dela é a
    # pseudo-perplexidade — reportada porque é a unidade em que a literatura de
    # MLM conversa, não porque acrescente informação.
    perda: float
    perplexidade: float
    acerto: float
    tokens_avaliados: int
    # A decomposição que carrega a evidência de mecanismo.
    perda_equacao: float
    acerto_equacao: float
    tokens_equacao: int
    perda_prosa: float
    acerto_prosa: float
    tokens_prosa: int
    # Perda POR SEQUÊNCIA, que é o que permite o teste pareado. Sem isto a
    # comparação seria entre duas médias soltas, e a diferença sumiria no ruído
    # exatamente como sumiu no G1 a 256 candidatos.
    perda_por_sequencia: list[float] = field(default_factory=list)
    # Quantas sequências o regime de equação de fato tratou. Sem isto um regime
    # que recaiu inteiro em aleatório é indistinguível de um que funcionou.
    fracao_tratada: float = 0.0


def conferir_disjuncao(treino: Path, avaliacao: Path) -> dict:
    """Levanta se os dois preparos compartilharem partições.

    ⚠️ Esta é a guarda mais importante do módulo, e a mais fácil de esquecer: o
    binário de avaliação e o de treino têm o mesmo formato e o mesmo nome de
    arquivo. Apontar o script para `phienc_dados` por engano não produz erro
    nenhum — produz perplexidade baixa e um veredito animador sobre dado visto.
    """
    import json

    def _partes(raiz: Path) -> tuple[set[str], dict]:
        caminho = Path(raiz) / NOME_MANIFESTO
        if not caminho.exists():
            raise FileNotFoundError(
                f"{caminho} não existe — sem manifesto não há como saber de que "
                "partições este binário veio, e portanto não há como provar que a "
                "avaliação é disjunta do treino.")
        man = json.loads(caminho.read_text(encoding="utf-8"))
        usadas = man.get("partes_usadas")
        if not usadas:
            raise ValueError(
                f"{caminho} não lista `partes_usadas`. Ele foi gravado por uma versão "
                "do preparador anterior ao registro da LISTA (só a contagem), e com "
                "sorteio saber que 9 partes entraram não permite reconstruir quais — "
                "refaça a preparação para poder provar a disjunção.")
        return set(usadas), man

    a, man_a = _partes(treino)
    b, man_b = _partes(avaliacao)
    comum = sorted(a & b)
    if comum:
        raise ValueError(
            f"treino e avaliação compartilham {len(comum)} partição(ões): "
            f"{comum[:5]}{'…' if len(comum) > 5 else ''}. A perplexidade sairia medida "
            "em dado visto. Prepare o conjunto de avaliação com `--excluir-de` "
            "apontando para o manifesto do treino.")
    return {"particoes_treino": len(a), "particoes_avaliacao": len(b),
            "disjuntas": True,
            "tokenizer_treino": man_a.get("tokenizer"),
            "tokenizer_avaliacao": man_b.get("tokenizer")}


def escolher_sequencias(fluxo: Fluxo, n: int, semente: int) -> list[int]:
    """Sorteia índices de sequência. **Não** é um prefixo.

    ⚠️ `range(n)` seria a sétima ocorrência da armadilha que a §5.2 do artigo
    cataloga: as sequências estão na ordem em que as partições foram lidas, então
    um prefixo é um punhado de partições inteiras — uma amostra por conglomerado
    disfarçada de amostra aleatória.
    """
    if n > fluxo.n_seq:
        raise ValueError(
            f"pedidas {n} sequências e o conjunto tem {fluxo.n_seq}. Medir menos do "
            "que se pede muda o protocolo em silêncio — reduza o `n` explicitamente.")
    rng = np.random.default_rng([semente, 0xA5])
    return sorted(int(i) for i in rng.choice(fluxo.n_seq, size=n, replace=False))


def mascaras_do_regime(fluxo: Fluxo, indices: list[int], regime: str, *,
                       taxa: float, semente: int,
                       n_vocab: int) -> tuple[list[dict], float]:
    """Calcula as máscaras UMA vez. Os dois braços recebem estes mesmos arrays."""
    if regime not in REGIMES:
        raise ValueError(f"regime {regime!r} desconhecido; conhecidos: {sorted(REGIMES)}")

    # ⚠️ Guarda de CONFIGURAÇÃO, achada por um teste que usava contexto 64.
    #
    # `_escolher_equacao` só trata uma equação de display com pelo menos
    # `MIN_TOKENS_TRATAMENTO` tokens que AINDA caiba no orçamento `taxa × contexto`.
    # Com contexto pequeno as duas condições se excluem: a 64 tokens e taxa 0,30 o
    # orçamento é 19 e o mínimo é 20, então **nenhuma equação pode ser escolhida,
    # nunca**. Toda sequência recai em mascaramento aleatório e o regime de equação
    # vira uma cópia do outro — dois regimes idênticos reportados como 2×2.
    #
    # Não há sintoma: a execução termina, os números saem, e a leitura "sem
    # diferença detectável" aparece porque a diferença não foi medida.
    orcamento = int(taxa * fluxo.cfg.contexto)
    if REGIMES[regime] > 0 and orcamento < MIN_TOKENS_TRATAMENTO:
        raise ValueError(
            f"com contexto {fluxo.cfg.contexto} e taxa {taxa}, o orçamento de máscara é "
            f"{orcamento} tokens e `MIN_TOKENS_TRATAMENTO` é {MIN_TOKENS_TRATAMENTO}: "
            "nenhuma equação pode ser tratada e o regime recairia INTEIRO em "
            f"aleatório. Use contexto ≥ {int(MIN_TOKENS_TRATAMENTO / taxa) + 1}.")

    cfg = ConfigMascara(taxa=taxa, p_equacao=REGIMES[regime], semente=semente)
    contadores = Contadores()
    lotes = []
    for ordem, indice in enumerate(indices):
        ids, ide, disp = fluxo.sequencia(indice)
        # Um gerador por sequência, puro em `(semente, regime, ordem)`: repetir a
        # execução reproduz máscara por máscara.
        rng = np.random.default_rng([semente, REGIMES[regime] == 1.0, ordem])
        entrada, alvos = mascarar(ids, ide, disp, cfg=cfg, rng=rng, id_mask=ID_MASK,
                                  n_vocab=n_vocab, ids_especiais=IDS_ESPECIAIS,
                                  contadores=contadores)
        lotes.append({"indice": indice, "entrada": entrada, "alvos": alvos,
                      # Token de equação em DISPLAY é o alvo do tratamento; é por
                      # ele que a decomposição separa, e não por matemática em geral.
                      "e_equacao": (ide >= 0) & disp})
    return lotes, contadores.fracao_tratada()


@torch.no_grad()
def medir(modelo, lotes: list[dict], nome: str, regime: str, fracao_tratada: float,
          *, dispositivo: str = "cpu", lote: int = 1) -> Medida:
    """Roda um braço sobre máscaras já calculadas."""
    dev = torch.device(dispositivo)
    somas = {"todos": [0.0, 0, 0], "equacao": [0.0, 0, 0], "prosa": [0.0, 0, 0]}
    por_sequencia = []

    for i in range(0, len(lotes), lote):
        bloco = lotes[i:i + lote]
        entrada = torch.from_numpy(np.stack([b["entrada"] for b in bloco])).to(dev)
        alvos = torch.from_numpy(np.stack([b["alvos"] for b in bloco])).to(dev)
        eq = torch.from_numpy(np.stack([b["e_equacao"] for b in bloco])).to(dev)

        logits = modelo(input_ids=entrada,
                        attention_mask=torch.ones_like(entrada)).logits
        perdas = F.cross_entropy(logits.transpose(1, 2), alvos,
                                 ignore_index=-100, reduction="none")
        avaliado = alvos != -100
        acertou = (logits.argmax(-1) == alvos) & avaliado

        for s in range(entrada.size(0)):
            m = avaliado[s]
            if not m.any():
                # Sequência sem nada mascarado não informa; incluí-la como perda 0
                # puxaria a média para baixo nos dois braços igualmente, mas
                # inventaria um par concordante no teste.
                continue
            por_sequencia.append(float(perdas[s][m].mean()))
            for classe, sel in (("todos", m),
                                ("equacao", m & eq[s]),
                                ("prosa", m & ~eq[s])):
                if sel.any():
                    somas[classe][0] += float(perdas[s][sel].sum())
                    somas[classe][1] += int(acertou[s][sel].sum())
                    somas[classe][2] += int(sel.sum())

    def _fecha(classe: str) -> tuple[float, float, int]:
        soma, acertos, n = somas[classe]
        return (soma / n if n else float("nan"),
                acertos / n if n else float("nan"), n)

    perda, acerto, n = _fecha("todos")
    p_eq, a_eq, n_eq = _fecha("equacao")
    p_pr, a_pr, n_pr = _fecha("prosa")
    return Medida(nome=nome, regime=regime, perda=perda,
                  perplexidade=float(np.exp(perda)) if n else float("nan"),
                  acerto=acerto, tokens_avaliados=n,
                  perda_equacao=p_eq, acerto_equacao=a_eq, tokens_equacao=n_eq,
                  perda_prosa=p_pr, acerto_prosa=a_pr, tokens_prosa=n_pr,
                  perda_por_sequencia=por_sequencia, fracao_tratada=fracao_tratada)


def comparar_medidas(controle: Medida, tratado: Medida) -> dict:
    """Teste de sinais pareado sobre a perda POR SEQUÊNCIA.

    As duas medidas vêm das mesmas máscaras nas mesmas sequências, então a
    comparação certa é pareada. O sinal é o análogo direto do McNemar que o resto
    do repositório usa: conta em quantas sequências cada braço teve perda menor e
    passa os discordantes pela mesma binomial exata.

    Empate exato de ponto flutuante é concordância e sai da conta — é o que o
    McNemar faz com o par que os dois acertam.
    """
    a, b = controle.perda_por_sequencia, tratado.perda_por_sequencia
    if len(a) != len(b):
        return {"erro": f"sequências diferentes: {len(a)} contra {len(b)}"}
    if not a:
        return {"erro": "nenhuma sequência com posição mascarada"}

    ganha_c = sum(1 for x, y in zip(a, b, strict=True) if x < y)
    ganha_t = sum(1 for x, y in zip(a, b, strict=True) if y < x)
    d = binomial_exata_bicaudal(ganha_c, ganha_t)
    d.update({"controle_melhor": ganha_c, "tratado_melhor": ganha_t,
              "delta_perda": tratado.perda - controle.perda,
              "delta_perda_equacao": tratado.perda_equacao - controle.perda_equacao,
              "delta_perda_prosa": tratado.perda_prosa - controle.perda_prosa})
    return d


def _leitura(regime: str, cmp: dict, fracao_tratada: float = 1.0) -> str:
    if REGIMES[regime] > 0 and fracao_tratada < MIN_FRACAO_TRATADA:
        return (f"INCONCLUSIVO — só {fracao_tratada:.1%} das sequências foram de "
                "fato tratadas; o regime recaiu em mascaramento aleatório e "
                "compararia os braços na tarefa do outro regime")
    if cmp.get("erro"):
        return f"INCONCLUSIVO — {cmp['erro']}"
    if cmp["discordantes"] < 20:
        return (f"INCONCLUSIVO — {cmp['discordantes']} sequências discordantes; "
                "aumente `--sequencias`")
    if cmp["p"] >= 0.05:
        return f"sem diferença detectável (p={cmp['p']:.3f})"
    melhor = "tratado" if cmp["tratado_melhor"] > cmp["controle_melhor"] else "controle"
    if regime == "aleatorio":
        return (f"no objetivo do CONTROLE, o {melhor} tem perda menor "
                f"(p={cmp['p']:.4f}) — é a checagem de custo, não de benefício")
    return (f"no objetivo do TRATADO, o {melhor} tem perda menor (p={cmp['p']:.4f}) "
            "— favorece o tratado por construção; ler junto com o outro regime")


def avaliar_ablacao(modelos: dict, fluxo: Fluxo, *, sequencias: int = 64,
                    taxa: float = 0.30, semente: int = 17, n_vocab: int = 40_960,
                    dispositivo: str = "cpu", lote: int = 1) -> dict:
    """O 2×2 completo. `modelos` é `{"controle": mod, "tratado": mod}`."""
    faltando = {"controle", "tratado"} - set(modelos)
    if faltando:
        raise ValueError(f"faltam os braços {sorted(faltando)}")

    indices = escolher_sequencias(fluxo, sequencias, semente)
    log.info("%d sequências sorteadas de %d disponíveis", len(indices), fluxo.n_seq)

    celulas: dict[str, dict[str, Medida]] = {}
    comparacoes, leituras = {}, {}
    for regime in REGIMES:
        lotes, ft = mascaras_do_regime(fluxo, indices, regime, taxa=taxa,
                                       semente=semente, n_vocab=n_vocab)
        celulas[regime] = {
            nome: medir(mod, lotes, nome, regime, ft, dispositivo=dispositivo, lote=lote)
            for nome, mod in modelos.items()}
        comparacoes[regime] = comparar_medidas(celulas[regime]["controle"],
                                               celulas[regime]["tratado"])
        leituras[regime] = _leitura(regime, comparacoes[regime], ft)
        log.info("regime %s · fracao_tratada %.3f · %s", regime, ft, leituras[regime])

    # A célula de mecanismo: tokens de equação sob mascaramento ALEATÓRIO.
    aleat = celulas["aleatorio"]
    mecanismo = {
        "controle_perda_equacao": aleat["controle"].perda_equacao,
        "tratado_perda_equacao": aleat["tratado"].perda_equacao,
        "delta": aleat["tratado"].perda_equacao - aleat["controle"].perda_equacao,
        "tokens": aleat["controle"].tokens_equacao,
        "o_que_significa": (
            "perda MENOR no tratado aqui é a evidência de mecanismo: mesma tarefa do "
            "controle, e a classe de token que o tratamento deveria ensinar. É a única "
            "célula do 2×2 cujo ganho não se explica por «foi nisso que ele treinou»."),
    }

    return {
        "tarefa": "mlm_em_texto_denso_em_equacoes",
        "protocolo": {"sequencias": len(indices), "contexto": fluxo.cfg.contexto,
                      "taxa_de_mascara": taxa, "semente": semente,
                      "dispositivo": dispositivo,
                      "mascaras": "calculadas uma vez e aplicadas aos dois braços"},
        "celulas": {r: {n: m.__dict__ for n, m in c.items()} for r, c in celulas.items()},
        "pareado": comparacoes,
        "leitura_por_regime": leituras,
        "mecanismo": mecanismo,
        "ressalva": RESSALVA,
    }


def carregar_fluxo(dados: Path, contexto: int | None = None) -> Fluxo:
    """Abre o conjunto de avaliação com o contexto do próprio preparo."""
    import json

    man = json.loads((Path(dados) / NOME_MANIFESTO).read_text(encoding="utf-8"))
    ctx = contexto or int(man.get("contexto", 8_192))
    return Fluxo(ConfigDados(raiz=Path(dados), contexto=ctx, sequencias=1))
