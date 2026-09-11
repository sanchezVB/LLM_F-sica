#!/usr/bin/env python3
"""Converte um checkpoint do ΦEnc em um diretório que `from_pretrained` carrega.

    .venv-treino\\Scripts\\python.exe scripts\\exportar_phienc.py \\
        --run data/processed/phienc_pretreino \\
        --para models/phienc-variante-A

## O bloqueio que isto resolve

O `laco.py` grava `torch.save({"passo": ..., "modelo": state_dict, "opt": ...})`.
É o formato certo para **retomar** o treino, e é inútil para avaliar: um
`state_dict` cru não carrega estrutura, nem configuração, nem tokenizer. Nenhum
dos três avaliadores do DOC-05 §11.2 consegue abrir isso —
`AutoModelForMaskedLM.from_pretrained` precisa de um `config.json` ao lado dos
pesos.

Sem este passo, treinar o ΦEnc produz um arquivo que só o próprio laço entende.

## ⚠️ A configuração vem do RUN, nunca de um nome na linha de comando

`--config PHIENC_150M` seria a interface óbvia e é a errada. Uma constante do
`config.py` pode ter mudado depois do treino; se a mudança não altera nenhuma
forma de tensor — janela local, `global_a_cada`, `contexto`, o próprio nome —, o
`load_state_dict` aceita tudo e o modelo exportado **não é o modelo treinado**.
Os pesos seriam os certos e o comportamento outro.

Então a configuração é reconstruída do `phienc.json` que o próprio run gravou. É a
mesma lição da `assinatura_do_manifesto`: um artefato internamente consistente não
prova ser o artefato certo, e a única defesa é a proveniência vir junto com ele.

## ⚠️ O tokenizer errado não parece um erro: parece um modelo ruim

O §11.2 treina seis variantes que diferem **só** no tokenizer, e três delas têm
vocabulários de tamanhos diferentes (A/B/E com 40.960, C com 32.768, D com
65.536). Exportar a variante C com o tokenizer de A dá ids fora de faixa ou, pior,
ids válidos que significam outra coisa: o MLM despenca e o bake-off registra "o
tokenizer C é ruim".

Aqui o tokenizer sai do manifesto de dados do próprio run, e o tamanho do
vocabulário é **conferido** contra a configuração. Divergência recusa a exportação.

## ⚠️ Round-trip conferido, não presumido

Depois de gravar, o diretório é reaberto e conferido de duas formas: **os pesos
tensor a tensor** (garantia forte, independente de kernel) e **os logits sobre uma
entrada fixa** (pega o que a igualdade de pesos não pega — um campo de
configuração que muda comportamento sem mudar forma nenhuma, como
`local_attention` ou `contexto`). As duas exigem igualdade exata.

Custa segundos e é o único jeito de a exportação ser um fato em vez de uma
suposição — `save_pretrained` seguido de `from_pretrained` pode perder um tensor
amarrado, mudar dtype ou reconstruir com outra configuração sem dizer nada, e o
sintoma disso é uma métrica um pouco pior, que é indistinguível de um resultado.

E a conferência usa a **mesma implementação de atenção** nos dois lados: medido
aqui em 2026-09-08, `eager` contra `sdpa` dá 1,2e-07 de diferença nos MESMOS pesos
— um ulp de float32, da ordem das somas no kernel. Ao ver esse número a tentação é
afrouxar a tolerância, e isso destruiria o teste: 1e-6 de folga também esconde um
tensor amarrado que voltou solto.

## ⚠️ Run que não terminou, ou que teve spike, não sai sem `--mesmo-assim`

Num bake-off de seis variantes, uma que divergiu e foi exportada em silêncio é
lida como "esse tokenizer é pior". A recusa é para que a diferença entre *modelo
ruim* e *treino que quebrou* não se perca entre o `.pt` e a tabela final.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from dataclasses import fields
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForMaskedLM, PreTrainedTokenizerFast

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8 as console_utf8  # noqa: E402
from phifm.core.schema.reprodutibilidade import (  # noqa: E402
    Entrada,
    gravar_manifesto_etapa,
    hash_arquivo,
)
from phifm.models.encoder.config import ESPECIAIS, ConfigEnc  # noqa: E402
from phifm.models.encoder.modelo import construir  # noqa: E402
from phifm.training.pretrain.laco import NOME_ESTADO, NOME_METRICAS  # noqa: E402

# ⚠️ No IMPORT, e não dentro do `main()`: o argparse imprime `--help` antes
# de qualquer código nosso, e `Φ` não existe em cp1252. Ver `phifm.core.console`.
console_utf8()


log = logging.getLogger("exportar_phienc")

# Os nomes dos especiais como o `transformers` os quer, mapeados dos ids que o
# `ESPECIAIS` fixa. O tokenizer treinado não guarda esses papéis — ele guarda os
# tokens —, e sem o mapeamento o `PreTrainedTokenizerFast` não sabe qual id é
# `[MASK]`, o que quebraria qualquer avaliação de MLM.
PAPEIS = {"pad_token": "pad_token_id", "unk_token": "unk",
          "cls_token": "cls_token_id", "sep_token": "sep_token_id",
          "mask_token": "mask"}


def config_do_run(manifesto: dict) -> ConfigEnc:
    """Reconstrói a `ConfigEnc` do que o run gravou. Ver a docstring do módulo.

    `como_dict()` acrescenta `parametros` e `fracao_de_embedding`, que são
    derivados e não são campos — passá-los ao construtor levantaria `TypeError`.
    """
    d = manifesto.get("modelo")
    if not isinstance(d, dict):
        raise SystemExit(
            f"{NOME_METRICAS} não tem a chave 'modelo' com a configuração do "
            "encoder. Sem ela não há como saber que modelo estes pesos são, e "
            "adivinhar por um nome é exatamente o que este script recusa.")
    validos = {f.name for f in fields(ConfigEnc)}
    # Campo obrigatório ausente levanta `TypeError` nomeando qual — melhor que
    # uma mensagem própria, porque nomeia o campo sem precisar mantê-la.
    return ConfigEnc(**{k: v for k, v in d.items() if k in validos})


def carregar_tokenizer(caminho: Path, vocab_esperado: int) -> PreTrainedTokenizerFast:
    """Embrulha o tokenizer treinado e CONFERE o tamanho do vocabulário.

    Ver o §"o tokenizer errado não parece um erro" na docstring do módulo.
    """
    if not caminho.exists():
        raise SystemExit(
            f"tokenizer ausente: {caminho}\nÉ o que o manifesto de dados do run "
            "nomeia. Exportar com outro mudaria o significado dos ids.")
    tok = Tokenizer.from_file(str(caminho))
    n = tok.get_vocab_size()
    if n != vocab_esperado:
        raise SystemExit(
            f"o tokenizer {caminho.name} tem {n:,} tokens e a configuração do "
            f"modelo declara vocab={vocab_esperado:,}.\n"
            "Não é um detalhe de forma: a tabela de embeddings tem o tamanho da "
            "configuração, então ids acima dela levantam e ids abaixo apontam "
            "para outro token. O §11.2 tem variantes de 32.768, 40.960 e 65.536 "
            "— este é o erro que faz um tokenizer bom parecer ruim.")
    especiais = {}
    for papel, chave in PAPEIS.items():
        t = tok.id_to_token(ESPECIAIS[chave])
        if t is None:
            raise SystemExit(
                f"o id {ESPECIAIS[chave]} ({papel}) não existe em {caminho.name}; "
                "o tokenizer não é o que este projeto treinou")
        especiais[papel] = t
    # ⚠️ `model_input_names` sem `token_type_ids`, e não é cosmético.
    #
    # O default do `PreTrainedTokenizerFast` inclui `token_type_ids`, e o
    # `ModernBertModel.forward()` NÃO aceita esse argumento. Qualquer consumidor
    # que faça `mod(**tok(texto))` — o `avaliar_encoders.py` faz, e é o avaliador
    # de recuperação do §11.2 — morre com `TypeError`.
    #
    # Pego em 2026-09-08 pelo teste de ponta a ponta num modelo minúsculo, o que
    # é o lugar barato de pegar: o caro seria descobrir depois das 44 h de T4 de
    # uma variante. O tokenizer do ModernBERT publicado declara os mesmos dois.
    envolvido = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        model_input_names=["input_ids", "attention_mask"],
        **especiais)
    log.info("tokenizer %s · %s tokens · %s", caminho.name, f"{n:,}",
             " ".join(f"{p}={t}" for p, t in especiais.items()))
    return envolvido


def conferir_saude(manifesto: dict, mesmo_assim: bool) -> list[str]:
    """Ressalvas do treino. Ver o §"run que não terminou" na docstring.

    Devolve a lista para o manifesto de etapa: exportar com `--mesmo-assim` é
    legítimo (uma variante interrompida por cota ainda pode valer uma medição),
    mas o artefato tem de carregar o motivo junto.
    """
    ressalvas = []
    if not manifesto.get("concluido"):
        passo = (manifesto.get("metricas") or {}).get("passo")
        ressalvas.append(
            f"treino NÃO concluído (parou no passo {passo}); "
            f"motivo registrado: {manifesto.get('motivo_da_parada') or 'nenhum'}")
    n = (manifesto.get("spike") or {}).get("n_spikes") or 0
    if n:
        ressalvas.append(
            f"{n} spike(s) de perda detectado(s); num bake-off isso pode ser lido "
            "como 'este tokenizer é pior' quando o que houve foi treino instável")
    if ressalvas and not mesmo_assim:
        raise SystemExit(
            "recusando exportar:\n  - " + "\n  - ".join(ressalvas) +
            "\n\nUse --mesmo-assim se a medição vale de qualquer forma; as "
            "ressalvas vão para o manifesto do artefato.")
    for r in ressalvas:
        log.warning("RESSALVA: %s", r)
    return ressalvas


def conferir_ida_e_volta(modelo, destino: Path, contexto: int,
                         atencao: str) -> float:
    """Reabre o que foi gravado e confere DUAS coisas. Ver a docstring do módulo.

    1. **Os pesos, tensor a tensor.** É a garantia forte e não depende de kernel.
    2. **Os logits sobre uma entrada fixa.** Pega o que a igualdade de pesos não
       pega: um campo de configuração que muda o comportamento sem mudar forma
       nenhuma — `local_attention`, `global_attn_every_n_layers`, o `contexto`.

    ## ⚠️ A mesma implementação de atenção nos dois lados

    `from_pretrained` sem `attn_implementation` escolhe `sdpa`, e comparar `sdpa`
    contra `eager` dá 1,2e-07 de diferença — um ulp de float32, vindo da ordem das
    somas no kernel, não dos pesos. Medido aqui em 2026-09-08.

    A tentação, ao ver esse número, é afrouxar a tolerância. Isso destruiria o
    teste: 1e-6 de folga também esconde um tensor amarrado que voltou solto. A
    correção é igualar o kernel e **manter o zero exato**.
    """
    n = min(64, contexto)
    g = torch.Generator().manual_seed(17)
    ids = torch.randint(0, modelo.config.vocab_size, (2, n), generator=g)
    mascara = torch.ones_like(ids)
    modelo = modelo.to("cpu").eval()
    de_volta = AutoModelForMaskedLM.from_pretrained(
        destino, attn_implementation=atencao).to("cpu").eval()

    antes, depois = modelo.state_dict(), de_volta.state_dict()
    faltando = set(antes) - set(depois)
    if faltando:
        raise SystemExit(
            f"{len(faltando)} tensor(es) não voltaram do disco: "
            f"{sorted(faltando)[:5]}\nUm tensor amarrado que volta solto — ou o "
            "contrário — muda o modelo sem mudar a forma de nada.")
    for k, v in antes.items():
        if not torch.equal(v, depois[k]):
            raise SystemExit(
                f"o tensor {k} mudou de valor na gravação "
                f"({v.dtype} -> {depois[k].dtype}). O artefato não é o "
                "checkpoint.")

    with torch.no_grad():
        a = modelo(input_ids=ids, attention_mask=mascara).logits
        b = de_volta(input_ids=ids, attention_mask=mascara).logits
    if a.shape != b.shape:
        raise SystemExit(f"forma dos logits mudou na ida e volta: {a.shape} vs "
                         f"{b.shape}")
    delta = float((a - b).abs().max())
    if delta != 0.0:
        raise SystemExit(
            f"os pesos são idênticos mas os logits do modelo reaberto diferem em "
            f"{delta:.3e}.\nCom os mesmos pesos, o mesmo kernel e a mesma entrada, "
            "zero é o único valor possível — então algum campo da configuração "
            "gravada muda o comportamento sem mudar forma nenhuma. Compare o "
            "`config.json` do destino com `config_do_encoder` em "
            "`phienc_exportado.json`.")
    log.info("ida e volta conferida: %d tensores idênticos, diferença máxima de "
             "logits = 0 (atenção %s nos dois lados)", len(antes), atencao)
    return delta


def conferir_tokenizer_com_modelo(destino: Path, atencao: str) -> None:
    """⚠️ `modelo(**tokenizer(texto))` tem de FUNCIONAR, e isso é uma terceira coisa.

    Pesos idênticos e logits idênticos não dizem nada sobre a compatibilidade
    entre o tokenizer gravado e o modelo gravado. Em 2026-09-08 o
    `PreTrainedTokenizerFast` embrulhado declarava `token_type_ids` por default e
    o `ModernBertModel.forward()` não aceita esse argumento: o artefato passava
    em toda conferência de pesos e morria com `TypeError` na primeira linha do
    avaliador de recuperação.

    Então a conferência é feita como o consumidor faz — `AutoModel`, não
    `AutoModelForMaskedLM`, porque é assim que o `avaliar_encoders.py` carrega
    (ele quer o corpo, não a cabeça de MLM).
    """
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(destino)
    corpo = AutoModel.from_pretrained(
        destino, attn_implementation=atencao).to("cpu").eval()
    lote = tok(["a energia cinética vale mv^2/2", "o tensor de Riemann"],
               padding=True, truncation=True, max_length=32,
               return_tensors="pt")
    try:
        with torch.no_grad():
            saida = corpo(**lote)
    except TypeError as e:
        raise SystemExit(
            f"o modelo não aceita o que o tokenizer produz: {e}\n"
            f"chaves do tokenizer: {sorted(lote)}\n"
            "É o defeito de 2026-09-08 (`token_type_ids`): o artefato passa em "
            "toda conferência de pesos e morre na primeira linha de quem for "
            "usá-lo. Ajuste `model_input_names` do tokenizer exportado.") from e
    h = getattr(saida, "last_hidden_state", None)
    if h is None:
        raise SystemExit(
            "a saída do corpo não tem `last_hidden_state` — é o que a média "
            "mascarada do avaliador do G1 consome")
    log.info("tokenizer × modelo conferidos: %s -> %s", sorted(lote),
             tuple(h.shape))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True,
                   help=f"diretório com {NOME_ESTADO} e {NOME_METRICAS}")
    p.add_argument("--para", type=Path, required=True,
                   help="destino do diretório no formato do `transformers`")
    p.add_argument("--tokenizer", type=Path, default=None,
                   help="padrão: o que o manifesto de dados do run nomeia")
    p.add_argument("--mesmo-assim", action="store_true",
                   help="exporta apesar de treino inacabado ou spikes")
    p.add_argument("--nota", default="",
                   help="o que a medição deste checkpoint diz")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8")

    estado, metricas = a.run / NOME_ESTADO, a.run / NOME_METRICAS
    for f in (estado, metricas):
        if not f.exists():
            raise SystemExit(f"{f} não existe — o run não gravou checkpoint")

    manifesto = json.loads(metricas.read_text(encoding="utf-8"))
    cfg = config_do_run(manifesto)
    ressalvas = conferir_saude(manifesto, a.mesmo_assim)

    tok_caminho = a.tokenizer or Path(
        (manifesto.get("dados") or {}).get("tokenizer") or "")
    if not str(tok_caminho):
        raise SystemExit(
            "o run não registrou qual tokenizer usou e --tokenizer não foi dado. "
            "Ver o §'o tokenizer errado não parece um erro'.")
    tok = carregar_tokenizer(Path(tok_caminho), cfg.vocab)

    # `construir` já confere a contagem de parâmetros contra a analítica: se o
    # `transformers` mudou a estrutura do ModernBERT, o erro sai aqui e não como
    # uma métrica estranha três horas depois.
    # `eager` e não `sdpa`: a exportação é em CPU e o kernel não muda os pesos.
    # O que importa é usar o MESMO nos dois lados da conferência — ver
    # `conferir_ida_e_volta`.
    ATENCAO = "eager"
    modelo = construir(cfg, torch.device("cpu"), atencao=ATENCAO)
    ck = torch.load(estado, map_location="cpu", weights_only=True)
    if "modelo" not in ck:
        raise SystemExit(f"{estado} não tem a chave 'modelo'")
    # ⚠️ `strict=True` explicitamente, e não por ser o padrão: com `strict=False`
    # um nome trocado deixaria uma camada com pesos ALEATÓRIOS e a exportação
    # terminaria com sucesso. O modelo rodaria e mediria mal.
    modelo.load_state_dict(ck["modelo"], strict=True)
    passo = ck.get("passo")
    log.info("pesos carregados do passo %s", passo)

    a.para.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(a.para)
    tok.save_pretrained(a.para)
    delta = conferir_ida_e_volta(modelo, a.para, cfg.contexto, ATENCAO)
    conferir_tokenizer_com_modelo(a.para, ATENCAO)

    proveniencia = {
        "origem": str(a.run),
        "passo": passo,
        "concluido": bool(manifesto.get("concluido")),
        "config_do_encoder": cfg.como_dict(),
        "tokenizer": str(tok_caminho),
        "tokenizer_sha": hash_arquivo(Path(tok_caminho)),
        "tokens_treinados": (manifesto.get("metricas") or {}).get("tokens"),
        "perda_final": (manifesto.get("metricas") or {}).get("perda"),
        "mascaramento": manifesto.get("mascaramento"),
        "spike": manifesto.get("spike"),
        "ressalvas": ressalvas,
        "ida_e_volta_delta_logits": delta,
        "nota": a.nota,
        "aviso": ("Exportado para AVALIAÇÃO. O estado do otimizador NÃO vem "
                  "aqui: para retomar o treino use o "
                  f"{NOME_ESTADO} do run de origem."),
    }
    (a.para / "phienc_exportado.json").write_text(
        json.dumps(proveniencia, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    gravar_manifesto_etapa(
        etapa="phienc_exportado",
        descricao=(f"ΦEnc do passo {passo} no formato do `transformers`, com "
                   "round-trip conferido"),
        raiz=a.para, base=RAIZ,
        # `Entrada` não tem campo de hash: as duas são externas a este passo, e
        # o `nota` é onde a proveniência de entrada externa mora.
        entradas=[
            Entrada(caminho=str(estado),
                    nota=f"checkpoint do laço de pré-treino, blake3 "
                         f"{hash_arquivo(estado)}"),
            Entrada(caminho=str(tok_caminho),
                    nota=f"tokenizer do §11.2, blake3 "
                         f"{hash_arquivo(Path(tok_caminho))}"),
        ],
        parametros=proveniencia,
    )

    print()
    print("=" * 74)
    print(f"  ΦEnc exportado -> {a.para}")
    print(f"  passo {passo} · {cfg.nome} · vocab {cfg.vocab:,} · "
          f"tokenizer {Path(tok_caminho).name}")
    print(f"  ida e volta: diferença máxima de logits = {delta}")
    for r in ressalvas:
        print(f"  ⚠️  {r}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
