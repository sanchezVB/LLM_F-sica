"""Avaliação de recuperação do ΦEnc — a primeira das três do DOC-05 §11.2.

    PYTHONPATH=src python scripts/avaliar_phienc.py \\
        --controle models/phienc-controle --tratado models/phienc-tratado

## ⚠️ O que este número É e o que ele NÃO É

O ΦEnc é um MLM. Ele **não foi treinado para embedding**, e agregar a média dos
estados ocultos de um MLM é uma linha de base fraca em recuperação — isso não é
defeito do nosso modelo, é a razão de o PhysBERT existir: ele faz SimCSE por cima
do MLM justamente porque o MLM cru não basta.

A medição do próprio repositório dá a escala do efeito:

    MiniLM-L6   23 M   treinado PARA embedding      nDCG@10 0,370
    PhysBERT   109 M   MLM de domínio + SimCSE      nDCG@10 0,275
    SciBERT    110 M   MLM, sem cabeça contrastiva  nDCG@10 0,207

Um MLM **sem nenhum ajuste contrastivo** fica abaixo do SciBERT, e é isso que se
deve esperar aqui. **Comparar o ΦEnc com a tabela do G1 seria ler o número
errado** — o G1 mede quem serve como recuperador pronto; isto mede se o
pré-treino mudou a representação.

**A comparação que esta avaliação existe para fazer é outra:** controle contra
tratado, os dois MLMs crus, os dois agregados por média, os dois no mesmo pool
com a mesma semente. Aí a fraqueza absoluta é comum aos dois e se cancela, e o
que sobra é o efeito do mascaramento consciente de equações — que é a hipótese do
DOC-07 §2.3, e a única coisa que o ΦEnc existe para testar.

Por isso a função de topo deste módulo é `comparar_bracos`, e não `avaliar`.

## Por que a métrica não é recalculada aqui

`avaliar_carregado`, em `phifm.eval.encoders`, é o único lugar do repositório onde
recall@k, MRR e nDCG@10 são computados. Este módulo só resolve o que o ΦEnc tem
de diferente — carregar pesos de um `state_dict` cru e envelopar um tokenizer que
é JSON do `tokenizers`, não diretório do `transformers` — e entrega o par
`(modelo, tokenizer)` para lá.

A alternativa seria um segundo avaliador, e este repositório já registrou o preço
disso duas vezes: dois caminhos que calculam a mesma coisa divergem em silêncio e
nada aponta qual está certo.

## As três guardas, e o que cada uma impede

1. **Config do checkpoint, não adivinhada.** A arquitetura sai do `phienc.json`
   que o treino gravou. Avaliar com uma config diferente carregaria pesos numa
   estrutura errada — e se as formas por acaso baterem, sai um número plausível.
2. **Contagem de parâmetros conferida.** O carregamento passa por `construir()`,
   que levanta se a estrutura do ModernBERT no `transformers` mudou.
3. **Tamanho do vocabulário conferido contra o do checkpoint.** O tokenizer é um
   argumento de linha de comando separado, então passar o de outra variante do
   bake-off é um erro de um caractere — e com vocabulário menor nada estoura: o
   modelo só nunca vê os ids que faltam, e a métrica sai pior sem explicação.

   ⚠️ A primeira versão desta guarda conferia `pad_token_id` do tokenizer contra
   o do modelo. Era **tautológica**: os nomes dos especiais são derivados desses
   mesmos ids logo acima, então a igualdade valia por construção e a guarda nunca
   podia falhar. Foi um teste que a pegou, e o conserto foi trocar o que ela
   compara — não apagá-la.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import polars as pl
import torch

from phifm.eval.encoders import Resultado, avaliar_carregado, comparar_pareado
from phifm.models.encoder.config import ESPECIAIS, ConfigEnc
from phifm.models.encoder.modelo import construir
from phifm.training.amostragem import SEMENTE_POOL

log = logging.getLogger(__name__)

NOME_ESTADO = "estado_pretreino.pt"
NOME_METRICAS = "phienc.json"

# Os cinco especiais, do id que `config.py` fixa para o nome que o
# `PreTrainedTokenizerFast` espera. A chave do `ESPECIAIS` não é uniforme
# (`pad_token_id` mas `unk`), então a tradução fica explícita aqui em vez de
# depender de sufixo.
ESPECIAIS_HF = {
    "pad_token": ESPECIAIS["pad_token_id"],
    "unk_token": ESPECIAIS["unk"],
    "cls_token": ESPECIAIS["cls_token_id"],
    "sep_token": ESPECIAIS["sep_token_id"],
    "mask_token": ESPECIAIS["mask"],
}

# Abaixo disto o braço tratado recaiu em MLM aleatório e a ablação não mediu o
# tratamento. Medido no corpus preparado: 0,903. O piso é folgado de propósito —
# ele existe para pegar recaída total (o defeito do `^` no padrão de display deu
# 0,000), não para julgar qualidade.
MIN_FRACAO_TRATADA = 0.50

# Acompanha TODO artefato desta avaliação, nos dois modos. É o número mais fácil
# de copiar para a linha errada de uma tabela: 0,08 de nDCG@10 parece um modelo
# quebrado se ninguém disser que um MLM cru não é um recuperador.
RESSALVA = (
    "MLM cru agregado por média é linha de base FRACA em recuperação — o SciBERT, "
    "que é MLM sem cabeça contrastiva, dá nDCG@10 0,207 no mesmo protocolo. O valor "
    "absoluto destes braços NÃO é comparável à tabela do G1, que mede recuperadores "
    "prontos. O que esta medição compara é um braço com o outro.")


def _ler_metricas(diretorio: Path) -> dict:
    caminho = diretorio / NOME_METRICAS
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} não existe. A avaliação precisa do `{NOME_METRICAS}` que o "
            "laço de pré-treino grava: é dele que saem a arquitetura, os contadores "
            "de mascaramento e os hiperparâmetros que a checagem de uma-variável "
            "compara. Um diretório só com os pesos não permite avaliar com veredito.")
    return json.loads(caminho.read_text(encoding="utf-8"))


def config_do_checkpoint(meta: dict) -> ConfigEnc:
    """Reconstrói a `ConfigEnc` gravada pelo treino.

    `como_dict()` acrescenta `parametros` e `fracao_de_embedding`, que são
    derivados e não são campos do dataclass — entram aqui como conferência, e não
    como argumento, para que uma divergência apareça em vez de ser reescrita.
    """
    d = dict(meta["modelo"])
    derivados = {"parametros": d.pop("parametros", None),
                 "fracao_de_embedding": d.pop("fracao_de_embedding", None)}
    cfg = ConfigEnc(**d)

    esperado = derivados["parametros"]
    if esperado and cfg.parametros()["total"] != esperado["total"]:
        raise ValueError(
            f"o checkpoint declara {esperado['total']:,} parâmetros e a mesma config "
            f"hoje dá {cfg.parametros()['total']:,}. A fórmula de `ConfigEnc."
            "parametros()` mudou desde o treino — os dois números não podem ser "
            "comparados sem decidir qual vale.")
    return cfg


def carregar_mlm(diretorio: Path, tokenizer: Path, dispositivo: str = "cpu"):
    """Devolve `(ModernBertForMaskedLM, tokenizador, metricas)`.

    A avaliação de MLM (`phifm.eval.mlm`) precisa da CABEÇA, porque o que ela mede
    é a probabilidade atribuída aos tokens mascarados. A de recuperação precisa do
    tronco. Carregar é a mesma coisa nos dois casos, e por isso é um lugar só.
    """
    diretorio, tokenizer = Path(diretorio), Path(tokenizer)
    meta = _ler_metricas(diretorio)
    cfg = config_do_checkpoint(meta)
    dev = torch.device(dispositivo)

    # `construir` confere a contagem de parâmetros contra a analítica — é a guarda
    # que pega uma mudança de estrutura no `transformers` antes de ela virar um
    # modelo silenciosamente de outro tamanho.
    completo = construir(cfg, dev)

    estado = torch.load(diretorio / NOME_ESTADO, map_location=dev, weights_only=True)
    if "modelo" not in estado:
        raise KeyError(
            f"{diretorio / NOME_ESTADO} não tem a chave 'modelo'. Chaves presentes: "
            f"{sorted(estado)}. Este não parece um checkpoint do laço de pré-treino.")
    completo.load_state_dict(estado["modelo"])
    log.info("%s · passo %s · carregado de %s",
             cfg.nome, estado.get("passo", "?"), diretorio)

    return completo.to(dev).eval(), carregar_tokenizer(tokenizer, cfg), meta


def carregar(diretorio: Path, tokenizer: Path, dispositivo: str = "cpu"):
    """Devolve `(tronco, tokenizador, metricas)` para `avaliar_carregado`.

    O tronco (`ModernBertModel`), e não a cabeça: quem recupera é a representação,
    e a projeção de saída amarrada não participa.
    """
    completo, tok, meta = carregar_mlm(diretorio, tokenizer, dispositivo)
    return completo.model, tok, meta


def carregar_tokenizer(caminho: Path, cfg: ConfigEnc):
    """Envelopa o JSON do `tokenizers` na interface que `_codificar` usa.

    ⚠️ **Nenhum pós-processador é imposto aqui.** Se o tokenizer treinado não
    insere `[CLS]`/`[SEP]`, a avaliação também não insere — que é o certo, porque
    o pré-treino leu binários produzidos pelo mesmo tokenizer. Forçar um
    envelope que o treino não usou criaria um desencontro treino/avaliação, e
    esse tipo de desencontro não levanta: ele só piora a métrica.

    O que de fato acontece fica registrado em `meta_do_tokenizer`, para ser lido
    no artefato em vez de suposto.
    """
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    base = Tokenizer.from_file(str(caminho))

    # ⚠️ A guarda que vale é esta, e não a que eu escrevi primeiro. A tentação era
    # conferir `tok.pad_token_id == ESPECIAIS["pad_token_id"]` — mas os nomes dos
    # especiais são DERIVADOS dos ids logo abaixo, então essa igualdade é
    # verdadeira por construção e a guarda nunca podia falhar. Um teste a pegou.
    #
    # O desencontro que de fato acontece é outro: o tokenizer é um argumento de
    # linha de comando separado do checkpoint, então avaliar com o tokenizer de
    # OUTRA variante do bake-off é um erro de um caractere. Vocabulários de
    # tamanhos diferentes é o sintoma que o denuncia — e se o do tokenizer for
    # menor, nada estoura: o modelo só nunca vê os ids que faltam, e a métrica
    # sai pior sem explicação.
    tamanho = base.get_vocab_size()
    if tamanho != cfg.vocab:
        raise ValueError(
            f"o tokenizer {caminho} tem vocabulário de {tamanho:,} e o checkpoint foi "
            f"treinado com {cfg.vocab:,}. Este é o tokenizer de outra variante — "
            "avaliar assim mede o modelo com uma entrada que ele nunca viu.")

    nomes = {}
    for chave, ident in ESPECIAIS_HF.items():
        token = base.id_to_token(ident)
        if token is None:
            raise ValueError(
                f"o tokenizer {caminho} não tem token no id {ident}, que o "
                f"`config.py` reserva para {chave}. O modelo grava esse id no "
                "`config.json` — um desencontro aqui faz o unpadding do ModernBERT "
                "descartar as posições erradas, sem erro.")
        nomes[chave] = token

    return PreTrainedTokenizerFast(tokenizer_object=base, model_max_length=cfg.contexto,
                                   **nomes)


def meta_do_tokenizer(tok) -> dict:
    """O que o tokenizer faz de fato, para o artefato registrar em vez de supor."""
    sonda = tok("x", add_special_tokens=True)["input_ids"]
    crua = tok("x", add_special_tokens=False)["input_ids"]
    return {
        "vocabulario": tok.vocab_size,
        "pad_token_id": tok.pad_token_id,
        "insere_especiais": len(sonda) != len(crua),
        "tokens_acrescentados": len(sonda) - len(crua),
    }


def avaliar(diretorio: Path, tokenizer: Path, val: pl.DataFrame, nome: str | None = None,
            *, dispositivo: str = "cpu", **kw) -> tuple[Resultado, dict]:
    """Mede um braço. Devolve `(Resultado, metricas_do_treino)`."""
    t0 = time.perf_counter()
    mod, tok, meta = carregar(diretorio, tokenizer, dispositivo)
    r = avaliar_carregado(mod, tok, nome or Path(diretorio).name, str(diretorio), val,
                          dispositivo=dispositivo, t0=t0, **kw)
    r.nosso = True
    return r, meta


# ── A ablação ───────────────────────────────────────────────────────────────

# Tudo o que precisa ser IGUAL entre os braços para a comparação isolar o
# tratamento. É a mesma disciplina do experimento T1c, onde só `--base` variava e
# um teste falhava se qualquer um dos outros seis escorregasse.
BLOCOS_IGUAIS = ("modelo", "treino", "dados")
# E o único campo que pode diferir dentro do bloco de mascaramento.
CAMPO_DO_TRATAMENTO = "p_equacao"


def conferir_uma_variavel(controle: dict, tratado: dict) -> list[str]:
    """Lista o que diverge entre os braços além do tratamento. Vazio é o esperado.

    Não levanta: a lista vai para o artefato e para o veredito, porque um braço
    com hiperparâmetro escorregado ainda produz números — e são esses números que
    alguém copiaria para uma tabela sem saber que medem duas coisas.
    """
    fora = []
    for bloco in BLOCOS_IGUAIS:
        a, b = controle.get(bloco), tratado.get(bloco)
        if a != b:
            campos = sorted({k for k in {**(a or {}), **(b or {})}
                             if (a or {}).get(k) != (b or {}).get(k)})
            fora.append(f"{bloco}: diverge em {campos}")

    ma, mb = controle.get("mascara") or {}, tratado.get("mascara") or {}
    for campo in sorted(set(ma) | set(mb)):
        if campo != CAMPO_DO_TRATAMENTO and ma.get(campo) != mb.get(campo):
            fora.append(f"mascara.{campo}: {ma.get(campo)} contra {mb.get(campo)}")
    return fora


def _fracao_tratada(meta: dict) -> float:
    return float((meta.get("mascaramento") or {}).get("fracao_tratada", 0.0))


def _p_equacao(meta: dict) -> float:
    return float((meta.get("mascara") or {}).get(CAMPO_DO_TRATAMENTO, 0.0))


def comparar_bracos(controle: Path, tratado: Path, tokenizer: Path, val: pl.DataFrame,
                    *, n: int = 2000, max_tokens: int = 192, lote: int = 16,
                    dispositivo: str = "cpu", semente: int = SEMENTE_POOL,
                    min_fracao_tratada: float = MIN_FRACAO_TRATADA) -> dict:
    """A ablação do DOC-07 §2.3, com as leituras registradas ANTES de medir.

    As quatro saídas possíveis, e nenhuma delas é "o tratamento funciona" por
    omissão:

    | desfecho | leitura |
    |---|---|
    | tratado vence o pareado | mascaramento consciente de equações **ajuda** |
    | controle vence o pareado | ele **atrapalha** — e o negativo é publicado |
    | empate com discordantes suficientes | **não há efeito detectável** nesta escala |
    | qualquer guarda reprovada | **INCONCLUSIVO** — o experimento não mediu o que diz |

    A quarta linha existe porque o pré-registro do T1c tinha exatamente esse
    buraco: com um braço ausente, o script imprimiu a conclusão que exigia aquele
    braço ter rodado e perdido. A lógica olhava só quem venceu e nunca quem
    falhou. Aqui, guarda reprovada **substitui** o veredito em vez de anotá-lo ao
    lado.
    """
    rc, meta_c = avaliar(controle, tokenizer, val, "controle (p_equacao=0)",
                         n=n, max_tokens=max_tokens, lote=lote,
                         dispositivo=dispositivo, semente=semente)
    rt, meta_t = avaliar(tratado, tokenizer, val, "tratado (equações mascaradas)",
                         n=n, max_tokens=max_tokens, lote=lote,
                         dispositivo=dispositivo, semente=semente)

    impedimentos = []
    divergencias = conferir_uma_variavel(meta_c, meta_t)
    if divergencias:
        impedimentos.append(
            "os braços diferem em mais do que o tratamento: " + "; ".join(divergencias))

    if _p_equacao(meta_c) != 0.0:
        impedimentos.append(
            f"o controle tem p_equacao={_p_equacao(meta_c)}, e controle é p_equacao=0 "
            "por definição — os dois braços estão tratados")

    ft = _fracao_tratada(meta_t)
    if ft < min_fracao_tratada:
        impedimentos.append(
            f"fracao_tratada do braço tratado é {ft:.3f}, abaixo do piso de "
            f"{min_fracao_tratada:.2f}: ele recaiu em MLM aleatório e o 'empate' que "
            "reportaria não diz nada sobre a hipótese")

    pareado = comparar_pareado(rc, rt)

    if impedimentos:
        veredito = "INCONCLUSIVO — " + "; ".join(impedimentos)
    elif pareado.get("erro"):
        veredito = f"INCONCLUSIVO — {pareado['erro']}"
    elif pareado["p"] < 0.05:
        vence = pareado["a"] if pareado["ganha_a"] > pareado["ganha_b"] else pareado["b"]
        ajuda = vence == rt.nome
        veredito = (
            f"o mascaramento consciente de equações {'AJUDA' if ajuda else 'ATRAPALHA'} "
            f"em recuperação (p={pareado['p']:.4f}, {pareado['discordantes']} "
            "discordantes)")
    elif pareado["discordantes"] < 20:
        veredito = (f"INCONCLUSIVO — só {pareado['discordantes']} itens discordantes; "
                    "aumente `n` antes de ler isto como empate")
    else:
        veredito = (f"sem efeito detectável nesta escala (p={pareado['p']:.3f}, "
                    f"{pareado['discordantes']} discordantes)")

    return {
        "tarefa": "recuperacao_por_citacao",
        "protocolo": {"n": n, "max_tokens": max_tokens, "semente": semente,
                      "agregacao": "media_mascarada", "dispositivo": dispositivo},
        "bracos": {"controle": rc.__dict__, "tratado": rt.__dict__},
        "mascaramento": {"controle": meta_c.get("mascaramento"),
                         "tratado": meta_t.get("mascaramento")},
        "p_equacao": {"controle": _p_equacao(meta_c), "tratado": _p_equacao(meta_t)},
        "uma_variavel": divergencias,
        "impedimentos": impedimentos,
        "pareado": pareado,
        "veredito": veredito,
        "ressalva": RESSALVA,
    }
