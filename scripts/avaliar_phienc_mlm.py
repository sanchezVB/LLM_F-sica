#!/usr/bin/env python3
"""MLM medido POR REGIÃO num ΦEnc exportado: token de equação contra prosa.

    .venv-treino\\Scripts\\python.exe scripts\\avaliar_phienc_mlm.py \\
        --modelo models/phienc-variante-A \\
        --dados data/processed/phienc_aval \\
        --out data/processed/avaliacao/phienc_mlm_A.json

A passagem pelo modelo. A parte pura — escolher as posições e agregar por região —
está em `phifm.eval.mlm_regiao`, testada na suíte rápida, e é lá que estão as duas
decisões que fazem a medição valer: a máscara da avaliação é **uniforme** (ignora
as marcas) e é **determinística em `(semente, sequência)`**, para os braços serem
pareados exatamente. Ver a docstring daquele módulo.

## ⚠️ Recusa dado que não é disjunto do treino

Medir MLM sobre o mesmo fluxo de tokens em que o modelo treinou mede memorização.
Entre seis variantes do §11.2 a comparação seguiria "justa" — todas medidas no
próprio texto de treino —, mas ordenaria capacidade de memorizar, e a hipótese do
DOC-07 §2.3 é sobre capacidade.

O manifesto da fatia de avaliação tem de dizer `disjunto_do_treino: true`, o que
só acontece quando ela foi preparada com
`preparar_dados_phienc.py --excluir-de <dir do treino>`.

## ⚠️ O contexto da avaliação é um parâmetro, e tem de ser o MESMO nos braços

O treino usa 8.192. Avaliar nesse comprimento em CPU é caro sem necessidade — a
acurácia de MLM por token não precisa da janela inteira —, então o padrão aqui é
512. Mas ele muda a dificuldade: mais contexto, mais informação para reconstruir o
token. Comparar duas variantes com contextos diferentes compararia duas tarefas.

O valor vai gravado no artefato e o comparador tem de conferir.

## ⚠️ Posições especiais e preenchimento ficam fora da máscara

`[CLS]`/`[SEP]` aparecem no meio do fluxo, porque as sequências são cortadas de um
binário contínuo de documentos concatenados. Mascará-los gastaria orçamento em
posições que nenhum modelo tem como acertar de verdade: a taxa cairia igual nos
dois braços, com ruído a mais.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import torch  # noqa: E402

from phifm import cache_hf_no_hd  # noqa: E402

cache_hf_no_hd()  # ⚠️ ANTES do transformers: o huggingface_hub lê o caminho do cache no import

from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

from phifm.core.console import utf8 as console_utf8  # noqa: E402
from phifm.eval.mlm_regiao import (  # noqa: E402
    FRACAO_MASCARA,
    Contagem,
    equacao_da_prova,
    fracao_de_equacao,
    posicoes_mascaradas,
    sequencias_sorteadas,
)
from phifm.training.embedding import escolher_dispositivo  # noqa: E402
from phifm.training.pretrain.dados import (  # noqa: E402
    BIT_MATH,
    NOME_MANIFESTO,
    ConfigDados,
    Fluxo,
    hash_de_tokenizer,
)

# ⚠️ No IMPORT, e não dentro do `main()`: o argparse imprime `--help` antes
# de qualquer código nosso, e `Φ` não existe em cp1252. Ver `phifm.core.console`.
console_utf8()

log = logging.getLogger("phienc-mlm")


def especiais_do_modelo(caminho) -> tuple[int, set[int]]:
    """`(id_de_mascara, ids_especiais)` — do tokenizer DO MODELO medido.

    ## ⚠️ Por que não de uma constante do projeto

    Isto era `ID_MASK = 4` e `posição proibida se id < 5`, que é a convenção das
    nossas variantes (`phifm.models.encoder.config.ESPECIAIS`). Num modelo de
    prateleira o `[MASK]` está noutro id e os especiais dele noutro conjunto:
    mascarar com o token errado e desproteger os especiais certos produziria um
    número com a cara certa.

    Quem define o que é máscara é o modelo que vai receber a entrada, então a
    fonte é o tokenizer dele.
    """
    tok = AutoTokenizer.from_pretrained(caminho)
    if tok.mask_token_id is None:
        raise SystemExit(
            f"o tokenizer de {caminho} não declara `mask_token`. Sem ele não há "
            "como fazer uma avaliação de MLM: a entrada mascarada seria um token "
            "qualquer.")
    return int(tok.mask_token_id), {int(x) for x in tok.all_special_ids}


def exigir_disjunto(dados: Path) -> dict:
    """Ver o §"recusa dado que não é disjunto do treino" na docstring."""
    man = dados / NOME_MANIFESTO
    if not man.exists():
        raise SystemExit(f"{man} não existe — a fatia não tem manifesto")
    m = json.loads(man.read_text(encoding="utf-8"))
    if not m.get("disjunto_do_treino"):
        raise SystemExit(
            f"{man} não declara `disjunto_do_treino: true`.\n"
            "Esta fatia pode se sobrepor ao treino, e nesse caso o número mede "
            "memorização com a cara de generalização. Prepare-a com:\n"
            "  preparar_dados_phienc.py --excluir-de <dir do treino> "
            f"--out {dados} --tokenizer <o mesmo do treino>")
    log.info("fatia disjunta: %d partes excluídas do treino em %s",
             len(m.get("partes_excluidas") or []), m.get("excluido_de"))
    return m


def previsao_nas_posicoes(modelo, entrada: torch.Tensor, att: torch.Tensor,
                          pos: torch.Tensor) -> torch.Tensor:
    """O argmax da cabeça de MLM **só nas posições mascaradas**.

    ⚠️ Não é otimização cosmética. A cabeça projeta 768 → 50.368 em TODAS as 1.024
    posições do contexto (39 GFLOP por sequência) quando ~150 estão mascaradas, e os
    logits ocupam 206 MB por chamada. Medido em 2026-09-22: no DirectML isso estoura
    a memória de vídeo — o alocador não devolve entre chamadas —, e em CPU custava
    0,51 sequência/s, ou 4 h 20 para as quatro medidas da regra.

    A conta é a MESMA: `head` e `decoder` são por token, então aplicá-los a um
    subconjunto de posições dá os mesmos logits daquelas posições. O teste
    `test_avaliar_phienc_mlm.py` compara com o caminho de logits cheios.

    Se o modelo não tiver as três peças separadas (`model`/`head`/`decoder`), cai no
    caminho cheio em vez de adivinhar a estrutura.
    """
    if not all(hasattr(modelo, n) for n in ("model", "head", "decoder")):
        logits = modelo(input_ids=entrada, attention_mask=att).logits[0]
        return logits[pos].argmax(-1)
    h = modelo.model(input_ids=entrada, attention_mask=att).last_hidden_state[0]
    return modelo.decoder(modelo.head(h[pos])).argmax(-1)


def avaliar(modelo, fluxo: Fluxo, n_seq: int, semente: int, dev,
            fracao: float, id_mask: int, ids_especiais: set[int],
            prova: str = "uniforme") -> tuple[Contagem, dict, list[int]]:
    """Uma passagem por `n_seq` sequências SORTEADAS. Devolve contagem, diagnóstico e índices.

    ⚠️ Sorteadas, e não `range(n_seq)` — ver `mlm_regiao.sequencias_sorteadas`.

    `prova`:
      - `uniforme` — a primária da regra do §2.3: 15% de posições uniformes,
        ignorando as marcas. Nenhum braço vê a própria política de treino.
      - `equacao` — a checagem de manipulação 2: UMA equação em display inteira
        escondida, escolhida como o treino do braço tratado escolhe. Todo token
        avaliado é de equação; a contagem fica toda na região `equacao`.
    """
    if prova not in ("uniforme", "equacao"):
        raise ValueError(f"prova {prova!r}: conhecidas são uniforme e equacao")
    c = Contagem()
    lista_especiais = sorted(ids_especiais)
    indices = sequencias_sorteadas(fluxo.n_seq, n_seq, semente).tolist()
    eq_vistas, t0 = [], time.perf_counter()
    for k, indice in enumerate(indices):
        a_, b_ = indice * fluxo.cfg.contexto, (indice + 1) * fluxo.cfg.contexto
        ids = np.asarray(fluxo.tokens[a_:b_], dtype=np.int64)
        marcas = np.asarray(fluxo.marcas[a_:b_])
        eq_vistas.append(fracao_de_equacao(marcas, BIT_MATH))

        if prova == "uniforme":
            proibidas = np.flatnonzero(
                np.isin(ids, lista_especiais)).astype(np.int64)
            pos = posicoes_mascaradas(ids.size, semente=semente, indice=indice,
                                      fracao=fracao, proibidas=proibidas)
        else:
            # `Fluxo.sequencia` e não as marcas cruas: é ela que descarta a
            # equação cortada nas DUAS pontas da janela.
            _, ide, disp = fluxo.sequencia(indice)
            pos = equacao_da_prova(ide, disp, ids, ids_especiais, semente, indice)
        if pos.size == 0:
            continue

        entrada = ids.copy()
        entrada[pos] = id_mask
        t = torch.from_numpy(entrada).unsqueeze(0).to(dev)
        att = torch.ones_like(t)
        with torch.no_grad():
            previsto = previsao_nas_posicoes(
                modelo, t, att, torch.from_numpy(pos).to(dev)).cpu().numpy()

        certo = (previsto == ids[pos]).astype(np.int64)
        # ⚠️ `marcas[pos]`, e não as marcas inteiras: `certo` e `em_equacao` têm
        # de estar na MESMA ordem, ou o acerto vai para a região errada. A
        # `Contagem.somar` levanta se as formas divergirem.
        if prova == "uniforme":
            em_eq = ((marcas[pos] & BIT_MATH) != 0).astype(np.int64)
        else:
            em_eq = np.ones_like(certo)
        c.somar(certo, em_eq, indice=indice)

        if k and k % 50 == 0:
            taxa = (k + 1) / (time.perf_counter() - t0)
            log.info("  %d/%d sequências · %.2f/s · faltam %.0f min",
                     k + 1, len(indices), taxa, (len(indices) - k) / taxa / 60)

    diag = {
        "sequencias_sorteadas": len(indices),
        "sequencias_com_mascara": len(c.por_sequencia),
        "fracao_de_equacao_media": (round(float(np.mean(eq_vistas)), 4)
                                    if eq_vistas else None),
        "segundos": round(time.perf_counter() - t0, 1),
    }
    return c, diag, indices


def main() -> int:
    p = argparse.ArgumentParser()
    # ⚠️ `str` e não `Path`: aceita id do Hub além de caminho local.
    #
    # `Path("answerdotai/ModernBERT-base")` vira `WindowsPath` e o `str()` dele sai
    # com barra invertida, que o `from_pretrained` não reconhece. É a SEGUNDA vez
    # que este defeito aparece no projeto — a primeira foi no
    # `diagnosticar_teto.py`, no mesmo dia. Onde um script aceita "modelo", o tipo
    # certo é `str`.
    p.add_argument("--modelo", required=True,
                   help="diretório exportado por `exportar_phienc.py`, ou id do "
                        "HuggingFace para medir uma base de fora")
    p.add_argument("--dados", type=Path, required=True,
                   help="fatia de avaliação, DISJUNTA do treino")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-sequencias", type=int, default=200)
    p.add_argument("--contexto", type=int, default=512,
                   help="ver o § sobre o contexto na docstring; o MESMO nos braços")
    p.add_argument("--fracao", type=float, default=FRACAO_MASCARA)
    p.add_argument("--semente", type=int, default=17,
                   help="a MESMA nos dois braços, ou o pareado não é pareado")
    p.add_argument("--dispositivo", default="auto")
    p.add_argument("--prova", choices=("uniforme", "equacao"), default="uniforme",
                   help="uniforme: a primária da regra do §2.3. equacao: a "
                        "checagem de manipulação 2, uma equação inteira escondida")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    for fluxo_io in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo_io.reconfigure(encoding="utf-8")

    man_dados = exigir_disjunto(a.dados)
    fluxo = Fluxo(ConfigDados(raiz=a.dados, contexto=a.contexto,
                              semente=a.semente))

    id_mask, ids_especiais = especiais_do_modelo(a.modelo)
    # ⚠️ A fatia tem de concordar com o modelo sobre o que é especial.
    #
    # O manifesto grava `id_cls`/`id_sep` usados para envolver cada documento. Se
    # eles não estiverem entre os especiais do modelo, a fatia foi preparada para
    # OUTRO tokenizer — e os ids do meio também significam outra coisa.
    for chave in ("id_cls", "id_sep"):
        ident = man_dados.get(chave)
        if ident is not None and int(ident) not in ids_especiais:
            raise SystemExit(
                f"a fatia envolve os documentos com {chave}={ident}, que NÃO é "
                f"especial para o tokenizer de {a.modelo} (especiais: "
                f"{sorted(ids_especiais)}). A fatia foi tokenizada para outro "
                "modelo, e os ids do meio também significam outra coisa.")
    log.info("especiais do modelo: máscara=%d · %d ids especiais",
             id_mask, len(ids_especiais))

    dev = escolher_dispositivo(a.dispositivo)
    modelo = AutoModelForMaskedLM.from_pretrained(
        a.modelo, attn_implementation="eager").to(dev).eval()
    log.info("modelo %s · vocab %s · contexto de avaliação %d · %s",
             a.modelo, f"{modelo.config.vocab_size:,}", a.contexto, dev)

    # ⚠️ O vocabulário do modelo tem de casar com o do binário. A fatia de
    # avaliação é tokenizada pela variante, e medir a variante A sobre tokens da
    # C daria ids que significam outra coisa — sem levantar, se a C for menor.
    if man_dados.get("tokenizer_sha"):
        # ⚠️ O hash do tokenizer que o MODELO carrega, na convenção da FATIA.
        #
        # Até 2026-09-16 isto comparava o `tokenizer_sha` do `phienc_exportado.json`
        # (BLAKE3, 64 caracteres) com o do manifesto da fatia (SHA-256, 16): dois
        # hashes do mesmo arquivo nunca batem, e a guarda recusaria todo ΦEnc
        # exportado. Ver `dados.hash_de_tokenizer`.
        #
        # Só um diretório local tem `tokenizer.json`; um id do Hub não tem, e nele a
        # guarda que vale é a dos ids especiais, feita acima.
        tok_modelo = Path(a.modelo) / "tokenizer.json"
        if tok_modelo.exists():
            do_modelo = hash_de_tokenizer(tok_modelo)
            if do_modelo != man_dados["tokenizer_sha"]:
                raise SystemExit(
                    f"o tokenizer do modelo ({do_modelo}) não é o desta fatia "
                    f"({man_dados['tokenizer_sha']}), na mesma convenção de hash.\n"
                    "Os ids significam coisas diferentes; o número sairia com a "
                    "cara de uma medição. Prepare a fatia com o tokenizer da "
                    "variante.")

    c, diag, indices = avaliar(modelo, fluxo, a.n_sequencias, a.semente, dev,
                               a.fracao, id_mask, ids_especiais, a.prova)
    d = c.como_dict()

    if not d["tokens_equacao"]:
        raise SystemExit(
            "nenhum token de equação foi mascarado nas sequências avaliadas — a "
            "diferença entre regiões não pode aparecer, por mais correta que a "
            "hipótese do DOC-07 §2.3 esteja. Confira a fatia.")

    resultado = {
        "modelo": str(a.modelo).replace("\\", "/"),
        "dados": str(a.dados).replace("\\", "/"),
        "disjunto_do_treino": True,
        "partes_excluidas": man_dados.get("partes_excluidas"),
        "tokenizer_sha": man_dados.get("tokenizer_sha"),
        # ⚠️ O protocolo vai junto. Duas variantes comparadas com contexto,
        # fração ou semente diferentes são duas tarefas diferentes, e o
        # comparador precisa poder RECUSAR em vez de subtrair.
        "protocolo": {"prova": a.prova, "amostragem": "sorteada",
                      "contexto": a.contexto, "fracao_mascara": a.fracao,
                      "semente": a.semente,
                      "n_sequencias_pedidas": a.n_sequencias,
                      "id_mascara": id_mask,
                      "n_ids_especiais": len(ids_especiais)},
        "diagnostico": diag,
        **d,
        # Os vetores por token, para o McNemar pareado entre braços. Sem eles a
        # comparação de dois braços vira duas proporções soltas.
        "acertos_por_token": c.acertos,
        "e_equacao_por_token": c.e_equacao,
        # ⚠️ O que o bootstrap por SEQUÊNCIA consome, e os índices sorteados,
        # para o comparador conferir que os dois braços viram as MESMAS.
        "indices_sorteados": indices,
        "por_sequencia": c.por_sequencia,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    print()
    print("=" * 74)
    print(f"  MLM · prova {a.prova.upper()} · {a.modelo} · contexto {a.contexto} · "
          f"{diag['sequencias_com_mascara']} de {diag['sequencias_sorteadas']} "
          "sequências sorteadas")
    print("=" * 74)
    print(f"  equação  {d['acuracia_equacao']:.4f}  "
          f"({d['tokens_equacao']:,} tokens)")
    if a.prova == "uniforme":
        print(f"  prosa    {d['acuracia_prosa']:.4f}  "
              f"({d['tokens_prosa']:,} tokens)")
        print(f"  total    {d['acuracia_total']:.4f}")
        print(f"  VANTAGEM EM EQUAÇÃO: {d['vantagem_em_equacao']:+.4f}")
    else:
        print("  (prova de equação inteira: todo token avaliado é de equação — é a "
              "checagem de manipulação 2, não a primária)")
    print()
    print(f"  fração de equação na fatia: {diag['fracao_de_equacao_media']}")
    print(f"  {d['nota']}")
    print("=" * 74)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
