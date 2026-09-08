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
from transformers import AutoModelForMaskedLM  # noqa: E402

from phifm.eval.mlm_regiao import (  # noqa: E402
    FRACAO_MASCARA,
    Contagem,
    fracao_de_equacao,
    posicoes_mascaradas,
)
from phifm.models.encoder.config import ESPECIAIS  # noqa: E402
from phifm.training.embedding import escolher_dispositivo  # noqa: E402
from phifm.training.pretrain.dados import (  # noqa: E402
    BIT_MATH,
    NOME_MANIFESTO,
    ConfigDados,
    Fluxo,
)

log = logging.getLogger("phienc-mlm")

ID_MASK = ESPECIAIS["mask"]
# Todo id abaixo disto é especial (`pad`, `unk`, `cls`, `sep`, `mask`).
N_ESPECIAIS = max(ESPECIAIS.values()) + 1


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


def avaliar(modelo, fluxo: Fluxo, n_seq: int, semente: int, dev,
            fracao: float) -> tuple[Contagem, dict]:
    """Uma passagem por `n_seq` sequências. Devolve a contagem e o diagnóstico."""
    c = Contagem()
    eq_vistas, t0 = [], time.perf_counter()
    for indice in range(n_seq):
        ids = np.asarray(fluxo.tokens[indice * fluxo.cfg.contexto:
                                      (indice + 1) * fluxo.cfg.contexto],
                         dtype=np.int64)
        marcas = np.asarray(fluxo.marcas[indice * fluxo.cfg.contexto:
                                         (indice + 1) * fluxo.cfg.contexto])
        if ids.size < fluxo.cfg.contexto:
            log.warning("sequência %d incompleta (%d tokens) — fim do fluxo",
                        indice, ids.size)
            break
        eq_vistas.append(fracao_de_equacao(marcas, BIT_MATH))

        proibidas = np.flatnonzero(ids < N_ESPECIAIS).astype(np.int64)
        pos = posicoes_mascaradas(ids.size, semente=semente, indice=indice,
                                  fracao=fracao, proibidas=proibidas)
        if pos.size == 0:
            continue

        entrada = ids.copy()
        entrada[pos] = ID_MASK
        t = torch.from_numpy(entrada).unsqueeze(0).to(dev)
        att = torch.ones_like(t)
        with torch.no_grad():
            logits = modelo(input_ids=t, attention_mask=att).logits[0]
        previsto = logits[torch.from_numpy(pos).to(dev)].argmax(-1).cpu().numpy()

        certo = (previsto == ids[pos]).astype(np.int64)
        # ⚠️ `marcas[pos]`, e não as marcas inteiras: `certo` e `em_equacao` têm
        # de estar na MESMA ordem, ou o acerto vai para a região errada. A
        # `Contagem.somar` levanta se as formas divergirem.
        em_eq = ((marcas[pos] & BIT_MATH) != 0).astype(np.int64)
        c.somar(certo, em_eq)

        if indice and indice % 50 == 0:
            taxa = (indice + 1) / (time.perf_counter() - t0)
            log.info("  %d/%d sequências · %.2f/s · faltam %.0f min",
                     indice + 1, n_seq, taxa, (n_seq - indice) / taxa / 60)

    diag = {
        "sequencias_avaliadas": len(eq_vistas),
        "fracao_de_equacao_media": (round(float(np.mean(eq_vistas)), 4)
                                    if eq_vistas else None),
        "segundos": round(time.perf_counter() - t0, 1),
    }
    return c, diag


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", type=Path, required=True,
                   help="diretório exportado por `exportar_phienc.py`")
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

    dev = escolher_dispositivo(a.dispositivo)
    modelo = AutoModelForMaskedLM.from_pretrained(
        a.modelo, attn_implementation="eager").to(dev).eval()
    log.info("modelo %s · vocab %s · contexto de avaliação %d · %s",
             a.modelo.name, f"{modelo.config.vocab_size:,}", a.contexto, dev)

    # ⚠️ O vocabulário do modelo tem de casar com o do binário. A fatia de
    # avaliação é tokenizada pela variante, e medir a variante A sobre tokens da
    # C daria ids que significam outra coisa — sem levantar, se a C for menor.
    if man_dados.get("tokenizer_sha"):
        prov = a.modelo / "phienc_exportado.json"
        if prov.exists():
            esperado = json.loads(prov.read_text(encoding="utf-8")).get(
                "tokenizer_sha")
            if esperado and esperado != man_dados["tokenizer_sha"]:
                raise SystemExit(
                    f"o modelo foi treinado com o tokenizer {esperado} e esta "
                    f"fatia foi tokenizada com {man_dados['tokenizer_sha']}.\n"
                    "Os ids significam coisas diferentes; o número sairia com a "
                    "cara de uma medição. Prepare a fatia com o tokenizer da "
                    "variante.")

    c, diag = avaliar(modelo, fluxo, a.n_sequencias, a.semente, dev, a.fracao)
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
        "protocolo": {"contexto": a.contexto, "fracao_mascara": a.fracao,
                      "semente": a.semente,
                      "n_sequencias_pedidas": a.n_sequencias},
        "diagnostico": diag,
        **d,
        # Os vetores por token, para o McNemar pareado entre braços. Sem eles a
        # comparação de dois braços vira duas proporções soltas.
        "acertos_por_token": c.acertos,
        "e_equacao_por_token": c.e_equacao,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    print()
    print("=" * 74)
    print(f"  MLM por região · {a.modelo.name} · contexto {a.contexto} · "
          f"{diag['sequencias_avaliadas']} sequências")
    print("=" * 74)
    print(f"  equação  {d['acuracia_equacao']:.4f}  "
          f"({d['tokens_equacao']:,} tokens)")
    print(f"  prosa    {d['acuracia_prosa']:.4f}  "
          f"({d['tokens_prosa']:,} tokens)")
    print(f"  total    {d['acuracia_total']:.4f}")
    print(f"  VANTAGEM EM EQUAÇÃO: {d['vantagem_em_equacao']:+.4f}")
    print()
    print(f"  fração de equação na fatia: {diag['fracao_de_equacao_media']}")
    print(f"  {d['nota']}")
    print("=" * 74)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
