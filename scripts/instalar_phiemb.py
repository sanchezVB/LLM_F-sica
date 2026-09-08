#!/usr/bin/env python3
"""Instala um ΦEmb treinado no Kaggle como o recuperador denso do sistema.

    .venv\\Scripts\\python.exe scripts\\instalar_phiemb.py --copiar \\
        --de data/processed/kaggle_t1a6m_saida/phiemb-minilm-t4-melhor \\
        --para models/phiemb-do-sistema \\
        --nota "T1a 6M sorteado, nDCG@10 0,60 no protocolo de teto 1,0" \\
        --ndcg-do-portao 0.6026 --teto-do-portao 1.0

## Por que isto é um script e não um `cp`

Um diretório de pesos que aparece em `models/` sem proveniência é um modelo que
ninguém sabe de onde veio nem por que está ali. Este grava o manifesto de etapa
junto, com o que a medição diz e o que ela **não** diz.

Espelha o `instalar_phirank.py`, e existe separado porque o que precisa ser
registrado é diferente: o ΦRank é julgado por acerto@1 no grupo, e o ΦEmb pelo
nDCG@10 do portão G1.

## ⚠️ O número do portão vem com o TETO, e isso não é decoração

Foi a lição mais caras desta linha de trabalho. Até 2026-09-06 o protocolo do G1
usava `val.head(2000)`, e nele **62% dos itens tinham alvo repetido**: colunas
byte-idênticas dão cosseno idêntico e o desempate do `argsort` é arbitrário. O
teto de um modelo perfeito era **nDCG@10 0,7562**, não 1,0 — e o G1.2 "passou" por
+0,003, que era ruído de desempate.

Um nDCG@10 gravado sem o teto do protocolo não diz se o limite é o modelo ou a
régua. Por isso `--ndcg-do-portao` e `--teto-do-portao` são obrigatórios, e o
script **recusa** um teto abaixo de 0,999.

## ⚠️ Trocar o recuperador invalida a referência do T1b/T1c

O nDCG 0,1666 do ΦRank foi medido sobre a fusão RRF **deste** recuperador. Trocar
os dois ao mesmo tempo mediria duas coisas, e é por isso que
`phifm.core.modelos.RECUPERADOR` é uma constante deliberada em vez de "o melhor
que houver". Depois de instalar, remedir a cadeia é parte da etapa — não um
detalhe para depois.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.core.modelos import RECUPERADOR  # noqa: E402
from phifm.core.schema.reprodutibilidade import (  # noqa: E402
    Entrada,
    gravar_manifesto_etapa,
)

EXIGIDOS = ("config.json", "model.safetensors", "tokenizer.json",
            "tokenizer_config.json")
# Abaixo disto há empate no pool e o desempate é arbitrário. Ver a docstring.
TETO_MINIMO = 0.999


def _config_do_treino(de: Path) -> tuple[dict | None, str | None]:
    """O `phiemb.json` mora no diretório do ÚLTIMO passo, não no `-melhor`.

    Na saída do notebook do Kaggle são irmãos: `phiemb-minilm-t4/phiemb.json` e
    `phiemb-minilm-t4-melhor/`. Procura nos dois e, se não achar, devolve o
    porquê — em vez de gravar um manifesto que finge ter a configuração.
    """
    candidatos = [de / "phiemb.json"]
    if de.name.endswith("-melhor"):
        candidatos.append(de.parent / de.name[: -len("-melhor")] / "phiemb.json")
    for c in candidatos:
        if c.exists():
            return json.loads(c.read_text(encoding="utf-8")), None
    return None, (
        "phiemb.json não encontrado em "
        + " nem ".join(str(c).replace("\\", "/") for c in candidatos)
        + ". A configuração do treino NÃO está registrada neste manifesto.")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--de", type=Path, required=True,
                   help="diretório `-melhor` baixado da saída do notebook")
    p.add_argument("--para", type=Path, required=True)
    p.add_argument("--nota", required=True,
                   help="a medição que justifica a instalação")
    p.add_argument("--ndcg-do-portao", type=float, required=True,
                   help="nDCG@10 no protocolo do G1 (2.000 candidatos), NÃO o da "
                        "avaliação interna do treino — os dois não são comparáveis")
    p.add_argument("--teto-do-portao", type=float, required=True,
                   help="teto de um modelo perfeito nesse pool. Ver a docstring: um "
                        "número sem o teto não diz se o limite é o modelo ou a régua")
    p.add_argument("--copiar", action="store_true",
                   help="copia os pesos; sem isto só (re)grava o manifesto sobre o "
                        "que já está em --para")
    a = p.parse_args()
    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8")

    if a.teto_do_portao < TETO_MINIMO:
        raise SystemExit(
            f"o teto declarado é {a.teto_do_portao:.4f}, abaixo de {TETO_MINIMO}. "
            "Um pool com alvo ou consulta repetidos empata no cosseno e o "
            "desempate é arbitrário — foi assim que o G1.2 'passou' por +0,003 em "
            "agosto. Meça com `preparar_pool` e reavalie antes de instalar.")

    if a.copiar:
        if not a.de.is_dir():
            raise SystemExit(f"{a.de} não é um diretório")
        a.para.mkdir(parents=True, exist_ok=True)
        for f in sorted(a.de.iterdir()):
            if f.is_file():
                shutil.copy2(f, a.para / f.name)

    faltando = [n for n in EXIGIDOS if not (a.para / n).exists()]
    if faltando:
        raise SystemExit(
            f"{a.para} está sem {faltando}. Um diretório de modelo incompleto falha "
            "no `from_pretrained` na hora da avaliação, longe daqui.")

    melhor_p = a.para / "melhor.json"
    if not melhor_p.exists():
        raise SystemExit(
            f"{melhor_p} não existe. Sem ele não se sabe em que passo o checkpoint "
            "foi eleito nem por qual critério — e o critério já divergiu neste "
            "projeto (o `-gc-melhor` foi eleito por MRR quando o portão pede "
            "nDCG@10).")
    melhor = json.loads(melhor_p.read_text(encoding="utf-8"))
    treino, ausencia = _config_do_treino(a.de if a.copiar else a.para)

    base = melhor.get("base") or (treino or {}).get("base")
    if not base:
        raise SystemExit(
            "nem `melhor.json` nem `phiemb.json` declaram a `base`. Sem ela o "
            "modelo entra no sistema sem se saber de que pré-treino ele partiu.")

    substitui = str(a.para).replace("\\", "/") != RECUPERADOR

    gravar_manifesto_etapa(
        etapa="phiemb_do_sistema",
        descricao=f"ΦEmb do sistema, de {base}",
        raiz=a.para,
        entradas=[Entrada(caminho=str(a.de), nota="saída do notebook do Kaggle")],
        parametros={
            "script": "scripts/instalar_phiemb.py",
            "base": base,
            "justificativa_medida": a.nota,
            "ndcg_10_no_portao": a.ndcg_do_portao,
            "teto_do_protocolo": a.teto_do_portao,
            "n_candidatos_do_portao": 2000,
            "passo_do_checkpoint": melhor.get("passo"),
            "criterio_do_checkpoint": melhor.get("criterio"),
            "metricas_internas_do_checkpoint": {
                k: melhor.get(k) for k in
                ("ndcg_10", "recall_1", "recall_10", "mrr", "n_candidatos")},
            "config_do_treino": (treino or {}).get("config"),
            "pares_por_s": (treino or {}).get("metricas", {}).get("pares_por_s"),
            "lacuna_de_registro": ausencia,
            "constante_do_sistema": RECUPERADOR,
            "substitui_a_constante": substitui,
            "o_que_a_medicao_NAO_diz": (
                "que este ΦEmb sirva fora de recuperação por citação. O benchmark é "
                "NOSSO, a noção de relevância é 'foi citado' — proxy enviesada e sem "
                "juízo humano —, e o nDCG@10 tem UM relevante por consulta. As "
                "métricas internas do checkpoint são a 1.000 candidatos e NÃO são "
                "comparáveis com o número do portão, a 2.000: confundir os dois "
                "protocolos é o que produziu o G1.2 falso de 2026-08-27. E o número "
                "do portão só significa algo com o teto ao lado: no protocolo "
                "anterior ele era 0,7562."),
        },
        registros=melhor.get("passo", 0))

    tam = sum(f.stat().st_size for f in a.para.iterdir() if f.is_file())
    print("=" * 74)
    print(f"  {a.para}")
    print(f"  base            : {base}")
    print(f"  passo           : {melhor.get('passo')} · critério "
          f"{melhor.get('criterio')}")
    print(f"  portão G1       : nDCG@10 {a.ndcg_do_portao:.4f} de um teto de "
          f"{a.teto_do_portao:.4f} (2.000 candidatos)")
    print(f"  bytes           : {tam/1e6:.1f} MB")
    print(f"  nota            : {a.nota}")
    if ausencia:
        print(f"  ⚠️ {ausencia}")
    if substitui:
        print(f"  ⚠️ `phifm.core.modelos.RECUPERADOR` ainda aponta para "
              f"{RECUPERADOR}.")
        print("     Instalar os pesos NÃO troca o recuperador do sistema — a "
              "constante é")
        print("     uma edição deliberada, e trocar exige remedir a cadeia do "
              "T1b/T1c,")
        print("     cuja referência (nDCG 0,1666) foi medida com o recuperador "
              "antigo.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
