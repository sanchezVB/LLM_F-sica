#!/usr/bin/env python3
"""Sprint S3 · 4d — peS2o e OpenWebMath filtrados pelo `is_physics`.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/filtrar_hf.py --fonte openwebmath

Baixa, filtra e apaga arquivo por arquivo: nunca há mais de um em disco. Ver a
docstring de `phifm.corpus.slices.hf_filtrado` para o limiar e para o que NENHUM
número nosso responde sobre texto de web.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.core.env import contato_obrigatorio  # noqa: E402
from phifm.core.sistema import impedir_suspensao, liberar_suspensao  # noqa: E402
from phifm.corpus.slices.hf_filtrado import LIMIAR, filtrar  # noqa: E402
from phifm.corpus.slices.retomada import feitas  # noqa: E402
from phifm.core.schema.reprodutibilidade import (  # noqa: E402
    Entrada,
    gravar_manifesto_etapa,
)

FONTES = {
    "openwebmath": ("open-web-math/open-web-math", (".parquet",)),
    "pes2o": ("allenai/peS2o", (".parquet", ".json.gz", ".jsonl.gz")),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fonte", choices=sorted(FONTES), required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--modelo", type=Path, default=Path("models/isphysics-clf"))
    p.add_argument("--limiar", type=float, default=LIMIAR)
    p.add_argument("--max-arquivos", type=int, default=None)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    ds, ext = FONTES[a.fonte]
    out = a.out or Path(f"data/processed/{a.fonte}_fisica")

    impedir_suspensao()
    try:
        f = filtrar(ds, out, a.modelo, limiar=a.limiar, max_arquivos=a.max_arquivos,
                    ext=ext, contato=contato_obrigatorio())
    finally:
        liberar_suspensao()

    print("\n" + "=" * 72)
    print(f"{a.fonte} filtrado · limiar {a.limiar} · {f.arquivos_lidos} arquivos")
    print()
    print(f"  vistos            : {f.vistos:,}")
    print(f"  aceitos           : {f.aceitos:,}  ({100*f.taxa:.2f}%)")
    print(f"  lidos da rede     : {f.bytes_lidos/1e9:.1f} GB")
    print(f"  texto aceito      : {f.caracteres_aceitos/1e9:.2f} G caracteres")
    print(f"  ≈ tokens          : {f.caracteres_aceitos/4/1e9:.2f} B (a ~4 chars/token)")
    print(f"  falhas            : {len(f.falhas)}")
    for x in f.falhas[:5]:
        print(f"    · {x}")
    if f.dominios:
        print()
        print("  DOMÍNIOS do que foi aceito — o sinal objetivo sobre o que entrou:")
        for d, n in f.dominios.most_common(15):
            print(f"    {n:>8,}  {d}")
    print()
    print("=" * 72)
    print("⚠️ Os falsos positivos de 1,5%–13,6% que justificam o limiar 0,9 foram")
    print("   medidos em RESUMOS DO ARXIV. Isto é texto de web e paper completo —")
    print("   outra distribuição. A amostra abaixo existe para um humano decidir;")
    print("   inventar uma taxa de contaminação daqui seria extrapolar demais.")

    out.mkdir(parents=True, exist_ok=True)
    # ⚠️ `vistos`/`aceitos`/`dominios` são desta EXECUÇÃO, não do acumulado. Uma
    # coleta retomável roda várias vezes, e cada uma sobrescreve este arquivo com
    # os seus próprios números — enquanto os parquets acumulam.
    #
    # Isso já produziu leitura errada: o relatório comparava `aceitos` (233.079,
    # da última execução) com a contagem dos parquets (860.521, acumulada) e
    # concluía "em curso" para uma coleta CONCLUÍDA. A comparação era entre coisas
    # de naturezas diferentes, então diria "em curso" para sempre depois do
    # segundo lançamento.
    #
    # `total_unidades` e `concluido` existem para responder "acabou?" sem
    # comparação nenhuma: unidades feitas contra unidades que a fonte publica.
    feitas_ate_agora = feitas(out)
    (out / "_filtragem.json").write_text(json.dumps(
        {"fonte": f.fonte, "revisao": f.revisao, "limiar": a.limiar,
         "arquivos": f.arquivos_lidos,
         "vistos": f.vistos, "aceitos": f.aceitos, "taxa": round(f.taxa, 5),
         "bytes_lidos": f.bytes_lidos, "caracteres": f.caracteres_aceitos,
         "unidades_feitas": len(feitas_ate_agora),
         "total_unidades": f.total_unidades,
         "concluido": bool(f.total_unidades
                           and len(feitas_ate_agora) >= f.total_unidades),
         "escopo_dos_numeros": "esta execução, não o acumulado",
         "dominios": dict(f.dominios.most_common(60)), "falhas": f.falhas},
        indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "_amostra_para_revisao.json").write_text(
        json.dumps(f.amostra, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\namostra de {len(f.amostra)} documentos → {out/'_amostra_para_revisao.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
