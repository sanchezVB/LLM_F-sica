#!/usr/bin/env python3
"""ΦEmb — fine-tune contrastivo sobre pares de citação (DOC-07 §3).

Roda na venv de TREINO, que é Python 3.12: o `torch-directml` não suporta 3.14.

    .venv-treino/Scripts/python.exe scripts/train_embedding.py --max-pares 20000
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polars as pl  # noqa: E402

from phifm.training.embedding import BASE_PADRAO, Config, TreinadorEmb  # noqa: E402
from phifm.core.sistema import impedir_suspensao, liberar_suspensao  # noqa: E402


def _com_negativos_dificeis(a) -> tuple:
    """Carrega o parquet minerado e acrescenta `proibidos` por âncora.

    ⚠️ `proibidos` é montado AQUI, da tabela inteira de arestas, e não vem do
    parquet minerado. Custa uma passada de ~3 s em 6,56 M linhas, e a razão de não
    guardá-lo no artefato é que ele é derivado: guardá-lo criaria uma segunda
    cópia da verdade, que pode ficar velha se as arestas mudarem.

    Ele existe porque os negativos são COMPARTILHADOS no lote: a âncora `i` é
    pontuada contra o negativo difícil da âncora `j`, e esse documento pode ser
    citação verdadeira de `i`. Sem a máscara, o modelo é penalizado por acertar.
    """
    treino = pl.read_parquet(a.negativos)
    if a.max_pares:
        treino = treino.head(a.max_pares)
    proib = (pl.scan_parquet(a.pares / "pares_treino.parquet")
             .group_by("arxiv_id")
             .agg(pl.col("arxiv_citado").unique().alias("proibidos"))
             .collect())
    antes = treino.height
    treino = treino.join(proib, on="arxiv_id", how="left")
    # `join` à esquerda não pode perder nem criar linha. Se perdesse, o treino
    # rodaria sobre outro conjunto e a comparação com o campeão deixaria de isolar
    # a dificuldade dos negativos.
    if treino.height != antes:
        raise SystemExit(f"a junção mudou o número de linhas: {antes:,} -> "
                         f"{treino.height:,}. Não treine sobre isto.")
    vazios = int(treino["negativos_id"].list.len().eq(0).sum())
    logging.info("negativos difíceis de %s · %s pares · %s sem negativo (%.2f%%)",
                 a.negativos, f"{treino.height:,}", f"{vazios:,}",
                 100 * vazios / max(treino.height, 1))
    return treino, treino.height


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pares", type=Path, default=Path("data/processed/pares"))
    p.add_argument("--out", type=Path, default=Path("models/phiemb"))
    p.add_argument("--base", default=BASE_PADRAO)
    p.add_argument("--lote", type=int, default=16,
                   help="lote LÓGICO: é dele que saem os negativos do InfoNCE")
    p.add_argument("--sub-lote", type=int, default=None,
                   help="liga o GradCache. É o que vai à GPU de cada vez; o `--lote` "
                        "passa a ser só lógico. `--lote 512 --sub-lote 128` dá 511 "
                        "negativos com a memória de 128. Ver test_gradcache.py")
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-pares", type=int, default=None,
                   help="teto de pares nesta execução; 1,65 M leva ~90 h aqui")
    p.add_argument("--passos-aval", type=int, default=500)
    p.add_argument("--n-candidatos", type=int, default=256,
                   help="pool da avaliação; 256 dá erro padrão de ±0,031")
    p.add_argument("--dispositivo", default="auto",
                   choices=["auto", "cuda", "dml", "cpu"])
    # ⚠️ `--sem-amp` desliga precisão mista. Existe por dois motivos concretos, não
    # por completude: (a) o GradCache com AMP produziria gradiente errado em
    # silêncio e é recusado, então quem precisa de lote lógico grande num CUDA
    # desliga aqui; (b) para comparar contra os treinos já feitos, que foram todos
    # em fp32 — mudar precisão E dados ao mesmo tempo não isola nada, que é o erro
    # que este projeto cometeu ao mudar base e lote juntos.
    p.add_argument("--sem-amp", action="store_true",
                   help="desliga fp16; obrigatório com --sub-lote em CUDA")
    p.add_argument("--negativos", type=Path, default=None,
                   help="parquet de negativos difíceis (scripts/minerar_negativos.py); "
                        "SUBSTITUI os pares, porque já os contém")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    # ⚠️ Leitura PREGUIÇOSA com o corte antes do `collect`. `pl.read_parquet`
    # carregava os 6,56 M de pares inteiros — 2,46 GB comprimidos que expandem
    # para 8+ GB de texto — e só depois `--max-pares` cortava. Medido em
    # 2026-08-07: 1,0 GB de RAM livre, 31 GB de swap, e o processo morria antes
    # do primeiro passo sem deixar traceback. O corte tem de acontecer no plano,
    # não depois dele.
    if a.negativos:
        treino, total = _com_negativos_dificeis(a)
    else:
        plano = pl.scan_parquet(a.pares / "pares_treino.parquet")
        total = plano.select(pl.len()).collect().item()
        treino = (plano.head(a.max_pares) if a.max_pares else plano).collect()
    val = pl.read_parquet(a.pares / "pares_validacao.parquet")
    logging.info("pares: %s de %s disponíveis · %s validação",
                 f"{treino.height:,}", f"{total:,}", f"{val.height:,}")

    cfg = Config(base=a.base, lote=a.lote, sub_lote=a.sub_lote,
                 max_tokens=a.max_tokens, lr=a.lr,
                 max_pares=a.max_pares, passos_aval=a.passos_aval,
                 n_candidatos=a.n_candidatos, dispositivo=a.dispositivo, amp=not a.sem_amp)
    if a.sub_lote:
        logging.info("GradCache ligado: lote lógico %d (%d negativos) com a memória "
                     "de %d", a.lote, a.lote - 1, a.sub_lote)
    # ⚠️ Sem isto o treino morre quando a máquina dorme. Medido em 2026-08-14:
    # parou no passo 150 de 781 porque o Windows suspendeu às 02h49 — e ninguém
    # segurava, porque só as COLETAS chamavam isto. Enquanto uma coleta rodava o
    # treino pegava carona; quando todas terminaram, ele ficou sem proteção.
    impedir_suspensao()
    try:
        m = TreinadorEmb(cfg).treinar(treino, val, a.out)
    finally:
        liberar_suspensao()

    antes = m.historico[0]
    print(f"\n{'':22} {'recall@1':>10} {'recall@10':>11} {'MRR':>8}")
    print(f"{'antes (base)':22} {antes['recall_1']:10.3f} {antes['recall_10']:11.3f} {antes['mrr']:8.3f}")
    print(f"{'depois (ΦEmb)':22} {m.recall_1:10.3f} {m.recall_10:11.3f} {m.mrr:8.3f}")
    d1 = m.recall_1 - antes["recall_1"]
    print(f"{'ganho':22} {d1:+10.3f} {m.recall_10-antes['recall_10']:+11.3f} "
          f"{m.mrr-antes['mrr']:+8.3f}")
    # ⚠️ Formato `passo N | CONCLUIDO` de propósito, igual ao das linhas de
    # progresso. As linhas de progresso saem a cada 50 passos, então a última é
    # `passo 750` num treino de 781 — e um vigia que lê `passo (\d+)` do log
    # anuncia "750 de 781", que se lê como morte 31 passos antes do fim.
    #
    # Custou exatamente isso em 2026-08-16: o treino completou os 781 e a
    # notificação disse que não. O total tem de estar no mesmo formato que o
    # progresso, senão quem lê o log distingue os dois e quem faz grep não.
    print(f"\npasso {m.passo} | CONCLUIDO | {m.pares_por_s:.1f} pares/s · → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
