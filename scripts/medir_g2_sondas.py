#!/usr/bin/env python3
"""As sondas de contaminação S1 e S2 do G2 (DOC-12 §3.7.9) no llama-server do G2.

    # DEV: 300 do UG-dev, o OlympiadBench-zh e os _sol do SciBench, para conferir o caminho
    .venv\\Scripts\\python.exe scripts\\medir_g2_sondas.py dev --modelo 1.7B
    # TESTE: os 2.528 itens de física; exige o congelamento e registra o acesso (nota "sondas")
    .venv\\Scripts\\python.exe scripts\\medir_g2_sondas.py teste --modelo 1.7B --braco base

As fichas geradas ficam em `g2_saidas/` (fora do git, retomáveis). O agregado vai para
`data/processed/avaliacao/g2_sondas_<fase>_<modelo>_<braco>.json`. No teste, ele leva só os
contadores e o SHA-256 das listas de ids; as listas (por item) ficam lacradas fora do git,
em `g2_saidas/sondas_teste_<modelo>_<braco>_listas.json`, com exatamente os bytes do hash
(§3.7.3: até o veredito, nada se lê por item). No dev, as listas vão no próprio agregado.
As sondas usam todos os itens do teste, inclusive os que o autoteste tira da correção: elas
medem memorização do enunciado e da solução, não dependem do gabarito. Sai com 1 se faltar
sonda (a retomada refaz só o que falta).

Um slot só, um pedido por vez, sempre na mesma ordem. Com 4 slots, o servidor junta no mesmo
lote pedidos de slots diferentes, e a conta da geração gulosa muda conforme a companhia: no
dev, duas rodadas de 4 slots com os mesmos prompts deram gerações diferentes em 42% das
sondas, e 11 de 785 sinais trocaram. Com 1 slot, servidor novo e a mesma ordem, o dev
refeito deu as mesmas 785 gerações, ficha a ficha. A ordem conta: o estado que um pedido
deixa no servidor mexe no arredondamento do seguinte. Numa ordem diferente, ~10% das
gerações mudam em alguma ficha depois da 8ª, e nenhum sinal mudou em 262. Uma retomada (um
servidor novo no meio) é uma ordem diferente dali em diante.
"""
from __future__ import annotations

import argparse
import http.client
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2, g2_geracao, g2_itens, g2_sondas  # noqa: E402

AVAL = RAIZ / "data/processed/avaliacao"
SAIDAS = AVAL / "g2_saidas"
MODELOS = {"1.7B": ("Qwen3-1.7B-Base", RAIZ / "models/base/gguf/Qwen3-1.7B-Base-Q8_0.gguf"),
           "8B": ("Qwen3-8B-Base", RAIZ / "models/base/gguf/Qwen3-8B-Base-Q8_0.gguf")}
# O ΦGen entra por emenda, com o GGUF dele: até lá, um --braco errado gastaria o acesso do
# ΦGen sondando a base.
BRACOS = ("base",)
N_UG_DEV = 300
SLOTS = 1                     # a geração gulosa só repete sem pedidos de outros slots no lote
ERROS_SEGUIDOS = 3            # servidor morto ou travado: para, e a retomada sobe outro
CODIGO = ("src/phifm/eval/g2_sondas.py", "scripts/medir_g2_sondas.py")


def itens_da_fase(fase: str) -> list[g2_itens.Item]:
    if fase == "teste":
        return [i for b in g2.BENCH_G2_1 for i in g2_itens.carregar(b, "teste")]
    ug = sorted(g2_itens.carregar("ugphysics", "dev"), key=lambda i: i.id)
    ug = random.Random(g2.SEMENTE_G2 + 700).sample(ug, min(N_UG_DEV, len(ug)))
    return ug + g2_itens.carregar("olympiadbench", "dev") + g2_itens.carregar("scibench", "dev")


def ja_feitas(saida: Path, hc: str) -> dict[tuple[str, str, str], dict]:
    """As sondas gravadas com este hash e sem 'erro' (a última vale), depois de aparar a
    cauda de uma queda de energia."""
    if not saida.exists():
        return {}
    cortados = g2_sondas.aparar(saida)
    if cortados:
        print(f"   {saida.name}: {cortados} bytes de linha cortada aparados", flush=True)
    return {(d["bench"], d["item_id"], d["sonda"]): d for d in g2_geracao.ler_jsonl(saida)
            if d.get("hash_config") == hc and "erro" not in d}


def rodar(fase: str, modelo: str, braco: str, itens: list[g2_itens.Item]) -> dict:
    nome, gguf = MODELOS[modelo]
    kl = json.loads((AVAL / f"g2_kl_{modelo}.json").read_text(encoding="utf-8"))
    pedido = g2_sondas.pedido({p: g2.sha256_arquivo(RAIZ / p) for p in CODIGO})
    rotulo = f"sondas_{fase}_{modelo}_{braco}"
    saida = SAIDAS / f"{rotulo}.jsonl"
    SAIDAS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with g2_geracao.ServidorG2(gguf, slots=SLOTS, log=SAIDAS / f"{rotulo}.servidor.log") as srv:
        hc = g2_geracao.hash_da_configuracao(srv, kl["gguf"]["q8_0"]["sha256"], g2_geracao.GPU, pedido)
        teto = g2_sondas.contexto_do_slot(srv.url)
        feitas = ja_feitas(saida, hc)              # antes do livro: uma queda não gasta acesso
        if fase == "teste":
            g2.exigir_aceite_e_congelamento()
            continua = g2.medida_aberta(modelo, braco, hc, nota="sondas")
            g2.registrar_acesso(modelo, braco, "retomada" if continua else "medida", hc,
                                nota="sondas")
        fila = [(i, s) for i in itens for s in g2_sondas.SONDAS if (i.bench, i.id, s) not in feitas]
        # cada execução (um servidor novo) é uma ordem nova dali em diante: fica marcada
        execucao = datetime.now(UTC).isoformat(timespec="seconds")
        erros = seguidos = 0
        for n, (i, s) in enumerate(fila, start=1):
            try:
                r = g2_sondas.sondar(srv.url, i.enunciado, i.solucao, s, pedido, teto=teto)
                seguidos = 0
            except (OSError, ValueError, KeyError, http.client.HTTPException) as e:
                # o pedido falhou: a retomada refaz; sem o texto do item, só o porquê
                r = {"erro": type(e).__name__, "http": getattr(e, "code", None),
                     "msg": str(e)[:200]}
                erros, seguidos = erros + 1, seguidos + 1
            linha = {"bench": i.bench, "item_id": i.id, "sonda": s, **r, "hash_config": hc,
                     "execucao": execucao, "ordem": n}
            with saida.open("a", encoding="utf-8") as f:
                f.write(g2_geracao.linha_jsonl(linha) + "\n")
            if seguidos and (seguidos >= ERROS_SEGUIDOS or not srv.no_ar()):
                print(f"   {seguidos} erros seguidos ou servidor fora do ar: paro, e a retomada "
                      "sobe um servidor novo", flush=True)
                break
            if n % 250 == 0:
                print(f"   {n}/{len(fila)} sondas · {erros} erros · {time.time() - t0:.0f} s",
                      flush=True)
    registros = ja_feitas(saida, hc)
    linhas = [d for d in g2_geracao.ler_jsonl(saida) if d.get("hash_config") == hc]
    execucoes = {}
    for d in linhas:
        execucoes[d.get("execucao", "?")] = execucoes.get(d.get("execucao", "?"), 0) + 1
    esperados = [(i.bench, i.id, s) for i in itens for s in g2_sondas.SONDAS]
    faltam = sum(k not in registros for k in esperados)
    agregado = g2_sondas.agregar([registros[k] for k in esperados if k in registros])
    listas = agregado.pop("listas")
    if fase == "teste":                            # o lacre: por item, só fora do git
        (SAIDAS / f"{rotulo}_listas.json").write_bytes(g2_sondas.canonico(listas))
    else:
        agregado["listas"] = listas
    resultado = {"o_que_e": f"G2, sondas de contaminação S1 e S2, fase {fase} (DOC-12 §3.7.9 A)",
                 "modelo": nome, "braco": braco, "hash_config": hc,
                 "codigo_sha256": pedido["sondas"]["codigo"], "parametros": g2_sondas.parametros(),
                 "slots": SLOTS, "contexto_do_slot": teto, "segundos": round(time.time() - t0),
                 "itens": len(itens), "registros": len(esperados) - faltam, "faltam": faltam,
                 "linhas_de_erro": sum("erro" in d for d in linhas), "execucoes": execucoes,
                 **agregado}
    destino = AVAL / f"g2_sondas_{fase}_{modelo}_{braco}.json"
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return resultado


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("fase", choices=("dev", "teste"))
    p.add_argument("--modelo", choices=tuple(MODELOS), default="1.7B")
    p.add_argument("--braco", choices=BRACOS, default="base")
    a = p.parse_args()
    if a.fase == "teste":
        g2.exigir_aceite_e_congelamento()          # antes de LER o teste
    itens = itens_da_fase(a.fase)
    print(f"sondas {a.fase}: {len(itens)} itens × {len(g2_sondas.SONDAS)} sondas", flush=True)
    r = rodar(a.fase, a.modelo, a.braco, itens)
    for b, ss in r["por_benchmark"].items():
        print(f"{b}: " + " · ".join(
            f"{s} {x['suspeitos']}/{x.get('aplicaveis', x.get('com_sonda_aplicavel'))} "
            f"(≥{g2_sondas.TRECHO_LONGO}: {x['trecho_longo']})" for s, x in ss.items()))
    print(f"nulo ({r['nulo']}): {r['fracao_no_nulo']} · listas {r['sha256_listas'][:12]}… · "
          f"faltam {r['faltam']} · {r['segundos']} s")
    return 0 if r["faltam"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
