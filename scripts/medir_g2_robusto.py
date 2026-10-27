#!/usr/bin/env python3
"""Executor experimental separado do protocolo oficial congelado.

A medida de física do G2 (DOC-12 §3.7): gera, corrige e agrega.

    # DEV (permitido agora): itens de dev, k=1, para afinar extrator e corretor
    .venv\\Scripts\\python.exe scripts\\medir_g2_robusto.py dev --modelo 1.7B --iteracao 1 [--ug 300]

    # TESTE: recusa sem o aceite da regra, sem o congelamento e sem o livro de acessos
    .venv\\Scripts\\python.exe scripts\\medir_g2_robusto.py teste --modelo 1.7B --braco base

Na fase de dev, cada iteração usa ~300 itens sorteados do UG-dev (com `--ug 0`, o UG-dev
inteiro), os 115 do OlympiadBench-zh e os `_sol` do SciBench que não viraram exemplo. Itens
que viraram exemplo não são medidos.

Saídas:
- as gerações, em `data/processed/avaliacao/g2_saidas/` (fora do git);
- os agregados, em `data/processed/avaliacao/g2_<fase>_<modelo>_<rótulo>.json` (versionados):
  PASS, sem resposta, truncado, INCONCLUSIVE e ERROR por benchmark, e as fichas geradas.

O teste só lê agregados até o veredito (§3.7.3, lacre). O GPQA não existe aqui (decisão do
dono, 2026-10-19).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2, g2_itens, g2_prerequisitos, g2_prompt  # noqa: E402
from phifm.eval import g2_geracao_robusta as g2_geracao
from phifm.eval.g2_correcao import CorretorIsolado  # noqa: E402
from phifm.eval.g2_corretor import Veredito  # noqa: E402

AVAL = RAIZ / "data/processed/avaliacao"
SAIDAS = AVAL / "g2_saidas"
MODELOS = {"1.7B": ("Qwen3-1.7B-Base", RAIZ / "models/base/gguf/Qwen3-1.7B-Base-Q8_0.gguf"),
           "8B": ("Qwen3-8B-Base", RAIZ / "models/base/gguf/Qwen3-8B-Base-Q8_0.gguf")}
K_TESTE = {"ugphysics": 3, "olympiadbench": 8, "phybench": 16, "scibench": 8}
UG_DEV_POR_ITERACAO = 300


def artefato_do_braco(modelo: str, braco: str, gguf: Path | None = None,
                      kl: Path | None = None, manifesto_cpt: Path | None = None
                      ) -> tuple[str, Path, Path]:
    """Um rótulo novo não transforma os pesos da base em um candidato treinado."""
    if modelo not in MODELOS:
        raise ValueError("tamanho de modelo desconhecido")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", braco):
        raise ValueError("braço deve usar apenas letras, números, _ e -")
    nome, base = MODELOS[modelo]
    kl_base = AVAL / f"g2_kl_{modelo}.json"
    if braco == "base":
        if any(p is not None for p in (gguf, kl, manifesto_cpt)):
            raise ValueError("o braço base usa os artefatos oficiais; use outro braço para CPT")
        return nome, base, kl_base
    if gguf is None or kl is None or manifesto_cpt is None:
        raise ValueError("um candidato CPT exige --gguf, --kl e --manifesto-cpt explícitos")
    if gguf.resolve() == base.resolve() or kl.resolve() == kl_base.resolve():
        raise ValueError("um candidato CPT não pode reutilizar os artefatos da base")
    return f"ΦGen-CPT ({nome}; {braco})", gguf, kl


def aplicar_exclusoes(itens: list[g2_itens.Item], exclusoes: dict[str, set[str]]
                       ) -> list[g2_itens.Item]:
    return [i for i in itens if i.id not in exclusoes.get(i.bench, set())]


def _exemplos() -> dict[str, list[g2_itens.Item]]:
    escolha = json.loads((AVAL / "g2_exemplos.json").read_text(encoding="utf-8"))["exemplos"]
    por_id = {i.id: i for i in g2_itens.carregar("ugphysics", "dev")
              + g2_itens.carregar("scibench", "dev")}
    return {b: [por_id[i] for i in e["ids"]] for b, e in escolha.items()}


def _exemplos_do(item: g2_itens.Item, exemplos: dict[str, list[g2_itens.Item]]
                 ) -> list[g2_itens.Item]:
    if item.bench != "scibench":
        return exemplos[item.bench]
    return [e for e in exemplos["scibench"] if e.extra["livro"] == item.extra["livro"]]


def itens_de_dev(n_ug: int, iteracao: int, exemplos: dict[str, list[g2_itens.Item]]
                 ) -> list[g2_itens.Item]:
    usados = {e.id for es in exemplos.values() for e in es}
    ug = sorted((i for i in g2_itens.carregar("ugphysics", "dev") if i.id not in usados),
                key=lambda i: i.id)
    if n_ug:
        ug = random.Random(g2.SEMENTE_G2 + 100 + iteracao).sample(ug, min(n_ug, len(ug)))
    oly = g2_itens.carregar("olympiadbench", "dev")
    sci = [i for i in g2_itens.carregar("scibench", "dev") if i.id not in usados]
    return ug + oly + sci


def chaves_esperadas(itens: list[g2_itens.Item], k: dict[str, int]) -> set[tuple[str, str, int]]:
    ids = {(i.bench, i.id) for i in itens}
    if len(ids) != len(itens):
        raise ValueError("itens duplicados na medição")
    if not itens:
        raise ValueError("conjunto de itens vazio")
    if any(type(k.get(i.bench)) is not int or k[i.bench] < 1 for i in itens):
        raise ValueError("k deve ser inteiro positivo para cada benchmark")
    return {(i.bench, i.id, j) for i in itens for j in range(k[i.bench])}


def _agregar(itens: list[g2_itens.Item], amostras: list[dict], vereditos: dict,
             k: dict[str, int]) -> dict:
    amostras = g2_geracao.unificar_amostras(amostras)
    esperadas = chaves_esperadas(itens, k)
    g2_geracao.exigir_completude(amostras, esperadas)
    if set(vereditos) != esperadas:
        raise ValueError("correção incompleta ou com vereditos inesperados")
    por_bench: dict[str, dict] = {}
    for bench in sorted({i.bench for i in itens}):
        linhas = [a for a in amostras if a["bench"] == bench]
        cont = collections.Counter(vereditos[(a["bench"], a["item_id"], a["j"])] for a in linhas)
        n = len(linhas)
        fichas = [a["fichas"] for a in linhas]
        por_bench[bench] = {
            "itens": len({a["item_id"] for a in linhas}), "amostras": n,
            "acerto": round(cont[Veredito.PASS] / n, 4) if n else None,
            "fracoes": {str(v): round(c / n, 4) for v, c in sorted(cont.items())},
            "fichas_media": round(statistics.mean(fichas), 1) if fichas else None,
            "fichas_p95": (sorted(fichas)[int(0.95 * (len(fichas) - 1))] if fichas else None),
        }
    return por_bench


def rodar(fase: str, modelo: str, rotulo: str, itens: list[g2_itens.Item],
          exemplos: dict[str, list[g2_itens.Item]], k: dict[str, int], braco: str, *,
          gguf: Path | None = None, kl: Path | None = None,
          manifesto_cpt: Path | None = None, executavel: Path | None = None,
          gpu: str = "RX 7600 (Vulkan)") -> dict:
    if fase not in ("dev", "teste"):
        raise ValueError("fase deve ser dev ou teste")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", rotulo):
        raise ValueError("rótulo deve usar apenas letras, números, _ e -")
    nome, arquivo_gguf, arquivo_kl = artefato_do_braco(modelo, braco, gguf, kl, manifesto_cpt)
    confere = {"raiz": RAIZ, "fase": fase, "modelo": modelo, "braco": braco,
               "gguf": arquivo_gguf, "kl": arquivo_kl, "manifesto_cpt": manifesto_cpt,
               "executavel": executavel}
    requisitos = g2_prerequisitos.exigir(**confere)
    itens = aplicar_exclusoes(itens, requisitos.exclusoes)
    relatorio_kl = json.loads(arquivo_kl.read_text(encoding="utf-8"))
    pedido = g2_geracao.amostragem()
    saida = SAIDAS / f"{fase}_{modelo}_{braco}_{rotulo}.jsonl"
    esperadas = chaves_esperadas(itens, k)
    pedidos = [(i.bench, i.id, j, g2_prompt.montar_prompt(i, _exemplos_do(i, exemplos)))
               for i in itens for j in range(k[i.bench])]
    log = SAIDAS / f"{fase}_{modelo}_{braco}_{rotulo}.servidor.log"
    destino = AVAL / f"g2_{fase}_{modelo}_{braco}_{rotulo}.json"
    t0 = time.time()
    srv = g2_geracao.ServidorG2(arquivo_gguf, log=log,
                               executavel=executavel or g2_geracao.EXECUTAVEL)
    hc = g2_geracao.hash_da_configuracao(srv, relatorio_kl["gguf"]["q8_0"]["sha256"],
                                        gpu, pedido)
    g2_prerequisitos.exigir(**confere, hash_config=hc)
    codigo = g2.codigo_do_corretor(RAIZ, extras=tuple(
        f"src/phifm/eval/{n}.py" for n in ("g2_prompt", "g2_geracao", "g2_correcao")))
    artefatos = {"codigo": codigo, "preregistro": g2.sha256_arquivo(g2.PREREGISTRO),
                "kl": g2.sha256_arquivo(arquivo_kl), "exclusoes": {
                    b: sorted(ids) for b, ids in requisitos.exclusoes.items()}}
    if manifesto_cpt is not None:
        artefatos["manifesto_cpt"] = g2.sha256_arquivo(manifesto_cpt)
    he = g2_geracao.hash_da_execucao(hc, pedidos, g2_prompt.PARADAS, pedido,
                                   artefatos=artefatos)
    # Recusar mistura antes de iniciar o servidor ou registrar acesso ao teste.
    anteriores = g2_geracao.carregar_amostras(saida, hc, he)
    if destino.exists():
        final = json.loads(destino.read_text(encoding="utf-8"))
        if (final.get("hash_execucao") != he or not saida.exists()
                or final.get("sha256_amostras") != g2_geracao.sha256_bytes(saida)):
            raise ValueError("resultado já publicado com outra identidade ou log alterado")
        g2_geracao.exigir_completude(anteriores, esperadas)
        # ERROR também é resultado: depois de publicar, não se sorteia outra tentativa
        # por conhecer a nota. Recuperação operacional acontece antes do agregado final.
        return final
    if fase == "teste":
        g2.exigir_aceite_e_congelamento()
        continua = bool(anteriores)
        g2.registrar_acesso(modelo, braco, "retomada" if continua else "medida", hc)
    SAIDAS.mkdir(parents=True, exist_ok=True)
    with srv:
        lote = g2_geracao.gerar_lote(srv.url, pedidos, g2_prompt.PARADAS, pedido, saida, hc,
                                    artefatos=artefatos)
        for feitas, a in enumerate(lote, start=1):
            if feitas % 25 == 0:
                print(f"   {feitas}/{len(pedidos)} amostras · {time.time() - t0:.0f} s · "
                      f"última: {a.bench} {a.fichas} fichas ({a.parada})", flush=True)
    por_id = {(i.bench, i.id): i for i in itens}
    amostras = g2_geracao.carregar_amostras(saida, hc, he)
    g2_geracao.exigir_completude(amostras, esperadas)
    # A correção roda DEPOIS da geração, com o servidor já parado (a máquina sem outra carga
    # pesada), num processo que pode ser morto (g2_correcao).
    vereditos = {}
    with CorretorIsolado() as cor:
        for a in amostras:
            if a["parada"] == "erro":
                v = Veredito.ERROR
            elif a["truncada"]:
                v = Veredito.TRUNCADO
            else:
                v = cor.corrigir(por_id[(a["bench"], a["item_id"])], a["texto"])
            vereditos[(a["bench"], a["item_id"], a["j"])] = v
        mortes = cor.mortes
    resultado = {
        "o_que_e": f"G2, fase {fase}: agregados por benchmark (DOC-12 §3.7); as gerações ficam "
                   "fora do git",
        "modelo": nome, "braco": braco, "rotulo": rotulo, "hash_config": hc,
        "hash_execucao": he,
        "sha256_amostras": g2_geracao.sha256_bytes(saida),
        "exclusoes_por_benchmark": {b: len(ids) for b, ids in requisitos.exclusoes.items()},
        "codigo_sha256": codigo,
        "correcoes_mortas_no_teto": mortes,
        "k": k, "segundos": round(time.time() - t0),
        "por_benchmark": _agregar(itens, amostras, vereditos, k),
    }
    if fase == "dev":       # só no dev: no teste, nada por item até o veredito (lacre)
        resultado["vereditos_por_amostra"] = {f"{b}|{i}|{j}": str(v)
                                              for (b, i, j), v in sorted(vereditos.items())}
    temporario = destino.with_suffix(".json.tmp")
    temporario.write_text(json.dumps(resultado, ensure_ascii=False, indent=1) + "\n",
                          encoding="utf-8")
    temporario.replace(destino)
    return resultado


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("fase", choices=("dev", "teste"))
    p.add_argument("--modelo", choices=tuple(MODELOS), default="1.7B")
    p.add_argument("--braco", default="base")
    p.add_argument("--gguf", type=Path, help="GGUF do candidato CPT (obrigatório fora da base)")
    p.add_argument("--kl", type=Path, help="conferência Q8/F16 do candidato CPT")
    p.add_argument("--manifesto-cpt", type=Path, help="auditoria do corpus usada pelo candidato")
    p.add_argument("--executavel", type=Path, help="llama-server da revisão fixada")
    p.add_argument("--gpu", default="RX 7600 (Vulkan)", help="hardware/backend da execução")
    p.add_argument("--iteracao", type=int, default=1)
    p.add_argument("--ug", type=int, default=UG_DEV_POR_ITERACAO,
                   help="itens do UG-dev por iteração (0 = todos)")
    a = p.parse_args()
    _, gguf, kl = artefato_do_braco(a.modelo, a.braco, a.gguf, a.kl, a.manifesto_cpt)
    requisitos = g2_prerequisitos.exigir(
        RAIZ, fase=a.fase, modelo=a.modelo, braco=a.braco, gguf=gguf, kl=kl,
        manifesto_cpt=a.manifesto_cpt, executavel=a.executavel)
    exemplos = _exemplos()
    if a.fase == "dev":
        itens = itens_de_dev(a.ug, a.iteracao, exemplos)
        k = dict.fromkeys(("ugphysics", "olympiadbench", "scibench"), 1)
        rotulo = f"it{a.iteracao}" + ("_ugcompleto" if a.ug == 0 else "")
    else:
        g2.exigir_aceite_e_congelamento()          # antes de LER qualquer coisa do teste
        itens = [i for b in g2.BENCH_G2_1 for i in g2_itens.carregar(b, "teste")]
        k, rotulo = K_TESTE, "veredito"
    itens = aplicar_exclusoes(itens, requisitos.exclusoes)
    print(f"{a.fase}: {len(itens)} itens, k={k}")
    r = rodar(a.fase, a.modelo, rotulo, itens, exemplos, k, a.braco, gguf=a.gguf, kl=a.kl,
              manifesto_cpt=a.manifesto_cpt, executavel=a.executavel, gpu=a.gpu)
    for b, x in r["por_benchmark"].items():
        print(f"{b}: acerto {x['acerto']} · {x['fracoes']} · fichas média {x['fichas_media']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
