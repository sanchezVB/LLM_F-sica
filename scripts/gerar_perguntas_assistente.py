#!/usr/bin/env python3
"""Os itens da medida do assistente (DOC-13 §9.1) e a guarda I1.

    # histórico: as versões escritas pelo Qwen, nos artigos de desenvolvimento
    .venv-treino\\Scripts\\python.exe scripts\\gerar_perguntas_assistente.py --etapa dev

    # 0. o conjunto formal é escrito pelo Claude (decisão do dono, DOC-13 §9.1):
    #    exportar um lote de resumos SEM título, escrever, importar pelas guardas
    .venv\\Scripts\\python.exe scripts\\gerar_perguntas_assistente.py --exportar primario -n 25
    .venv\\Scripts\\python.exe scripts\\gerar_perguntas_assistente.py --importar <respostas.json>

    # 1. as 40 da revisão (31 + 9) e a folha para o dono julgar
    .venv\\Scripts\\python.exe scripts\\gerar_perguntas_assistente.py --etapa revisao

    # 2. o dono julga e baixa o JSON; a apuração diz se I1 passou
    .venv-treino\\Scripts\\python.exe scripts\\gerar_perguntas_assistente.py --apurar <veredictos.json>

    # 3. só com I1 aprovada: o conjunto inteiro (500 + 150)
    .venv-treino\\Scripts\\python.exe scripts\\gerar_perguntas_assistente.py --etapa completa

As perguntas e os gabaritos ficam em `data/processed/assistente/`, fora do git: são o
conjunto de teste (DOC-11 §8.1). Vai para o git só o agregado, em
`data/processed/avaliacao/assistente_itens_<etapa>.json` e `assistente_i1.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8  # noqa: E402

utf8()

DIR = RAIZ / "data/processed/assistente"
FOLHA = DIR / "revisao_perguntas.html"
AVALIACAO = RAIZ / "data/processed/avaliacao"
I1_AGREGADO = RAIZ / "data/processed/avaliacao/assistente_i1.json"

MOTIVOS = {"valida": "válida", "confusa": "pergunta confusa ou dependente do artigo",
           "fora_do_resumo": "a resposta não está no resumo",
           "gabarito_errado": "gabarito errado"}

PAGINA = r"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Revisão das perguntas</title>
<style>
:root{--bg:#faf9f7;--fg:#1c1b19;--sutil:#6b6862;--linha:#e0ddd6;--cartao:#fff;
--sim:#1a7f4b;--nao:#b3261e;--realce:#f3f0e8}
@media (prefers-color-scheme:dark){:root{--bg:#16151a;--fg:#eceaf0;--sutil:#9b98a3;
--linha:#2f2d36;--cartao:#1e1d24;--sim:#4ec98a;--nao:#f2837b;--realce:#26242e}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.6 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
.env{max-width:52rem;margin:0 auto;padding:1.5rem 1rem 4rem}
h1{font-size:1.25rem;margin:0 0 .25rem}
.sutil{color:var(--sutil);font-size:.875rem}
.barra{height:6px;background:var(--linha);border-radius:3px;overflow:hidden;margin:1rem 0 .5rem}
.barra>div{height:100%;background:var(--sim);width:0;transition:width .2s}
.cartao{background:var(--cartao);border:1px solid var(--linha);border-radius:10px;
padding:1.1rem 1.2rem;margin:1rem 0}
.rotulo{font-size:.75rem;text-transform:uppercase;letter-spacing:.06em;color:var(--sutil);
margin:1rem 0 .25rem}
.rotulo:first-child{margin-top:0}
.pergunta{font-size:1.1rem;font-weight:600}
.resumo{white-space:pre-wrap;font:15px/1.65 ui-serif,Georgia,serif}
details{margin-top:.6rem}
summary{cursor:pointer}
details .resumo{margin-top:.4rem;color:var(--sutil)}
.acoes{display:flex;gap:.5rem;flex-wrap:wrap;margin:1rem 0 .25rem}
button{font:inherit;padding:.55rem 1rem;border-radius:8px;border:1px solid var(--linha);
background:var(--cartao);color:var(--fg);cursor:pointer}
button:hover{background:var(--realce)}
button.sim{border-color:var(--sim);color:var(--sim);font-weight:600}
button.nao{border-color:var(--nao);color:var(--nao)}
kbd{font:12px ui-monospace,monospace;border:1px solid var(--linha);border-radius:4px;
padding:1px 5px;color:var(--sutil)}
.aviso{border-left:3px solid var(--linha);padding:.5rem 0 .5rem .9rem;color:var(--sutil);
font-size:.875rem;margin:1rem 0}
.fim{text-align:center;padding:2rem 0}
</style></head><body><div class=env>
<h1>As perguntas da medida do assistente servem?</h1>
<p class=sutil>Cada pergunta foi escrita pelo Claude lendo só o resumo do artigo, sem o
título. Quem vai responder é o Qwen, que nunca viu o resumo. O resumo aparece
<b>traduzido</b> pelo Claude; o original em inglês abre logo abaixo dele e é o que vale
em caso de dúvida. Uma
pergunta é <b>válida</b> se: (1) se entende sozinha, sem ver o artigo; (2) a resposta
está no resumo; (3) o gabarito está certo. Leia a pergunta ANTES do resumo — é assim
que o assistente vai recebê-la.</p>
<div class=barra><div id=preenche></div></div>
<p class=sutil id=progresso></p>
<div id=area></div>
<div class=acoes id=acoes>
  <button class=sim onclick="julga('valida')">Válida <kbd>1</kbd></button>
  <button class=nao onclick="julga('confusa')">Confusa / depende do artigo <kbd>2</kbd></button>
  <button class=nao onclick="julga('fora_do_resumo')">Resposta não está no resumo <kbd>3</kbd></button>
  <button class=nao onclick="julga('gabarito_errado')">Gabarito errado <kbd>4</kbd></button>
  <button onclick="volta()">Voltar <kbd>&larr;</kbd></button>
</div>
<div class=aviso>Regra escrita antes (DOC-13 §9.1): se menos de <b>__MINIMO__ de __N__</b>
forem válidas, o gerador de perguntas é refeito antes de qualquer outra etapa. As
perguntas revisadas ficam no conjunto como estão — não há conserto item a item.</div>
<div class=acoes>
  <button onclick=baixar()>Baixar veredictos (JSON)</button>
  <button onclick=zerar()>Começar de novo</button>
</div>
<p class=sutil>Depois de baixar, é só me avisar onde o arquivo ficou.</p>
</div>
<script>
const ITENS = __DADOS__;
const CHAVE = "revisao_perguntas_" + "__ASSINATURA__";
const NOMES = __MOTIVOS__;
let v = {}, i = 0;
try { const g = localStorage.getItem(CHAVE); if (g) { const o = JSON.parse(g);
  v = o.v || {}; i = o.i || 0; } } catch (e) {}
function salva(){ try { localStorage.setItem(CHAVE, JSON.stringify({v:v, i:i})); } catch (e) {} }
function pinta(){
  const feitos = Object.keys(v).length;
  document.getElementById("preenche").style.width = (100*feitos/ITENS.length).toFixed(1) + "%";
  const validas = Object.values(v).filter(x => x === "valida").length;
  document.getElementById("progresso").textContent =
    feitos + " de " + ITENS.length + " julgadas · " + validas + " válidas";
  const area = document.getElementById("area");
  if (i >= ITENS.length){
    area.innerHTML = "<div class=fim><b>Fim.</b><br>Baixe os veredictos abaixo.</div>";
    document.getElementById("acoes").style.display = "none";
    return;
  }
  const d = ITENS[i];
  area.innerHTML = "<div class=cartao><p class=sutil>pergunta " + (i+1) + " de " +
    ITENS.length + "</p><div class=rotulo>pergunta</div><div class=pergunta id=p></div>" +
    "<div class=rotulo>gabarito</div><div id=g></div>" +
    "<div class=rotulo>resumo do artigo</div><div class=resumo id=r></div>" +
    (d.resumo_pt ? "<details><summary class=sutil>original em inglês</summary>" +
                   "<div class=resumo id=ro></div></details>" : "") +
    "<p class=sutil id=l></p></div>";
  // textContent, nunca innerHTML: pergunta, gabarito e resumo vêm de modelo e corpus.
  document.getElementById("p").textContent = d.pergunta;
  document.getElementById("g").textContent = d.gabarito;
  document.getElementById("r").textContent = d.resumo_pt ?
    d.titulo_pt + "\n\n" + d.resumo_pt : d.titulo + "\n\n" + d.resumo;
  if (d.resumo_pt) document.getElementById("ro").textContent = d.titulo + "\n\n" + d.resumo;
  document.getElementById("l").textContent = "arXiv:" + d.arxiv_id;
  document.getElementById("acoes").style.display = "flex";
}
function julga(x){ if (i >= ITENS.length) return; v[String(i)] = x; i++; salva(); pinta(); }
function volta(){ if (i > 0){ i--; delete v[String(i)]; salva(); pinta(); } }
function zerar(){ if (confirm("Apagar todos os julgamentos?")){ v = {}; i = 0; salva(); pinta(); } }
function baixar(){
  const linhas = Object.entries(v).map(([k, x]) => ({indice: +k, veredicto: x,
    motivo: NOMES[x], arxiv_id: ITENS[+k].arxiv_id, estrato: ITENS[+k].estrato}));
  const b = new Blob([JSON.stringify({assinatura_amostra: "__ASSINATURA__",
    n_amostra: ITENS.length, veredictos: linhas}, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b);
  a.download = "veredictos_perguntas.json";
  a.click();
}
addEventListener("keydown", e => {
  const m = {"1": "valida", "2": "confusa", "3": "fora_do_resumo", "4": "gabarito_errado"};
  if (m[e.key]) julga(m[e.key]); else if (e.key === "ArrowLeft") volta();
});
pinta();
</script></body></html>
"""


def montar_folha(aceitas, spine: Path, semente: int, minimo: int,
                 traducoes: dict | None = None) -> tuple[str, str]:
    """A folha com as `aceitas`, embaralhadas: julgar 31 de um estrato e depois 9 do
    outro confundiria o estrato com o cansaço de quem julga.

    `traducoes` ({arxiv_id: {titulo, resumo}}, feitas pelo Claude a pedido do dono) põe o
    resumo em português na frente, com o original ao lado. Tem de cobrir TODAS as
    perguntas: metade traduzida e metade não mudaria a leitura no meio da revisão."""
    import numpy as np
    import polars as pl

    meta = (pl.scan_parquet(spine).select("arxiv_id", "title", "abstract")
            .filter(pl.col("arxiv_id").is_in([t.arxiv_id for t in aceitas])).collect())
    por_id = {r["arxiv_id"]: r for r in meta.iter_rows(named=True)}
    itens = [{"arxiv_id": t.arxiv_id, "estrato": t.estrato, "pergunta": t.pergunta,
              "gabarito": t.gabarito, "titulo": " ".join((por_id[t.arxiv_id]["title"] or "").split()),
              "resumo": " ".join((por_id[t.arxiv_id]["abstract"] or "").split())}
             for t in aceitas]
    if traducoes is not None:
        faltam = [x["arxiv_id"] for x in itens if x["arxiv_id"] not in traducoes]
        if faltam:
            raise SystemExit(f"traduções faltando para {len(faltam)} perguntas: {faltam[:5]}")
        for x in itens:
            x["titulo_pt"] = traducoes[x["arxiv_id"]]["titulo"]
            x["resumo_pt"] = traducoes[x["arxiv_id"]]["resumo"]
    itens = [itens[k] for k in np.random.default_rng(semente).permutation(len(itens))]
    dados = json.dumps(itens, ensure_ascii=False)
    # A assinatura entra na chave do localStorage: regenerar a folha com OUTRAS
    # perguntas não pode herdar os julgamentos da anterior.
    assinatura = hashlib.sha256(dados.encode("utf-8")).hexdigest()[:16]
    return (PAGINA.replace("__DADOS__", dados).replace("__ASSINATURA__", assinatura)
            .replace("__MOTIVOS__", json.dumps(MOTIVOS, ensure_ascii=False))
            .replace("__MINIMO__", str(minimo)).replace("__N__", str(len(itens)))), assinatura


def _resumo_por_situacao(tentativas) -> dict[str, dict[str, int]]:
    por: dict[str, dict[str, int]] = {}
    for t in tentativas:
        por.setdefault(t.estrato, {}).setdefault(t.situacao, 0)
        por[t.estrato][t.situacao] += 1
    return por


def _dev_com_qwen(a, m) -> int:
    """O histórico: as versões 1–3 escritas pelo Qwen, nos artigos de desenvolvimento."""
    from phifm.rag.llm import ModeloLocal

    cotas = m.cotas_de_revisao()
    cache = DIR / f"tentativas_{m.assinatura_gerador()[:12]}_{m.SEMENTE_DEV}.jsonl"
    ordem = m.sortear(a.pares_validacao, a.spine, m.SEMENTE_DEV)
    modelo = ModeloLocal()
    if not modelo.no_ar():
        modelo.iniciar(log=RAIZ / "data/processed/llama_server.log")
    try:
        tentativas = m.gerar(modelo, ordem, a.spine, cotas, cache)
    finally:
        modelo.parar()
    aceitas = [t for t in tentativas if t.situacao == "aceita"]
    (DIR / "itens_dev.json").write_text(
        json.dumps([asdict(t) for t in aceitas], ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(aceitas)} aceitas em {len(tentativas)} · {_resumo_por_situacao(tentativas)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exportar", choices=["primario", "pos_corte"],
                   help="grava o próximo lote de resumos (sem título) para o Claude escrever")
    p.add_argument("-n", type=int, default=25, help="tamanho do lote exportado")
    p.add_argument("--importar", type=Path, help="lote escrito pelo Claude, para conferir")
    p.add_argument("--etapa", choices=["dev", "revisao", "completa"])
    p.add_argument("--apurar", type=Path, help="JSON baixado da folha de revisão")
    p.add_argument("--spine", type=Path, default=RAIZ / "data/processed/spine.parquet")
    p.add_argument("--pares-validacao", type=Path,
                   default=RAIZ / "data/processed/pares/pares_validacao.parquet")
    a = p.parse_args()

    from phifm.eval import assistente as m

    if a.apurar:
        veredictos = json.loads(a.apurar.read_text(encoding="utf-8"))
        folha = json.loads((AVALIACAO / "assistente_itens_revisao.json")
                           .read_text(encoding="utf-8")).get("assinatura_folha")
        if veredictos.get("assinatura_amostra") != folha:
            raise SystemExit(f"os veredictos são de outra folha ({veredictos.get('assinatura_amostra')}"
                             f" ≠ {folha}): julgamentos colados nas perguntas erradas.")
        r = m.apurar_i1(veredictos)
        r["assinatura_amostra"] = veredictos.get("assinatura_amostra")
        I1_AGREGADO.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"I1: {r['validas']} válidas de {r['julgadas']} (mínimo {r['minimo']}) → "
              f"{'PASSA' if r['passa'] else 'NÃO PASSA'}")
        for k, n in r["invalidas_por_motivo"].items():
            print(f"   {MOTIVOS.get(k, k)}: {n}")
        if not r["completa"]:
            print("⚠️ revisão incompleta: faltam perguntas por julgar.")
        print(f"→ {I1_AGREGADO.relative_to(RAIZ)}")
        return 0

    if a.etapa == "dev":
        return _dev_com_qwen(a, m)

    # O cache do conjunto formal é por AUTOR e regras: trocar qualquer um e reaproveitar
    # tentativas antigas misturaria versões.
    assinatura = m.assinatura_autor()
    cache = DIR / f"tentativas_{m.AUTOR}_{assinatura[:12]}_{m.SEMENTE}.jsonl"
    ordem = m.ordem_formal(a.pares_validacao, a.spine)

    if a.exportar:
        import polars as pl

        lote = m.pendentes(ordem, m.ler_cache(cache), a.exportar, a.n)
        resumos = dict(pl.scan_parquet(a.spine).select("arxiv_id", "abstract")
                       .filter(pl.col("arxiv_id").is_in([x["arxiv_id"] for x in lote]))
                       .collect().iter_rows())
        for x in lote:   # SÓ o resumo: o título fica de fora, como para o Qwen
            x["resumo"] = " ".join((resumos[x["arxiv_id"]] or "").split())
        destino = DIR / f"lote_{a.exportar}_{lote[0]['ordem']:04d}.json"
        destino.write_text(json.dumps(lote, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(lote)} resumos ({a.exportar}, ordem {lote[0]['ordem']}–{lote[-1]['ordem']})"
              f" → {destino}")
        return 0

    if a.importar:
        respostas = json.loads(a.importar.read_text(encoding="utf-8"))
        novas = m.importar(respostas, ordem, a.spine, cache)
        print(f"{len(novas)} tentativas registradas · {_resumo_por_situacao(novas)}")
        for t in novas:
            if t.situacao not in ("aceita", "nenhuma"):
                print(f"   caiu ({t.situacao}): {t.pergunta}")
        feitas = m.ler_cache(cache)
        for est in ("primario", "pos_corte"):
            n = sum(1 for (e, _), t in feitas.items() if e == est and t.situacao == "aceita")
            print(f"   {est}: {n} aceitas no total")
        return 0

    if a.etapa is None:
        p.error("--exportar, --importar, --etapa ou --apurar")
    if a.etapa == "completa":
        i1 = json.loads(I1_AGREGADO.read_text(encoding="utf-8")) if I1_AGREGADO.exists() else {}
        if not i1.get("passa"):
            raise SystemExit("I1 não foi aprovada (ou não foi apurada). A regra do DOC-13 "
                             "§9.1 proíbe fechar o conjunto antes da revisão das 40.")
        cotas = {"primario": m.N_PRIMARIO, "pos_corte": m.N_POS_CORTE}
    else:
        cotas = m.cotas_de_revisao()

    feitas = m.ler_cache(cache)
    aceitas = m.aceitas_em_ordem(ordem, feitas, cotas)
    ultimas = {e: max(t.ordem for t in aceitas if t.estrato == e) for e in cotas}
    tentativas = [t for (e, _), t in feitas.items() if t.ordem <= ultimas[e]]
    (DIR / f"itens_{a.etapa}.json").write_text(
        json.dumps([asdict(t) for t in aceitas], ensure_ascii=False, indent=1), encoding="utf-8")
    agregado = {
        "etapa": a.etapa, "regra": "DOC-13 §9.1", "autor": m.AUTOR, "semente": m.SEMENTE,
        "corte": m.CORTE, "cotas": cotas,
        "tentativas_por_estrato": _resumo_por_situacao(tentativas),
        "sobreposicao_media_aceitas": round(sum(t.sobreposicao for t in aceitas)
                                            / max(len(aceitas), 1), 4),
        "assinatura_autor": assinatura,
        "sha256_itens": hashlib.sha256(json.dumps([asdict(t) for t in aceitas],
                                                  ensure_ascii=False).encode()).hexdigest(),
    }
    if a.etapa == "revisao":
        arq = DIR / "traducoes_revisao.json"
        traducoes = json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else None
        pagina, assinatura_folha = montar_folha(aceitas, a.spine, m.SEMENTE,
                                                m.MINIMO_VALIDAS_I1, traducoes)
        if traducoes:
            agregado["traducao"] = "resumos em português (Claude), original ao lado"
        FOLHA.write_text(pagina, encoding="utf-8")
        agregado["assinatura_folha"] = assinatura_folha
    (AVALIACAO / f"assistente_itens_{a.etapa}.json").write_text(
        json.dumps(agregado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(aceitas)} itens · {agregado['tentativas_por_estrato']}")
    if a.etapa == "revisao":
        print(f"→ folha: {FOLHA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
