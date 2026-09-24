#!/usr/bin/env python3
"""A medida do assistente, depois dos itens: braços, juiz, I3, R e a regra (DOC-13 §9.1).

    # 1. os três braços nos 650 itens (GPU, horas; retomável) e o juiz
    .venv-treino\\Scripts\\python.exe scripts\\medir_assistente.py --rodar --julgar
    .venv-treino\\Scripts\\python.exe scripts\\medir_assistente.py --rodar --limite 20   # medir o tempo

    # 2. I3: o dono julga 100 respostas; a apuração diz se o juiz vale
    .venv-treino\\Scripts\\python.exe scripts\\medir_assistente.py --folha-i3
    .venv-treino\\Scripts\\python.exe scripts\\medir_assistente.py --apurar-i3 <veredictos.json>

    # 3. R: o dono classifica as respostas de B que o juiz não deu como certas
    .venv-treino\\Scripts\\python.exe scripts\\medir_assistente.py --folha-r
    .venv-treino\\Scripts\\python.exe scripts\\medir_assistente.py --apurar-r <veredictos.json>

    # 4. a regra
    .venv-treino\\Scripts\\python.exe scripts\\medir_assistente.py --decidir

Respostas, julgamentos e folhas ficam em `data/processed/assistente/`, fora do git:
carregam as perguntas do conjunto de teste (DOC-11 §8.1). Vão para o git só os
agregados em `data/processed/avaliacao/assistente_*.json`.

⚠️ Os números pelo juiz não são impressos antes de I3: se o juiz não concordar com o
dono, a regra manda consertá-lo "antes de qualquer leitura".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8  # noqa: E402

utf8()

DIR = RAIZ / "data/processed/assistente"
AVALIACAO = RAIZ / "data/processed/avaliacao"
ITENS = DIR / "itens_completa.json"
FOLHA_I3 = DIR / "folha_i3.html"
FOLHA_R = DIR / "folha_r.html"
TRADUCOES_R = DIR / "traducoes_r.json"

OPCOES_I3 = [
    {"valor": "certo", "rotulo": "Certa", "classe": "sim",
     "ajuda": "diz o que o gabarito diz, sem contradizê-lo"},
    {"valor": "parcial", "rotulo": "Parcial", "classe": "",
     "ajuda": "só parte do gabarito, ou uma versão mais vaga, sem contradizer"},
    {"valor": "errado", "rotulo": "Errada", "classe": "nao",
     "ajuda": "outra resposta, ou contradiz o gabarito"},
    {"valor": "absteve", "rotulo": "Não respondeu", "classe": "",
     "ajuda": "diz que não sabe / que a informação não está disponível"}]
OPCOES_R = [
    {"valor": "modelo_errou", "rotulo": "O modelo errou", "classe": "nao",
     "ajuda": "a resposta não diz o que o gabarito diz"},
    {"valor": "juiz_errou", "rotulo": "O juiz errou", "classe": "sim",
     "ajuda": "a resposta estava certa"},
    {"valor": "item_invalido", "rotulo": "O item é inválido", "classe": "",
     "ajuda": "pergunta mal feita ou gabarito errado"}]

PAGINA = r"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITULO__</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
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
.pergunta{font-size:1.1rem;font-weight:600}
.texto{white-space:pre-wrap}
.resumo{white-space:pre-wrap;font:15px/1.65 ui-serif,Georgia,serif}
details{margin-top:1rem}
summary{cursor:pointer;color:var(--sutil);font-size:.875rem}
.acoes{display:flex;gap:.5rem;flex-wrap:wrap;margin:1rem 0 .25rem}
button{font:inherit;padding:.55rem 1rem;border-radius:8px;border:1px solid var(--linha);
background:var(--cartao);color:var(--fg);cursor:pointer;text-align:left}
button:hover{background:var(--realce)}
button.sim{border-color:var(--sim);color:var(--sim);font-weight:600}
button.nao{border-color:var(--nao);color:var(--nao)}
button small{display:block;color:var(--sutil);font-size:.75rem;font-weight:400}
kbd{font:12px ui-monospace,monospace;border:1px solid var(--linha);border-radius:4px;
padding:1px 5px;color:var(--sutil)}
.aviso{border-left:3px solid var(--linha);padding:.5rem 0 .5rem .9rem;color:var(--sutil);
font-size:.875rem;margin:1rem 0}
.fim{text-align:center;padding:2rem 0}
</style></head><body><div class=env>
<h1>__TITULO__</h1>
__INTRO__
<div class=barra><div id=preenche></div></div>
<p class=sutil id=progresso></p>
<div id=area></div>
<div class=acoes id=acoes></div>
<div class=acoes>
  <button onclick=baixar()>Baixar veredictos (JSON)</button>
  <button onclick=zerar()>Começar de novo</button>
</div>
<p class=sutil>Depois de baixar, é só me avisar onde o arquivo ficou.</p>
</div>
<script>
const ITENS = __DADOS__;
const OPCOES = __OPCOES__;
const CHAVE = "__PREFIXO___" + "__ASSINATURA__";
// `t` guarda o horário de cada julgamento; a apuração recusa folha julgada rápido demais
// para ter sido lida (DOC-13 §9.1, I1: um script já "julgou" 40 itens em um segundo).
let v = {}, t = {}, i = 0, guarda = true;
try { const g = localStorage.getItem(CHAVE); if (g) { const o = JSON.parse(g);
  v = o.v || {}; t = o.t || {}; i = o.i || 0; }
  localStorage.setItem(CHAVE + "_teste", "1"); localStorage.removeItem(CHAVE + "_teste");
} catch (e) { guarda = false; }
function salva(){ try { localStorage.setItem(CHAVE, JSON.stringify({v:v, t:t, i:i})); } catch (e) {} }
const acoes = document.getElementById("acoes");
OPCOES.forEach((o, k) => { const b = document.createElement("button");
  if (o.classe) b.className = o.classe;
  b.append(o.rotulo + " "); const kb = document.createElement("kbd"); kb.textContent = k + 1;
  b.append(kb); const s = document.createElement("small"); s.textContent = o.ajuda; b.append(s);
  b.onclick = () => julga(o.valor); acoes.append(b); });
const vb = document.createElement("button"); vb.innerHTML = "Voltar <kbd>&larr;</kbd>";
vb.onclick = volta; acoes.append(vb);
const limpa = s => String(s).replace(/\\(emph|textit|textbf)\{([^{}]*)\}/g, "$2");
function pinta(){
  const feitos = Object.keys(v).length;
  document.getElementById("preenche").style.width = (100*feitos/ITENS.length).toFixed(1) + "%";
  document.getElementById("progresso").textContent =
    feitos + " de " + ITENS.length + " julgadas" +
    (guarda ? "" : " · ⚠️ este navegador NÃO guarda o progresso: não recarregue a página " +
                   "no meio (abra o arquivo no Chrome com dois cliques para ele guardar)");
  const area = document.getElementById("area");
  if (i >= ITENS.length){
    area.innerHTML = "<div class=fim><b>Fim.</b><br>Baixe os veredictos abaixo.</div>";
    acoes.style.display = "none";
    return;
  }
  // textContent, nunca innerHTML: pergunta, resposta e resumo vêm de modelo e corpus.
  const c = document.createElement("div"); c.className = "cartao";
  const n = document.createElement("p"); n.className = "sutil";
  n.textContent = "item " + (i+1) + " de " + ITENS.length; c.append(n);
  let alvo = c;
  for (const f of ITENS[i].campos) {
    if (f.recolhido) { const d = document.createElement("details");
      const s = document.createElement("summary"); s.textContent = f.rotulo; d.append(s);
      const x = document.createElement("div"); x.className = f.classe || "texto";
      x.textContent = limpa(f.texto); d.append(x); c.append(d); continue; }
    const r = document.createElement("div"); r.className = "rotulo"; r.textContent = f.rotulo;
    const x = document.createElement("div"); x.className = f.classe || "texto";
    x.textContent = limpa(f.texto); c.append(r, x);
  }
  area.replaceChildren(c);
  acoes.style.display = "flex";
  if (window.renderMathInElement) renderMathInElement(area, {
    delimiters: [{left: "$$", right: "$$", display: false},
                 {left: "$", right: "$", display: false}],
    throwOnError: false, strict: "ignore"});
}
function julga(x){ if (i >= ITENS.length) return; v[String(i)] = x; t[String(i)] = Date.now();
  i++; salva(); pinta(); }
function volta(){ if (i > 0){ i--; delete v[String(i)]; delete t[String(i)]; salva(); pinta(); } }
function zerar(){ if (confirm("Apagar todos os julgamentos?")){ v = {}; t = {}; i = 0; salva();
  pinta(); } }
function baixar(){
  const linhas = Object.entries(v).map(([k, x]) => ({indice: +k, veredicto: x,
    id: ITENS[+k].id, ms: t[k] || null}));
  const b = new Blob([JSON.stringify({assinatura_amostra: "__ASSINATURA__",
    n_amostra: ITENS.length, veredictos: linhas}, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b);
  a.download = "__ARQUIVO__";
  a.click();
}
addEventListener("keydown", e => {
  const k = parseInt(e.key, 10);
  if (k >= 1 && k <= OPCOES.length) julga(OPCOES[k-1].valor);
  else if (e.key === "ArrowLeft") volta();
});
pinta();
addEventListener("load", pinta);
</script></body></html>
"""

INTRO_I3 = """<p class=sutil>Cada item traz uma pergunta, o <b>gabarito</b> (tirado do resumo do
artigo que responde a ela) e a resposta do assistente, sem as marcas de citação. Julgue
<b>só contra o gabarito</b>, como o juiz automático faz: a resposta diz o que ele diz?
Você não sabe de qual braço veio cada resposta, nem o que o juiz disse — de propósito.</p>
<div class=aviso>Regra escrita antes (DOC-13 §9.1, I3): o seu julgamento e o do juiz são
comparados pelo κ de Cohen em <b>certa × o resto</b>, a variável que decide. Abaixo de
<b>__KAPPA__</b>, o juiz não vale, e o resultado fica NÃO DECIDIDO até ele ser
consertado.</div>"""

INTRO_R = """<p class=sutil>São as respostas do braço <b>B</b> — o artigo certo estava
entre as fontes — que o juiz <b>não</b> deu como certas. Para cada uma: o modelo errou
mesmo, o juiz errou (a resposta estava certa), ou o item não presta (pergunta mal feita
ou gabarito errado)? O resumo do artigo de onde saiu a pergunta está no fim do cartão,
traduzido; o original abre logo abaixo.</p>
<div class=aviso>Regra escrita antes (DOC-13 §9.1, R): "juiz errou" conta como acerto
do modelo; "item inválido" sai da conta. Sem esta revisão, pergunta ruim e juiz errado
contariam como erro do gerador.</div>"""


def _sha_itens(itens: list[dict]) -> str:
    return hashlib.sha256(json.dumps(itens, ensure_ascii=False).encode()).hexdigest()


def carregar_itens() -> tuple[list[dict], str]:
    """Os 650 itens, conferidos contra o sha256 versionado no git."""
    itens = json.loads(ITENS.read_text(encoding="utf-8"))
    sha = _sha_itens(itens)
    esperado = json.loads((AVALIACAO / "assistente_itens_completa.json")
                          .read_text(encoding="utf-8"))["sha256_itens"]
    if sha != esperado:
        raise SystemExit(f"os itens em {ITENS} não são os versionados ({sha[:12]} ≠ "
                         f"{esperado[:12]}): alguém mexeu no conjunto depois de fechado.")
    return itens, sha


def caminhos(sha_itens: str):
    from phifm.eval import assistente_bracos as b
    from phifm.rag.llm import MODELO

    ab = b.assinatura_bracos(MODELO.name, sha_itens)
    aj = b.assinatura_juiz(MODELO.name)
    return DIR / f"bracos_{ab[:12]}.jsonl", DIR / f"juiz_{aj[:12]}.jsonl", ab, aj


def montar_folha(titulo: str, intro: str, opcoes: list[dict], itens: list[dict],
                 prefixo: str, arquivo: str) -> tuple[str, str]:
    dados = json.dumps(itens, ensure_ascii=False)
    assinatura = hashlib.sha256(dados.encode("utf-8")).hexdigest()[:16]
    pagina = (PAGINA.replace("__TITULO__", titulo).replace("__INTRO__", intro)
              .replace("__OPCOES__", json.dumps(opcoes, ensure_ascii=False))
              .replace("__PREFIXO__", prefixo).replace("__ARQUIVO__", arquivo)
              .replace("__ASSINATURA__", assinatura).replace("__DADOS__", dados))
    return pagina, assinatura


def _ler_folha(caminho: Path, agregado: Path, n_esperado: int) -> dict:
    """O JSON baixado de uma folha, conferido: é desta folha, está completo e foi lido."""
    from phifm.eval.assistente import MS_MINIMO_POR_JULGAMENTO, julgamentos_rapidos

    veredictos = json.loads(caminho.read_text(encoding="utf-8"))
    folha = json.loads(agregado.read_text(encoding="utf-8"))["assinatura_folha"]
    if veredictos.get("assinatura_amostra") != folha:
        raise SystemExit(f"os veredictos são de outra folha ({veredictos.get('assinatura_amostra')}"
                         f" ≠ {folha}).")
    lista = veredictos["veredictos"]
    if len(lista) != n_esperado:
        raise SystemExit(f"{len(lista)} julgamentos de {n_esperado}: a folha está incompleta.")
    rapidos = julgamentos_rapidos(lista)
    if rapidos is None or rapidos > n_esperado // 4:
        raise SystemExit(f"{rapidos} julgamentos a menos de {MS_MINIMO_POR_JULGAMENTO} ms do "
                         "anterior: isso não é leitura. A folha não vale.")
    return {v["id"]: v["veredicto"] for v in lista}


def _modelo_e_assistente(a, com_busca: bool):
    from phifm.eval.assistente import K_FONTES, SEMENTE
    from phifm.rag.assistente import Assistente
    from phifm.rag.llm import ModeloLocal

    # A busca primeiro: se ela falhar, não fica um servidor órfão ocupando a GPU.
    busca = None
    if com_busca:
        from phifm.retrieval.indice import Busca

        busca = Busca(a.indice, dispositivo="cpu")
    modelo = ModeloLocal()
    if not modelo.no_ar():
        print("subindo o llama-server (Qwen3-8B na GPU)...", flush=True)
        modelo.iniciar(log=RAIZ / "data/processed/llama_server.log")
    if busca is None:
        return modelo, None
    return modelo, Assistente(busca, modelo, a.spine, k=K_FONTES, semente=SEMENTE)


def rodar(a, itens, cache_b: Path) -> None:
    from phifm.eval import assistente_bracos as b

    modelo, assistente = _modelo_e_assistente(a, com_busca=True)
    feitos0 = len({(r.estrato, r.arxiv_id) for r in b.ler_respostas(cache_b)})
    alvo = len(itens) if a.limite is None else min(len(itens), feitos0 + a.limite)
    t0, n = time.perf_counter(), [0]

    def ao_terminar(item, regs):
        n[0] += 1
        feitos = feitos0 + n[0]
        s = (time.perf_counter() - t0) / n[0]
        seg = " · ".join(f"{r.braco} {r.segundos:.0f}s" + ("=A" if r.igual_a_A else "")
                         for r in regs)
        print(f"[{feitos}/{len(itens)}] {item['estrato']} {item['arxiv_id']} · {seg} · "
              f"P {'em ' + str(regs[0].posicao_p) if regs[0].posicao_p else 'fora'} · "
              f"{s:.0f} s/item · faltam ~{(alvo - feitos) * s / 3600:.1f} h", flush=True)

    try:
        b.rodar(assistente, itens, cache_b, a.limite, ao_terminar)
    finally:
        modelo.parar()


def julgar(a, itens, cache_b: Path, cache_j: Path) -> None:
    from phifm.eval import assistente_bracos as b

    respostas = b.ler_respostas(cache_b)
    modelo, _ = _modelo_e_assistente(a, com_busca=False)
    t0, n = time.perf_counter(), [0]
    feitos0 = len(b.ler_julgamentos(cache_j))

    def ao_julgar(r, d):
        n[0] += 1
        if n[0] % 25 == 0:
            print(f"   {feitos0 + n[0]} julgamentos · "
                  f"{(time.perf_counter() - t0) / n[0]:.1f} s cada", flush=True)

    try:
        julg = b.julgar_todas(modelo, itens, respostas, cache_j, ao_julgar)
    finally:
        modelo.parar()
    ilegiveis = sum(1 for d in julg.values() if d["veredicto"] == "ilegivel")
    # Só a contagem: os acertos pelo juiz esperam I3 (ver o topo do arquivo).
    print(f"{len(julg)} textos julgados · {ilegiveis} fora do formato")


def resumo_mecanico(itens, respostas, sha_itens: str, ab: str) -> dict:
    """O que não depende do juiz: pode ser lido e versionado antes de I3."""
    import numpy as np

    saida = {"regra": "DOC-13 §9.1", "sha256_itens": sha_itens, "assinatura_bracos": ab,
             "itens_rodados": len({(r.estrato, r.arxiv_id) for r in respostas}),
             "itens_total": len(itens), "por_estrato": {}}
    for estrato in ("primario", "pos_corte"):
        rs = [r for r in respostas if r.estrato == estrato]
        if not rs:
            continue
        por = {x: [r for r in rs if r.braco == x] for x in "ABC"}
        saida["por_estrato"][estrato] = {
            "itens": len(por["A"]),
            "p_entre_as_fontes_A": round(float(np.mean([r.posicao_p is not None
                                                        for r in por["A"]])), 4),
            "p_citada_quando_presente": {
                x: round(float(np.mean([r.p_citada for r in por[x] if r.posicao_p])), 4)
                for x in "AB" if any(r.posicao_p for r in por[x])},
            "respostas_com_citacao_removida": {x: sum(1 for r in por[x] if r.removidas)
                                               for x in "ABC"},
            "segundos_medianos": {x: round(float(np.median([r.segundos for r in por[x]
                                                             if not r.igual_a_A])), 1)
                                  for x in "ABC" if any(not r.igual_a_A for r in por[x])},
        }
    return saida


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rodar", action="store_true", help="os três braços (GPU)")
    p.add_argument("--limite", type=int, help="no máximo N itens novos nesta chamada")
    p.add_argument("--julgar", action="store_true", help="o juiz em todas as respostas (GPU)")
    p.add_argument("--folha-i3", action="store_true")
    p.add_argument("--apurar-i3", type=Path)
    p.add_argument("--folha-r", action="store_true")
    p.add_argument("--apurar-r", type=Path)
    p.add_argument("--decidir", action="store_true")
    p.add_argument("--indice", type=Path, default=RAIZ / "data/processed/indice_busca")
    p.add_argument("--spine", type=Path, default=RAIZ / "data/processed/spine.parquet")
    a = p.parse_args()

    from phifm.eval import assistente_bracos as b

    itens, sha_itens = carregar_itens()
    cache_b, cache_j, ab, aj = caminhos(sha_itens)
    agregado_i3 = AVALIACAO / "assistente_i3.json"
    agregado_r = AVALIACAO / "assistente_r.json"

    if a.rodar:
        rodar(a, itens, cache_b)
        respostas = b.ler_respostas(cache_b)
        resumo = resumo_mecanico(itens, respostas, sha_itens, ab)
        (AVALIACAO / "assistente_bracos.json").write_text(
            json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(resumo["por_estrato"], ensure_ascii=False, indent=1))
    if a.julgar:
        julgar(a, itens, cache_b, cache_j)

    if a.folha_i3 or a.apurar_i3 or a.folha_r or a.apurar_r or a.decidir:
        respostas = b.ler_respostas(cache_b)
        if len({(r.estrato, r.arxiv_id) for r in respostas}) < len(itens):
            raise SystemExit("os braços não rodaram em todos os itens (--rodar).")
        julg = b.ler_julgamentos(cache_j)
        veredictos = b.veredictos_por_braco(itens, respostas, julg)
        por_id = {(i["estrato"], i["arxiv_id"]): i for i in itens}

        def id_de(r) -> str:
            return f"{r.braco}:{r.estrato}:{r.arxiv_id}"

    if a.folha_i3 or a.apurar_i3:
        amostra = b.amostra_i3(respostas)
        if a.folha_i3:
            dados = [{"id": id_de(r), "campos": [
                {"rotulo": "pergunta", "texto": por_id[(r.estrato, r.arxiv_id)]["pergunta"],
                 "classe": "pergunta"},
                {"rotulo": "gabarito", "texto": por_id[(r.estrato, r.arxiv_id)]["gabarito"]},
                {"rotulo": "resposta do assistente",
                 "texto": b.sem_citacoes(r.texto).strip() or "(resposta vazia)"}]}
                for r in amostra]
            pagina, assinatura = montar_folha(
                "O juiz concorda com você?", INTRO_I3.replace("__KAPPA__", str(b.KAPPA_MINIMO)),
                OPCOES_I3, dados, "i3", "veredictos_i3.json")
            FOLHA_I3.write_text(pagina, encoding="utf-8")
            agregado_i3.write_text(json.dumps(
                {"regra": "DOC-13 §9.1 · I3", "n": len(dados), "assinatura_folha": assinatura,
                 "assinatura_juiz": aj}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{len(dados)} respostas → {FOLHA_I3}")
        else:
            humanos = _ler_folha(a.apurar_i3, agregado_i3, len(amostra))
            juiz = {id_de(r): veredictos[r.braco][(r.estrato, r.arxiv_id)] for r in amostra}
            ids = [id_de(r) for r in amostra]
            r3 = b.apurar_i3([humanos[i] for i in ids], [juiz[i] for i in ids])
            anterior = json.loads(agregado_i3.read_text(encoding="utf-8"))
            anterior.update(r3)
            agregado_i3.write_text(json.dumps(anterior, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
            print(f"I3: κ (certa × resto) = {r3['kappa_certo']} · κ nas 4 = "
                  f"{r3['kappa_4_categorias']} · mínimo {r3['minimo']} → "
                  f"{'PASSA' if r3['passa'] else 'NÃO PASSA'}")

    if a.folha_r or a.apurar_r:
        i3 = json.loads(agregado_i3.read_text(encoding="utf-8")) if agregado_i3.exists() else {}
        if "passa" not in i3:
            raise SystemExit("R vem depois de I3: se o juiz não valer, os erros de B mudam.")
        amostra = b.amostra_r(veredictos["B"])
        if a.folha_r:
            dados = _dados_r(a, amostra, respostas, julg, por_id)
            pagina, assinatura = montar_folha(
                "Os erros do braço B são do modelo?", INTRO_R, OPCOES_R, dados, "r",
                "veredictos_r.json")
            FOLHA_R.write_text(pagina, encoding="utf-8")
            agregado_r.write_text(json.dumps(
                {"regra": "DOC-13 §9.1 · R", "nao_certos_B_primario":
                 sum(1 for k, v in veredictos["B"].items() if k[0] == "primario" and v != "certo"),
                 "revisados": len(dados), "assinatura_folha": assinatura},
                ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{len(dados)} respostas → {FOLHA_R}")
        else:
            classes = _ler_folha(a.apurar_r, agregado_r, len(amostra))
            anterior = json.loads(agregado_r.read_text(encoding="utf-8"))
            anterior["por_classe"] = {c: sum(1 for v in classes.values() if v == c)
                                      for c in b.CLASSES_R}
            agregado_r.write_text(json.dumps(anterior, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
            (DIR / "classes_r.json").write_text(json.dumps(classes, ensure_ascii=False,
                                                           indent=1), encoding="utf-8")
            print(f"R: {anterior['por_classe']}")

    if a.decidir:
        i3 = json.loads(agregado_i3.read_text(encoding="utf-8"))
        classes = json.loads((DIR / "classes_r.json").read_text(encoding="utf-8"))
        revisao = {tuple(k.split(":")[1:]): v for k, v in classes.items()}
        resultado = b.decidir(veredictos, revisao, i3)
        resultado["secundarias"] = b.secundarias(itens, respostas, veredictos)
        resultado["sha256_itens"], resultado["assinatura_bracos"] = sha_itens, ab
        resultado["assinatura_juiz"] = aj
        (AVALIACAO / "assistente_resultado.json").write_text(
            json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: resultado[k] for k in (
            "situacao", "gerador", "busca", "acerto_B_com_R", "ic_acerto_B_com_R",
            "lacuna_da_busca", "ic_lacuna_da_busca")}, ensure_ascii=False, indent=1))
    if not any((a.rodar, a.julgar, a.folha_i3, a.apurar_i3, a.folha_r, a.apurar_r, a.decidir)):
        p.error("nada a fazer: --rodar, --julgar, --folha-i3, --apurar-i3, --folha-r, "
                "--apurar-r ou --decidir")
    return 0


def _dados_r(a, amostra, respostas, julg, por_id) -> list[dict]:
    """Os cartões de R: pergunta, gabarito, a resposta com as citações, as fontes que o
    modelo recebeu, o que o juiz disse e o resumo de P (traduzido pelo Claude, com o
    original ao lado — como na revisão de I1)."""
    import polars as pl

    from phifm.eval import assistente_bracos as b

    traducoes = (json.loads(TRADUCOES_R.read_text(encoding="utf-8"))
                 if TRADUCOES_R.exists() else None)
    if traducoes is not None:
        faltam = [k[1] for k in amostra if k[1] not in traducoes]
        if faltam:
            raise SystemExit(f"traduções faltando para {len(faltam)} itens: {faltam[:5]}")
    rb = {(r.estrato, r.arxiv_id): r for r in respostas if r.braco == "B"}
    ids_fontes = sorted({f for k in amostra for f in rb[k].fontes})
    meta = {r["arxiv_id"]: r for r in pl.scan_parquet(a.spine)
            .select("arxiv_id", "title", "abstract")
            .filter(pl.col("arxiv_id").is_in(ids_fontes)).collect().iter_rows(named=True)}
    dados = []
    for k in amostra:
        r, it = rb[k], por_id[k]
        j = julg[b.chave_do_julgamento(it["pergunta"], it["gabarito"], r.texto)]
        fontes = "\n".join(
            f"[{n}] {' '.join((meta[f]['title'] or '').split())}"
            + ("   ← o artigo de onde saiu a pergunta" if f == k[1] else "")
            for n, f in enumerate(r.fontes, start=1))
        original = (" ".join((meta[k[1]]["title"] or "").split()) + "\n\n"
                    + " ".join((meta[k[1]]["abstract"] or "").split()))
        campos = [
            {"rotulo": "pergunta", "texto": it["pergunta"], "classe": "pergunta"},
            {"rotulo": "gabarito", "texto": it["gabarito"]},
            {"rotulo": "resposta do assistente (braço B)", "texto": r.texto or "(vazia)"},
            {"rotulo": "o juiz disse", "texto": f"{j['veredicto']} — {j['motivo']}"},
            {"rotulo": "fontes que o modelo recebeu", "texto": fontes},
        ]
        if traducoes is not None:
            t = traducoes[k[1]]
            campos += [{"rotulo": "resumo do artigo (tradução)",
                        "texto": f"{t['titulo']}\n\n{t['resumo']}", "classe": "resumo"},
                       {"rotulo": "original em inglês", "texto": original, "classe": "resumo",
                        "recolhido": True}]
        else:
            campos.append({"rotulo": "resumo do artigo", "texto": original, "classe": "resumo"})
        dados.append({"id": f"B:{k[0]}:{k[1]}", "campos": campos})
    return dados


if __name__ == "__main__":
    raise SystemExit(main())
