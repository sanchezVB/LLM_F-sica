#!/usr/bin/env python3
"""Gera uma folha de revisão HTML para a amostra de contaminação do peS2o.

    .venv\\Scripts\\python.exe scripts\\folha_de_revisao.py

Produz `data/processed/pes2o_fisica/revisao.html` — um arquivo único, sem rede,
que se abre com dois cliques e grava o progresso no navegador. No fim, o botão
**Baixar veredictos** salva um JSON que `scripts/apurar_revisao.py` transforma em
taxa de falso positivo com intervalo de confiança.

## Por que este julgamento não pode ser automatizado

A taxa de falso positivo do classificador `isphysics` foi medida em **1,5–13,6%**, e
foi essa medição que justificou o limiar 0,9. Mas ela foi feita em **resumos do
arXiv**, e o corpus tem 92,1% dos seus tokens em **texto pleno de paper** — outra
distribuição de texto, outro conjunto de fontes, e a taxa não transfere de graça.

Só uma pessoa que entende de Física decide se um documento é de Física. Um segundo
classificador julgando o primeiro herdaria os mesmos vieses e daria uma concordância
alta que não mede nada.

O que o computador pode fazer é o resto: sortear, esconder o rótulo do classificador
até depois do julgamento, contar, e calcular o intervalo. É o que este script faz.

## ⚠️ O escore fica ESCONDIDO até o julgamento

Ver "0,98" antes de julgar ancora a resposta — é o viés de ancoragem, e num
julgamento de 400 itens ele domina. A folha revela o escore **depois** de o botão ser
clicado, para que a pessoa possa ver onde discordou do modelo sem que isso contamine
o julgamento seguinte.

## ⚠️ A amostra é ESTRATIFICADA, e a folha conta os estratos separado

O peS2o filtrado tem dois regimes, e eles não são a mesma população:

    resumo (s2ag)        3.626.168 docs   65,6% dos docs    7,9% dos tokens
    texto pleno (s2orc)  1.900.163 docs   34,4% dos docs   92,1% dos tokens

Um número único misturando os dois não é nem a precisão em resumos nem a precisão em
papers, e o peso muda por um fator de nove conforme se conte documento ou token. Por
isso a folha mostra uma linha por estrato, e a combinação pelo peso em tokens fica em
`apurar_revisao.py`, com a conta à vista.

Os estratos vêm **embaralhados**: julgar 200 resumos e depois 200 papers confundiria
o efeito do estrato com o cansaço de quem julga.

## Quantos documentos bastam

Alvo pré-comprometido: **200 por estrato**. Com 10 falsos positivos em 200 (5%), o
intervalo de Wilson é **2,7% a 9,0%**. Wilson é assimétrico, então um "±" seria
mentira arredondada. O alvo é declarado antes porque **parar quando o número fica
bonito é viés de parada opcional** — a folha mostra o intervalo ao vivo, e mostra
também esse aviso.

## Onde o arquivo fica

No HD, junto da amostra, como todo dado deste projeto. Nada sobe para nenhum lugar:
o julgamento é sobre o corpus de pesquisa do usuário e a decisão de divulgar não é
um detalhe de implementação.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

AMOSTRA = Path("data/processed/pes2o_fisica/_amostra_para_revisao.json")
# ⚠️ Um numero so, substituido na pagina. Repetido em dois lugares ele
# diverge, e o alvo pre-comprometido passa a depender de qual voce leu.
ALVO_POR_ESTRATO = 200

PAGINA = r"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Revisao da amostra do peS2o</title>
<style>
:root{--bg:#faf9f7;--fg:#1c1b19;--sutil:#6b6862;--linha:#e0ddd6;--cartao:#fff;
--sim:#1a7f4b;--nao:#b3261e;--talvez:#8a6d1f;--realce:#f3f0e8}
@media (prefers-color-scheme:dark){:root{--bg:#16151a;--fg:#eceaf0;--sutil:#9b98a3;
--linha:#2f2d36;--cartao:#1e1d24;--sim:#4ec98a;--nao:#f2837b;--talvez:#d9b64e;
--realce:#26242e}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.6 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
.env{max-width:52rem;margin:0 auto;padding:1.5rem 1.25rem 4rem}
h1{font-size:1.25rem;margin:0 0 .25rem}
.sutil{color:var(--sutil);font-size:.875rem}
.barra{height:6px;background:var(--linha);border-radius:3px;overflow:hidden;margin:1rem 0 .5rem}
.barra>div{height:100%;background:var(--sim);width:0;transition:width .2s}
.cartao{background:var(--cartao);border:1px solid var(--linha);border-radius:10px;
padding:1.1rem 1.2rem;margin:1rem 0}
.texto{white-space:pre-wrap;font:15px/1.65 ui-serif,Georgia,serif;max-height:26rem;
overflow-y:auto}
.acoes{display:flex;gap:.5rem;flex-wrap:wrap;margin:1rem 0 .25rem}
button{font:inherit;padding:.55rem 1rem;border-radius:8px;border:1px solid var(--linha);
background:var(--cartao);color:var(--fg);cursor:pointer}
button:hover{background:var(--realce)}
button.sim{border-color:var(--sim);color:var(--sim);font-weight:600}
button.nao{border-color:var(--nao);color:var(--nao);font-weight:600}
button.talvez{border-color:var(--talvez);color:var(--talvez)}
kbd{font:12px ui-monospace,monospace;border:1px solid var(--linha);border-radius:4px;
padding:1px 5px;color:var(--sutil)}
table{border-collapse:collapse;font-size:.875rem;margin:.5rem 0}
td,th{padding:.25rem .75rem .25rem 0;text-align:left}
.aviso{border-left:3px solid var(--talvez);padding:.5rem 0 .5rem .9rem;
color:var(--sutil);font-size:.875rem;margin:1rem 0}
.escore{font:13px ui-monospace,monospace;color:var(--sutil)}
.fim{text-align:center;padding:2rem 0}
</style></head><body><div class=env>
<h1>Esta amostra do peS2o e de Fisica?</h1>
<p class=sutil>O classificador disse que sim, com escore &ge; 0,9. A pergunta e se ele
acertou. O escore fica escondido ate voce julgar &mdash; ver "0,98" antes ancora a
resposta.</p>
<p class=sutil>A amostra e <b>estratificada</b>: metade resumo (titulo + abstract),
metade texto pleno de paper. Os dois vem embaralhados. Por documento os resumos sao
a maioria do corpus; por token sao uma fracao pequena, e por isso os dois estratos
sao contados separado.</p>
<div class=barra><div id=preenche></div></div>
<p class=sutil id=progresso></p>
<div id=area></div>
<div class=acoes id=acoes>
  <button class=sim  onclick="julga('fisica')">E de Fisica <kbd>1</kbd></button>
  <button class=nao  onclick="julga('nao')">NAO e de Fisica <kbd>2</kbd></button>
  <button class=talvez onclick="julga('duvida')">Nao sei dizer <kbd>3</kbd></button>
  <button onclick="volta()">Voltar <kbd>&larr;</kbd></button>
</div>
<p class=sutil id=ultima></p>
<div id=painel></div>
<div class=aviso>
  Alvo pre-comprometido: <b>__ALVO__ documentos POR ESTRATO</b>. Parar antes porque o
  numero ficou bonito e vies de parada opcional &mdash; o intervalo abaixo e honesto
  so se voce nao usar ele para decidir quando parar.
</div>
<div class=acoes>
  <button onclick=baixar()>Baixar veredictos (JSON)</button>
  <button onclick=zerar()>Comecar de novo</button>
</div>
<p class=sutil>Depois de baixar:
<code>.venv\Scripts\python.exe scripts\apurar_revisao.py --veredictos &lt;arquivo&gt;</code></p>
</div>
<script>
const DOCS = __DADOS__;
const CHAVE = "revisao_pes2o_" + "__ASSINATURA__";
let v = {}, i = 0;
try { const g = localStorage.getItem(CHAVE); if (g) { const o = JSON.parse(g);
  v = o.v || {}; i = o.i || 0; } } catch (e) {}

function salva(){ try { localStorage.setItem(CHAVE, JSON.stringify({v:v, i:i}));
  } catch (e) {} }

function wilson(k, n){
  if (!n) return null;
  const z = 1.959964, p = k / n, d = 1 + z*z/n;
  const c = (p + z*z/(2*n)) / d;
  const m = z * Math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d;
  return [Math.max(0, c-m), Math.min(1, c+m)];
}

function esc(s){ return String(s == null ? "" : s)
  .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

// ⚠️ O cartao do documento em julgamento NAO pode tocar o escore. Ver o teste
// `test_o_escore_nao_e_renderizado_antes_do_julgamento`: ver "0,98" antes de
// julgar ancora a resposta, e o que sai deixa de medir precisao do filtro.
function montaCartao(d){
  const tam = d.n_chars ? esc(d.n_chars.toLocaleString("pt-BR")) + " caracteres" : "";
  return "<div class=cartao><p class=sutil>documento " + (i+1) + " de " +
    DOCS.length + " · <b>" + esc(d.estrato || d.url) + "</b>" +
    (tam ? " · " + tam : "") + "</p><div class=texto id=t1></div>" +
    (d.meio ? "<p class=sutil style='margin:.9rem 0 .2rem'>trecho do miolo, " +
              "a 45% do documento</p><div class=texto id=t2></div>" : "") +
    "</div>";
}

// Por estrato, e nao um numero so. Os dois estratos tem tamanhos muito
// diferentes em token, e a combinacao pelo peso e feita por `apurar_revisao.py`
// — aqui a mistura daria um numero que nao e nem um nem outro.
function montaPainel(){
  const g = {};
  Object.entries(v).forEach(function(par){
    const e = DOCS[+par[0]].estrato || "amostra";
    (g[e] = g[e] || []).push(par[1]);
  });
  const chaves = Object.keys(g).sort();
  if (!chaves.length) return "";
  let h = "<table><tr><th>estrato</th><th>julgados</th><th>nao e Fisica</th>" +
          "<th>taxa</th><th>Wilson 95%</th><th>\"nao sei\"</th></tr>";
  chaves.forEach(function(e){
    const lista = g[e];
    const n = lista.filter(function(x){ return x !== "duvida"; }).length;
    const k = lista.filter(function(x){ return x === "nao"; }).length;
    const du = lista.filter(function(x){ return x === "duvida"; }).length;
    const ic = wilson(k, n);
    h += "<tr><td>" + esc(e) + "</td><td>" + n + "</td><td>" + k + "</td><td><b>" +
      (n ? (100*k/n).toFixed(1) + "%" : "&mdash;") + "</b></td><td>" +
      (ic ? (100*ic[0]).toFixed(1) + "% a " + (100*ic[1]).toFixed(1) + "%"
          : "&mdash;") + "</td><td>" + du + "</td></tr>";
  });
  return h + "</table><p class=sutil>Por documento os resumos sao a maioria do " +
    "corpus; por TOKEN sao uma fracao pequena. A combinacao dos estratos pelo " +
    "peso em tokens e feita por <code>apurar_revisao.py</code> &mdash; misturar " +
    "os dois aqui daria um numero que nao e nem um nem outro.</p>";
}

function pinta(){
  const feitos = Object.keys(v).length;
  document.getElementById("preenche").style.width =
    (100 * feitos / DOCS.length).toFixed(1) + "%";
  document.getElementById("progresso").textContent =
    feitos + " de " + DOCS.length + " julgados · alvo __ALVO__ por estrato";

  if (i >= DOCS.length){
    document.getElementById("area").innerHTML =
      "<div class=fim><b>Fim da amostra.</b><br>Baixe os veredictos abaixo.</div>";
    document.getElementById("acoes").style.display = "none";
  } else {
    const d = DOCS[i];
    document.getElementById("area").innerHTML = montaCartao(d);
    // textContent, nunca innerHTML: o texto vem do corpus.
    document.getElementById("t1").textContent = d.inicio;
    if (d.meio) document.getElementById("t2").textContent = d.meio;
    document.getElementById("acoes").style.display = "flex";
  }

  const ant = i > 0 && v[String(i-1)] ? i-1 : null;
  // `toFixed(3)`: um escore de 1,0 sai como `1` e le mal ao lado de `0,912`.
  document.getElementById("ultima").innerHTML = ant === null ? "" :
    "documento " + (ant+1) + ": voce disse <b>" + v[String(ant)] +
    "</b> · o classificador dava <b>" + Number(DOCS[ant].score).toFixed(3) +
    "</b>";

  document.getElementById("painel").innerHTML = montaPainel();
}

function julga(x){ if (i >= DOCS.length) return; v[String(i)] = x; i++; salva(); pinta(); }
function volta(){ if (i > 0){ i--; delete v[String(i)]; salva(); pinta(); } }
function zerar(){ if (confirm("Apagar todos os julgamentos?")){ v = {}; i = 0;
  salva(); document.getElementById("acoes").style.display = "flex"; pinta(); } }

function baixar(){
  const linhas = Object.entries(v).map(([k, x]) =>
    ({indice: +k, veredicto: x, score: DOCS[+k].score, url: DOCS[+k].url,
      estrato: DOCS[+k].estrato || null, n_chars: DOCS[+k].n_chars || null}));
  const b = new Blob([JSON.stringify({assinatura_amostra: "__ASSINATURA__",
    n_amostra: DOCS.length, alvo_pre_comprometido: __ALVO__,
    veredictos: linhas}, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b);
  a.download = "veredictos_revisao.json";
  a.click();
}

addEventListener("keydown", e => {
  if (e.key === "1") julga("fisica");
  else if (e.key === "2") julga("nao");
  else if (e.key === "3") julga("duvida");
  else if (e.key === "ArrowLeft") volta();
});
pinta();
</script></body></html>
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--amostra", type=Path, default=AMOSTRA)
    p.add_argument("--out", type=Path, default=None,
                   help="por omissão, `revisao.html` ao lado da amostra")
    a = p.parse_args()
    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8")

    if not a.amostra.exists():
        raise SystemExit(
            f"{a.amostra} não existe. Ela é gerada pelo filtro do peS2o; sem a "
            "amostra não há o que revisar.")
    bruto = a.amostra.read_text(encoding="utf-8")
    docs = json.loads(bruto)
    if not isinstance(docs, list) or not docs:
        raise SystemExit(f"{a.amostra} não é uma lista de documentos")
    faltando = [k for k in ("score", "url", "inicio") if k not in docs[0]]
    if faltando:
        raise SystemExit(
            f"a amostra não tem {faltando}. A folha mostra o texto e esconde o "
            "escore; sem essas chaves ela não pode fazer nem uma coisa nem outra.")

    # ⚠️ A assinatura entra na chave do localStorage. Sem ela, regenerar a folha
    # sobre uma amostra DIFERENTE herdaria os julgamentos da anterior — 400 veredictos
    # colados nos documentos errados, e nada avisaria.
    assinatura = hashlib.sha256(bruto.encode("utf-8")).hexdigest()[:16]

    saida = a.out or a.amostra.parent / "revisao.html"
    pagina = (PAGINA
              .replace("__DADOS__", json.dumps(docs, ensure_ascii=False))
              .replace("__ALVO__", str(ALVO_POR_ESTRATO))
              .replace("__ASSINATURA__", assinatura))
    saida.write_text(pagina, encoding="utf-8")

    escores = [float(d["score"]) for d in docs]
    estratos = Counter(d.get("estrato", "sem estrato") for d in docs)
    print("=" * 70)
    print(f"  {len(docs)} documentos · escore {min(escores):.3f} a {max(escores):.3f}")
    for est, n in sorted(estratos.items()):
        print(f"    {est:14s} {n:>4d}")
    if len(estratos) == 1 and "sem estrato" in estratos:
        print("  ⚠️ amostra SEM estrato. Se ela veio do filtro, cobre 0,67% do "
              "corpus e é")
        print("     100% resumo — ver `scripts/amostrar_para_revisao.py`.")
    print(f"  assinatura da amostra: {assinatura}")
    print(f"  -> {saida}  ({saida.stat().st_size/1e3:.0f} KB)")
    print("=" * 70)
    print("  Abra no navegador. Teclas: 1 = é Física · 2 = não é · 3 = não sei.")
    print("  O progresso fica no navegador; o botão baixa o JSON dos veredictos.")
    print(f"  Alvo pré-comprometido: {ALVO_POR_ESTRATO} documentos POR ESTRATO. "
          "A 5% observados,")
    print("  o intervalo de Wilson fica em 2,7%–9,0% em cada um.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
