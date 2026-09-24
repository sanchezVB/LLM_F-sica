"""A busca no navegador: uma página e uma rota JSON, servidas SÓ para esta máquina.

Biblioteca padrão e nada mais (`http.server`): um servidor web de verdade seria uma
dependência sem ganho para um usuário local. O índice carrega uma vez e fica em memória;
cada consulta responde em dezenas de milissegundos.

⚠️ Escuta em `127.0.0.1`, e não em `0.0.0.0`: ninguém na rede enxerga a página. Expor o
índice para fora é uma decisão, não um padrão.

⚠️ As consultas passam por um lock: o `ThreadingHTTPServer` atende cada requisição numa
thread, e o modelo na DirectML não foi feito para ser chamado de várias ao mesmo tempo.
Um usuário só não percebe a fila.
"""
from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

MAX_K = 50
MAX_CONSULTA = 5_000  # caracteres; um resumo inteiro cabe com folga

PAGINA = r"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Busca em Física</title>
<style>
  :root { --fundo:#fbfbf9; --texto:#1d1d1b; --suave:#6b6b66; --borda:#e2e1dc;
          --destaque:#1f5fa8; --cartao:#ffffff; }
  @media (prefers-color-scheme: dark) {
    :root { --fundo:#161615; --texto:#ecebe6; --suave:#9d9c96; --borda:#2e2d2a;
            --destaque:#7fb0ea; --cartao:#1e1e1c; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--fundo); color:var(--texto);
         font:16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 820px; margin: 0 auto; padding: 32px 16px 64px; }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  .sub { color: var(--suave); margin: 0 0 20px; font-size: .95rem; }
  form { display: flex; flex-direction: column; gap: 8px; }
  textarea { width: 100%; min-height: 88px; padding: 12px; font: inherit; color: inherit;
             background: var(--cartao); border: 1px solid var(--borda); border-radius: 8px;
             resize: vertical; }
  .linha { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  button { padding: 10px 18px; font: inherit; border: 0; border-radius: 8px;
           background: var(--destaque); color: #fff; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  .dica { color: var(--suave); font-size: .85rem; }
  #estado { color: var(--suave); font-size: .9rem; margin: 18px 0 8px; min-height: 1.2em; }
  ol { list-style: none; padding: 0; margin: 0; }
  li { background: var(--cartao); border: 1px solid var(--borda); border-radius: 8px;
       padding: 14px 16px; margin-bottom: 10px; }
  li a { color: var(--destaque); text-decoration: none; font-weight: 600; }
  li a:hover { text-decoration: underline; }
  .meta { color: var(--suave); font-size: .88rem; margin-top: 4px; }
</style>
</head>
<body>
<main>
  <h1>Busca em artigos de Física</h1>
  <p class="sub">__N__ artigos do arXiv, ordenados por similaridade de significado.</p>
  <form id="f">
    <textarea id="q" placeholder="Escreva o assunto, ou cole o título e o resumo de um artigo…"></textarea>
    <div class="linha">
      <button id="b" type="submit">Buscar</button>
      <span class="dica">Colar título + resumo de um artigo funciona melhor: foi assim que o modelo aprendeu.</span>
    </div>
  </form>
  <div id="estado"></div>
  <ol id="r"></ol>
</main>
<script>
const f = document.getElementById("f"), q = document.getElementById("q"),
      b = document.getElementById("b"), estado = document.getElementById("estado"),
      r = document.getElementById("r");
function esc(s) { return String(s ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
f.addEventListener("submit", async (e) => {
  e.preventDefault();
  const texto = q.value.trim();
  if (!texto) return;
  b.disabled = true; estado.textContent = "Buscando…"; r.innerHTML = "";
  try {
    const t0 = performance.now();
    const resp = await fetch("/api/buscar?k=15&q=" + encodeURIComponent(texto));
    const dados = await resp.json();
    if (!resp.ok) throw new Error(dados.erro || resp.status);
    estado.textContent = dados.resultados.length + " resultados em " +
      Math.round(performance.now() - t0) + " ms";
    r.innerHTML = dados.resultados.map(x => `<li>
      <a href="${esc(x.link)}" target="_blank" rel="noopener">${esc(x.titulo)}</a>
      <div class="meta">${esc((x.autores || []).join(", "))}${(x.autores||[]).length >= 3 ? " et al." : ""}
        · ${esc(x.ano ?? "—")} · ${esc(x.categoria ?? "—")} · similaridade ${x.escore.toFixed(3)}</div>
    </li>`).join("");
  } catch (err) {
    estado.textContent = "Erro: " + err.message;
  } finally { b.disabled = false; }
});
q.addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) f.requestSubmit(); });
</script>
</body>
</html>
"""


def criar_servidor(busca, porta: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """O servidor pronto, sem iniciar. `porta=0` escolhe uma livre (é o que o teste usa)."""
    trava = threading.Lock()
    pagina = PAGINA.replace("__N__", f"{busca.vetores.shape[0]:,}".replace(",", "."))

    class Manipulador(BaseHTTPRequestHandler):
        def _responder(self, status: int, corpo: bytes, tipo: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)

        def _json(self, status: int, dados: dict) -> None:
            self._responder(status, json.dumps(dados, ensure_ascii=False).encode("utf-8"),
                            "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 — nome imposto pelo http.server
            url = urlparse(self.path)
            if url.path == "/":
                self._responder(HTTPStatus.OK, pagina.encode("utf-8"),
                                "text/html; charset=utf-8")
                return
            if url.path != "/api/buscar":
                self._json(HTTPStatus.NOT_FOUND, {"erro": "rota desconhecida"})
                return
            args = parse_qs(url.query)
            consulta = (args.get("q") or [""])[0].strip()
            if not consulta:
                self._json(HTTPStatus.BAD_REQUEST, {"erro": "consulta vazia"})
                return
            if len(consulta) > MAX_CONSULTA:
                self._json(HTTPStatus.BAD_REQUEST,
                           {"erro": f"consulta com mais de {MAX_CONSULTA} caracteres"})
                return
            try:
                k = max(1, min(MAX_K, int((args.get("k") or ["10"])[0])))
            except ValueError:
                self._json(HTTPStatus.BAD_REQUEST, {"erro": "k tem de ser um número"})
                return
            with trava:
                resultados = busca.buscar(consulta, k=k)
            self._json(HTTPStatus.OK, {"consulta": consulta, "resultados": [
                {"posicao": x.posicao, "escore": round(x.escore, 4),
                 "arxiv_id": x.arxiv_id, "titulo": x.titulo, "ano": x.ano,
                 "categoria": x.categoria, "autores": x.autores, "link": x.link}
                for x in resultados]})

        def log_message(self, formato: str, *args) -> None:
            return  # o terminal fica limpo; erros de verdade levantam

    return ThreadingHTTPServer((host, porta), Manipulador)
