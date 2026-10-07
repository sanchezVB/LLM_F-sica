"""A busca — e, se ligado, o assistente — no navegador, servidos SÓ para esta máquina.

Biblioteca padrão e nada mais (`http.server`): um servidor web de verdade seria uma
dependência sem ganho para um usuário local. O índice carrega uma vez e fica em memória.

## As rotas

- `GET /` — a página: aba de busca e, com assistente, aba de pergunta;
- `GET /api/buscar?q=…&k=…` — os artigos mais próximos;
- `GET /api/estado` — se há assistente e se o modelo está carregado;
- `POST /api/perguntar` (JSON `{"pergunta": …}`) — a resposta com fontes, citações
  conferidas pelo portão e as frases sem fonte. Só existe com assistente.

## Segurança — local não quer dizer que só você chama

⚠️ Escuta em `127.0.0.1`, e não em `0.0.0.0`: ninguém na rede enxerga a página. Expor o
índice para fora é uma decisão, não um padrão.

⚠️ Mas qualquer site aberto NESTE navegador pode mandar pedidos para `127.0.0.1`. Duas
guardas: o cabeçalho `Host` tem de ser o deste servidor (um domínio que resolve para
127.0.0.1 — *DNS rebinding* — é recusado), e `/api/perguntar` só aceita
`Content-Type: application/json`, que o navegador não manda entre origens sem uma
pré-verificação que este servidor não autoriza. Sem isso, uma página qualquer poria a GPU
para trabalhar.

## Concorrência

⚠️ As consultas passam por uma trava: o `ThreadingHTTPServer` atende cada requisição numa
thread, e o modelo na DirectML não foi feito para ser chamado de várias ao mesmo tempo. A
busca que o assistente faz passa pela MESMA trava — e só ela: uma busca espera o ~1 s da
recuperação de uma pergunta, não os ~30 s da resposta. Perguntas vão uma de cada vez (o
modelo roda com `-np 1`).
"""
from __future__ import annotations

import json
import math
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

MAX_K = 50
MAX_CONSULTA = 5_000  # caracteres; um resumo inteiro cabe com folga
MAX_PERGUNTA = 2_000
MAX_CORPO = 16_384    # bytes do POST

PAGINA = r"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Busca em Física</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
  :root { --fundo:#fbfbf9; --texto:#1d1d1b; --suave:#6b6b66; --borda:#e2e1dc;
          --destaque:#1f5fa8; --no-destaque:#ffffff; --cartao:#ffffff; --alerta:#9a5b00;
          --realce:#f1efe8; }
  @media (prefers-color-scheme: dark) {
    :root { --fundo:#161615; --texto:#ecebe6; --suave:#9d9c96; --borda:#2e2d2a;
            --destaque:#7fb0ea; --no-destaque:#10161d; --cartao:#1e1e1c; --alerta:#e0a54a;
            --realce:#262522; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--fundo); color:var(--texto);
         font:16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 820px; margin: 0 auto; padding: 32px 16px 64px; }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  .sub { color: var(--suave); margin: 0 0 16px; font-size: .95rem; }
  .abas { display: flex; gap: 4px; margin: 0 0 12px; border-bottom: 1px solid var(--borda); }
  .aba { padding: 8px 14px; border: 0; background: none; color: var(--suave); font: inherit;
         cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -1px; }
  .aba[aria-selected="true"] { color: var(--texto); border-bottom-color: var(--destaque);
                               font-weight: 600; }
  form { display: flex; flex-direction: column; gap: 8px; }
  textarea { width: 100%; min-height: 88px; padding: 12px; font: inherit; color: inherit;
             background: var(--cartao); border: 1px solid var(--borda); border-radius: 8px;
             resize: vertical; }
  .linha { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  button.acao { padding: 10px 18px; font: inherit; border: 0; border-radius: 8px;
                background: var(--destaque); color: var(--no-destaque); cursor: pointer; }
  button.acao:disabled { opacity: .5; cursor: default; }
  .dica { color: var(--suave); font-size: .85rem; }
  .estado { color: var(--suave); font-size: .9rem; margin: 18px 0 8px; min-height: 1.2em; }
  ol { list-style: none; padding: 0; margin: 0; }
  li { background: var(--cartao); border: 1px solid var(--borda); border-radius: 8px;
       padding: 14px 16px; margin-bottom: 10px; }
  li a, .resposta a { color: var(--destaque); text-decoration: none; font-weight: 600; }
  li a:hover, .resposta a:hover { text-decoration: underline; }
  .meta { color: var(--suave); font-size: .88rem; margin-top: 4px; }
  .resposta { background: var(--cartao); border: 1px solid var(--borda); border-radius: 8px;
              padding: 16px 18px; white-space: pre-wrap; margin-bottom: 16px; }
  .resposta a { font-weight: 500; }
  h2 { font-size: 1rem; margin: 20px 0 8px; }
  li.citada { border-left: 3px solid var(--destaque); }
  li:target { background: var(--realce); }
  details { margin-top: 6px; }
  summary { cursor: pointer; color: var(--suave); font-size: .88rem; }
  details p { margin: 6px 0 0; font-size: .92rem; }
  .aviso { border-left: 3px solid var(--alerta); padding: 8px 12px; margin: 12px 0;
           font-size: .92rem; background: var(--cartao); border-radius: 0 8px 8px 0; }
  .aviso ul { margin: 6px 0 0; padding-left: 18px; }
  [hidden] { display: none !important; }
</style>
</head>
<body>
<main>
  <h1>Busca em artigos de Física</h1>
  <p class="sub">__N__ artigos do arXiv, ordenados por similaridade de significado.</p>
  <div class="abas" role="tablist" id="abas" __ABAS__>
    <button class="aba" role="tab" aria-selected="true" data-painel="p-busca">Buscar artigos</button>
    <button class="aba" role="tab" aria-selected="false" data-painel="p-pergunta">Perguntar</button>
  </div>

  <section id="p-busca">
    <form id="f">
      <textarea id="q" placeholder="Escreva o assunto, ou cole o título e o resumo de um artigo…"></textarea>
      <div class="linha">
        <button class="acao" id="b" type="submit">Buscar</button>
        <span class="dica">Colar título + resumo de um artigo funciona melhor: foi assim que o modelo aprendeu.</span>
      </div>
    </form>
    <div class="estado" id="estado"></div>
    <ol id="r"></ol>
  </section>

  <section id="p-pergunta" hidden>
    <form id="fp">
      <textarea id="qp" placeholder="Pergunte em português ou inglês — por exemplo: o que limita a coerência de qubits supercondutores?"></textarea>
      <div class="linha">
        <button class="acao" id="bp" type="submit">Perguntar</button>
        <span class="dica" id="dica-modelo"></span>
      </div>
    </form>
    <div class="aviso">O assistente responde só com os artigos que a busca achou, e cita
      cada afirmação. <b>A qualidade ainda está sendo medida</b>: a citação aponta para um
      artigo de verdade, mas confira se ele diz o que a frase afirma.</div>
    <div class="estado" id="estado-p"></div>
    <div id="rp"></div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
function esc(s) { return String(s ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
function formulas(el) {
  if (window.renderMathInElement) renderMathInElement(el, {
    delimiters: [{left: "$$", right: "$$", display: false}, {left: "$", right: "$", display: false}],
    throwOnError: false, strict: "ignore"});
}

// ── abas ──
document.querySelectorAll(".aba").forEach(a => a.addEventListener("click", () => {
  document.querySelectorAll(".aba").forEach(x => {
    const ativa = x === a;
    x.setAttribute("aria-selected", ativa);
    $(x.dataset.painel).hidden = !ativa;
  });
  (a.dataset.painel === "p-pergunta" ? $("qp") : $("q")).focus();
}));

// ── busca ──
$("f").addEventListener("submit", async (e) => {
  e.preventDefault();
  const texto = $("q").value.trim();
  if (!texto) return;
  $("b").disabled = true; $("estado").textContent = "Buscando…"; $("r").innerHTML = "";
  try {
    const t0 = performance.now();
    const resp = await fetch("/api/buscar?k=15&q=" + encodeURIComponent(texto));
    const dados = await resp.json();
    if (!resp.ok) throw new Error(dados.erro || resp.status);
    $("estado").textContent = dados.resultados.length + " resultados em " +
      Math.round(performance.now() - t0) + " ms";
    $("r").innerHTML = dados.resultados.map(x => `<li>
      <a href="${esc(x.link)}" target="_blank" rel="noopener">${esc(x.titulo)}</a>
      <div class="meta">${esc((x.autores || []).join(", "))}${(x.autores||[]).length >= 3 ? " et al." : ""}
        · ${esc(x.ano ?? "—")} · ${esc(x.categoria ?? "—")} · similaridade ${x.escore.toFixed(3)}</div>
    </li>`).join("");
    formulas($("r"));
  } catch (err) {
    $("estado").textContent = "Erro: " + err.message;
  } finally { $("b").disabled = false; }
});
$("q").addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("f").requestSubmit(); });

// ── pergunta ──
async function estado() {
  try {
    const d = await (await fetch("/api/estado")).json();
    if (!d.assistente) return;
    $("abas").hidden = false;
    $("dica-modelo").textContent = d.modelo_no_ar
      ? "Modelo carregado: ~30–50 s por resposta."
      : "A primeira pergunta carrega o modelo na GPU (~1–2 min). Ele se desliga sozinho depois de "
        + d.ocioso_min + " min sem uso.";
  } catch (e) {}
}
function comCitacoes(texto, n) {
  // [n] vira link para a fonte n; o resto do texto é escapado antes.
  return esc(texto).replace(/\[(\d+)\]/g, (m, k) =>
    (+k >= 1 && +k <= n) ? `<a href="#fonte-${k}">[${k}]</a>` : m)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}
function htmlFontes(fontes, final) {
  return `<h2>Fontes que o modelo leu</h2><ol>` + fontes.map(f => `
    <li id="fonte-${f.numero}" class="${f.citada ? "citada" : ""}">
      [${f.numero}] <a href="${esc(f.link)}" target="_blank" rel="noopener">${esc(f.titulo)}</a>
      <div class="meta">${esc(f.ano ?? "—")} · arXiv:${esc(f.arxiv_id)}${final
        ? " · " + (f.citada ? "citada na resposta" : "não citada") : ""}</div>
      <details><summary>resumo</summary><p>${esc(f.resumo)}</p></details>
    </li>`).join("") + `</ol>`;
}
function mostrarFinal(d) {
  // O texto que PASSOU pelo portão substitui o que foi aparecendo: uma citação
  // inventada pode ter sido escrita e depois removida.
  let html = `<div class="resposta">${comCitacoes(d.texto, d.fontes.length)}</div>`;
  if (d.removidas.length) html += `<div class="aviso">O portão removeu citações a fontes
    que não existiam: ${d.removidas.map(k => "[" + k + "]").join(", ")}.</div>`;
  if (d.frases_sem_fonte.length) html += `<div class="aviso">Frases sem citação — não confie
    nelas sem conferir:<ul>${d.frases_sem_fonte.map(f => "<li>" + esc(f) + "</li>").join("")}</ul></div>`;
  $("rp").innerHTML = html + htmlFontes(d.fontes, true);
  formulas($("rp"));
  $("estado-p").textContent = "Respondida em " + d.segundos.toFixed(0) + " s";
}
$("fp").addEventListener("submit", async (e) => {
  e.preventDefault();
  const pergunta = $("qp").value.trim();
  if (!pergunta) return;
  $("bp").disabled = true; $("rp").innerHTML = "";
  const t0 = performance.now();
  let fase = "Buscando…";
  const relogio = setInterval(() => { if (fase) $("estado-p").textContent =
    fase + " " + Math.round((performance.now() - t0) / 1000) + " s"; }, 500);
  try {
    const resp = await fetch("/api/perguntar", {method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify({pergunta, fluxo: true})});
    if (!resp.ok) { const d = await resp.json(); throw new Error(d.erro || resp.status); }
    const leitor = resp.body.getReader(), dec = new TextDecoder();
    let resto = "", caixa = null, terminou = false;
    while (true) {
      const {value, done} = await leitor.read();
      if (done) break;
      resto += dec.decode(value, {stream: true});
      let i;
      while ((i = resto.indexOf("\n")) >= 0) {
        const linha = resto.slice(0, i); resto = resto.slice(i + 1);
        if (!linha.trim()) continue;
        const d = JSON.parse(linha);
        if (d.tipo === "fontes") {
          fase = "Escrevendo…";
          $("rp").innerHTML = `<div class="resposta" id="rascunho"></div>` + htmlFontes(d.fontes, false);
          caixa = $("rascunho");
        } else if (d.tipo === "pedaco" && caixa) {
          caixa.textContent += d.texto;
        } else if (d.tipo === "fim") {
          terminou = true; fase = null; mostrarFinal(d);
        } else if (d.tipo === "erro") {
          throw new Error(d.erro);
        }
      }
    }
    if (!terminou) throw new Error("a resposta parou no meio");
  } catch (err) {
    $("estado-p").textContent = "Erro: " + err.message;
  } finally { clearInterval(relogio); $("bp").disabled = false; estado(); }
});
$("qp").addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("fp").requestSubmit(); });
estado();
</script>
</body>
</html>
"""


class _BuscaTravada:
    """A busca do assistente passando pela trava da rota de busca."""

    def __init__(self, busca, trava: threading.Lock):
        self._busca, self._trava = busca, trava

    def buscar(self, *a, **kw):
        with self._trava:
            return self._busca.buscar(*a, **kw)

    def __getattr__(self, nome):
        return getattr(self._busca, nome)


def _fonte_json(f, citadas: list[int]) -> dict:
    return {"numero": f.numero, "arxiv_id": f.arxiv_id, "titulo": f.titulo, "ano": f.ano,
            "resumo": f.resumo, "link": f.link, "citada": f.numero in citadas,
            "escore": None if math.isnan(f.escore) else round(f.escore, 4)}


def _resposta_json(r, segundos: float) -> dict:
    return {"pergunta": r.pergunta, "texto": r.texto, "consulta": r.consulta,
            "fontes": [_fonte_json(f, r.citadas) for f in r.fontes],
            "citadas": r.citadas, "removidas": r.removidas,
            "frases_sem_fonte": r.frases_sem_fonte, "segundos": round(segundos, 1)}


def criar_servidor(busca, porta: int = 8765, host: str = "127.0.0.1", assistente=None,
                   ocioso_min: int | None = None) -> ThreadingHTTPServer:
    """O servidor pronto, sem iniciar. `porta=0` escolhe uma livre (é o que o teste usa).

    `assistente` é um `phifm.rag.assistente.Assistente` (ou qualquer coisa com
    `responder(pergunta)` e `busca`); sem ele, a página é só a busca, como antes.
    """
    trava_busca, trava_pergunta = threading.Lock(), threading.Lock()
    # As abas vêm no HTML, e não depois de `/api/estado`: com o modelo desligado, essa rota
    # espera a recusa da porta dele (~2 s no Windows), e a aba aparecia atrasada.
    pagina = (PAGINA.replace("__N__", f"{busca.vetores.shape[0]:,}".replace(",", "."))
              .replace("__ABAS__", "" if assistente is not None else "hidden"))
    if assistente is not None:
        assistente.busca = _BuscaTravada(assistente.busca, trava_busca)

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

        def _host_valido(self) -> bool:
            porta_real = self.server.server_address[1]
            permitidos = {f"127.0.0.1:{porta_real}", f"localhost:{porta_real}"}
            if self.headers.get("Host", "") in permitidos:
                return True
            self._json(HTTPStatus.FORBIDDEN, {"erro": "Host não permitido"})
            return False

        def do_GET(self) -> None:  # noqa: N802 — nome imposto pelo http.server
            if not self._host_valido():
                return
            url = urlparse(self.path)
            if url.path == "/":
                self._responder(HTTPStatus.OK, pagina.encode("utf-8"),
                                "text/html; charset=utf-8")
                return
            if url.path == "/api/estado":
                modelo = getattr(assistente, "modelo", None)
                self._json(HTTPStatus.OK, {
                    "assistente": assistente is not None,
                    "modelo_no_ar": bool(modelo is not None and modelo.no_ar()),
                    "ocioso_min": ocioso_min})
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
            with trava_busca:
                resultados = busca.buscar(consulta, k=k)
            self._json(HTTPStatus.OK, {"consulta": consulta, "resultados": [
                {"posicao": x.posicao, "escore": round(x.escore, 4),
                 "arxiv_id": x.arxiv_id, "titulo": x.titulo, "ano": x.ano,
                 "categoria": x.categoria, "autores": x.autores, "link": x.link}
                for x in resultados]})

        def do_POST(self) -> None:  # noqa: N802
            if not self._host_valido():
                return
            if urlparse(self.path).path != "/api/perguntar" or assistente is None:
                self._json(HTTPStatus.NOT_FOUND, {"erro": "rota desconhecida"})
                return
            tipo = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if tipo != "application/json":
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                           {"erro": "o corpo tem de ser application/json"})
                return
            try:
                tamanho = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                tamanho = -1
            if not 0 < tamanho <= MAX_CORPO:
                self._json(HTTPStatus.BAD_REQUEST, {"erro": "corpo vazio ou grande demais"})
                return
            try:
                corpo = json.loads(self.rfile.read(tamanho))
                pergunta = str(corpo.get("pergunta", "")).strip()
                fluxo = bool(corpo.get("fluxo", False))
            except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
                self._json(HTTPStatus.BAD_REQUEST, {"erro": "JSON inválido"})
                return
            if not pergunta or len(pergunta) > MAX_PERGUNTA:
                self._json(HTTPStatus.BAD_REQUEST,
                           {"erro": f"pergunta vazia ou com mais de {MAX_PERGUNTA} caracteres"})
                return
            if fluxo and hasattr(assistente, "responder_em_fluxo"):
                self._responder_em_fluxo(pergunta)
                return
            t0 = time.perf_counter()
            try:
                with trava_pergunta:
                    r = assistente.responder(pergunta)
            except SystemExit as e:  # o cliente do modelo sinaliza servidor travado assim
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"erro": str(e)})
                return
            self._json(HTTPStatus.OK, _resposta_json(r, time.perf_counter() - t0))

        def _responder_em_fluxo(self, pergunta: str) -> None:
            """Uma linha JSON por evento (NDJSON), escrita assim que o evento sai. Sem
            `Content-Length`: o corpo termina quando a conexão fecha (HTTP/1.0). Se o
            navegador desconectar no meio, o gerador é FECHADO — ele segura a trava do
            modelo até acabar."""
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def enviar(d: dict) -> None:
                self.wfile.write((json.dumps(d, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()

            t0 = time.perf_counter()
            with trava_pergunta:
                eventos = assistente.responder_em_fluxo(pergunta)
                try:
                    for tipo, dado in eventos:
                        if tipo == "fontes":
                            consulta, fontes = dado
                            enviar({"tipo": "fontes", "consulta": consulta,
                                    "fontes": [_fonte_json(f, []) for f in fontes]})
                        elif tipo == "pedaco":
                            enviar({"tipo": "pedaco", "texto": dado})
                        else:
                            enviar({"tipo": "fim", **_resposta_json(
                                dado, time.perf_counter() - t0)})
                except SystemExit as e:
                    enviar({"tipo": "erro", "erro": str(e)})
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                finally:
                    eventos.close()

        def log_message(self, formato: str, *args) -> None:
            return  # o terminal fica limpo; erros de verdade levantam

    return ThreadingHTTPServer((host, porta), Manipulador)
