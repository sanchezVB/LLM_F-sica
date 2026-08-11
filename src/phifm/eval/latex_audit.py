"""S3b — auditoria de preservação de LaTeX (DOC-02 §3.2, critério C1 do DOC-03).

Responde a pergunta que precede qualquer decisão de gastar: **as fatias de terceiro
degradaram as equações?**

O DOC-02 é explícito sobre o risco e sobre a consequência:

> Essas fatias foram processadas pelo pipeline de terceiros, não pelo nosso.
> Parte da estrutura de LaTeX pode ter sido degradada — exatamente o que o
> Minerva identificou como decisivo. Se a degradação exceder ~10%, o bulk pago
> do arXiv (~US$ 100–180 de egress) passa a se justificar.

Custo desta medição: US$ 0. Ela precede o gasto.

## O desenho, e a inversão que economiza 4 h

O caminho ingênuo é sortear 2.000 identificadores e procurá-los no RedPajama — o
que obriga a varrer os 81 GB dos 100 shards. O caminho certo é o inverso:
**pegar os papers do primeiro shard** e buscar as fontes deles no arXiv. Mesmo
rigor amostral, um centésimo do custo de leitura.

## Por que 200 papers e não os 2.000 do documento

Buscar 2.000 tarballs de `/e-print/` um a um é volume que o arXiv pede para ir
pelo S3 em lote, não pelo endpoint individual. E 200 papers já dão precisão de
sobra: com ~50 equações cada, são ~10 mil equações, e o erro padrão da taxa de
preservação fica em ~0,003. O erro padrão POR PAPER, que é o que limita, fica em
~2% — suficiente para separar 5% de 15% de degradação, que é a decisão em jogo.

Escalar para 2.000 é trocar 10 min por 100 min de coleta para ganhar um fator
√10 de precisão que a decisão não usa. Se o resultado cair na faixa ambígua de
8–12%, aí vale escalar — e o parâmetro existe para isso.

## O que conta como "equação preservada"

Comparação por **forma canônica**, não por string. `T = 2\\pi\\sqrt{L/g}` e
`T=2\\pi(L/g)^{1/2}` são a mesma equação, e um pipeline que reescreveu a notação
sem perder conteúdo NÃO degradou nada. Só o canonicalizador (DOC-03 §3) sabe
disso; casamento de texto contaria a reescrita como perda e inflaria a
degradação.
"""

from __future__ import annotations

import io
import json
import logging
import re
import tarfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

from phifm.core.latex.canonical import hash_canonico
from phifm.core.latex.extrair import extrair_equacoes, juntar_fontes
from phifm.core.latex.macros import preparar
from phifm.core.schema.manifest import RateLimit
from phifm.corpus.acquire.base import CONTACT, PoliteSession

log = logging.getLogger(__name__)

INDICE_REDPAJAMA = (
    "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T"
    "/resolve/main/urls/arxiv.txt"
)
EPRINT = "https://arxiv.org/e-print/"

# Ver §"por que 200" na docstring.
N_PAPERS = 200
# Só o começo do shard: 3 MB dão ~54 registros, então 12 MB bastam para 200.
BYTES_DO_SHARD = 16 * 1024 * 1024


@dataclass
class Comparacao:
    arxiv_id: str
    eq_fonte: int = 0
    eq_redpajama: int = 0
    preservadas: int = 0
    erro: str = ""

    @property
    def taxa(self) -> float | None:
        return self.preservadas / self.eq_fonte if self.eq_fonte else None


@dataclass
class Auditoria:
    comparacoes: list[Comparacao] = field(default_factory=list)

    @property
    def uteis(self) -> list[Comparacao]:
        """Papers com equações na fonte. Sem equação não há o que preservar, e
        incluí-los na média puxaria o resultado para um lado arbitrário."""
        return [c for c in self.comparacoes if not c.erro and c.eq_fonte > 0]

    def resumo(self) -> dict:
        u = self.uteis
        if not u:
            return {"erro": "nenhum paper comparável"}
        tf = sum(c.eq_fonte for c in u)
        tp = sum(c.preservadas for c in u)
        # Por equação (agregado) e por paper (média das taxas). Os dois importam:
        # o primeiro é a taxa global, o segundo revela se a perda se concentra em
        # poucos papers ou está espalhada.
        por_eq = tp / tf
        taxas = sorted(c.taxa for c in u)
        med = taxas[len(taxas) // 2]
        return {
            "papers_comparados": len(u),
            "papers_com_erro": sum(1 for c in self.comparacoes if c.erro),
            "equacoes_na_fonte": tf,
            "equacoes_preservadas": tp,
            "preservacao_por_equacao": round(por_eq, 4),
            "preservacao_mediana_por_paper": round(med, 4),
            "degradacao_por_equacao": round(1 - por_eq, 4),
            "papers_abaixo_de_90pc": sum(1 for t in taxas if t < 0.90),
            "limiar_do_doc02": 0.10,
            "veredito": (
                "RedPajama SERVE — degradação abaixo do limiar de 10%"
                if 1 - por_eq <= 0.10 else
                "RedPajama DEGRADA além do limiar — o bulk pago do arXiv "
                "(US$ 100–180) passa a se justificar (DOC-02 §3.2)"
            ),
        }


def amostrar_redpajama(http: PoliteSession, n: int = N_PAPERS) -> dict[str, str]:
    """`arxiv_id` → texto, do começo do primeiro shard."""
    url = http.get(INDICE_REDPAJAMA, timeout=60).text.splitlines()[0].strip()
    r = http.get(url, timeout=300, headers={"Range": f"bytes=0-{BYTES_DO_SHARD}"})
    # A última linha da faixa quase certamente está cortada.
    linhas = r.text.split("\n")[:-1]
    out: dict[str, str] = {}
    for l in linhas:
        try:
            d = json.loads(l)
        except json.JSONDecodeError:
            continue
        aid = (d.get("meta") or {}).get("arxiv_id")
        if aid and d.get("text"):
            out[aid] = d["text"]
        if len(out) >= n:
            break
    log.info("RedPajama: %d papers amostrados de %.0f MB do primeiro shard",
             len(out), len(r.content) / 1e6)
    return out


def baixar_fonte(http: PoliteSession, arxiv_id: str) -> dict[str, str]:
    """Arquivos `.tex` da submissão. Aceita tar.gz e .tex avulso gzipado."""
    r = http.get(EPRINT + arxiv_id, timeout=120)
    bruto = r.content
    try:
        with tarfile.open(fileobj=io.BytesIO(bruto), mode="r:*") as tf:
            return {
                m.name: tf.extractfile(m).read().decode("utf-8", "replace")
                for m in tf.getmembers()
                if m.isfile() and m.name.lower().endswith(".tex")
            }
    except tarfile.ReadError:
        pass
    # Submissão de um arquivo só: gzip de .tex, ou .tex cru.
    import gzip
    for desc in (lambda b: gzip.decompress(b), lambda b: b):
        try:
            return {"main.tex": desc(bruto).decode("utf-8", "replace")}
        except Exception:
            continue
    raise ValueError("formato de submissão não reconhecido")


def comparar_um(http: PoliteSession, arxiv_id: str, texto_rp: str) -> Comparacao:
    c = Comparacao(arxiv_id)
    try:
        fonte = juntar_fontes(baixar_fonte(http, arxiv_id))
    except Exception as exc:
        c.erro = f"{type(exc).__name__}: {str(exc)[:80]}"
        return c

    # ⚠️ Expandir as macros do autor ANTES de extrair. Sem isto a medição
    # confunde notação com conteúdo: a fonte escreve `\Ecal_\mu`, o RedPajama
    # escreve `\mathcal{E}_\mu`, e a equação conta como perdida. Medido no
    # 1607.04520: das que NÃO casavam, 97% usavam macro; das que casavam, 20%.
    eq_fonte = extrair_equacoes(preparar(fonte))
    # O texto do RedPajama pode trazer o preâmbulo, então as macros dele também
    # são expandidas — com as definições que ELE tiver, não as da fonte.
    eq_rp = extrair_equacoes(preparar(texto_rp), remover_comentario=False)
    c.eq_fonte, c.eq_redpajama = len(eq_fonte), len(eq_rp)

    # Conjuntos de formas canônicas. Duplicatas dentro do mesmo documento não
    # devem inflar nem o numerador nem o denominador.
    try:
        hf = {hash_canonico(e) for e in eq_fonte}
        hr = {hash_canonico(e) for e in eq_rp}
    except Exception as exc:
        c.erro = f"canonicalização: {str(exc)[:80]}"
        return c
    c.eq_fonte = len(hf)
    c.eq_redpajama = len(hr)
    c.preservadas = len(hf & hr)
    return c


def auditar(n: int = N_PAPERS, contato: str = CONTACT) -> Auditoria:
    # 1 req/3 s é o pedido do arXiv (DOC-02 §8.2) e vale para `/e-print/` também.
    http = PoliteSession(RateLimit(requests_per_second=1 / 3, max_retries=6,
                                   backoff_max_s=300), contato)
    amostra = amostrar_redpajama(http, n)
    a = Auditoria()
    for i, (aid, texto) in enumerate(amostra.items(), 1):
        c = comparar_um(http, aid, texto)
        a.comparacoes.append(c)
        if i % 20 == 0 or c.erro:
            t = f"{c.taxa:.2f}" if c.taxa is not None else "—"
            log.info("%d/%d · %s · fonte %d eq · rp %d eq · preservação %s %s",
                     i, len(amostra), aid, c.eq_fonte, c.eq_redpajama, t,
                     f"· {c.erro}" if c.erro else "")
    return a


def salvar(a: Auditoria, destino: Path) -> dict:
    destino.parent.mkdir(parents=True, exist_ok=True)
    r = a.resumo()
    destino.write_text(
        json.dumps({"resumo": r, "por_paper": [asdict(c) for c in a.comparacoes]},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")
    return r
