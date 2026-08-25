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
import tarfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl

from phifm.core.latex.canonical import hash_canonico
from phifm.core.latex.extrair import extrair_equacoes, montar_documento
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
# ⚠️ O shard do RedPajama-arXiv é o arXiv INTEIRO, não Física: medido, 52% dos
# registros amostrados não estão no nosso spine. Para chegar a N papers de Física
# é preciso ler ~2x isso. 16 MB davam 271 registros e só 103 de Física — perto do
# limite para decidir, e foi por isso que o intervalo de confiança cruzou o
# limiar. 64 MB dão ~1.100 registros.
BYTES_DO_SHARD = 64 * 1024 * 1024


@dataclass
class Comparacao:
    arxiv_id: str
    eq_fonte: int = 0
    eq_redpajama: int = 0
    preservadas: int = 0
    erro: str = ""
    # Como a fonte foi montada. "seguido" = o principal foi achado e os `\input`
    # resolvidos, então a contagem é fiel. "concatenado" = não havia
    # `\documentclass` e juntamos tudo, então a contagem pode estar INFLADA por
    # arquivos que o documento não inclui. Sem este campo os dois casos entravam
    # na mesma média e o resultado era chamado de "degradação".
    modo: str = ""
    tex_ignorados: int = 0

    @property
    def taxa(self) -> float | None:
        return self.preservadas / self.eq_fonte if self.eq_fonte else None

    @property
    def confiavel(self) -> bool:
        return self.modo != "concatenado"


def _bootstrap_ausencia(
    papers: list[Comparacao], reamostras: int = 20_000, seed: int = 17
) -> tuple[float | None, float | None, float | None]:
    """IC 95% da fração de ausência, reamostrando PAPERS.

    Devolve `(None, None, None)` com menos de 30 papers: um IC de bootstrap sobre
    amostra pequena é largo e enviesado, e apresentá-lo dá aparência de rigor a
    um número que não tem.
    """
    if len(papers) < 30:
        return None, None, None
    import random
    rng = random.Random(seed)

    def frac(am: list[Comparacao]) -> float:
        tf = sum(c.eq_fonte for c in am)
        return sum(max(0, c.eq_fonte - c.eq_redpajama) for c in am) / tf if tf else 0.0

    n = len(papers)
    bs = sorted(frac([papers[rng.randrange(n)] for _ in range(n)])
                for _ in range(reamostras))
    lo, hi = bs[int(0.025 * reamostras)], bs[int(0.975 * reamostras)]
    return lo, hi, sum(1 for b in bs if b > 0.10) / reamostras


@dataclass
class Auditoria:
    comparacoes: list[Comparacao] = field(default_factory=list)

    @property
    def uteis(self) -> list[Comparacao]:
        """Papers com equações na fonte. Sem equação não há o que preservar, e
        incluí-los na média puxaria o resultado para um lado arbitrário."""
        return [c for c in self.comparacoes if not c.erro and c.eq_fonte > 0]

    @property
    def fieis(self) -> list[Comparacao]:
        """Papers em que o documento principal foi identificado e os `\\input`
        resolvidos. Só nestes a contagem da fonte é o que o LaTeX veria.

        Nos outros caímos em concatenar o pacote, e aí um rascunho esquecido no
        tarball vira "equação que o RedPajama perdeu". **É este subconjunto que
        decide**, porque a decisão em jogo é gastar US$ 100–180.
        """
        return [c for c in self.uteis if c.confiavel]

    def resumo(self) -> dict:
        u = self.fieis
        todos = self.uteis
        if not u:
            return {"erro": f"nenhum paper com fonte montada de forma fiel "
                            f"({len(todos)} comparáveis, todos por concatenação)"}
        tf = sum(c.eq_fonte for c in u)
        tp = sum(c.preservadas for c in u)
        # Por equação (agregado) e por paper (média das taxas). Os dois importam:
        # o primeiro é a taxa global, o segundo revela se a perda se concentra em
        # poucos papers ou está espalhada.
        por_eq = tp / tf
        taxas = sorted(c.taxa for c in u)
        med = taxas[len(taxas) // 2]
        # ── decomposição, que é o que torna o número acionável ────────────
        #
        # Um número único de "degradação" mistura duas coisas com consequências
        # opostas, e a decisão de gastar US$ 180 depende de qual delas domina:
        #
        #   AUSENTES    o RedPajama tem MENOS equações que a fonte. É perda de
        #               conteúdo: a equação não está lá de forma nenhuma. Só isto
        #               justifica pagar o bulk.
        #   DISCORDAM   contagens parecidas e formas canônicas diferentes. É
        #               notação — ou resíduo do NOSSO comparador. Pagar por isto
        #               seria comprar a solução de um problema nosso.
        #
        # Medido em 199 papers: 13,4% ausentes e 14,0% discordantes. Reportar
        # 27,4% e mandar gastar seria juntar as duas.
        ausentes = sum(max(0, c.eq_fonte - c.eq_redpajama) for c in u)
        discordam = sum(max(0, min(c.eq_fonte, c.eq_redpajama) - c.preservadas) for c in u)
        f_aus, f_dis = ausentes / tf, discordam / tf

        # ── incerteza, sem a qual a estimativa pontual não decide ─────────
        #
        # A unidade de amostragem é o PAPER, não a equação: sorteamos papers do
        # shard, e as equações de um mesmo paper compartilham o destino que o
        # pipeline do RedPajama lhe deu. Tratar 19 mil equações como 19 mil
        # observações independentes daria um intervalo falsamente estreito.
        #
        # Daí o bootstrap reamostrar PAPERS. Ele é largo de propósito: poucos
        # papers densos em equações dominam a razão, e essa instabilidade é real,
        # não defeito do método.
        lo, hi, p_acima = _bootstrap_ausencia(u)

        if p_acima is not None and lo <= 0.10 <= hi:
            v = (f"INDECISO — ausência de {100*f_aus:.1f}%, IC 95% "
                 f"[{100*lo:.1f}%, {100*hi:.1f}%], cruza o limiar de 10%. "
                 f"{100*p_acima:.0f}% das reamostragens ficam acima. "
                 f"Ampliar a amostra decide; gastar agora seria apostar.")
        elif f_aus <= 0.10 and f_aus + f_dis > 0.10:
            v = (f"INCONCLUSIVO — perda REAL de {100*f_aus:.1f}% está abaixo do limiar, "
                 f"mas {100*f_dis:.1f}% de discordância de notação impede afirmar. "
                 "Resolver o comparador antes de decidir.")
        elif f_aus > 0.10:
            v = (f"RedPajama DEGRADA — perda real de {100*f_aus:.1f}%, IC 95% "
                 f"[{100*lo:.1f}%, {100*hi:.1f}%], inteiramente acima do limiar de "
                 "10%; o bulk pago do arXiv (US$ 100–180) se justifica (DOC-02 §3.2)")
        else:
            v = (f"RedPajama SERVE — perda real de {100*f_aus:.1f}%, IC 95% "
                 f"[{100*lo:.1f}%, {100*hi:.1f}%], abaixo do limiar de 10%")

        # Contraste com a medição ANTIGA, sobre todos os papers. Se os dois
        # números divergirem muito, a diferença É o viés da concatenação — e
        # deixar isso visível é o que impede a próxima pessoa (ou eu) de voltar a
        # confiar na média suja.
        tf_t = sum(c.eq_fonte for c in todos)
        contraste = round(1 - sum(c.preservadas for c in todos) / tf_t, 4) if tf_t else None

        return {
            "papers_comparados": len(u),
            "papers_por_concatenacao_excluidos": len(todos) - len(u),
            "papers_com_erro": sum(1 for c in self.comparacoes if c.erro),
            "equacoes_na_fonte": tf,
            "equacoes_preservadas": tp,
            "preservacao_por_equacao": round(por_eq, 4),
            "preservacao_mediana_por_paper": round(med, 4),
            "degradacao_total": round(1 - por_eq, 4),
            "degradacao_por_ausencia": round(f_aus, 4),
            "degradacao_por_discordancia": round(f_dis, 4),
            # Sobre TODOS os comparáveis, inclusive os concatenados. Serve de
            # contraste, não de resultado.
            "degradacao_total_sem_filtro": contraste,
            "papers_abaixo_de_90pc": sum(1 for t in taxas if t < 0.90),
            "ausencia_ic95": [None if lo is None else round(lo, 4),
                              None if hi is None else round(hi, 4)],
            "ausencia_p_acima_do_limiar": None if p_acima is None else round(p_acima, 4),
            "limiar_do_doc02": 0.10,
            "veredito": v,
        }


def amostrar_redpajama(http: PoliteSession, n: int = N_PAPERS,
                       cache: Path | None = None) -> dict[str, str]:
    """`arxiv_id` → texto, do começo do primeiro shard.

    Cacheado: a faixa de 16 MB é a MESMA a cada rodada, e reiterar no comparador
    não deve rebaixá-la. Cachear também garante que rodadas sucessivas comparem
    a mesma amostra — se o shard mudasse, a comparação entre rodadas perderia
    sentido sem aviso.
    """
    guardado = cache / f"redpajama_shard0_{BYTES_DO_SHARD}.jsonl" if cache else None
    veio_do_cache = guardado is not None and guardado.exists()
    if veio_do_cache:
        linhas = guardado.read_text(encoding="utf-8").split("\n")
    else:
        url = http.get(INDICE_REDPAJAMA, timeout=60).text.splitlines()[0].strip()
        r = http.get(url, timeout=300, headers={"Range": f"bytes=0-{BYTES_DO_SHARD}"})
        # A última linha da faixa quase certamente está cortada.
        linhas = r.text.split("\n")[:-1]
        if guardado is not None:
            guardado.parent.mkdir(parents=True, exist_ok=True)
            guardado.write_text("\n".join(linhas), encoding="utf-8")
    out: dict[str, str] = {}
    for linha in linhas:
        try:
            d = json.loads(linha)
        except json.JSONDecodeError:
            continue
        aid = (d.get("meta") or {}).get("arxiv_id")
        if aid and d.get("text"):
            out[aid] = d["text"]
        if len(out) >= n:
            break
    # A checagem tem de ser ANTES da gravação: testar `exists()` depois de
    # escrever faz o log dizer "do cache" na própria rodada que criou o cache.
    log.info("RedPajama: %d papers amostrados de %d linhas do primeiro shard%s",
             len(out), len(linhas), " (do cache)" if veio_do_cache else " (baixado)")
    return out


def baixar_fonte(http: PoliteSession, arxiv_id: str,
                 cache: Path | None = None) -> dict[str, str]:
    """Arquivos `.tex` da submissão. Aceita tar.gz e .tex avulso gzipado.

    O cache guarda o tarball CRU. Existe por cortesia antes de por velocidade:
    iterar no comparador não deve custar 199 novas requisições ao arXiv a cada
    tentativa (princípio A5, DOC-02 §1). O tarball é o insumo fiel — reextrair
    dele é grátis e não depende de como a extração estava escrita na hora.
    """
    guardado = cache / f"{arxiv_id}.tar" if cache else None
    if guardado is not None and guardado.exists():
        bruto = guardado.read_bytes()
    else:
        r = http.get(EPRINT + arxiv_id, timeout=120)
        bruto = r.content
        if guardado is not None:
            guardado.parent.mkdir(parents=True, exist_ok=True)
            guardado.write_bytes(bruto)
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


def comparar_um(http: PoliteSession, arxiv_id: str, texto_rp: str,
                cache: Path | None = None) -> Comparacao:
    c = Comparacao(arxiv_id)
    try:
        doc = montar_documento(baixar_fonte(http, arxiv_id, cache))
    except Exception as exc:
        c.erro = f"{type(exc).__name__}: {str(exc)[:80]}"
        return c
    c.modo, c.tex_ignorados = doc.modo, len(doc.ignorados)

    # ⚠️ Expandir as macros do autor ANTES de extrair. Sem isto a medição
    # confunde notação com conteúdo: a fonte escreve `\Ecal_\mu`, o RedPajama
    # escreve `\mathcal{E}_\mu`, e a equação conta como perdida. Medido no
    # 1607.04520: das que NÃO casavam, 97% usavam macro; das que casavam, 20%.
    eq_fonte = extrair_equacoes(preparar(doc.texto))
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


def so_fisica(amostra: dict[str, str], spine: Path) -> dict[str, str]:
    """Restringe a amostra aos papers de Física, pelo spine.

    ## O defeito que isto corrige

    O shard do RedPajama-arXiv é o **arXiv inteiro**. Eu amostrava dele e
    reportava "degradação do RedPajama" como se fosse resposta à pergunta do
    DOC-02, que é sobre equações de **Física**. Medido: 52% da amostra não estava
    no spine — metade do resultado vinha de math, cs e afins.

    Não era só população errada, era população com patologia própria. O paper que
    mais contribuía para a perda (`1607.04847`, 1.418 equações na fonte contra
    331 no RedPajama) é de teoria de grafos, e as "equações perdidas" eram tabelas
    de ciclos de permutação num apêndice — listas de inteiros como
    `(52,0,22,47,31,...)`. O RedPajama descartou o apêndice, o que para Física não
    diz nada.

    Restringir muda os números na direção que importa:

        população          total   ausência   discordância
        arXiv inteiro      26,4%     13,4%          13,0%
        só Física          19,1%     14,2%           4,9%

    A discordância cai de 13% para 4,9% — o comparador foi escrito para notação
    de Física e funciona nela. E a ausência, que é o que decide o gasto, fica
    ligeiramente MAIOR.
    """
    ids = pl.Series(list(amostra))
    fis = set(
        pl.scan_parquet(spine)
        .filter(pl.col("arxiv_id").is_in(ids.implode()))
        .select("arxiv_id")
        .collect()["arxiv_id"]
    )
    fora = len(amostra) - len(fis)
    log.info("amostra: %d de %d são Física pelo spine (%d descartados: o shard "
             "do RedPajama é o arXiv inteiro)", len(fis), len(amostra), fora)
    return {k: v for k, v in amostra.items() if k in fis}


def auditar(n: int = N_PAPERS, contato: str = CONTACT,
            cache: Path | None = None,
            spine: Path | None = None) -> Auditoria:
    # 1 req/3 s é o pedido do arXiv (DOC-02 §8.2) e vale para `/e-print/` também.
    http = PoliteSession(RateLimit(requests_per_second=1 / 3, max_retries=6,
                                   backoff_max_s=300), contato)
    # Amostra com folga: só ~metade do shard é Física, e o filtro vem depois.
    amostra = amostrar_redpajama(http, n * 3 if spine else n, cache)
    if spine is not None:
        amostra = dict(list(so_fisica(amostra, spine).items())[:n])
    a = Auditoria()
    for i, (aid, texto) in enumerate(amostra.items(), 1):
        c = comparar_um(http, aid, texto, cache)
        a.comparacoes.append(c)
        if i % 20 == 0 or c.erro:
            t = f"{c.taxa:.2f}" if c.taxa is not None else "—"
            log.info("%d/%d · %s · fonte %d eq · rp %d eq · preservação %s · %s%s",
                     i, len(amostra), aid, c.eq_fonte, c.eq_redpajama, t, c.modo,
                     f" · {c.erro}" if c.erro else "")
    modos: dict[str, int] = {}
    for c in a.comparacoes:
        modos[c.modo or "erro"] = modos.get(c.modo or "erro", 0) + 1
    log.info("montagem da fonte: %s", modos)
    return a


def salvar(a: Auditoria, destino: Path) -> dict:
    destino.parent.mkdir(parents=True, exist_ok=True)
    r = a.resumo()
    destino.write_text(
        json.dumps({"resumo": r, "por_paper": [asdict(c) for c in a.comparacoes]},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")
    return r
