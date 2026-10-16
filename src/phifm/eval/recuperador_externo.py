"""O recuperador contra os de fora, cada um como os autores mandam (DOC-13 §9.12, ACEITA
em 2026-10-16).

As constantes abaixo SÃO a regra pré-registrada; o DOC-13 §9.12 é o texto dela.

## O que a revisão aponta

A Tabela 6 do artigo põe todos os modelos no MESMO protocolo: média dos tokens, 192
tokens, `título. resumo`, cosseno. É a escolha honesta entre modelos treinados para ela —
e um handicap para os que não foram. O SPECTER2 e o SciNCL foram treinados com o token
[CLS], `título [SEP] resumo`, até 512 tokens, e são medidos pela distância euclidiana. A
pergunta é se o ΦEmb continua à frente quando cada um roda como os autores mandam.

## Duas partes

1. **Na nossa tarefa (decide):** o pool de 2.000 pares da Tabela 6, o mesmo sorteio. O
   ΦEmb no protocolo dele; o SPECTER2 (adaptador de proximidade) e o SciNCL na
   configuração recomendada. Antes de tudo, o caminho novo tem de REPRODUZIR a Tabela 6
   para dois modelos já medidos — senão a diferença pode ser do código, não do modelo.
2. **Na tarefa deles (relatada):** as quatro tarefas de proximidade do SciDocs, avaliadas
   pelo código do SciRepEval. O ΦEmb, a base dele (MiniLM-L6) e os dois de fora. Os
   vetores são gerados aqui — o harness oficial manda tudo para CUDA, e a máquina do dono
   é AMD —, e a avaliação (os pares, as métricas, a distância) é a deles.

Este módulo não importa torch no topo: a configuração, o texto, a ordenação e a leitura
rodam na suíte rápida. Só `codificar` precisa da venv de treino.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np
import polars as pl

from phifm.core.modelos import RECUPERADOR
from phifm.eval.statistics.proporcao import binomial_exata_bicaudal
from phifm.training.amostragem import SEMENTE_POOL, preparar_pool

# ── a regra (DOC-13 §9.12) ───────────────────────────────────────────────────
# As duas partes recusam enquanto isto for False. ACEITA pelo dono em 2026-10-16, como
# escrita.
REGRA_9_12_ACEITA = True
N_POOL = 2000                     # o protocolo da Tabela 6
N_BOOTSTRAP = 10_000
SEMENTE_BOOTSTRAP = 20261018
# A guarda de reprodução: o nDCG@10 que a Tabela 6 gravou (g1_resultado.json), e a folga.
# O ΦEmb do sistema é a cópia instalada do run de 6 M (`phiemb-minilm-6m-sorteado-melhor`
# na tabela; ver `phifm.core.modelos`): se não for o mesmo peso, a guarda pega.
# 0,0005 porque o desempate de cossenos quase iguais muda com a ordem da soma (BLAS do
# numpy contra o do torch, CPU contra GPU) — e uma diferença de código daria muito mais.
TABELA_6 = {"sentence-transformers/all-MiniLM-L6-v2": 0.4761127829551697,
            RECUPERADOR: 0.6223083138465881}
TOLERANCIA_REPRODUCAO = 0.0005
# O harness, fixo no commit lido em 2026-10-16 (o `main` daquele dia).
SCIREPEVAL_URL = "https://github.com/allenai/scirepeval"
SCIREPEVAL_COMMIT = "e04594e6401cb68fb4ddf6daa801fb219cccc0bb"
SCIDOCS_META = ("allenai/scirepeval", "scidocs_view_cite_read")
SCIDOCS_TAREFAS = {"SciDocs Cite": "scidocs_cite", "SciDocs CoCite": "scidocs_cocite",
                   "SciDocs CoView": "scidocs_view", "SciDocs CoRead": "scidocs_read"}
SCIDOCS_METRICAS = ("map", "ndcg")


@dataclass(frozen=True)
class Config:
    """Um modelo e o jeito de lê-lo. Trocar qualquer campo é medir outra coisa."""
    nome: str
    caminho: str
    pooling: str                  # "media" | "cls"
    formato: str                  # "ponto": `título. resumo` · "sep": `título[SEP]resumo`
    max_tokens: int
    similaridade: str             # "cosseno" | "euclidiana"
    adaptador: str | None = None  # o SPECTER2 é a base + um adaptador por tarefa
    papel: str = ""               # "nosso" | "cientifico" | "geral" | "base"

    @property
    def chave(self) -> str:
        """Para nome de arquivo: o mesmo modelo lido de dois jeitos são duas chaves."""
        base = re.sub(r"[^A-Za-z0-9]+", "-", f"{self.caminho}-{self.adaptador or ''}").strip("-")
        return f"{base}_{self.pooling}_{self.formato}_{self.max_tokens}_{self.similaridade}"


NOSSO = Config("ΦEmb (o do sistema, 6 M pares)", RECUPERADOR, "media", "ponto", 192, "cosseno",
               papel="nosso")
MINILM = Config("all-MiniLM-L6-v2 (a base do ΦEmb)", "sentence-transformers/all-MiniLM-L6-v2",
                "media", "ponto", 192, "cosseno", papel="base")
# Como os autores mandam: o cartão de cada modelo e o harness em que foram medidos.
SPECTER2 = Config("SPECTER2 (adaptador de proximidade)", "allenai/specter2_base", "cls", "sep",
                  512, "euclidiana", adaptador="allenai/specter2", papel="cientifico")
SCINCL = Config("SciNCL", "malteos/scincl", "cls", "sep", 512, "euclidiana", papel="cientifico")
CIENTIFICOS = (SPECTER2, SCINCL)


def no_protocolo_da_tabela_6(c: Config) -> Config:
    return replace(c, nome=f"{c.nome} · protocolo da Tabela 6", pooling="media",
                   formato="ponto", max_tokens=192, similaridade="cosseno")


# O GTE-large é a margem do artigo (+0,044), medida a 192 tokens; ele lê 512.
GTE_512 = Config("GTE-large a 512 tokens", "thenlper/gte-large", "media", "ponto", 512,
                 "cosseno", papel="geral")
NA_NOSSA_TAREFA = (MINILM, NOSSO, *CIENTIFICOS, *map(no_protocolo_da_tabela_6, CIENTIFICOS),
                   GTE_512)
NA_TAREFA_DELES = (NOSSO, MINILM, *CIENTIFICOS)


def _limpo(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def texto(titulo: str | None, resumo: str | None, formato: str, sep: str = "[SEP]") -> str:
    """`ponto` é byte a byte o texto de treino do ΦEmb (`textos_de_documentos`); `sep` é o
    do cartão do SPECTER2 e do SciNCL: título, o [SEP] do tokenizador, resumo."""
    if formato == "ponto":
        return re.sub(r"\s+", " ", f"{titulo or ''}. {resumo or ''}").strip()
    if formato == "sep":
        return f"{_limpo(titulo)}{sep}{_limpo(resumo)}"
    raise ValueError(f"formato {formato!r}")


def pool_com_campos(val: pl.DataFrame, spine: pl.LazyFrame, n: int = N_POOL,
                    semente: int = SEMENTE_POOL) -> pl.DataFrame:
    """O pool da Tabela 6 — o mesmo `preparar_pool` — com título e resumo de cada lado,
    do spine. Recusa se o texto refeito não for o do pool: aí o par não é o mesmo."""
    pool = preparar_pool(val, n, semente).with_row_index("_ordem")
    ids = sorted(set(pool["arxiv_id"].to_list()) | set(pool["arxiv_citado"].to_list()))
    meta = (spine.filter(pl.col("arxiv_id").is_in(ids)).select("arxiv_id", "title", "abstract")
            .unique("arxiv_id").collect())
    faltam = sorted(set(ids) - set(meta["arxiv_id"].to_list()))
    if faltam:
        raise ValueError(f"{len(faltam)} artigos do pool fora do spine (o primeiro: {faltam[0]})")
    # Junção e ordem de volta pelo índice: a ordem das linhas É o pareamento consulta-alvo.
    pool = (pool.join(meta.rename({"title": "titulo_a", "abstract": "resumo_a"}),
                      on="arxiv_id", how="left")
            .join(meta.rename({"arxiv_id": "arxiv_citado", "title": "titulo_p",
                               "abstract": "resumo_p"}), on="arxiv_citado", how="left")
            .sort("_ordem").drop("_ordem"))
    for lado, col in (("a", "ancora"), ("p", "positivo")):
        refeito = [texto(t, r, "ponto") for t, r in
                   zip(pool[f"titulo_{lado}"], pool[f"resumo_{lado}"], strict=True)]
        diferentes = sum(x != y for x, y in zip(refeito, pool[col], strict=True))
        if diferentes:
            raise ValueError(f"{diferentes} textos de '{col}' refeitos do spine não batem com o "
                             "pool: o spine não é o que gerou os pares")
    return pool


def carregar(c: Config, dispositivo: str = "cpu"):
    """`(modelo, tokenizador, dispositivo)` de `c`. O SPECTER2 é a base com o adaptador
    de proximidade ativo, como no cartão dele — o pacote `adapters` mora numa venv à parte
    (ver o script), porque fixa a versão do `transformers`."""
    from transformers import AutoModel, AutoTokenizer

    from phifm.training.embedding import escolher_dispositivo

    dev = escolher_dispositivo(dispositivo)
    tok = AutoTokenizer.from_pretrained(c.caminho)
    if c.adaptador:
        from adapters import AutoAdapterModel

        mod = AutoAdapterModel.from_pretrained(c.caminho)
        mod.load_adapter(c.adaptador, source="hf", load_as="tarefa", set_active=True)
    else:
        mod = AutoModel.from_pretrained(c.caminho, attn_implementation="eager")
    return mod.to(dev).eval(), tok, dev


def codificar(c: Config, titulos: list[str], resumos: list[str], dispositivo: str = "cpu",
              lote: int = 16, carregado=None) -> np.ndarray:
    """Os vetores de `c`, na configuração de `c`. A média passa pelo MESMO `_codificar` da
    Tabela 6 (que normaliza); o [CLS] segue o cartão do SPECTER2 — sem `token_type_ids`,
    preenchimento só até o maior do lote — e só normaliza se a similaridade for o cosseno.
    `carregado` reaproveita o modelo entre partes de um corpo grande."""
    import torch
    import torch.nn.functional as F

    mod, tok, dev = carregado or carregar(c, dispositivo)
    textos = [texto(t, r, c.formato, tok.sep_token) for t, r in zip(titulos, resumos,
                                                                     strict=True)]
    if c.pooling == "media":
        from phifm.eval.encoders import _codificar

        return _codificar(mod, tok, textos, dev, c.max_tokens, lote).float().numpy()
    saidas = []
    with torch.no_grad():
        for i in range(0, len(textos), lote):
            b = tok(textos[i:i + lote], padding=True, truncation=True, max_length=c.max_tokens,
                    return_tensors="pt", return_token_type_ids=False)
            h = mod(**{k: v.to(dev) for k, v in b.items()}).last_hidden_state[:, 0, :]
            saidas.append(h.float().cpu())
    v = torch.cat(saidas)
    return (F.normalize(v, dim=-1) if c.similaridade == "cosseno" else v).numpy()


# ── a ordenação e as métricas (sem torch) ────────────────────────────────────


def posicoes(va: np.ndarray, vp: np.ndarray, similaridade: str) -> np.ndarray:
    """A posição (1 = primeiro) do alvo de cada consulta entre os `n` candidatos. Empate
    exato conta a favor do alvo; com textos distintos, é raro e igual para todos."""
    va, vp = np.asarray(va, dtype=np.float64), np.asarray(vp, dtype=np.float64)
    if similaridade == "cosseno":
        sim = va @ vp.T
    elif similaridade == "euclidiana":
        # −‖a − p‖², que ordena como −‖a − p‖: a raiz é monótona.
        sim = 2 * va @ vp.T - (va * va).sum(1)[:, None] - (vp * vp).sum(1)[None, :]
    else:
        raise ValueError(f"similaridade {similaridade!r}")
    alvo = np.diag(sim)
    return 1 + (sim > alvo[:, None]).sum(axis=1)


def ndcg10_por_item(pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos)
    return np.where(pos <= 10, 1.0 / np.log2(pos + 1.0), 0.0)


def metricas(pos: np.ndarray) -> dict:
    pos = np.asarray(pos)
    return {"ndcg_10": round(float(ndcg10_por_item(pos).mean()), 4),
            "recall_1": round(float((pos == 1).mean()), 4),
            "recall_10": round(float((pos <= 10).mean()), 4),
            "mrr": round(float((1.0 / pos).mean()), 4), "n": int(pos.size)}


def comparar(pos_a: np.ndarray, pos_b: np.ndarray, n_bootstrap: int = N_BOOTSTRAP,
             semente: int = SEMENTE_BOOTSTRAP) -> dict:
    """A − B em nDCG@10, com IC 95% por bootstrap pareado por consulta; McNemar exato em
    recall@1, relatado como na Tabela 6."""
    d = ndcg10_por_item(pos_a) - ndcg10_por_item(pos_b)
    rng = np.random.default_rng(semente)
    reps = d[rng.integers(0, d.size, (n_bootstrap, d.size))].mean(axis=1)
    ic = (round(float(np.percentile(reps, 2.5)), 4), round(float(np.percentile(reps, 97.5)), 4))
    a1, b1 = np.asarray(pos_a) == 1, np.asarray(pos_b) == 1
    return {"ndcg_10_A_menos_B": round(float(d.mean()), 4), "ic95": ic,
            "leitura": "A_acima" if ic[0] > 0 else "A_abaixo" if ic[1] < 0 else
            "nao_se_distingue",
            "recall_1_mcnemar (relatado)": binomial_exata_bicaudal(int((a1 & ~b1).sum()),
                                                                   int((b1 & ~a1).sum()))}


def comparar_por_consulta(a: dict, b: dict, n_bootstrap: int = N_BOOTSTRAP,
                          semente: int = SEMENTE_BOOTSTRAP) -> dict:
    """A − B numa métrica por consulta (`{(tarefa, consulta): valor}`, o nDCG do SciDocs),
    nas consultas que os dois têm, com IC por bootstrap pareado. Relatado."""
    chaves = sorted(set(a) & set(b))
    if not chaves:
        raise ValueError("nenhuma consulta em comum")
    d = np.array([a[k] - b[k] for k in chaves], dtype=float)
    reps = d[np.random.default_rng(semente).integers(0, d.size, (n_bootstrap, d.size))].mean(1)
    ic = (round(float(np.percentile(reps, 2.5)), 4), round(float(np.percentile(reps, 97.5)), 4))
    return {"n_consultas": len(chaves), "A_menos_B": round(float(d.mean()), 4), "ic95": ic,
            "leitura": "A_acima" if ic[0] > 0 else "A_abaixo" if ic[1] < 0 else
            "nao_se_distingue"}


def conferir_reproducao(medidos: dict[str, float]) -> dict:
    """A guarda: o caminho novo reproduz a Tabela 6 nos dois modelos já medidos nela."""
    linhas = {c: {"tabela_6": round(v, 4), "agora": round(medidos[c], 4),
                  "passa": abs(medidos[c] - v) <= TOLERANCIA_REPRODUCAO}
              for c, v in TABELA_6.items()}
    return {"modelos": linhas, "tolerancia": TOLERANCIA_REPRODUCAO,
            "passa": all(x["passa"] for x in linhas.values())}


def decidir(reproducao: dict, contra: dict[str, dict]) -> str:
    """`contra`: a comparação ΦEmb − cada científico na configuração recomendada."""
    if not reproducao["passa"]:
        return "invalida"
    leituras = [d["leitura"] for d in contra.values()]
    if any(x == "A_abaixo" for x in leituras):
        return "abaixo_de_algum"
    if all(x == "A_acima" for x in leituras):
        return "supera_os_dois"
    return "nao_se_distingue_de_algum"
