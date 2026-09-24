"""O índice de busca: cada artigo de Física do arXiv, embutido pelo ΦEmb do sistema.

Construir uma vez (`construir`), buscar muitas (`Busca`). Tudo local e em disco: com ~1,6
milhão de vetores de 384 dimensões em float16 (~1,2 GB), a busca por força bruta cabe na
memória e leva frações de segundo — um banco vetorial seria uma dependência sem ganho
nesta escala.

## ⚠️ Três coisas que um índice pode errar sem dar erro nenhum

1. **O texto.** O documento é embutido pelo MESMO texto com que o encoder foi treinado —
   `phifm.training.pairs.textos_de_documentos`, a única definição dele no projeto.
   Conferido em 2026-09-24: idêntico aos 88.807 documentos citados da validação. Outro
   texto não quebraria nada; daria uma busca um pouco pior, sem causa visível.
2. **A conta.** Os vetores saem de `phifm.eval.encoders._codificar` — a mesma média
   mascarada e normalização das medições do G1. Um índice com outra agregação mediria
   uma coisa e serviria outra.
3. **O modelo.** O manifesto grava o blake3 dos pesos, e a `Busca` recusa consultar com
   outro modelo: uma consulta embutida por um encoder e comparada com vetores de outro
   devolve ordens plausíveis e erradas.

## Retomada

A indexação completa leva cerca de 1 hora na RX 7600. Os vetores vão para um `.npy`
mapeado em memória, bloco a bloco, e `progresso.json` guarda quantos documentos já estão
gravados. Rodar de novo continua de onde parou; o resultado é o mesmo de uma execução
contínua, porque cada documento é embutido sozinho no seu lote e a ordem é a da tabela.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

NOME_VETORES = "vetores.npy"
NOME_DOCUMENTOS = "documentos.parquet"
NOME_PROGRESSO = "progresso.json"
NOME_MANIFESTO = "manifesto.json"
MAX_TOKENS = 192
REGRA_DO_TEXTO = "phifm.training.pairs.textos_de_documentos"


def _silenciar_barras() -> None:
    """A barra "Loading weights" do `transformers` 5 imprime uma linha por tensor."""
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
    except Exception:  # noqa: BLE001 — cosmético; a falha não pode derrubar o índice
        pass


def hash_do_modelo(modelo: Path) -> str:
    """blake3 dos pesos: é a identidade do encoder que gerou os vetores."""
    from phifm.core.schema.reprodutibilidade import hash_arquivo

    pesos = Path(modelo) / "model.safetensors"
    if not pesos.exists():
        raise FileNotFoundError(f"{pesos} não existe — o índice precisa dos pesos para "
                                "gravar a identidade do modelo")
    return hash_arquivo(pesos)


def documentos(spine: Path) -> pl.DataFrame:
    """Os documentos a indexar, na ordem da tabela, com o que a busca mostra."""
    from phifm.training.pairs import textos_de_documentos

    s = pl.scan_parquet(spine)
    textos = textos_de_documentos(s)
    meta = s.select("arxiv_id", "title", "year", "primary_category",
                    pl.col("authors").list.head(3).alias("autores"))
    return textos.join(meta, on="arxiv_id", how="left", maintain_order="left").collect()


def construir(spine: Path, modelo: Path, saida: Path, *, dispositivo: str = "auto",
              lote: int = 64, bloco: int = 20_000, limite: int | None = None,
              max_blocos: int | None = None) -> dict:
    """Embute os documentos e grava o índice em `saida`. Retoma se já houver progresso.

    `limite` indexa só os primeiros N (para ensaio). `max_blocos` para depois de N blocos
    — existe para o teste de retomada, que precisa de uma interrupção reproduzível.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    from phifm.eval.encoders import _codificar
    from phifm.training.embedding import escolher_dispositivo

    saida.mkdir(parents=True, exist_ok=True)
    identidade = hash_do_modelo(modelo)
    docs = documentos(spine)
    if limite is not None:
        docs = docs.head(limite)
    n = docs.height

    _silenciar_barras()
    dev = escolher_dispositivo(dispositivo)
    tok = AutoTokenizer.from_pretrained(modelo)
    mod = AutoModel.from_pretrained(modelo, attn_implementation="eager").to(dev).eval()
    dim = int(mod.config.hidden_size)

    prog_arq = saida / NOME_PROGRESSO
    vetores_arq = saida / NOME_VETORES
    feitos = 0
    if prog_arq.exists():
        prog = json.loads(prog_arq.read_text(encoding="utf-8"))
        # ⚠️ Retomar sobre outro modelo, outra tabela ou outro tamanho misturaria vetores
        # de dois índices num arquivo só — recusado, em vez de continuado.
        if (prog["modelo_blake3"], prog["n"], prog["dim"]) != (identidade, n, dim):
            raise SystemExit(
                f"{saida} tem um índice parcial de OUTRO modelo ou outra tabela "
                f"({prog['modelo_blake3'][:12]}…, n={prog['n']:,}, dim={prog['dim']}). "
                "Apague a pasta ou use outra saída.")
        feitos = int(prog["feitos"])
        vetores = np.lib.format.open_memmap(vetores_arq, mode="r+")
        log.info("retomando o índice: %s de %s já gravados", f"{feitos:,}", f"{n:,}")
    else:
        docs.select("arxiv_id", "title", "year", "primary_category", "autores").write_parquet(
            saida / NOME_DOCUMENTOS)
        vetores = np.lib.format.open_memmap(vetores_arq, mode="w+", dtype=np.float16,
                                            shape=(n, dim))

    textos = docs["texto"]
    blocos_feitos = 0
    with torch.no_grad():
        while feitos < n:
            fim = min(feitos + bloco, n)
            v = _codificar(mod, tok, textos[feitos:fim].to_list(), dev, MAX_TOKENS, lote)
            vetores[feitos:fim] = v.numpy().astype(np.float16)
            vetores.flush()
            feitos = fim
            prog_arq.write_text(json.dumps({"feitos": feitos, "n": n, "dim": dim,
                                            "modelo_blake3": identidade}), encoding="utf-8")
            log.info("  %s / %s documentos", f"{feitos:,}", f"{n:,}")
            blocos_feitos += 1
            if max_blocos is not None and blocos_feitos >= max_blocos and feitos < n:
                return {"feitos": feitos, "n": n, "interrompido": True}

    manifesto = {
        "documentos": n, "dimensao": dim, "dtype": "float16", "max_tokens": MAX_TOKENS,
        "modelo": str(modelo), "modelo_blake3": identidade,
        "regra_do_texto": REGRA_DO_TEXTO,
        "agregacao": "phifm.eval.encoders._codificar (média mascarada, normalizada)",
        "tabela": str(spine), "limite": limite,
        "criado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (saida / NOME_MANIFESTO).write_text(json.dumps(manifesto, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    return {"feitos": feitos, "n": n, "interrompido": False}


@dataclass
class Resultado:
    posicao: int
    escore: float
    arxiv_id: str
    titulo: str
    ano: int | None
    categoria: str | None
    autores: list[str]

    @property
    def link(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


class Busca:
    """Carrega o índice na memória e responde consultas por similaridade de cosseno."""

    def __init__(self, indice: Path, modelo: Path | None = None, dispositivo: str = "auto"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        from phifm.training.embedding import escolher_dispositivo

        man_arq = indice / NOME_MANIFESTO
        if not man_arq.exists():
            raise SystemExit(f"{indice} não tem {NOME_MANIFESTO}: o índice não terminou de "
                             "ser construído. Rode a indexação de novo; ela retoma.")
        self.manifesto = json.loads(man_arq.read_text(encoding="utf-8"))
        modelo = Path(modelo or self.manifesto["modelo"])
        if hash_do_modelo(modelo) != self.manifesto["modelo_blake3"]:
            raise SystemExit(
                f"o modelo {modelo} não é o que gerou este índice "
                f"({self.manifesto['modelo_blake3'][:12]}…). Uma consulta de um encoder "
                "comparada com vetores de outro devolve uma ordem plausível e errada.")
        _silenciar_barras()
        self.dev = escolher_dispositivo(dispositivo)
        self.tok = AutoTokenizer.from_pretrained(modelo)
        self.mod = AutoModel.from_pretrained(
            modelo, attn_implementation="eager").to(self.dev).eval()
        self._torch = torch
        # float16 em memória (~1,2 GB para o índice inteiro); a conta vai em float32.
        self.vetores = np.load(indice / NOME_VETORES)
        self.docs = pl.read_parquet(indice / NOME_DOCUMENTOS)
        if self.docs.height != self.vetores.shape[0]:
            raise SystemExit("documentos e vetores do índice têm tamanhos diferentes")

    def embutir(self, consulta: str) -> np.ndarray:
        from phifm.eval.encoders import _codificar

        with self._torch.no_grad():
            v = _codificar(self.mod, self.tok, [consulta], self.dev, MAX_TOKENS, 1)
        return v[0].numpy().astype(np.float32)

    def buscar(self, consulta: str, k: int = 10, bloco: int = 200_000) -> list[Resultado]:
        q = self.embutir(consulta)
        melhores_i: list[np.ndarray] = []
        melhores_s: list[np.ndarray] = []
        for ini in range(0, self.vetores.shape[0], bloco):
            s = self.vetores[ini:ini + bloco].astype(np.float32) @ q
            kk = min(k, s.size)
            idx = np.argpartition(-s, kk - 1)[:kk]
            melhores_i.append(idx + ini)
            melhores_s.append(s[idx])
        todos_i, todos_s = np.concatenate(melhores_i), np.concatenate(melhores_s)
        ordem = np.argsort(-todos_s, kind="stable")[:k]
        saida = []
        for pos, j in enumerate(ordem, start=1):
            linha = self.docs.row(int(todos_i[j]), named=True)
            saida.append(Resultado(pos, float(todos_s[j]), linha["arxiv_id"], linha["title"],
                                   linha["year"], linha["primary_category"],
                                   list(linha["autores"] or [])))
        return saida
