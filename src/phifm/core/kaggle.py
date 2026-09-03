"""Registro dos experimentos que rodam na cota gratuita do Kaggle.

Existe porque `empacotar_kaggle.py` e `publicar_kaggle.py` precisam concordar sobre
os mesmos nomes, e a primeira versão dos dois tinha os slugs como constantes de
módulo em CADA script. Acrescentar um segundo experimento por cópia duplicaria
também a lição de cada erro já pago lá — `machine_shape`, `.zip.bin`, o pin da
imagem docker — e a cópia divergiria em silêncio.

Este módulo é de propósito sem torch, sem polars e sem rede: ele é só o contrato de
nomes, e por isso entra na suíte rápida.

## O invariante que este módulo faz valer, e só onde ele vale

No `kernels push` o Kaggle **deriva o slug do TÍTULO** e derruba tudo que não for
`[a-z0-9]`. Medido em 2026-08-24: o título "PhiFM T1a - PhiEmb" virou
`phifm-t1a-emb` — o Φ e o hífen solto sumiram —, o slug derivado deixou de casar com
o `id` declarado (`phifm-t1a-phiemb`) e a CLI devolveu 409 depois do upload.
`conferir()` compara `slug_derivado(titulo)` com o slug declarado e levanta ANTES de
subir nada.

⚠️ **Isto NÃO vale para o dataset**, e eu quase gravei a regra errada. A primeira
versão deste módulo checava os dois, e a suíte reprovou em segundos: o dataset do
T1a está publicado desde 2026-08-24 com o título "PhiFM T1a — pares de citação
arXiv" e o slug `phifm-t1a-pares-citacao`, que aquele título não deriva — travessão,
cedilha e a palavra "arXiv" a mais. Para `datasets create` quem manda é o campo
`id`; só o `kernels push` deriva do título.

Generalizar uma lição para além do que ela mediu tem custo: aqui teria proibido um
título legível em português por um problema que o dataset não tem.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Como o Kaggle transforma título em slug: baixa a caixa, e tudo que não é
# alfanumérico ASCII colapsa num hífen. Não é adivinhação — é o que reproduz o
# `phifm-t1a-emb` observado.
_NAO_SLUG = re.compile(r"[^a-z0-9]+")


def slug_derivado(titulo: str) -> str:
    return _NAO_SLUG.sub("-", titulo.lower()).strip("-")


@dataclass(frozen=True)
class Experimento:
    """Um experimento de GPU: o que empacotar, com que nome, e qual célula roda."""

    nome: str
    titulo_dados: str
    slug_dados: str
    titulo_notebook: str
    slug_notebook: str
    # Caminhos RELATIVOS à raiz do repositório. Absolutos aqui fariam o registro
    # depender da máquina, e ele é lido tanto pelos scripts quanto pelos testes.
    pacote: str
    fonte_celula: str
    # ⚠️ Lista DECLARADA do que o pacote contém. Montar o manifesto de `iterdir()`
    # já atestou 175 KB de código obsoleto como se fizesse parte do pacote
    # (2026-08-24). O que não está aqui é removido do diretório de saída.
    arquivos: tuple[str, ...]
    # Scripts do repositório que entram no `phifm_src.zip.bin`, para o notebook
    # chamar exatamente o que roda na máquina local em vez de reimplementar.
    scripts: tuple[str, ...]
    # Diretórios de modelo (relativos à raiz) que sobem junto. Vazio quando os
    # pesos são públicos e o Kaggle os baixa do HuggingFace — subir 90 MB de
    # `all-MiniLM-L6-v2` seria pagar banda por algo que já está lá.
    modelos: tuple[str, ...] = ()
    # `owner/repo` do GitHub. Quando preenchido, o código NÃO viaja no dataset: o
    # notebook baixa o tarball do commit exato.
    #
    # ⚠️ Isto existe por uma falha medida em 2026-09-03. O Kaggle **fixa a versão do
    # dataset** no momento em que ela é anexada ao kernel, e `kernels push` não
    # re-resolve para a mais recente — o `dataset_sources` do metadado nem carrega
    # número de versão. Resultado: o conserto do fp16 subiu numa versão nova, o
    # `datasets status` disse `ready`, e o notebook rodou 15 min sobre o código
    # ANTIGO. Ele delatou na saída (`git_sha: 73088dc` contra o conserto em
    # `68fe86e`), que foi a única razão de eu perceber.
    #
    # Com o código vindo do GitHub num SHA, não há versão a fixar: o notebook é
    # reempurrado a cada publicação. E os dados podem ficar no dataset justamente
    # porque não mudam — era o zip de 188 KB que mudava dentro dos 457 MB.
    repo: str | None = None

    def conferir(self) -> None:
        """Só o notebook. Ver a ressalva na docstring do módulo."""
        obtido = slug_derivado(self.titulo_notebook)
        if obtido != self.slug_notebook:
            raise ValueError(
                f"{self.nome}: o título do notebook {self.titulo_notebook!r} deriva "
                f"o slug {obtido!r}, e o declarado é {self.slug_notebook!r}. O "
                f"`kernels push` usa o derivado e rejeita o id com 409 — depois de "
                f"receber os arquivos. Ajuste o título ou o slug para coincidirem.")


T1A = Experimento(
    nome="t1a",
    # Título como estava publicado desde 2026-08-24; o slug vem do `id`, não dele.
    titulo_dados="PhiFM T1a — pares de citação arXiv",
    slug_dados="phifm-t1a-pares-citacao",
    titulo_notebook="PhiFM T1a Gpu",
    slug_notebook="phifm-t1a-gpu",
    pacote="data/processed/kaggle_t1a",
    fonte_celula="kaggle/t1a_phiemb.py",
    arquivos=("pares_treino.parquet", "pares_validacao.parquet",
              "phifm_src.zip.bin"),
    scripts=("train_embedding.py",),
)

T1C = Experimento(
    nome="t1c",
    titulo_dados="PhiFM T1c — ΦRank de base diferente",
    slug_dados="phifm-t1c-rerank-bases",
    titulo_notebook="PhiFM T1c Rerank",
    slug_notebook="phifm-t1c-rerank",
    pacote="data/processed/kaggle_t1c",
    fonte_celula="kaggle/t1c_phirank.py",
    # Sem `phifm_src.zip.bin`: o código vem do GitHub (ver `repo`).
    arquivos=("pares_do_recuperador_limpos.parquet", "pares_validacao.parquet",
              "modelos.zip.bin"),
    scripts=("train_rerank.py", "avaliar_t1b.py"),
    repo="sanchezVB/LLM_F-sica",
    # ⚠️ O ΦEmb vai junto e é o `phiemb-minilm-melhor`, NÃO o `-t4-melhor`.
    # O resultado de referência do T1b (nDCG 0,1584) foi medido com este, e trocar
    # o recuperador ao mesmo tempo que o reranqueador mediria duas coisas.
    #
    # O `phirank-rrf-melhor` é o CONTROLE: é o reranqueador que empatou com a fusão
    # (p=0,118), e ele precisa ser reavaliado no MESMO número de consultas que os
    # novos, senão a comparação de poder estatístico fica torta.
    modelos=("models/phiemb-minilm-melhor", "models/phirank-rrf-melhor"),
)

EXPERIMENTOS: dict[str, Experimento] = {e.nome: e for e in (T1A, T1C)}


def obter(nome: str) -> Experimento:
    try:
        exp = EXPERIMENTOS[nome]
    except KeyError:
        raise SystemExit(
            f"experimento {nome!r} não existe. Conhecidos: "
            f"{sorted(EXPERIMENTOS)}") from None
    exp.conferir()
    return exp
