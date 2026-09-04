"""Fluxo de dados SEM ESTADO, calculado de `(semente, passo)`. DOC-08 §7.2.

O documento é direto sobre por que isto merece um módulo próprio:

> Retomar os pesos é fácil. Retomar **a posição exata no fluxo de dados** é onde a
> maioria das implementações falha silenciosamente — e o efeito é revisitar ou pular
> dados, quebrando a política de épocas e tornando a execução irreprodutível.

A solução dele, implementada aqui: **o fluxo não guarda estado, ele calcula**. Qual
sequência corresponde ao passo `n` é uma função pura de `(semente, n)`. Retomar do
passo 60.000 produz exatamente a mesma sequência que uma execução contínua
produziria, e o batch ofensor de um *spike* (§6.1) fica identificado pelo número do
passo — reproduzível sem reexecutar nada antes dele.

## O formato em disco, e por que não parquet

O corpus tokenizado é um `memmap` plano de `uint16`:

| arquivo | dtype | bytes/token | o quê |
|---|---|---|---|
| `tokens.u16.bin` | uint16 | 2 | os ids. V=40.960 cabe em uint16 (< 65.536) |
| `marcas.u8.bin` | uint8 | 1 | 3 bits: é matemática, é display, começa equação |

`uint16` porque **40.960 < 65.536**: usar int32 dobraria os 21 GB para 42 GB sem
ganhar nada. E plano em vez de parquet porque o acesso é aleatório por offset — um
`scan_parquet` por passo releria e descomprimiria um grupo de linhas inteiro para
pegar 8.192 tokens.

## Os três bits, e a propriedade que eles dão de graça

    bit 0 (1)  é matemática
    bit 1 (2)  é display  (só faz sentido com o bit 0)
    bit 2 (4)  COMEÇA uma equação

O `id_equacao` que o mascarador espera é reconstruído por `cumsum` do bit 2 dentro da
janela. Isso resolve dois problemas de uma vez:

1. **ids únicos dentro da sequência** sem guardar um contador global de 4 bytes por
   token (que custaria 42 GB);
2. ⚠️ **uma equação cortada pela fronteira da janela fica FORA do tratamento**, de
   graça: se a janela começa no meio de uma equação, aquele trecho não tem o bit 2 e
   o `cumsum` o deixa com id −1. Mascarar "a equação inteira" quando só metade dela
   está na janela seria mascarar metade e chamar de inteira.

## O que este módulo NÃO faz

Não tokeniza. Isso é `scripts/preparar_dados_phienc.py`, uma vez, offline. Tokenizar
no laço de treino gastaria CPU competindo com o carregamento e tornaria a vazão
dependente do número de workers — e o `(semente, passo)` deixaria de determinar o
conteúdo se o tokenizer mudasse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Os três bits das marcas. `uint8` sobra: 5 bits livres para o que vier.
BIT_MATH = 1
BIT_DISPLAY = 2
BIT_INICIO = 4

NOME_TOKENS = "tokens.u16.bin"
NOME_MARCAS = "marcas.u8.bin"
NOME_MANIFESTO = "MANIFESTO_DADOS.json"


def marcas_de(id_equacao: np.ndarray, e_display: np.ndarray) -> np.ndarray:
    """Empacota `(id_equacao, e_display)` nos três bits. Usado na preparação.

    >>> import numpy as np
    >>> ide = np.array([-1, 0, 0, -1, 1, 1], dtype=np.int32)
    >>> disp = np.array([False, True, True, False, False, False])
    >>> marcas_de(ide, disp).tolist()
    [0, 7, 3, 0, 5, 1]
    """
    m = np.zeros(id_equacao.size, dtype=np.uint8)
    dentro = id_equacao >= 0
    m[dentro] |= BIT_MATH
    m[dentro & e_display] |= BIT_DISPLAY
    # Começa equação onde o id muda e o novo id é válido. O primeiro token conta
    # como início se já estiver dentro de uma equação.
    muda = np.empty(id_equacao.size, dtype=bool)
    muda[0] = dentro[0] if id_equacao.size else False
    if id_equacao.size > 1:
        muda[1:] = dentro[1:] & (id_equacao[1:] != id_equacao[:-1])
    m[muda] |= BIT_INICIO
    return m


def desempacotar(marcas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(id_equacao, e_display)` a partir das marcas de UMA janela.

    ⚠️ Uma equação sem o bit de início dentro da janela — porque a janela a cortou
    pela metade — recebe id −1 e fica fora do tratamento. Ver a docstring do módulo.

    >>> import numpy as np
    >>> m = np.array([0, 7, 3, 0, 5, 1], dtype=np.uint8)
    >>> ide, disp = desempacotar(m)
    >>> ide.tolist(), disp.tolist()
    ([-1, 0, 0, -1, 1, 1], [False, True, True, False, False, False])
    """
    math = (marcas & BIT_MATH).astype(bool)
    inicio = (marcas & BIT_INICIO).astype(bool)
    # Número da equação = quantos inícios vieram até aqui, menos 1. Onde não houve
    # nenhum início ainda (janela cortou o começo), o valor é −1 e permanece.
    ide = np.cumsum(inicio, dtype=np.int64) - 1
    ide[~math] = -1
    # E o trecho antes do primeiro início fica −1 mesmo sendo matemática: é a
    # equação truncada, que não pode ser tratada como inteira.
    ide[math & (ide < 0)] = -1
    return ide.astype(np.int32), (marcas & BIT_DISPLAY).astype(bool)


@dataclass(frozen=True)
class ConfigDados:
    raiz: Path
    contexto: int = 8_192
    # Sequências por MICRO-passo. Os ~2 M tokens por passo do DOC-08 §4 vêm de
    # acumulação de gradiente, não de um lote que caiba na memória de uma vez.
    sequencias: int = 1
    semente: int = 17

    def __post_init__(self) -> None:
        if self.contexto <= 0 or self.sequencias <= 0:
            raise ValueError("contexto e sequencias têm de ser positivos")


class Fluxo:
    """Sequências determinísticas em `(semente, passo)`. Não guarda posição."""

    def __init__(self, cfg: ConfigDados) -> None:
        self.cfg = cfg
        man = cfg.raiz / NOME_MANIFESTO
        if not man.exists():
            raise SystemExit(
                f"{man} não existe. Rode `scripts/preparar_dados_phienc.py` — o "
                "laço de treino não tokeniza, de propósito (ver a docstring do "
                "módulo).")
        self.manifesto = json.loads(man.read_text(encoding="utf-8"))
        self.tokens = np.memmap(cfg.raiz / NOME_TOKENS, dtype=np.uint16, mode="r")
        self.marcas = np.memmap(cfg.raiz / NOME_MARCAS, dtype=np.uint8, mode="r")
        if self.tokens.size != self.marcas.size:
            raise SystemExit(
                f"{NOME_TOKENS} tem {self.tokens.size:,} tokens e {NOME_MARCAS} tem "
                f"{self.marcas.size:,} marcas. Os dois são gravados juntos; um "
                "desencontro significa preparação interrompida no meio — refaça.")
        declarado = self.manifesto.get("tokens")
        if declarado is not None and declarado != int(self.tokens.size):
            raise SystemExit(
                f"o manifesto declara {declarado:,} tokens e o arquivo tem "
                f"{self.tokens.size:,}. Treinar sobre um prefixo silencioso mudaria "
                "a política de épocas sem avisar.")
        self.n_seq = int(self.tokens.size // cfg.contexto)
        if self.n_seq == 0:
            raise SystemExit(
                f"{self.tokens.size:,} tokens não dão uma sequência de "
                f"{cfg.contexto}. Prepare mais dados ou reduza o contexto.")
        self._cache_epoca: tuple[int, np.ndarray] | None = None

    # ── a permutação, que é o coração do determinismo ───────────────────────

    def _ordem(self, epoca: int) -> np.ndarray:
        """Permutação das sequências desta época. Pura em `(semente, época)`.

        ⚠️ Uma permutação por época, e não um sorteio por passo. Com sorteio
        independente por passo o modelo veria a mesma sequência duas vezes antes de
        ver outras — o que quebra a política de épocas do DOC-06 §2.4 sem nada
        avisar. Com permutação, uma época é exatamente uma passagem.

        O cache guarda UMA época: 1,29 M de sequências em int64 são 10 MB, e
        recalcular a permutação a cada passo custaria mais que guardá-la.
        """
        if self._cache_epoca is not None and self._cache_epoca[0] == epoca:
            return self._cache_epoca[1]
        rng = np.random.default_rng([self.cfg.semente, epoca])
        ordem = rng.permutation(self.n_seq)
        self._cache_epoca = (epoca, ordem)
        return ordem

    def indices_do_passo(self, passo: int) -> list[int]:
        """Quais sequências formam o micro-lote do passo `passo`.

        Função pura: mesma `(semente, passo)`, mesma resposta, em qualquer execução.
        É isto que o DOC-08 §7.2 pede e o que identifica o batch de um spike.
        """
        if passo < 0:
            raise ValueError(f"passo={passo} negativo")
        base = passo * self.cfg.sequencias
        saida = []
        for i in range(self.cfg.sequencias):
            n = base + i
            epoca, pos = divmod(n, self.n_seq)
            saida.append(int(self._ordem(epoca)[pos]))
        return saida

    def epoca_do_passo(self, passo: int) -> float:
        """Quantas épocas já passaram ao FIM deste passo, com fração."""
        return ((passo + 1) * self.cfg.sequencias) / self.n_seq

    # ── o lote ──────────────────────────────────────────────────────────────

    def sequencia(self, indice: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(ids, id_equacao, e_display)` da sequência `indice`."""
        c = self.cfg.contexto
        a, b = indice * c, (indice + 1) * c
        ids = np.asarray(self.tokens[a:b], dtype=np.int64)
        ide, disp = desempacotar(np.asarray(self.marcas[a:b]))
        return ids, ide, disp

    def lote(self, passo: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(ids, id_equacao, e_display)` empilhados, `(sequencias, contexto)`."""
        partes = [self.sequencia(i) for i in self.indices_do_passo(passo)]
        return (np.stack([p[0] for p in partes]),
                np.stack([p[1] for p in partes]),
                np.stack([p[2] for p in partes]))

    def tokens_por_passo(self) -> int:
        return self.cfg.sequencias * self.cfg.contexto

    def como_dict(self) -> dict:
        return {"tokens": int(self.tokens.size), "sequencias": self.n_seq,
                "contexto": self.cfg.contexto,
                "sequencias_por_passo": self.cfg.sequencias,
                "tokens_por_micro_passo": self.tokens_por_passo(),
                "semente": self.cfg.semente,
                "preparacao": self.manifesto.get("git_sha"),
                "tokenizer": self.manifesto.get("tokenizer")}
