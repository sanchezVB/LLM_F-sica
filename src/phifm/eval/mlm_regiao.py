"""MLM medido POR REGIÃO: token de equação contra token de prosa. Sem torch.

Uma das três medidas que o DOC-05 §11.2 exige, e a que discrimina a hipótese do
DOC-07 §2.3 — a de que mascarar equações inteiras ensina o modelo a *fechar* uma
equação em vez de completar um símbolo isolado. Se a hipótese vale, o braço
tratado ganha **nos tokens de equação** e empata na prosa. Um número agregado
misturaria os dois e não distinguiria nada.

Aqui mora só a parte pura: escolher as posições mascaradas e agregar o resultado.
A passagem pelo modelo fica em `scripts/avaliar_phienc_mlm.py`, na venv de treino.
É a mesma separação de `training/amostragem.py`, e pelo mesmo motivo: a guarda
abaixo é o que faz a medição valer, e uma guarda só serve se o teste dela roda na
suíte rápida.

## ⚠️ O mascaramento da AVALIAÇÃO é neutro, e isso não é um detalhe

O braço tratado é treinado com uma política própria: equação inteira para
`[MASK]`, sem o 80/10/10. Avaliar com **essa** política testaria o modelo tratado
na distribuição em que ele treinou e o controle numa distribuição estranha a ele.
O tratado ganharia por construção, e o número teria a cara de uma medição.

Então a avaliação mascara **posições uniformes**, ignorando as marcas — e usa as
marcas apenas para *reportar* onde cada acerto caiu. Nenhum dos dois braços vê a
sua própria política de treino.

A consequência é que este teste é **conservador para a hipótese**: se o
mascaramento por span ensina algo, ele tem de aparecer mesmo quando a prova é
feita com máscara pontual. Se aparecer só com máscara por span, é memorização de
formato, não capacidade — e essa distinção é a razão de a medida existir.

## ⚠️ As MESMAS posições para os dois braços

`posicoes_mascaradas` é determinística em `(semente, sequência)`. Dois modelos
avaliados com a mesma semente vêem exatamente as mesmas máscaras, o que permite
o McNemar pareado sobre "acertou este token" — muito mais sensível que comparar
duas taxas soltas, e é o que `eval/statistics` já usa no resto do projeto.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Fração de tokens mascarados na avaliação. 15% é a convenção do BERT, e o valor
# não é sintonizável aqui de propósito: mudá-lo entre braços mudaria a
# dificuldade, e mudá-lo entre execuções tornaria os números incomparáveis.
FRACAO_MASCARA = 0.15


def posicoes_mascaradas(n_tokens: int, semente: int, indice: int,
                        fracao: float = FRACAO_MASCARA,
                        proibidas: np.ndarray | None = None) -> np.ndarray:
    """Posições a mascarar — UNIFORMES, ignorando as marcas de equação.

    Ver o §"o mascaramento da avaliação é neutro" na docstring do módulo.

    Determinística em `(semente, indice)`: o mesmo par devolve as mesmas posições
    em qualquer execução e em qualquer máquina, para os braços serem pareados
    exatamente.

    `proibidas` são posições que não podem ser mascaradas — tokens especiais e
    preenchimento. Mascarar preenchimento gastaria orçamento de máscara em
    posições que nenhum modelo tem como acertar, e a taxa cairia igual nos dois
    braços mas com ruído a mais.
    """
    if n_tokens <= 0:
        return np.empty(0, dtype=np.int64)
    livres = np.arange(n_tokens, dtype=np.int64)
    if proibidas is not None and proibidas.size:
        livres = livres[~np.isin(livres, proibidas)]
    if livres.size == 0:
        return np.empty(0, dtype=np.int64)
    quantos = max(1, int(round(livres.size * fracao)))
    quantos = min(quantos, livres.size)
    # `default_rng((semente, indice))` e não `semente + indice`: somar colide —
    # (17, 5) e (18, 4) dariam a MESMA sequência, e duas sequências vizinhas
    # receberiam máscaras correlacionadas.
    rng = np.random.default_rng((semente, indice))
    return np.sort(rng.choice(livres, size=quantos, replace=False))


@dataclass
class Contagem:
    """Acertos e totais, separados por região. `equacao` vem das marcas."""

    acertos_equacao: int = 0
    total_equacao: int = 0
    acertos_prosa: int = 0
    total_prosa: int = 0
    # Um vetor por token avaliado, na ordem em que foram vistos: 1 acertou, 0
    # errou. É o que o McNemar pareado consome — sem ele, dois braços só podem
    # ser comparados como proporções soltas, que é muito menos sensível.
    acertos: list[int] = field(default_factory=list)
    e_equacao: list[int] = field(default_factory=list)

    def somar(self, certo: np.ndarray, em_equacao: np.ndarray) -> None:
        if certo.shape != em_equacao.shape:
            raise ValueError(
                f"certo tem {certo.shape} e em_equacao tem {em_equacao.shape}; "
                "com formas diferentes o acerto seria atribuído à região errada")
        eq = em_equacao.astype(bool)
        self.acertos_equacao += int(certo[eq].sum())
        self.total_equacao += int(eq.sum())
        self.acertos_prosa += int(certo[~eq].sum())
        self.total_prosa += int((~eq).sum())
        self.acertos.extend(int(x) for x in certo.astype(int))
        self.e_equacao.extend(int(x) for x in eq.astype(int))

    def como_dict(self) -> dict:
        def taxa(a: int, t: int) -> float | None:
            return round(a / t, 4) if t else None

        return {
            "acuracia_equacao": taxa(self.acertos_equacao, self.total_equacao),
            "acuracia_prosa": taxa(self.acertos_prosa, self.total_prosa),
            "acuracia_total": taxa(self.acertos_equacao + self.acertos_prosa,
                                   self.total_equacao + self.total_prosa),
            "tokens_equacao": self.total_equacao,
            "tokens_prosa": self.total_prosa,
            # ⚠️ A diferença entre as regiões é a QUANTIDADE de interesse, e ela
            # tem de sair calculada em vez de o leitor subtrair duas linhas.
            "vantagem_em_equacao": (
                round(self.acertos_equacao / self.total_equacao
                      - self.acertos_prosa / self.total_prosa, 4)
                if self.total_equacao and self.total_prosa else None),
            "nota": (
                "Máscara UNIFORME, ignorando as marcas — as marcas só dizem onde "
                "cada acerto caiu. Avaliar com a política de máscara do braço "
                "tratado o testaria na distribuição do próprio treino e o "
                "controle numa estranha a ele, e o tratado ganharia por "
                "construção. Por isso este teste é CONSERVADOR para a hipótese "
                "do DOC-07 §2.3."),
        }


def fracao_de_equacao(marcas: np.ndarray, bit_math: int) -> float:
    """Quanto da sequência é token de equação — o denominador do interesse.

    Serve de conferência de sanidade: se vier perto de zero, a avaliação está
    rodando sobre texto sem equação e a diferença entre regiões não pode
    aparecer, por mais que a hipótese esteja certa.
    """
    if marcas.size == 0:
        return 0.0
    return float(((marcas & bit_math) != 0).mean())
