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

## ⚠️ A LINHA DE BASE: token de equação já é MUITO mais fácil, sem tratamento

Medido em 2026-09-10, primeira execução da medida num modelo real. O
**ModernBERT-base** — que nunca viu mascaramento consciente de equação — sobre 2.000
sequências da fatia disjunta (39,4% de token de equação):

    acurácia em equação   0,8765
    acurácia em prosa     0,7480
    vantagem em equação   +0,1286

LaTeX é redundante: fechado um `rac{`, o `}{` e o `}` vêm quase de graça, e os
nomes de símbolo repetem dentro da mesma expressão. **Token de equação é
intrinsecamente mais previsível que prosa**, e por larga margem.

A consequência para o DOC-07 §2.3 é direta e fácil de errar: **uma
`vantagem_em_equacao` positiva no braço tratado NÃO é evidência da hipótese** — ela
já é positiva sem tratamento nenhum. A quantidade que testa a hipótese é a
**diferença entre braços** dessa vantagem, uma diferença de diferenças:

    (vantagem do tratado) − (vantagem do controle)

Reportar só o número de dentro de um braço convidaria a ler +0,13 como sucesso do
mascaramento por span quando é propriedade do LaTeX.

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
    # `[indice, acertos_eq, total_eq, acertos_prosa, total_prosa]` por sequência,
    # quando o chamador diz qual sequência é. É o que o bootstrap por SEQUÊNCIA da
    # ablação consome — ver `diferenca_das_diferencas`.
    por_sequencia: list[list[int]] = field(default_factory=list)

    def somar(self, certo: np.ndarray, em_equacao: np.ndarray,
              indice: int | None = None) -> None:
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
        if indice is not None:
            self.por_sequencia.append([
                int(indice), int(certo[eq].sum()), int(eq.sum()),
                int(certo[~eq].sum()), int((~eq).sum())])

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
            # ⚠️ A linha de base é +0,1286 SEM tratamento (ModernBERT-base,
            # 2026-09-10). Sem isto ao lado, um leitor toma a vantagem de dentro
            # de um braço por evidência da hipótese.
            "linha_de_base_sem_tratamento": 0.1286,
            "nota": (
                "⚠️ A `vantagem_em_equacao` já é +0,1286 num modelo SEM "
                "tratamento (ModernBERT-base): token de LaTeX é intrinsecamente "
                "mais previsível que prosa. O que testa a hipótese do DOC-07 §2.3 "
                "é a DIFERENÇA ENTRE BRAÇOS desta vantagem, não o valor dela. "
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


# ── a ablação do DOC-07 §2.3: dois braços, a regra da célula `t2eq_tratado` ──
#
# Acrescentado em 2026-09-16, com os dois braços treinados e ANTES de medir. A
# regra da célula (`kaggle/t2eq_tratado.py`) fixa a primária — a diferença das
# diferenças — e o teste: bootstrap pareado por SEQUÊNCIA. Este módulo guardava
# os acertos por token, e o McNemar sobre eles contaria como independentes tokens
# da mesma sequência, que compartilham contexto, notação e tópico. É a Falha 2 do
# artigo das armadilhas, e ela deu lá um intervalo 11x estreito demais.

def sequencias_sorteadas(n_disponiveis: int, n: int, semente: int) -> np.ndarray:
    """Índices de sequência SORTEADOS, em ordem crescente. NÃO um prefixo.

    ⚠️ O avaliador media `range(n)` até 2026-09-16. Numa fatia de uma parte só, as
    2.000 primeiras sequências são os ~4% primeiros documentos dela, na ordem de
    ingestão: uma amostra por conglomerado com cara de amostra aleatória — a
    armadilha de amostragem por posição que o projeto já catalogou seis vezes. A
    versão das avaliações escrita no Mac evitava isto; a de `main` não.

    Os dois braços recebem os MESMOS índices, porque o sorteio depende só de
    `(n_disponiveis, n, semente)`.
    """
    if n > n_disponiveis:
        raise ValueError(
            f"pedidas {n} sequências e a fatia tem {n_disponiveis}. Medir menos do "
            "que se pede muda o protocolo em silêncio — reduza `n` explicitamente.")
    if n <= 0:
        raise ValueError(f"n={n}: nada a medir")
    rng = np.random.default_rng((semente, 0x5E9))
    return np.sort(rng.choice(n_disponiveis, size=n, replace=False)).astype(np.int64)


def equacao_da_prova(ide: np.ndarray, disp: np.ndarray, ids: np.ndarray,
                     ids_especiais, semente: int, indice: int,
                     taxa: float = 0.30) -> np.ndarray:
    """As posições de UMA equação em display inteira, escolhida como o treino escolhe.

    É a checagem de manipulação 2 da regra: a prova na política de treino do braço
    tratado. Usa `_escolher_equacao` do próprio mascaramento — mesmo piso de
    `MIN_TOKENS_TRATAMENTO`, mesmo orçamento, mesma recusa de equação cortada pela
    janela —, e não uma reimplementação dele.

    Vazio quando a sequência não tem equação elegível. Determinístico em
    `(semente, indice)`, igual nos dois braços.
    """
    from phifm.training.pretrain.mascaramento import _escolher_equacao

    mascaravel = ~np.isin(ids, np.asarray(sorted(ids_especiais)))
    n_alvo = int(round(taxa * int(mascaravel.sum())))
    if n_alvo == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng((semente, indice, 3))
    return np.sort(_escolher_equacao(ide, disp, mascaravel, n_alvo, rng, None))


def _somas_bootstrap(tabela: np.ndarray, semente: int, n_boot: int,
                     lote: int = 500) -> np.ndarray:
    """Somas das colunas de `tabela` sob `n_boot` reamostras de LINHAS (sequências).

    As MESMAS reamostras para quem chamar com a mesma `semente` e o mesmo número de
    linhas: é o que torna o bootstrap PAREADO entre os braços.
    """
    n = tabela.shape[0]
    rng = np.random.default_rng((semente, 0xB007))
    saidas = []
    for inicio in range(0, n_boot, lote):
        k = min(lote, n_boot - inicio)
        pesos = rng.multinomial(n, np.full(n, 1.0 / n), size=k)
        saidas.append(pesos @ tabela)
    return np.vstack(saidas)


def _tabela(por_sequencia: list, colunas: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(por_sequencia, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != colunas + 1:
        raise ValueError(
            f"esperava linhas [indice, {colunas} contagens], recebi forma {arr.shape}")
    return arr[:, 0].astype(np.int64), arr[:, 1:]


def _mesmas_sequencias(ic: np.ndarray, it: np.ndarray) -> None:
    if ic.shape != it.shape or not np.array_equal(ic, it):
        raise ValueError(
            "controle e tratado não foram medidos nas MESMAS sequências, na mesma "
            "ordem. O pareamento compararia textos diferentes, e o número sairia "
            "com a cara de uma comparação.")


def diferenca_das_diferencas(controle: list, tratado: list, *, semente: int = 17,
                             n_boot: int = 10_000) -> dict:
    """A primária da regra do §2.3, com IC de 95% por bootstrap pareado por sequência.

    Cada entrada é uma lista de `[indice, acertos_eq, total_eq, acertos_prosa,
    total_prosa]` por sequência. A quantidade é

        (acc_eq − acc_prosa) do TRATADO  −  (acc_eq − acc_prosa) do CONTROLE

    com as acurácias AGREGADAS sobre os tokens (a mesma conta de
    `Contagem.como_dict`), e não a média das acurácias por sequência — uma sequência
    com 3 tokens de equação pesaria o mesmo que uma com 300.

    ⚠️ Por que diferença de diferenças: token de equação já é +0,1286 mais fácil que
    prosa SEM tratamento nenhum. Um tratado que melhora tudo por igual tem diferença
    das diferenças ZERO, e é exatamente isso que ele deve ter — melhorar tudo não é
    a hipótese do §2.3.
    """
    ic, cc = _tabela(controle, 4)
    it, ct = _tabela(tratado, 4)
    _mesmas_sequencias(ic, it)
    if cc[:, 1].sum() == 0 or cc[:, 3].sum() == 0:
        raise ValueError("sem token de equação ou sem token de prosa mascarado: a "
                         "diferença entre regiões não existe")

    def did(sc: np.ndarray, st: np.ndarray) -> np.ndarray:
        van_c = sc[..., 0] / sc[..., 1] - sc[..., 2] / sc[..., 3]
        van_t = st[..., 0] / st[..., 1] - st[..., 2] / st[..., 3]
        return van_t - van_c

    ponto = float(did(cc.sum(0), ct.sum(0)))
    # O mesmo `semente` e o mesmo número de linhas: as MESMAS reamostras nos dois.
    bc = _somas_bootstrap(cc, semente, n_boot)
    bt = _somas_bootstrap(ct, semente, n_boot)
    dist = did(bc, bt)
    lo, hi = (float(x) for x in np.percentile(dist, [2.5, 97.5]))
    sc, st = cc.sum(0), ct.sum(0)
    return {
        "quantidade": "(acc_eq − acc_prosa) do tratado − (acc_eq − acc_prosa) do controle",
        "diferenca_das_diferencas": round(ponto, 5),
        "ic95": [round(lo, 5), round(hi, 5)],
        "cruza_zero": bool(lo <= 0.0 <= hi),
        "controle": {"acuracia_equacao": round(sc[0] / sc[1], 5),
                     "acuracia_prosa": round(sc[2] / sc[3], 5),
                     "vantagem_em_equacao": round(sc[0] / sc[1] - sc[2] / sc[3], 5)},
        "tratado": {"acuracia_equacao": round(st[0] / st[1], 5),
                    "acuracia_prosa": round(st[2] / st[3], 5),
                    "vantagem_em_equacao": round(st[0] / st[1] - st[2] / st[3], 5)},
        "sequencias": int(ic.size),
        "tokens_equacao": int(sc[1]),
        "tokens_prosa": int(sc[3]),
        "reamostras": n_boot,
        "teste": "bootstrap pareado por SEQUÊNCIA, as mesmas reamostras nos dois braços",
    }


def diferenca_de_acuracia(controle: list, tratado: list, *, semente: int = 17,
                          n_boot: int = 10_000) -> dict:
    """`acc_tratado − acc_controle`, agregada, com IC por bootstrap pareado por sequência.

    Entradas: `[indice, acertos, total]` por sequência. É a conta da checagem de
    manipulação 2 — a prova com a equação inteira escondida.
    """
    ic, cc = _tabela(controle, 2)
    it, ct = _tabela(tratado, 2)
    _mesmas_sequencias(ic, it)
    if cc[:, 1].sum() == 0:
        raise ValueError("nenhum token avaliado")
    ponto = float(ct[:, 0].sum() / ct[:, 1].sum() - cc[:, 0].sum() / cc[:, 1].sum())
    bc = _somas_bootstrap(cc, semente, n_boot)
    bt = _somas_bootstrap(ct, semente, n_boot)
    dist = bt[:, 0] / bt[:, 1] - bc[:, 0] / bc[:, 1]
    lo, hi = (float(x) for x in np.percentile(dist, [2.5, 97.5]))
    return {
        "diferenca": round(ponto, 5),
        "ic95": [round(lo, 5), round(hi, 5)],
        "cruza_zero": bool(lo <= 0.0 <= hi),
        "acuracia_controle": round(float(cc[:, 0].sum() / cc[:, 1].sum()), 5),
        "acuracia_tratado": round(float(ct[:, 0].sum() / ct[:, 1].sum()), 5),
        "sequencias": int(ic.size),
        "tokens": int(cc[:, 1].sum()),
        "reamostras": n_boot,
    }


def ler_pela_regra(primaria: dict, checagem_2: dict, fracao_tratada: float,
                   escala: str = "0,6 B") -> dict:
    """A leitura da regra de `kaggle/t2eq_tratado.py`, e nada além dela.

    As checagens de manipulação vêm ANTES: reprovada qualquer uma, a primária não
    se lê. "O tratado vence" na checagem 2 é o IC da diferença inteiro acima de
    zero — fixado aqui, antes de medir.

    ⚠️ `escala` entra no TEXTO do desfecho porque a leitura de um "não decidido"
    depende dela: "a 0,4 B não dá para ver" é uma afirmação sobre o orçamento, e
    deixá-la fixa faria o artefato do caminho B (0,4 B) declarar a escala do §2.3 a
    48 M (0,6 B). Quem chama passa a do run que mediu.
    """
    checagens = {
        "1_fracao_tratada": {"valor": fracao_tratada, "minimo": 0.50,
                             "aprovada": bool(fracao_tratada >= 0.50)},
        "2_prova_com_equacao_inteira": {
            "diferenca": checagem_2["diferenca"], "ic95": checagem_2["ic95"],
            "aprovada": bool(checagem_2["ic95"][0] > 0.0)},
    }
    if not all(c["aprovada"] for c in checagens.values()):
        return {"checagens": checagens, "desfecho": "RUN INVÁLIDO",
                "leitura": ("uma checagem de manipulação reprovou: o run não testa o "
                            "DOC-07 §2.3, e a primária NÃO se lê.")}
    lo, hi = primaria["ic95"]
    if lo > 0:
        desfecho, leitura = "TRATADO À FRENTE", (
            "o tratado ganha em equação ALÉM do que ganha em prosa, na prova que "
            "favorece o controle. A hipótese do §2.3 fica sustentada a "
            f"{escala}.")
    elif hi < 0:
        desfecho, leitura = "CONTROLE À FRENTE", (
            "o tratamento piora equação relativamente à prosa. É o negativo que o "
            "DOC-07 manda publicar, com a escala de "
            f"{escala} declarada ao lado.")
    else:
        desfecho, leitura = "NÃO DECIDIDO", (
            "o IC cruza zero. NÃO é o negativo do DOC-07: é "
            f"'a {escala} não dá para ver', registrado como não decidido.")
    return {"checagens": checagens, "desfecho": desfecho, "leitura": leitura}
