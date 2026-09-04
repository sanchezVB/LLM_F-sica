"""Detecção de *loss spike* com resposta automática. DOC-08 §6.1.

> Um *loss spike* não detectado na hora 200 de uma execução de 300 horas queima
> US$ 2.200. A automação disso vale mais que qualquer escolha de hardware.

E aqui vale mais ainda que no documento, por um motivo que o §6.1 não previa: o
DOC-08 §4 pede **bf16** e nenhuma das duas GPUs disponíveis tem bf16 (a T4 é Turing,
o DirectML não expõe). Em **fp16** a faixa dinâmica é menor e é exatamente o regime
em que os spikes aparecem — a detecção deixa de ser seguro e passa a ser requisito.

Puro, sem torch: o laço passa `(passo, perda, norma_do_gradiente)` e recebe um
veredito. Isso deixa os testes na suíte rápida e permite reproduzir uma sequência de
perdas de uma execução real sem GPU.

## O critério, do §6.1

    perda > μ + 4σ da janela móvel de 100 passos
      OU
    norma do gradiente > 10× a mediana móvel

## ⚠️ Duas decisões que o documento não especifica e que mudam o comportamento

**1. As estatísticas são da janela ANTES do passo julgado.** Incluir o próprio valor
inflaria μ e σ e um spike grande esconderia a si mesmo: com janela 100, um valor 50×
maior que os outros levanta σ o suficiente para ficar dentro de 4σ dele próprio.

**2. O valor do spike NÃO entra na janela.** Se entrasse, poluiria a estatística
pelos 100 passos seguintes — o limiar subiria e uma sequência de spikes passaria
inteira. A janela é a memória do que é *normal*, e um spike não é.

## A resposta, e o que este módulo faz e não faz

O §6.1 lista cinco passos. Este módulo decide **quando** e **quanto** (o veredito e
o fator de LR); quem executa o rollback e o pulo de batches é o laço, porque essas
duas coisas mexem em pesos e em disco.

O quinto passo — "3 spikes em 5.000 passos, parar e exigir intervenção humana" — é
`Veredito.exigir_humano`, e o laço tem de levantar nele. Um projeto de uma pessoa que
treina durante o sono precisa que essa parada seja automática.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConfigSpike:
    janela: int = 100
    sigmas: float = 4.0
    fator_grad: float = 10.0
    passos_lr_reduzida: int = 500
    fator_lr: float = 0.5
    max_spikes: int = 3
    janela_max_spikes: int = 5_000
    # Abaixo disto a janela não tem estatística: 4σ sobre 3 pontos não significa
    # nada, e os primeiros passos de um treino têm perda descendo rápido de
    # propósito — julgá-los como spike pararia todo treino no começo.
    minimo_para_julgar: int = 30

    def __post_init__(self) -> None:
        if self.janela < 2:
            raise ValueError(f"janela={self.janela} não dá estatística")
        if not 0.0 < self.fator_lr < 1.0:
            raise ValueError(f"fator_lr={self.fator_lr} tem de estar em (0, 1)")
        if self.minimo_para_julgar < 2:
            raise ValueError("minimo_para_julgar < 2 não dá desvio padrão")


@dataclass(frozen=True)
class Veredito:
    e_spike: bool
    motivo: str = ""
    exigir_humano: bool = False
    # Estatísticas do momento, para o log dizer POR QUE parou.
    media: float = 0.0
    desvio: float = 0.0
    limiar: float = 0.0


@dataclass
class Detector:
    cfg: ConfigSpike = field(default_factory=ConfigSpike)
    _perdas: deque[float] = field(default_factory=deque, repr=False)
    _normas: deque[float] = field(default_factory=deque, repr=False)
    _spikes: list[int] = field(default_factory=list)
    _reduzir_ate: int = -1

    def __post_init__(self) -> None:
        self._perdas = deque(maxlen=self.cfg.janela)
        self._normas = deque(maxlen=self.cfg.janela)

    # ── julgar ──────────────────────────────────────────────────────────────

    def observar(self, passo: int, perda: float, norma_grad: float) -> Veredito:
        """Julga o passo e, se for normal, o incorpora à janela.

        ⚠️ A ordem importa: julga contra a janela ANTERIOR e só então acrescenta.
        Ver a decisão 1 na docstring do módulo.
        """
        if not math.isfinite(perda) or not math.isfinite(norma_grad):
            # NaN/inf não é spike estatístico, é aritmética morta. Não há janela que
            # justifique continuar, e em fp16 isso é o modo de falha mais comum.
            return self._marcar(
                passo, f"perda={perda} norma={norma_grad} não é finito — o treino "
                       "morreu, não desviou")

        v = self._julgar(perda, norma_grad)
        if v.e_spike:
            return self._marcar(passo, v.motivo, v.media, v.desvio, v.limiar)
        self._perdas.append(perda)
        self._normas.append(norma_grad)
        return v

    def _julgar(self, perda: float, norma: float) -> Veredito:
        n = len(self._perdas)
        if n < self.cfg.minimo_para_julgar:
            return Veredito(e_spike=False)

        media = sum(self._perdas) / n
        var = sum((x - media) ** 2 for x in self._perdas) / max(n - 1, 1)
        desvio = math.sqrt(var)
        limiar = media + self.cfg.sigmas * desvio
        if perda > limiar:
            return Veredito(
                e_spike=True, media=media, desvio=desvio, limiar=limiar,
                motivo=(f"perda {perda:.4f} > μ+{self.cfg.sigmas:g}σ = {limiar:.4f} "
                        f"(μ={media:.4f}, σ={desvio:.4f}, janela={n})"))

        ordenadas = sorted(self._normas)
        mediana = ordenadas[n // 2] if n % 2 else (
            (ordenadas[n // 2 - 1] + ordenadas[n // 2]) / 2)
        teto = self.cfg.fator_grad * mediana
        if mediana > 0 and norma > teto:
            return Veredito(
                e_spike=True, media=mediana, limiar=teto,
                motivo=(f"norma do gradiente {norma:.3f} > "
                        f"{self.cfg.fator_grad:g}× a mediana móvel {mediana:.3f}"))
        return Veredito(e_spike=False)

    def _marcar(self, passo: int, motivo: str, media: float = 0.0,
                desvio: float = 0.0, limiar: float = 0.0) -> Veredito:
        # ⚠️ O valor do spike NÃO entra na janela. Ver a decisão 2 da docstring.
        self._spikes.append(passo)
        self._reduzir_ate = passo + self.cfg.passos_lr_reduzida
        recentes = [p for p in self._spikes
                    if passo - p <= self.cfg.janela_max_spikes]
        exigir = len(recentes) >= self.cfg.max_spikes
        if exigir:
            motivo += (f". {len(recentes)} spikes em {self.cfg.janela_max_spikes} "
                       "passos — é sintoma de problema sistêmico, não de um batch "
                       "ruim. O DOC-08 §6.1 manda parar e exigir intervenção humana.")
        return Veredito(e_spike=True, motivo=motivo, exigir_humano=exigir,
                        media=media, desvio=desvio, limiar=limiar)

    # ── a resposta ──────────────────────────────────────────────────────────

    def fator_lr(self, passo: int) -> float:
        """`fator_lr` da config durante `passos_lr_reduzida` após um spike; 1,0 fora.

        É o passo 4 do §6.1: "reduzir a LR em 50% por 500 passos, depois restaurar".
        """
        return self.cfg.fator_lr if passo <= self._reduzir_ate else 1.0

    def como_dict(self) -> dict:
        return {"spikes": list(self._spikes), "n_spikes": len(self._spikes),
                "janela_cheia": len(self._perdas),
                "reduzir_lr_ate": self._reduzir_ate,
                "criterio": (f"perda > μ+{self.cfg.sigmas:g}σ da janela de "
                             f"{self.cfg.janela}, ou norma > "
                             f"{self.cfg.fator_grad:g}× a mediana móvel"),
                "nota": ("estatísticas calculadas na janela ANTES do passo julgado, "
                         "e o valor do spike não entra na janela — senão um spike "
                         "grande esconde a si mesmo e polui os 100 passos seguintes")}


# ── o agendamento, que mora aqui porque a resposta a spike o modula ─────────


def plano_wsd(total: int, frac_warmup: float = 0.03,
              frac_decay: float = 0.10) -> tuple[int, int]:
    """`(passos_warmup, passos_decay)` a partir das frações do DOC-08 §4.

    ⚠️ Chame isto **uma vez**, no início do treino, e guarde os dois inteiros no
    checkpoint. As frações do documento são para *dimensionar* o plano, não para
    serem reavaliadas a cada passo — ver a docstring de `lr_wsd`.
    """
    if total <= 0:
        raise ValueError(f"total={total} tem de ser positivo")
    if not 0.0 <= frac_warmup < 1.0 or not 0.0 <= frac_decay < 1.0:
        raise ValueError("frac_warmup e frac_decay têm de estar em [0, 1)")
    if frac_warmup + frac_decay >= 1.0:
        raise ValueError(
            f"warmup {frac_warmup} + decay {frac_decay} não deixa fase estável; o "
            "WSD sem platô é só um cosseno mal feito")
    return max(int(total * frac_warmup), 1), max(int(total * frac_decay), 1)


def lr_wsd(passo: int, total: int, *, pico: float, passos_warmup: int,
           passos_decay: int, piso: float = 0.0) -> float:
    """Warmup-Stable-Decay do DOC-08 §4, em vez de cosseno.

    > O cosseno exige fixar o número total de passos **antes** de começar. O WSD
    > mantém LR constante e decai só no fim — o que permite parar em qualquer ponto
    > com um decay curto e obter um modelo utilizável, **retomar e estender o treino
    > sem descartar o agendamento**, e fazer a fase de annealing coincidir com o
    > decay.

    ⚠️ **`passos_warmup` é ABSOLUTO, não uma fração do total, e isso não é detalhe.**

    A primeira versão desta função recebia `frac_warmup=0.03` e calculava o warmup
    como `total × 0,03`. Com isso, estender o treino de 10 mil para 200 mil passos
    alongava o warmup de 300 para 6.000 — e a LR do passo 5.000, **que já havia sido
    dado com o pico**, passaria a valer 8,3e-4. O agendamento deixava de ser
    extensível, que é o argumento inteiro do WSD sobre o cosseno.

    Foi um teste que pegou isto (`test_o_plato_e_a_razao_de_ser_do_wsd`), e a
    tentação era ajustar o número esperado no teste. Use `plano_wsd()` uma vez para
    derivar os dois inteiros das frações do documento, guarde-os no checkpoint, e
    estender o treino passa a mover **só** o começo do decay.

    >>> w, d = plano_wsd(1000)
    >>> lr_wsd(0, 1000, pico=1e-3, passos_warmup=w, passos_decay=d)
    0.0
    >>> round(lr_wsd(30, 1000, pico=1e-3, passos_warmup=w, passos_decay=d), 8)
    0.001
    >>> round(lr_wsd(500, 1000, pico=1e-3, passos_warmup=w, passos_decay=d), 8)
    0.001
    >>> round(lr_wsd(950, 1000, pico=1e-3, passos_warmup=w, passos_decay=d), 8)
    0.00049495
    >>> lr_wsd(999, 1000, pico=1e-3, passos_warmup=w, passos_decay=d)
    0.0

    E o platô é o mesmo com qualquer total, que é a propriedade que importa:

    >>> [lr_wsd(5000, t, pico=1e-3, passos_warmup=300, passos_decay=1000)
    ...  for t in (10_000, 50_000, 200_000)]
    [0.001, 0.001, 0.001]
    """
    if total <= 0:
        raise ValueError(f"total={total} tem de ser positivo")
    if passos_warmup < 1 or passos_decay < 1:
        raise ValueError(
            f"passos_warmup={passos_warmup} e passos_decay={passos_decay} têm de "
            "ser >= 1; use `plano_wsd()` para derivá-los das frações do §4")
    if passos_warmup + passos_decay >= total:
        raise ValueError(
            f"warmup {passos_warmup} + decay {passos_decay} não deixa fase estável "
            f"em {total} passos; o WSD sem platô é só um cosseno mal feito")
    if passo < passos_warmup:
        return pico * passo / passos_warmup
    inicio_decay = total - passos_decay
    if passo < inicio_decay:
        return pico
    # Decay linear até o piso. Linear e não cosseno: o ponto do WSD é que o fim é
    # curto e previsível, e um cosseno no trecho final reintroduziria a dependência
    # do total que ele existe para evitar.
    restante = (total - 1 - passo) / max(total - 1 - inicio_decay, 1)
    return piso + (pico - piso) * max(restante, 0.0)
