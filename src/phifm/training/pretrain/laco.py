"""O laço de pré-treino do ΦEnc. DOC-08 §4, §6.1, §7.

Costura as quatro peças que moram separadas de propósito:

| peça | módulo | por que separada |
|---|---|---|
| a hipótese | `mascaramento.py` | numpy puro, testável na suíte rápida |
| o fluxo | `dados.py` | sem estado, calculado de `(semente, passo)` |
| o detector | `spike.py` | puro, reproduz uma sequência de perdas sem GPU |
| o modelo | `models/encoder/` | configuração conferida contra o `transformers` |

Este arquivo é o único que importa torch e o único que não tem teste unitário — o
que ele tem é um teste de fumaça que roda 3 passos num modelo minúsculo e confere
que a perda desce, que o checkpoint retoma no mesmo lugar, e que um spike injetado
dispara o rollback.

## Os hiperparâmetros, do DOC-08 §4

| | valor | nota |
|---|---|---|
| otimizador | AdamW `β=(0,9, 0,98)`, `ε=1e-6` | `β₂=0,98` é padrão em encoders |
| LR de pico | 1e-3 | modelo pequeno tolera LR alta |
| agendamento | WSD, warmup 3%, decay 10% | ver `spike.lr_wsd` |
| weight decay | 0,01, **sem** em norms e bias | |
| clipping | 1,0 | |
| máscara | 30% | ModernBERT: os 15% do BERT são subótimos |

## ⚠️ bf16 não existe aqui, e isso muda o que o laço precisa fazer

O §4 pede "bf16, mestre fp32". A T4 é Turing e o DirectML não expõe bf16 — então é
**fp16 com `GradScaler`**, cuja faixa dinâmica é menor. Duas consequências:

1. a detecção de spike do §6.1 deixa de ser seguro e passa a ser **requisito**;
2. `GradScaler` pode pular passos (quando acha inf/nan nos gradientes), e um passo
   pulado **não é um spike** — confundir os dois faria o detector disparar rollback
   por comportamento normal do scaler. O laço só chama o detector quando o passo de
   fato aconteceu.

## O orçamento de tokens por passo

O §4 pede **~2 M tokens por passo**. Com contexto 8.192 isso é 244 sequências, que
não caberiam em 16 GB — vêm de acumulação de gradiente:
`sequencias_por_micro_passo × acumulacao`. O laço registra os dois e o produto, para
que uma comparação entre execuções não confunda "lote maior" com "acumulou mais".

## Duas GPUs, e por que o resultado tem de ser o MESMO de uma

A cota do Kaggle conta horas de sessão, e a sessão "T4 x2" tem duas placas. Rodar em
`torchrun --nproc_per_node 2` quase dobra os tokens por hora de cota. O que não pode
mudar é o experimento: um braço treinado em duas GPUs tem de ver os mesmos dados, as
mesmas máscaras e o mesmo gradiente que em uma — senão uma comparação entre braços
rodados de jeitos diferentes mede o jeito de rodar.

- `acumulacao` continua sendo a do passo INTEIRO. Cada processo faz
  `acumulacao / processos` micro-passos, e o processo `r` pega os micro-passos
  `r·(acumulacao/processos) + j` do passo: os dois juntos cobrem exatamente os índices
  que um processo só cobriria.
- A máscara sai de `(semente, índice do micro-passo)`, e não de um gerador que avança
  a cada chamada. ⚠️ **Mudou em 2026-09-17**: até ali era um `default_rng(semente)`
  único, cujo estado dependia de quantas máscaras já tinham sido sorteadas. Com dois
  processos isso daria a MESMA sequência de sorteios nos dois, e uma retomada de
  checkpoint já sorteava máscaras diferentes das de uma execução contínua. A
  distribuição das máscaras não muda; a realização muda, então os braços já treinados
  (T2a, §2.3) não se reproduzem sorteio a sorteio com o código novo.
- A perda e a norma do gradiente que o detector de spike vê são a MÉDIA entre os
  processos, e a vazão que decide abortar por tempo também: uma decisão que um
  processo toma e o outro não é um processo esperando para sempre.
- Só o processo 0 grava. Os contadores do mascaramento são SOMADOS entre os processos
  antes de ir ao JSON — senão `fracao_tratada` descreveria metade do treino.

`tests/regression/test_laco_ddp.py` prova por equivalência: dois processos em CPU
(`gloo`) terminam com os mesmos pesos que um.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch

from phifm.models.encoder.config import ConfigEnc, flops_de_treino
from phifm.models.encoder.modelo import construir
from phifm.training.pretrain.dados import Fluxo
from phifm.training.pretrain.mascaramento import (
    ConfigMascara,
    Contadores,
    mascarar,
)
from phifm.training.pretrain.spike import (
    ConfigSpike,
    Detector,
    lr_wsd,
    plano_wsd,
)

log = logging.getLogger(__name__)

NOME_ESTADO = "estado_pretreino.pt"
NOME_METRICAS = "phienc.json"


@dataclass(frozen=True)
class Distribuicao:
    """Em que processo este treinador roda, e de quantos. `(0, 1)` é um processo só."""

    rank: int = 0
    mundo: int = 1

    @property
    def principal(self) -> bool:
        return self.rank == 0


def iniciar_distribuicao() -> tuple[Distribuicao, torch.device]:
    """Lê o ambiente do `torchrun` e inicia o grupo de processos.

    Sem `WORLD_SIZE` > 1, devolve `(0, 1)` e o dispositivo de sempre, sem iniciar nada.
    """
    mundo = int(os.environ.get("WORLD_SIZE", "1"))
    if mundo <= 1:
        return Distribuicao(), torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank, local = int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        dev, backend = torch.device("cuda", local), "nccl"
    else:
        dev, backend = torch.device("cpu"), "gloo"
    torch.distributed.init_process_group(backend)
    return Distribuicao(rank, mundo), dev


@dataclass(frozen=True)
class ConfigTreino:
    total_passos: int
    acumulacao: int = 1
    lr_pico: float = 1e-3
    frac_warmup: float = 0.03
    frac_decay: float = 0.10
    wd: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-6
    clip: float = 1.0
    amp: bool = True
    passos_log: int = 50
    passos_estado: int = 500
    # Ids do tokenizer. Ver `models/encoder/config.ESPECIAIS`.
    id_mask: int = 4
    ids_especiais: tuple[int, ...] = (0, 1, 2, 3, 4)
    # ⚠️ Teto de horas da SESSÃO, e ele aborta em vez de avisar.
    #
    # `horas_estimadas` existia neste módulo e não era chamada em lugar nenhum: o
    # laço sabia projetar o custo e nunca fazia nada com a projeção. Num ambiente
    # onde a sessão morre no relógio — o Kaggle desliga às 9 h e o checkpoint não
    # retoma entre sessões — uma projeção não usada é um run que descobre aos 85%
    # que não cabia.
    #
    # O dimensionamento de um run por FLOPs é uma ESTIMATIVA de MFU: a 48 M de
    # parâmetros a banda de memória manda mais que o tensor core, e 15% contra 25%
    # de MFU é a diferença entre 5,9 h e 8,9 h. A medida real chega na primeira
    # janela de log, minutos depois de começar; desligar ali custa esses minutos.
    #
    # `None` desliga a guarda, que é o certo para quem roda em máquina própria.
    limite_horas: float | None = None

    def __post_init__(self) -> None:
        if self.total_passos <= 0 or self.acumulacao <= 0:
            raise ValueError("total_passos e acumulacao têm de ser positivos")


@dataclass
class Metricas:
    passo: int = 0
    perda: float = 0.0
    lr: float = 0.0
    norma_grad: float = 0.0
    tokens: int = 0
    tokens_por_s: float = 0.0
    mfu: float = 0.0
    epoca: float = 0.0
    passos_pulados_pelo_scaler: int = 0
    rollbacks: int = 0
    historico: list[dict] = field(default_factory=list)


class Treinador:
    def __init__(self, cfg_enc: ConfigEnc, cfg: ConfigTreino,
                 cfg_mascara: ConfigMascara, fluxo: Fluxo,
                 cfg_spike: ConfigSpike | None = None,
                 dev: torch.device | None = None,
                 pico_flops: float = 65e12,
                 distribuicao: Distribuicao | None = None,
                 modelo: torch.nn.Module | None = None) -> None:
        self.cfg_enc, self.cfg, self.cfg_mascara, self.fluxo = (
            cfg_enc, cfg, cfg_mascara, fluxo)
        self.dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dist = distribuicao or Distribuicao()
        if cfg.acumulacao % self.dist.mundo:
            raise ValueError(
                f"acumulação {cfg.acumulacao} não se divide por {self.dist.mundo} "
                "processos. Ela é a do passo inteiro; cada processo faz uma fração "
                "dela, e uma fração que não é inteira mudaria o lote do experimento.")
        self.acum_local = cfg.acumulacao // self.dist.mundo
        # `pico_flops`: 65 TFLOPS é o fp16 nominal da T4. Serve só para a MFU do
        # §9.3 — nominal, não medido, e a MFU relativa a um número nominal é um
        # limite superior otimista. Está aqui para o log dizer de onde saiu.
        self.pico_flops = pico_flops

        torch.manual_seed(cfg_mascara.semente)
        # `modelo` vindo de fora é o pré-treino CONTINUADO (ADR-0003, caminho B): os
        # pesos são de uma base pronta e a arquitetura é a dela. Do zero, `construir`.
        self.modelo = modelo if modelo is not None else construir(cfg_enc, self.dev)
        if self.dist.mundo > 1:
            from torch.nn.parallel import DistributedDataParallel
            self.modelo = DistributedDataParallel(
                self.modelo,
                device_ids=[self.dev.index] if self.dev.type == "cuda" else None)
        self.opt = torch.optim.AdamW(
            self._grupos(), lr=cfg.lr_pico, betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps)
        self.amp = cfg.amp and self.dev.type == "cuda"
        self.escala = torch.amp.GradScaler("cuda") if self.amp else None
        self.detector = Detector(cfg_spike or ConfigSpike())
        # ⚠️ Derivado UMA vez e guardado no checkpoint. Recalcular as frações a cada
        # retomada mudaria a LR de passos já dados — ver `spike.lr_wsd`.
        self.passos_warmup, self.passos_decay = plano_wsd(
            cfg.total_passos, cfg.frac_warmup, cfg.frac_decay)
        self.contadores = Contadores()
        log.info("WSD: warmup %d · platô %d · decay %d · pico %.1e",
                 self.passos_warmup,
                 cfg.total_passos - self.passos_warmup - self.passos_decay,
                 self.passos_decay, cfg.lr_pico)

    # ── os grupos de parâmetros ─────────────────────────────────────────────

    def _grupos(self) -> list[dict]:
        """Weight decay em matrizes, nenhum em norms e bias. DOC-08 §4.

        ⚠️ Aplicar decay em `LayerNorm.weight` empurra o ganho para zero, o que
        equivale a apagar a normalização devagar. O treino não quebra: ele fica
        pior de um jeito que nenhuma métrica aponta.
        """
        com, sem = [], []
        for nome, p in self._nucleo().named_parameters():
            if not p.requires_grad:
                continue
            (sem if p.ndim <= 1 or nome.endswith(".bias") else com).append(p)
        log.info("weight decay em %d tensores, isento em %d", len(com), len(sem))
        return [{"params": com, "weight_decay": self.cfg.wd},
                {"params": sem, "weight_decay": 0.0}]

    # ── um micro-passo ──────────────────────────────────────────────────────

    def _mascarar_lote(self, indice: int) -> tuple[torch.Tensor, torch.Tensor]:
        """O micro-passo `indice`, mascarado. Função de `(semente, indice)` só.

        ⚠️ O gerador é criado AQUI, por micro-passo — ver "Duas GPUs" na docstring do
        módulo. Um gerador único do treinador fazia a máscara depender de quantas
        tinham sido sorteadas antes.
        """
        ids, ide, disp = self.fluxo.lote(indice)
        rng = np.random.default_rng((self.cfg_mascara.semente, indice))
        entradas, alvos = [], []
        for s in range(ids.shape[0]):
            e, a = mascarar(
                ids[s], ide[s], disp[s], cfg=self.cfg_mascara, rng=rng,
                id_mask=self.cfg.id_mask, n_vocab=self.cfg_enc.vocab,
                ids_especiais=frozenset(self.cfg.ids_especiais),
                contadores=self.contadores)
            entradas.append(e)
            alvos.append(a)
        return (torch.from_numpy(np.stack(entradas)).to(self.dev),
                torch.from_numpy(np.stack(alvos)).to(self.dev))

    def _passo(self, passo: int) -> tuple[float, float, bool]:
        """Um passo de otimizador, com `acumulacao` micro-passos.

        Devolve `(perda, norma_do_gradiente, o_passo_aconteceu)`. O terceiro é
        `False` quando o `GradScaler` pulou — e um passo pulado **não é um spike**.
        """
        self.opt.zero_grad(set_to_none=True)
        perda_total = 0.0
        for micro in range(self.acum_local):
            # O processo `r` cobre a sua fatia dos micro-passos do passo inteiro.
            entrada, alvos = self._mascarar_lote(
                passo * self.cfg.acumulacao + self.dist.rank * self.acum_local + micro)
            # Sincronizar os gradientes só no último micro-passo: nos outros o DDP
            # faria um all-reduce por micro-passo para jogar fora.
            ultimo = micro == self.acum_local - 1
            sem_sync = (self.modelo.no_sync() if self.dist.mundo > 1 and not ultimo
                        else contextlib.nullcontext())
            with sem_sync:
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                    saida = self.modelo(input_ids=entrada, labels=alvos)
                    perda = saida.loss / self.acum_local
                if self.escala is not None:
                    self.escala.scale(perda).backward()
                else:
                    perda.backward()
            perda_total += float(perda.detach()) * self.acum_local

        lr = lr_wsd(passo, self.cfg.total_passos, pico=self.cfg.lr_pico,
                    passos_warmup=self.passos_warmup,
                    passos_decay=self.passos_decay)
        lr *= self.detector.fator_lr(passo)
        for g in self.opt.param_groups:
            g["lr"] = lr

        if self.escala is not None:
            self.escala.unscale_(self.opt)
        norma = float(torch.nn.utils.clip_grad_norm_(
            self._nucleo().parameters(), self.cfg.clip))
        if self.escala is not None:
            antes = self.escala.get_scale()
            self.escala.step(self.opt)
            self.escala.update()
            # O scaler pula o passo quando acha inf/nan e então REDUZ a escala. É
            # assim que se descobre que ele pulou: não há bandeira pública.
            aconteceu = self.escala.get_scale() >= antes
        else:
            self.opt.step()
            aconteceu = True
        # A MÉDIA entre processos: é o que o detector de spike vê, e os processos têm
        # de tomar a mesma decisão de rollback.
        return (self._media_entre_processos(perda_total / self.acum_local),
                self._media_entre_processos(norma), aconteceu)

    # ── os coletivos, que num processo só não fazem nada ────────────────────

    def _nucleo(self) -> torch.nn.Module:
        """O modelo sem o invólucro do DDP: é ele que grava e retoma."""
        return self.modelo.module if self.dist.mundo > 1 else self.modelo

    def _media_entre_processos(self, valor: float) -> float:
        if self.dist.mundo == 1:
            return valor
        t = torch.tensor([valor], dtype=torch.float64, device=self.dev)
        torch.distributed.all_reduce(t)
        return float(t.item()) / self.dist.mundo

    def contadores_globais(self) -> Contadores:
        """Os contadores do mascaramento SOMADOS entre os processos."""
        if self.dist.mundo == 1:
            return self.contadores
        nomes = [f.name for f in fields(Contadores) if not f.name.startswith("_")]
        t = torch.tensor([getattr(self.contadores, n) for n in nomes],
                         dtype=torch.int64, device=self.dev)
        torch.distributed.all_reduce(t)
        return Contadores(**dict(zip(nomes, (int(x) for x in t.tolist()), strict=True)))

    def _barreira(self) -> None:
        if self.dist.mundo > 1:
            torch.distributed.barrier()

    # ── o laço ──────────────────────────────────────────────────────────────

    def treinar(self, saida: Path) -> Metricas:
        saida.mkdir(parents=True, exist_ok=True)
        m = Metricas()
        passo = self.retomar(saida)
        tokens_por_passo = self.fluxo.tokens_por_passo() * self.cfg.acumulacao
        t0 = time.perf_counter()
        tokens_desde_log = 0
        self.modelo.train()

        while passo < self.cfg.total_passos:
            perda, norma, aconteceu = self._passo(passo)
            tokens_desde_log += tokens_por_passo

            if aconteceu:
                v = self.detector.observar(passo, perda, norma)
                if v.e_spike:
                    log.error("SPIKE no passo %d: %s", passo, v.motivo)
                    if v.exigir_humano:
                        self._gravar(saida, passo, m, motivo="spike sistêmico")
                        raise SystemExit(
                            f"{v.motivo}\n\nO estado ficou em {saida}. O DOC-08 "
                            "§6.1 manda intervenção humana aqui: olhe os batches "
                            f"dos passos {self.detector.como_dict()['spikes']} — o "
                            "fluxo é determinístico em (semente, passo), então eles "
                            "se reproduzem sem reexecutar nada antes.")
                    passo = self._rollback(saida, passo, m)
                    continue
            else:
                m.passos_pulados_pelo_scaler += 1

            if passo % self.cfg.passos_log == 0:
                dt = time.perf_counter() - t0
                # Média entre processos: é esta vazão que decide abortar por tempo,
                # e a decisão tem de ser a mesma em todos.
                m.tokens_por_s = self._media_entre_processos(
                    tokens_desde_log / max(dt, 1e-9))
                # Os tokens são do passo inteiro, e o pico é de UMA placa.
                m.mfu = (flops_de_treino(self.cfg_enc, m.tokens_por_s)
                         / (self.pico_flops * self.dist.mundo))
                m.passo, m.perda, m.lr, m.norma_grad = passo, perda, self.opt.param_groups[0]["lr"], norma
                m.epoca = self.fluxo.epoca_do_passo(passo * self.cfg.acumulacao)
                tratada = self.contadores_globais().fracao_tratada()
                if self.dist.principal:
                    log.info("passo %d | perda %.4f | lr %.2e | |g| %.3f | %.0f tok/s "
                             "| MFU %.1f%% | época %.3f | tratada %.3f",
                             passo, perda, m.lr, norma, m.tokens_por_s, 100 * m.mfu,
                             m.epoca, tratada)
                m.historico.append({k: v for k, v in asdict(m).items()
                                    if k != "historico"})
                self._conferir_orcamento_de_sessao(m, saida, passo)
                t0, tokens_desde_log = time.perf_counter(), 0

            passo += 1
            m.tokens += tokens_por_passo
            if passo % self.cfg.passos_estado == 0:
                self._gravar(saida, passo, m)

        m.passo = passo
        self._gravar(saida, passo, m, concluido=True)
        return m

    # ── estado ──────────────────────────────────────────────────────────────

    def _gravar(self, saida: Path, passo: int, m: Metricas,
                concluido: bool = False, motivo: str = "") -> None:
        """Grava o estado. Chamado em TODOS os processos; só o 0 escreve.

        ⚠️ Os contadores globais são um coletivo, então todos têm de passar por aqui;
        e a barreira no fim impede um processo de retomar de um arquivo pela metade.
        """
        contadores = self.contadores_globais()
        if not self.dist.principal:
            self._barreira()
            return
        torch.save({
            "passo": passo,
            "modelo": self._nucleo().state_dict(),
            "opt": self.opt.state_dict(),
            "escala": self.escala.state_dict() if self.escala else None,
            # ⚠️ O plano WSD vai no checkpoint. Recalculá-lo das frações numa
            # retomada com outro `total_passos` mudaria a LR de passos já dados.
            "passos_warmup": self.passos_warmup,
            "passos_decay": self.passos_decay,
            "detector": self.detector.como_dict(),
        }, saida / NOME_ESTADO)
        (saida / NOME_METRICAS).write_text(json.dumps({
            "modelo": self.cfg_enc.como_dict(),
            "treino": asdict(self.cfg),
            "mascara": asdict(self.cfg_mascara),
            "dados": self.fluxo.como_dict(),
            "tokens_por_passo": self.fluxo.tokens_por_passo() * self.cfg.acumulacao,
            "spike": self.detector.como_dict(),
            # ⚠️ Os contadores do mascaramento vão no JSON do treino, e não só no
            # log: `fracao_tratada` baixa invalida a ablação do DOC-07 §2.3, e um
            # resultado nulo sem esse número é indistinguível de tratamento ausente.
            "mascaramento": contadores.como_dict(),
            "metricas": {k: v for k, v in asdict(m).items() if k != "historico"},
            "historico": m.historico,
            "concluido": concluido,
            "motivo_da_parada": motivo,
            "amp": self.amp,
            "dispositivo": str(self.dev),
            "distribuicao": {
                "processos": self.dist.mundo,
                "acumulacao_por_processo": self.acum_local,
                "mascara": "(semente, índice do micro-passo) — desde 2026-09-17",
            },
            "ressalva_mfu": (f"MFU contra {self.pico_flops:.1e} FLOPS NOMINAIS; sem "
                             "FA-2 (a T4 é SM 7.5) o teto prático é bem menor, "
                             "então esta MFU é um limite superior otimista"),
        }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        self._barreira()

    def retomar(self, saida: Path) -> int:
        p = saida / NOME_ESTADO
        if not p.exists():
            return 0
        est = torch.load(p, map_location=self.dev, weights_only=False)
        self._nucleo().load_state_dict(est["modelo"])
        self.opt.load_state_dict(est["opt"])
        if self.escala is not None and est.get("escala"):
            self.escala.load_state_dict(est["escala"])
        self.passos_warmup = est.get("passos_warmup", self.passos_warmup)
        self.passos_decay = est.get("passos_decay", self.passos_decay)
        passo = int(est["passo"])
        log.info("retomado do passo %d (warmup %d, decay %d do checkpoint)",
                 passo, self.passos_warmup, self.passos_decay)
        return passo

    def _rollback(self, saida: Path, passo: int, m: Metricas) -> int:
        """Passos 2 e 3 do DOC-08 §6.1: voltar ao checkpoint e PULAR a janela.

        ⚠️ Pular é o que impede o laço de bater no mesmo batch para sempre. Sem o
        pulo, retomar do checkpoint refaria exatamente os mesmos passos — o fluxo é
        determinístico — e o spike voltaria no mesmo lugar, indefinidamente.
        """
        m.rollbacks += 1
        alvo = self.retomar(saida)
        # A janela suspeita é do checkpoint até o spike. Retomar dali e pular até
        # depois do spike descarta os batches que a produziram.
        salto = passo + 1
        if alvo >= salto:
            salto = alvo + 1
        log.warning("rollback: checkpoint no passo %d, pulando para %d "
                    "(%d batches descartados)", alvo, salto, salto - alvo)
        if m.rollbacks > 1 and alvo == 0:
            raise SystemExit(
                "segundo rollback sem nenhum checkpoint para voltar (o spike "
                f"aconteceu antes do passo {self.cfg.passos_estado}). Reduza "
                "`--passos-estado`, ou a LR de pico: sem checkpoint o rollback é só "
                "recomeçar do zero.")
        return salto

    def _conferir_orcamento_de_sessao(self, m: Metricas, saida: Path,
                                      passo: int) -> None:
        """Aborta quando a vazão MEDIDA diz que o run não cabe na sessão.

        ⚠️ A partir da SEGUNDA janela de log, e não da primeira.

        A primeira janela carrega o autotune do cuDNN, a primeira alocação do
        cache e a compilação dos kernels — ela mede devagar por construção, e
        abortar nela reprovaria runs que cabem. A segunda já é regime.

        ⚠️ Aborta, não avisa. O Kaggle desliga a sessão no relógio e este laço não
        retoma entre sessões: um run que projeta 10 h numa sessão de 9 h não vai
        entregar 90% do treino, vai entregar nada. Seguir seria gastar a sessão
        inteira para chegar ao mesmo lugar, só que horas depois.
        """
        if self.cfg.limite_horas is None or len(m.historico) < 2:
            return
        horas = self.horas_estimadas(m.tokens_por_s)
        if horas <= self.cfg.limite_horas:
            return
        # O checkpoint vai ao disco mesmo assim: se alguém decidir que a sessão
        # seguinte pode continuar à mão, o que já foi treinado está lá.
        self._gravar(saida, passo, m, motivo="acima do limite de horas")
        cabem = int(self.cfg.total_passos * self.cfg.limite_horas / horas)
        raise SystemExit(
            f"a {m.tokens_por_s:,.0f} tok/s medidos, os {self.cfg.total_passos:,} "
            f"passos levam {horas:.1f} h e o limite da sessão é "
            f"{self.cfg.limite_horas:.1f} h.\n\n"
            f"Caberiam ~{cabem:,} passos. ⚠️ Mas se este run é um BRAÇO de um "
            "experimento pareado, reduzir só este quebraria o orçamento igual: o "
            "outro braço tem de ser reduzido junto, e o que já rodou com o número "
            "antigo não vale mais.\n\n"
            f"O estado ficou em {saida}, no passo {passo}.")

    def horas_estimadas(self, tokens_por_s: float) -> float:
        tokens = (self.cfg.total_passos * self.fluxo.tokens_por_passo()
                  * self.cfg.acumulacao)
        return tokens / max(tokens_por_s, 1e-9) / 3600.0

    def resumo(self) -> dict:
        tokens = (self.cfg.total_passos * self.fluxo.tokens_por_passo()
                  * self.cfg.acumulacao)
        return {"tokens_totais": tokens,
                "flops": flops_de_treino(self.cfg_enc, tokens),
                "tokens_por_passo": tokens // self.cfg.total_passos,
                "parametros": self.cfg_enc.parametros()["total"],
                "epocas": tokens / max(self.fluxo.tokens.size, 1)}
