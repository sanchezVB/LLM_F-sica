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
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
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
                 pico_flops: float = 65e12) -> None:
        self.cfg_enc, self.cfg, self.cfg_mascara, self.fluxo = (
            cfg_enc, cfg, cfg_mascara, fluxo)
        self.dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # `pico_flops`: 65 TFLOPS é o fp16 nominal da T4. Serve só para a MFU do
        # §9.3 — nominal, não medido, e a MFU relativa a um número nominal é um
        # limite superior otimista. Está aqui para o log dizer de onde saiu.
        self.pico_flops = pico_flops

        torch.manual_seed(cfg_mascara.semente)
        self.modelo = construir(cfg_enc, self.dev)
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
        self.rng = np.random.default_rng(cfg_mascara.semente)
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
        for nome, p in self.modelo.named_parameters():
            if not p.requires_grad:
                continue
            (sem if p.ndim <= 1 or nome.endswith(".bias") else com).append(p)
        log.info("weight decay em %d tensores, isento em %d", len(com), len(sem))
        return [{"params": com, "weight_decay": self.cfg.wd},
                {"params": sem, "weight_decay": 0.0}]

    # ── um micro-passo ──────────────────────────────────────────────────────

    def _mascarar_lote(self, passo: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids, ide, disp = self.fluxo.lote(passo)
        entradas, alvos = [], []
        for s in range(ids.shape[0]):
            e, a = mascarar(
                ids[s], ide[s], disp[s], cfg=self.cfg_mascara, rng=self.rng,
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
        for micro in range(self.cfg.acumulacao):
            entrada, alvos = self._mascarar_lote(
                passo * self.cfg.acumulacao + micro)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                saida = self.modelo(input_ids=entrada, labels=alvos)
                perda = saida.loss / self.cfg.acumulacao
            if self.escala is not None:
                self.escala.scale(perda).backward()
            else:
                perda.backward()
            perda_total += float(perda.detach()) * self.cfg.acumulacao

        lr = lr_wsd(passo, self.cfg.total_passos, pico=self.cfg.lr_pico,
                    passos_warmup=self.passos_warmup,
                    passos_decay=self.passos_decay)
        lr *= self.detector.fator_lr(passo)
        for g in self.opt.param_groups:
            g["lr"] = lr

        if self.escala is not None:
            self.escala.unscale_(self.opt)
        norma = float(torch.nn.utils.clip_grad_norm_(
            self.modelo.parameters(), self.cfg.clip))
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
        return perda_total / self.cfg.acumulacao, norma, aconteceu

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
                m.tokens_por_s = tokens_desde_log / max(dt, 1e-9)
                m.mfu = (flops_de_treino(self.cfg_enc, m.tokens_por_s)
                         / self.pico_flops)
                m.passo, m.perda, m.lr, m.norma_grad = passo, perda, self.opt.param_groups[0]["lr"], norma
                m.epoca = self.fluxo.epoca_do_passo(passo * self.cfg.acumulacao)
                log.info("passo %d | perda %.4f | lr %.2e | |g| %.3f | %.0f tok/s "
                         "| MFU %.1f%% | época %.3f | tratada %.3f",
                         passo, perda, m.lr, norma, m.tokens_por_s, 100 * m.mfu,
                         m.epoca, self.contadores.fracao_tratada())
                m.historico.append({k: v for k, v in asdict(m).items()
                                    if k != "historico"})
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
        torch.save({
            "passo": passo,
            "modelo": self.modelo.state_dict(),
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
            "mascaramento": self.contadores.como_dict(),
            "metricas": {k: v for k, v in asdict(m).items() if k != "historico"},
            "historico": m.historico,
            "concluido": concluido,
            "motivo_da_parada": motivo,
            "amp": self.amp,
            "dispositivo": str(self.dev),
            "ressalva_mfu": (f"MFU contra {self.pico_flops:.1e} FLOPS NOMINAIS; sem "
                             "FA-2 (a T4 é SM 7.5) o teto prático é bem menor, "
                             "então esta MFU é um limite superior otimista"),
        }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def retomar(self, saida: Path) -> int:
        p = saida / NOME_ESTADO
        if not p.exists():
            return 0
        est = torch.load(p, map_location=self.dev, weights_only=False)
        self.modelo.load_state_dict(est["modelo"])
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
