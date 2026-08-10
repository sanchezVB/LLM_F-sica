"""ΦEmb — fine-tune contrastivo de um encoder sobre pares de citação (DOC-07 §3).

O objetivo é o Portão **G1**: bater o PhysBERT e os embedders gerais em
recuperação de Física. Este módulo treina e **mede**, porque treinar sem medir
não distingue progresso de ruído.

## Por que sobre encoder pronto, e não sobre o ΦEnc

O ΦEnc do DOC-07 §2 exige 15–30 B tokens de texto completo, que dependem do
Sprint S3. Temos 0,33 B (títulos e resumos). Um ΦEmb sobre encoder público
testa a **hipótese central** — o par de citação é supervisão suficiente — sem
esperar o S3. Se falhar aqui, falharia sobre o ΦEnc também, e descobrir isso
custando zero é o ponto.

## Restrições da máquina, medidas em 2026-08-07

O ROCm não funciona nesta GPU (registro em `setup/rocm_wsl.md`: a pilha HSA
enumera zero dispositivos). O **DirectML** funciona, com três ressalvas
descobertas por medição:

| Achado | Consequência |
|---|---|
| Backward do ModernBERT falha no DML | Base é SciBERT, não ModernBERT — perde-se o contexto de 8192, irrelevante para resumos de ~300 tokens |
| Só `attn_implementation="eager"` treina | `sdpa` quebra no backward com erro interno ilegível |
| Um passo contrastivo mantém DOIS grafos vivos | **Lote 8**, com gradient checkpointing |

**Sobre o teto de lote, e a medição que eu errei primeiro.** Medi a vazão com um
forward só e conclui "teto de 16". Errado pela metade: contrastivo codifica
âncora e positivo antes do backward, então lote 8 são 16 textos em memória. O
treino morreu pedindo 36 MB de VRAM.

**O teto é limite de QUALIDADE, não só de velocidade.** Contrastivo aprende de
`lote−1` negativos por âncora: **7 aqui**, contra 64–256 do que a literatura
usa. Acumulação de gradiente **não corrige** — ela soma gradientes de lotes
pequenos, não faz um lote pequeno ver mais negativos. É limite de conteúdo.

Vazão medida com checkpointing: **4,1 pares/s** → ~112 h por época sobre 1,65 M
pares. Uma 4090 alugada faria a mesma época em ~3 h por ~US$ 1, com lote 128 e
negativos de verdade. A escolha é do dono do projeto; este código roda nas duas.

**O que o piloto mostrou sobre retorno marginal.** Com 6.400 pares (0,4% do
conjunto) o recall@1 foi de 0,266 para 0,422. Entre os passos 400 e 800 o ganho
já caiu para +0,043. Fine-tune contrastivo converge em dezenas de milhares de
pares, não em milhões — então a época completa compra pouco sobre uma fração
dela, e `--max-pares` não é atalho, é a escolha informada.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)

BASE_PADRAO = "allenai/scibert_scivocab_uncased"


@dataclass
class Config:
    base: str = BASE_PADRAO
    max_tokens: int = 192          # resumos cabem; 256+ estoura com lote 16
    lote: int = 16                 # teto medido nos 8 GB da RX 7600
    lr: float = 2e-5
    temperatura: float = 0.05      # τ do InfoNCE; 0,05 é o usual em recuperação
    max_pares: int | None = None
    # Recomputa ativações no backward em vez de guardá-las. Custa ~30% de tempo
    # e devolve a maior parte da memória — a troca certa quando o lote é o
    # limite de QUALIDADE, não só de velocidade (ver docstring do módulo).
    checkpointing: bool = True
    passos_log: int = 50
    passos_aval: int = 500
    n_candidatos: int = 256   # pool da avaliação; ver Metricas.n_candidatos
    semente: int = 17
    dispositivo: str = "auto"      # auto | dml | cpu


@dataclass
class Metricas:
    passo: int = 0
    perda: float = 0.0
    recall_1: float = 0.0
    recall_10: float = 0.0
    mrr: float = 0.0
    # Sem isto a métrica é ininterpretável: recall@1 entre 128 candidatos é
    # muito mais fácil que entre 256. Comparar duas execuções com pools
    # diferentes é comparar coisas diferentes.
    n_candidatos: int = 0
    pares_por_s: float = 0.0
    historico: list[dict] = field(default_factory=list)


def escolher_dispositivo(pedido: str = "auto") -> torch.device:
    """DirectML se houver, CPU se não. Nunca falha em silêncio."""
    if pedido == "cpu":
        return torch.device("cpu")
    try:
        import torch_directml as dml

        if dml.device_count() > 0:
            log.info("dispositivo: %s (DirectML)", dml.device_name(0))
            return dml.device(0)
    except ImportError:
        pass
    if pedido == "dml":
        raise RuntimeError("DirectML pedido explicitamente e indisponível")
    log.warning("DirectML indisponível — caindo para CPU, o que será MUITO mais lento")
    return torch.device("cpu")


class ParesDataset(Dataset):
    """Mantém as colunas em Arrow e materializa uma linha por vez.

    ⚠️ A versão anterior fazia `df["ancora"].to_list()` nas duas colunas. Com
    6,5 M de pares isso são **13 milhões de strings Python** de ~1,5 KB cada —
    cerca de 20 GB, contra 15,9 GB de RAM na máquina. O treino morria em
    silêncio antes do primeiro passo, sem traceback, logo depois de registrar o
    dispositivo. Funcionou no piloto só porque `--max-pares 6400` cortava antes.

    Em Arrow os mesmos dados ocupam o tamanho do parquet descomprimido e cada
    `__getitem__` cria duas strings, não treze milhões.
    """

    def __init__(self, df: pl.DataFrame):
        self.a = df["ancora"].to_arrow()
        self.p = df["positivo"].to_arrow()
        self.n = df.height

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[str, str]:
        return self.a[i].as_py(), self.p[i].as_py()


def media_mascarada(saida, mascara) -> torch.Tensor:
    """Média sobre tokens reais, ignorando preenchimento.

    Usar o token `[CLS]` sem treino específico é pior: em encoder de MLM ele não
    foi otimizado para representar a sentença inteira. A média mascarada é a
    linha de base forte, e é o que o Sentence-BERT estabeleceu.
    """
    m = mascara.unsqueeze(-1).to(saida.dtype)
    return (saida * m).sum(1) / m.sum(1).clamp(min=1e-9)


class TreinadorEmb:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        torch.manual_seed(cfg.semente)
        self.dev = escolher_dispositivo(cfg.dispositivo)
        self.tok = AutoTokenizer.from_pretrained(cfg.base)
        # `eager` é obrigatório: `sdpa` quebra no backward do DirectML com erro
        # interno que nem decodifica como texto.
        self.mod = AutoModel.from_pretrained(cfg.base, attn_implementation="eager").to(self.dev)
        if cfg.checkpointing:
            # ⚠️ Um passo contrastivo mantém DOIS grafos vivos ao mesmo tempo —
            # âncora e positivo — antes do backward. Medir a vazão com um
            # forward só, como eu fiz primeiro, subestima a memória pela metade
            # e o treino morre com "não há memória de vídeo" pedindo 36 MB.
            self.mod.gradient_checkpointing_enable()
            self.mod.config.use_cache = False
        self.opt = torch.optim.AdamW(self.mod.parameters(), lr=cfg.lr)

    def _codificar(self, textos: list[str]) -> torch.Tensor:
        lote = self.tok(list(textos), padding="max_length", truncation=True,
                        max_length=self.cfg.max_tokens, return_tensors="pt")
        lote = {k: v.to(self.dev) for k, v in lote.items()}
        saida = self.mod(**lote).last_hidden_state
        return F.normalize(media_mascarada(saida, lote["attention_mask"]), dim=-1)

    def _perda(self, va: torch.Tensor, vp: torch.Tensor) -> torch.Tensor:
        """InfoNCE com negativos do próprio lote.

        A diagonal de `va @ vp.T` são os pares verdadeiros; o resto do lote são
        os negativos. Simétrica nas duas direções porque "A busca B" e "B busca
        A" são as duas consultas que o modelo vai receber em uso.
        """
        sim = va @ vp.T / self.cfg.temperatura
        alvo = torch.arange(sim.size(0), device=sim.device)
        return 0.5 * (F.cross_entropy(sim, alvo) + F.cross_entropy(sim.T, alvo))

    @torch.no_grad()
    def avaliar(self, val: pl.DataFrame, n: int | None = None) -> tuple[float, float, float]:
        """Recuperação num conjunto fechado: dado o texto A, achar o citado B.

        `recall@1`, `recall@10` e MRR sobre `n` candidatos. É a métrica do G1 em
        miniatura — o benchmark completo do DOC-11 vem depois, mas medir aqui
        já distingue aprendizado de ruído.
        """
        n = n or self.cfg.n_candidatos
        self.mod.eval()
        amostra = val.head(n)
        va = torch.cat([self._codificar(amostra["ancora"].to_list()[i:i + self.cfg.lote])
                        for i in range(0, amostra.height, self.cfg.lote)]).cpu()
        vp = torch.cat([self._codificar(amostra["positivo"].to_list()[i:i + self.cfg.lote])
                        for i in range(0, amostra.height, self.cfg.lote)]).cpu()
        sim = va @ vp.T
        alvo = torch.arange(sim.size(0))
        ordem = sim.argsort(dim=1, descending=True)
        posicao = (ordem == alvo.unsqueeze(1)).float().argmax(dim=1) + 1
        self.mod.train()
        return ((posicao == 1).float().mean().item(),
                (posicao <= 10).float().mean().item(),
                (1.0 / posicao).mean().item())

    def treinar(self, treino: pl.DataFrame, val: pl.DataFrame, saida: Path) -> Metricas:
        if self.cfg.max_pares:
            treino = treino.head(self.cfg.max_pares)
        # Gerador com semente explícita: a ordem dos lotes tem de ser a MESMA
        # entre execuções, senão retomar do passo N não significa nada — pularia
        # lotes diferentes dos que já foram vistos.
        g = torch.Generator().manual_seed(self.cfg.semente)
        carregador = DataLoader(ParesDataset(treino), batch_size=self.cfg.lote,
                                shuffle=True, drop_last=True, generator=g)
        m = Metricas()
        inicio = self.retomar(saida)

        # Linha de base ANTES de qualquer passo. Sem ela não há como afirmar que
        # o treino ajudou — só que o número final é X. Numa retomada isto mede o
        # ponto de partida real, não o encoder virgem, e o rótulo diz qual.
        r1, r10, mrr = self.avaliar(val)
        log.info("base %s | entre %d candidatos: recall@1 %.3f · recall@10 %.3f · MRR %.3f",
                 self.cfg.base, self.cfg.n_candidatos, r1, r10, mrr)
        m.n_candidatos = self.cfg.n_candidatos
        m.historico.append({"passo": inicio, "recall_1": r1, "recall_10": r10,
                            "mrr": mrr, "n_candidatos": m.n_candidatos, "perda": None,
                            "nota": "antes do treino" if not inicio else f"retomado do passo {inicio}"})

        t0, vistos, soma, n = time.perf_counter(), 0, 0.0, 0
        for passo, (a, p) in enumerate(carregador, start=1):
            if passo <= inicio:
                continue        # lote já consumido: pular é barato, refazer não
            perda = self._perda(self._codificar(a), self._codificar(p))
            perda.backward()
            self.opt.step()
            self.opt.zero_grad()
            soma += perda.item()
            n += 1
            vistos += len(a)

            if passo % self.cfg.passos_log == 0:
                m.pares_por_s = vistos / (time.perf_counter() - t0)
                log.info("passo %d | perda %.4f | %.1f pares/s", passo, soma / n, m.pares_por_s)
                soma, n = 0.0, 0

            if passo % self.cfg.passos_aval == 0:
                r1, r10, mrr = self.avaliar(val)
                log.info("  aval: recall@1 %.3f · recall@10 %.3f · MRR %.3f", r1, r10, mrr)
                m.historico.append({"passo": passo, "recall_1": r1, "recall_10": r10,
                                    "mrr": mrr, "perda": perda.item()})
                self.salvar(saida)
                self.salvar_estado(saida, passo)   # queda não custa o treino todo

            m.passo = passo

        r1, r10, mrr = self.avaliar(val)
        m.recall_1, m.recall_10, m.mrr = r1, r10, mrr
        m.perda = perda.item()
        self.salvar_estado(saida, m.passo)
        m.historico.append({"passo": m.passo, "recall_1": r1, "recall_10": r10,
                            "mrr": mrr, "perda": m.perda, "nota": "final"})
        self.salvar(saida, m, concluido=True)
        return m

    # ── retomada ──────────────────────────────────────────────────────────
    #
    # Uma época sobre 1,65 M pares leva ~112 h aqui. Nesta máquina processos
    # morrem por causa não identificada — cinco vezes em 2026-08-07 — então um
    # run de dias SEM retomada não é arriscado, é garantia de desperdício.
    #
    # O estado durável é modelo + otimizador + passo. Sem o otimizador, retomar
    # zera os momentos do Adam e o treino dá um salto de perda a cada queda;
    # sem o passo, refaz os mesmos lotes e nunca termina.

    def _estado(self) -> Path:
        return Path("estado_treino.pt")

    @staticmethod
    def _para_cpu(obj):
        """Copia recursivamente tensores para a CPU.

        ⚠️ Necessário para o estado do otimizador, não só para o modelo. O Adam
        guarda dois tensores por parâmetro (`exp_avg`, `exp_avg_sq`) — ~880 MB
        para 110 M de parâmetros — e eles ficam no dispositivo DirectML.
        `torch.save` sobre eles morre com `MemoryError` durante o pickle, sem
        mencionar dispositivo nenhum na mensagem.
        """
        if torch.is_tensor(obj):
            return obj.detach().to("cpu")
        if isinstance(obj, dict):
            return {k: TreinadorEmb._para_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(TreinadorEmb._para_cpu(v) for v in obj)
        return obj

    def salvar_estado(self, saida: Path, passo: int) -> None:
        saida.mkdir(parents=True, exist_ok=True)
        tmp = saida / (self._estado().name + ".tmp")
        torch.save(
            {"passo": passo,
             "modelo": self._para_cpu(self.mod.state_dict()),
             "otimizador": self._para_cpu(self.opt.state_dict()),
             "cfg": asdict(self.cfg)},
            tmp,
        )
        tmp.replace(saida / self._estado().name)   # atômico: nunca meio-arquivo

    def retomar(self, saida: Path) -> int:
        """Devolve o passo de onde continuar, ou 0 se não houver estado."""
        caminho = saida / self._estado().name
        if not caminho.exists():
            return 0
        est = torch.load(caminho, map_location="cpu", weights_only=False)
        self.mod.load_state_dict(est["modelo"])
        self.mod.to(self.dev)
        self.opt.load_state_dict(est["otimizador"])
        log.info("retomando do passo %s", f"{est['passo']:,}")
        return int(est["passo"])

    def salvar(self, saida: Path, m: Metricas | None = None,
               concluido: bool = False) -> None:
        """Grava uma CÓPIA em CPU. Nunca mexe no modelo em uso.

        ⚠️ A versão anterior fazia `self.mod.to("cpu").save_pretrained(...)` e
        depois `.to(self.dev)`. Isso **congelava o treino**: a partir do primeiro
        checkpoint a avaliação devolvia o mesmo número até a terceira decimal
        enquanto a perda continuava oscilando — sintoma clássico de otimizador
        atualizando tensores que o forward não usa mais.

        Comprovado por diagnóstico: sem a chamada de `salvar`, os pesos mudam e
        o recall@1 sobe de 0,344 para 0,523 em 100 passos. Com ela, empaca.

        Copiar o `state_dict` para CPU resolve e ainda mantém o artefato
        portátil — pesos com dispositivo DML dentro não recarregam em máquina
        sem DirectML.
        """
        saida.mkdir(parents=True, exist_ok=True)
        pesos = {k: v.detach().to("cpu") for k, v in self.mod.state_dict().items()}
        self.mod.save_pretrained(saida, state_dict=pesos)
        self.tok.save_pretrained(saida)
        meta = {"config": asdict(self.cfg), "base": self.cfg.base}
        # `completed_at` é a MESMA chave que o supervisor já procura nos
        # manifestos de coleta. Reusar o nome evita um segundo mecanismo de
        # "acabou" — e um mecanismo a menos é um ramo a menos sem teste.
        if concluido:
            from datetime import datetime
            meta["completed_at"] = datetime.now(UTC).isoformat()
        if m:
            meta["metricas"] = dict(asdict(m).items())
        (saida / "phiemb.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
