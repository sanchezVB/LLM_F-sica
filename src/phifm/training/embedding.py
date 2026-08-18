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

| Achado | Consequência | Vale no CUDA? |
|---|---|---|
| Backward do ModernBERT falha no DML | Base é SciBERT, não ModernBERT — perde-se o contexto de 8192, irrelevante para resumos de ~300 tokens | não testado |
| Só `attn_implementation="eager"` treina | `sdpa` quebra no backward com erro interno ilegível | **não** — no CUDA usa `sdpa` |
| Um passo contrastivo mantém DOIS grafos vivos | **Lote 8**, com gradient checkpointing | sim, é do algoritmo |
| Sem AMP utilizável | fp32 em tudo | **não** — no CUDA usa fp16 |

⚠️ As restrições da tabela são do **DirectML**, não do modelo, e a coluna da
direita existe porque eu quase as tratei como propriedades do problema. Ver
`escolher_dispositivo`: no CUDA duas delas caem, e é isso que torna o Kaggle
(30 h/semana de T4) uma rota diferente e não apenas uma máquina mais rápida.

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
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)

BASE_PADRAO = "allenai/scibert_scivocab_uncased"


def _vram_mb() -> float | None:
    """MB alocados na GPU, ou `None` se não houver como perguntar.

    ⚠️ Existe por causa de uma morte por VRAM no passo 800 do treino de 1,5 M
    pares — 226.492.416 bytes, que são exatamente 128 x 12 cabeças x 192 x 192 x
    4: os escores de atenção de UM lote. O mesmo lote 128 havia rodado 3.125
    passos num treino anterior sem estourar, então o teto não é o lote.

    O que se observou junto: a vazão caiu de forma monótona, 27,4 -> 24,9 pares/s
    ao longo de 800 passos, e `torch_directml` NÃO devolve memória num `del` —
    medido: 1 GB alocado continua marcado depois de liberar a referência. Ou seja,
    fragmentação crescente é o comportamento esperado do alocador, não anomalia.

    Sem este número, a próxima morte é outra vez inferência sobre um traceback.
    Com ele, é uma curva. Nunca levanta: um treino não pode morrer por causa do
    seu próprio medidor.
    """
    if torch.cuda.is_available():
        # No CUDA o número é confiável e é o total reservado pelo alocador — ao
        # contrário do contador parcial do DirectML, que ficou cravado em 271 MB
        # num cartão de 8 GB e só servia como tendência.
        return torch.cuda.memory_reserved(0) / 1e6
    try:
        import torch_directml as dml

        return float(sum(dml.gpu_memory(0)))
    except Exception:
        return None


def _agora() -> str:
    """Horário em UTC, ISO-8601. Um só ponto para não haver dois formatos."""
    return datetime.now(UTC).isoformat()


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
    # ─── GradCache ────────────────────────────────────────────────────────
    #
    # Desacopla o TAMANHO DO LOTE da memória. Sem isto o lote é limitado pelo que
    # cabe na GPU, e no InfoNCE o lote É o número de negativos — o limite de
    # QUALIDADE, não de velocidade. Medido nos 8 GB: SciBERT cabe em lote 8 (7
    # negativos), MiniLM em 128 (127). A literatura opera entre 64 e 256.
    #
    # `sub_lote` é o que realmente vai à GPU de cada vez; `lote` passa a ser o
    # lote LÓGICO, do qual saem os negativos. Com `sub_lote=32` e `lote=512` são
    # 511 negativos usando a memória de 32.
    #
    # `None` desliga o GradCache e usa o caminho direto — é o que os treinos
    # anteriores fizeram, e o que o teste de equivalência compara.
    sub_lote: int | None = None
    # Precisão mista. Só tem efeito no CUDA — ver o comentário em `__init__`.
    # No T4 do Kaggle é onde está o ganho: fp16 com `sdpa` não materializa a
    # matriz de atenção N×N, que é exatamente o tensor de 226 MB que estourou a
    # VRAM desta máquina duas vezes com lote 128.
    amp: bool = True
    passos_log: int = 50
    # ⚠️ Cadência do ESTADO retomável, independente da avaliação.
    #
    # O estado era salvo junto com a avaliação. Com `passos_aval=500` isso
    # significa que uma queda no passo 150 perde 150 passos — e foi exatamente o
    # que aconteceu em 2026-08-16: o treino morreu em silêncio no 150 e não havia
    # NADA para retomar, porque o primeiro estado só sairia no 500.
    #
    # As duas coisas têm custo muito diferente: a avaliação leva ~15 s (codifica
    # 2.000 textos), o estado leva ~2 s (grava pesos e momentos do Adam). Amarrar
    # a barata na cara foi economia no lugar errado.
    passos_estado: int = 100
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
    """CUDA se houver, DirectML se não, CPU por último. Nunca falha em silêncio.

    A ordem é essa porque as restrições deste módulo são todas do DirectML, não do
    modelo (ver a tabela na docstring): no CUDA voltam a valer `sdpa` no backward e
    autocast de verdade em fp16. O caminho DML fica intocado — a máquina do dono do
    projeto é uma RX 7600, e este código roda nas duas.

    Existe por causa do Kaggle: 30 h/semana de T4 resolvem o T1a e o T1b sem custo,
    e cada experimento que aqui leva 13 h com três mortes cabe numa sessão lá. Um
    experimento barato muda a economia das decisões que dependem dele.
    """
    if pedido == "cpu":
        return torch.device("cpu")
    if pedido in ("auto", "cuda") and torch.cuda.is_available():
        nome = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        log.info("dispositivo: %s (CUDA %d.%d, %d GB)", nome, *cap,
                 round(torch.cuda.get_device_properties(0).total_memory / 1e9))
        return torch.device("cuda:0")
    if pedido == "cuda":
        raise RuntimeError("CUDA pedido explicitamente e indisponível")
    try:
        import torch_directml as dml

        if dml.device_count() > 0:
            # ⚠️ `device_name` devolve string C de tamanho fixo, com NUL embutido:
            # `'AMD Radeon RX 7600\x00'`. Registrar cru mete bytes NUL no log, e
            # aí o `grep` classifica o arquivo inteiro como BINÁRIO e passa a
            # imprimir "Binary file matches" em vez das linhas.
            #
            # Custou um monitor: o vigia do treino ficou emitindo isso em vez das
            # avaliações, então eu tinha aviso sem conteúdo. Cinco NULs num log de
            # 14 KB de texto bastam.
            nome = dml.device_name(0).replace("\x00", "").strip()
            log.info("dispositivo: %s (DirectML)", nome)
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
        # `eager` é obrigatório NO DIRECTML: `sdpa` quebra no backward com erro
        # interno que nem decodifica como texto. No CUDA a restrição não existe, e
        # `sdpa` é onde está a diferença de velocidade e de memória — atenção sem
        # materializar a matriz N×N é justamente o tensor de 226 MB que matou o
        # treino de lote 128 duas vezes nesta máquina.
        self.atencao = "sdpa" if self.dev.type == "cuda" else "eager"
        self.mod = AutoModel.from_pretrained(
            cfg.base, attn_implementation=self.atencao).to(self.dev)
        # AMP só faz sentido onde há suporte de verdade. O DirectML não expõe
        # `GradScaler` utilizável, e ligar autocast lá deu queda silenciosa no
        # backward — então a decisão é pelo dispositivo, não por bandeira do
        # usuário: uma bandeira que quebra em metade dos dispositivos é armadilha.
        self.amp = self.dev.type == "cuda" and cfg.amp
        self.escala = torch.amp.GradScaler("cuda") if self.amp else None
        if self.amp:
            log.info("AMP fp16 ligado (%s) · atenção %s", self.dev, self.atencao)
        if cfg.checkpointing:
            # ⚠️ Um passo contrastivo mantém DOIS grafos vivos ao mesmo tempo —
            # âncora e positivo — antes do backward. Medir a vazão com um
            # forward só, como eu fiz primeiro, subestima a memória pela metade
            # e o treino morre com "não há memória de vídeo" pedindo 36 MB.
            self.mod.gradient_checkpointing_enable()
            self.mod.config.use_cache = False
        self.opt = torch.optim.AdamW(self.mod.parameters(), lr=cfg.lr)
        self._melhor_ndcg = -1.0  # reconstruído em `retomar`, ver o aviso lá

        # ⚠️ GradCache + AMP RECUSADO, não "não suportado em silêncio".
        #
        # O GradCache deriva a perda em relação a representações CACHEADAS e injeta
        # esse gradiente num segundo forward. Com `GradScaler` no meio, a escala
        # aplicada na fase 2 tem de ser desfeita antes da injeção da fase 3, senão
        # os gradientes entram multiplicados por um fator que muda a cada passo.
        #
        # O resultado disso não é uma exceção: é um treino que roda até o fim e
        # aprende outra coisa. Gradiente silenciosamente errado é a pior categoria
        # de defeito que este projeto pode ter, e a implementação correta pede um
        # teste de equivalência que ainda não existe.
        #
        # E a combinação não é necessária: o GradCache existe porque 8 GB não
        # cabiam o lote 128; num T4 de 16 GB com `sdpa` ele cabe direto. Quem
        # precisar das duas coisas ao mesmo tempo tem de implementar e provar por
        # equivalência, como o GradCache foi provado.
        if self.cfg.sub_lote and self.amp:
            raise RuntimeError(
                "GradCache (--sub-lote) com AMP não está implementado, e rodar "
                "assim produziria gradiente errado em silêncio. Escolha um: "
                "--sub-lote sem AMP (--sem-amp), ou AMP com o lote que couber "
                "direto na memória.")

    @property
    def lote_fisico(self) -> int:
        """Quantos textos vão à GPU de uma vez.

        ⚠️ Com GradCache, `cfg.lote` passa a ser LÓGICO — é de onde saem os
        negativos, não o que cabe na memória. Todo lugar que fatia para caber
        precisa deste, não daquele.

        Custou um treino: `avaliar` usava `cfg.lote` e, com `--lote 512`, tentou
        codificar 512 textos de uma vez na avaliação de linha de base, ANTES do
        primeiro passo. Morreu pedindo 576 MB — exatamente
        512 × 192 tokens × 1536 do intermediário × 4 bytes. O treino em si estava
        correto; quem não sabia da mudança de significado era um uso distante.
        """
        return self.cfg.sub_lote or self.cfg.lote

    def _codificar(self, textos: list[str]) -> torch.Tensor:
        lote = self.tok(list(textos), padding="max_length", truncation=True,
                        max_length=self.cfg.max_tokens, return_tensors="pt")
        lote = {k: v.to(self.dev) for k, v in lote.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            saida = self.mod(**lote).last_hidden_state
            v = media_mascarada(saida, lote["attention_mask"])
        # ⚠️ A normalização e a perda ficam em fp32, FORA do autocast. Sob fp16 a
        # norma de um vetor de 384 dimensões perde precisão o suficiente para o
        # cosseno mudar na terceira casa, e a métrica do portão G1 é decidida na
        # terceira casa — o G1.2 falhou por 0,005. Economizar precisão exatamente
        # onde a decisão acontece seria trocar o resultado pela velocidade.
        return F.normalize(v.float(), dim=-1)

    def _perda(self, va: torch.Tensor, vp: torch.Tensor) -> torch.Tensor:
        """InfoNCE com negativos do próprio lote.

        A diagonal de `va @ vp.T` são os pares verdadeiros; o resto do lote são
        os negativos. Simétrica nas duas direções porque "A busca B" e "B busca
        A" são as duas consultas que o modelo vai receber em uso.
        """
        sim = va @ vp.T / self.cfg.temperatura
        alvo = torch.arange(sim.size(0), device=sim.device)
        return 0.5 * (F.cross_entropy(sim, alvo) + F.cross_entropy(sim.T, alvo))

    def _passo_gradcache(self, ancoras: list[str], positivos: list[str]) -> torch.Tensor:
        """Um passo de InfoNCE com lote lógico maior que a memória. (Gao et al., 2021)

        ## O que ele resolve

        No InfoNCE o lote É o conjunto de negativos, então limitar o lote pela
        memória limita a QUALIDADE. O truque: as representações são minúsculas
        (`lote × 384` floats) mas o GRAFO que as produziu é enorme. GradCache
        separa os dois.

        ## As três fases

        1. **Cachear.** Forward de cada pedaço SEM grafo, guardando só as
           representações. Memória: um pedaço.
        2. **Perda.** Calcular o InfoNCE nas representações do lote INTEIRO e
           derivar em relação a elas. É aqui que todos os negativos entram, e
           custa quase nada — a matriz de similaridade é `lote × lote`.
        3. **Propagar.** Refazer cada pedaço COM grafo e injetar o gradiente
           cacheado daquele fatia via `autograd.backward(reps, grad)`.

        Custa um forward extra por pedaço (~1,6× o tempo) e devolve a memória.

        ## ⚠️ O forward tem de ser REPRODUTÍVEL entre as fases 1 e 3

        É o ponto onde uma implementação errada falha em SILÊNCIO. Se o dropout
        sortear máscaras diferentes nos dois forwards, o gradiente cacheado não
        corresponde à ativação recomputada — o treino roda, a perda cai, e os
        gradientes estão errados. Nada no log denuncia.

        A semente por pedaço força a mesma máscara nas duas fases.
        `tests/regression/test_gradcache.py` compara os gradientes com o caminho
        direto e é o que autoriza usar isto.
        """
        sub = self.cfg.sub_lote or len(ancoras)
        pedacos = [(ancoras[i:i + sub], positivos[i:i + sub])
                   for i in range(0, len(ancoras), sub)]

        # ── fase 1: representações sem grafo ──────────────────────────────
        sementes = [self.cfg.semente * 1_000_003 + i for i in range(len(pedacos))]
        ra, rp = [], []
        with torch.no_grad():
            for (ca, cp), s in zip(pedacos, sementes):
                torch.manual_seed(s)
                ra.append(self._codificar(ca))
                rp.append(self._codificar(cp))
        va = torch.cat(ra).detach().requires_grad_(True)
        vp = torch.cat(rp).detach().requires_grad_(True)

        # ── fase 2: perda no lote LÓGICO inteiro ──────────────────────────
        perda = self._perda(va, vp)
        perda.backward()
        ga, gp = va.grad, vp.grad

        # ── fase 3: refazer com grafo e injetar o gradiente ───────────────
        i = 0
        for (ca, cp), s in zip(pedacos, sementes):
            n = len(ca)
            torch.manual_seed(s)
            torch.autograd.backward([self._codificar(ca), self._codificar(cp)],
                                    [ga[i:i + n], gp[i:i + n]])
            i += n
        return perda.detach()

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
        lf = self.lote_fisico
        va = torch.cat([self._codificar(amostra["ancora"].to_list()[i:i + lf])
                        for i in range(0, amostra.height, lf)]).cpu()
        vp = torch.cat([self._codificar(amostra["positivo"].to_list()[i:i + lf])
                        for i in range(0, amostra.height, lf)]).cpu()
        sim = va @ vp.T
        alvo = torch.arange(sim.size(0))
        ordem = sim.argsort(dim=1, descending=True)
        posicao = (ordem == alvo.unsqueeze(1)).float().argmax(dim=1) + 1
        self.mod.train()
        # nDCG@10 com UM relevante por consulta: 1/log2(1+posição) dentro do
        # top-10, zero fora. É a métrica do portão G1 (DOC-00 §5), e ela sai de
        # graça das posições que já temos.
        dcg = torch.where(posicao <= 10,
                          1.0 / torch.log2(posicao.float() + 1.0),
                          torch.zeros_like(posicao, dtype=torch.float))
        return ((posicao == 1).float().mean().item(),
                (posicao <= 10).float().mean().item(),
                (1.0 / posicao).mean().item(),
                dcg.mean().item())

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
        # o treino ajudou — só que o número final é X.
        #
        # ⚠️ Numa RETOMADA isto não mede a base: `retomar` já carregou os nossos
        # pesos, então o número é o do modelo treinado até `inicio`. O rótulo
        # dizia "base <nome>" nos dois casos, e eu li 0,513 como "a base melhorou
        # de 0,454" antes de perceber que era o nosso modelo no passo 800. O
        # comentário aqui afirmava que "o rótulo diz qual" — dizia no `historico`,
        # não no log, que é onde alguém olha.
        r1, r10, mrr, ndcg = self.avaliar(val)
        log.info("%s | entre %d candidatos: recall@1 %.3f · recall@10 %.3f · "
                 "MRR %.3f · nDCG@10 %.3f",
                 f"base {self.cfg.base}" if not inicio
                 else f"ponto de partida (nosso modelo no passo {inicio:,})",
                 self.cfg.n_candidatos, r1, r10, mrr, ndcg)
        m.n_candidatos = self.cfg.n_candidatos
        m.historico.append({"passo": inicio, "recall_1": r1, "recall_10": r10,
                            "mrr": mrr, "ndcg_10": ndcg, "n_candidatos": m.n_candidatos, "perda": None,
                            "nota": "antes do treino" if not inicio else f"retomado do passo {inicio}"})

        t0, vistos, soma, n = time.perf_counter(), 0, 0.0, 0
        for passo, (a, p) in enumerate(carregador, start=1):
            if passo <= inicio:
                continue        # lote já consumido: pular é barato, refazer não
            if self.cfg.sub_lote:
                perda = self._passo_gradcache(list(a), list(p))
            else:
                perda = self._perda(self._codificar(a), self._codificar(p))
                if self.escala is not None:
                    self.escala.scale(perda).backward()
                else:
                    perda.backward()
            if self.escala is not None:
                self.escala.step(self.opt)
                self.escala.update()
            else:
                self.opt.step()
            self.opt.zero_grad()
            soma += perda.item()
            n += 1
            vistos += len(a)

            if passo % self.cfg.passos_log == 0:
                m.pares_por_s = vistos / (time.perf_counter() - t0)
                v = _vram_mb()
                log.info("passo %d | perda %.4f | %.1f pares/s%s", passo, soma / n,
                         m.pares_por_s, "" if v is None else f" | VRAM {v:,.0f} MB")
                soma, n = 0.0, 0

            m.passo = passo

            if passo % self.cfg.passos_aval == 0:
                r1, r10, mrr, ndcg = self.avaliar(val)
                log.info("  aval: recall@1 %.3f · recall@10 %.3f · MRR %.3f · "
                         "nDCG@10 %.3f", r1, r10, mrr, ndcg)
                m.recall_1, m.recall_10, m.mrr = r1, r10, mrr
                m.historico.append({"passo": passo, "recall_1": r1, "recall_10": r10,
                                    "mrr": mrr, "ndcg_10": ndcg, "perda": perda.item()})
                # `m` vai junto: sem ele o `phiemb.json` do checkpoint não levava
                # `metricas`, e a curva existia só no log. Conferido em execução —
                # `historico` chegou vazio ao json depois de 27 avaliações.
                self.salvar(saida, m, passo=passo)
                self._talvez_melhor(saida, m, passo, r1, r10, mrr, ndcg)

            # Estado FORA do bloco de avaliação: cadência própria, mais curta.
            # Ver o comentário de `passos_estado` em Config.
            if passo % self.cfg.passos_estado == 0:
                self.salvar_estado(saida, passo)

        r1, r10, mrr, ndcg = self.avaliar(val)
        m.recall_1, m.recall_10, m.mrr = r1, r10, mrr
        m.perda = perda.item()
        self.salvar_estado(saida, m.passo)
        m.historico.append({"passo": m.passo, "recall_1": r1, "recall_10": r10,
                            "mrr": mrr, "perda": m.perda, "nota": "final"})
        self.salvar(saida, m, concluido=True)
        return m

    # ── melhor checkpoint ─────────────────────────────────────────────────
    #
    # ⚠️ `salvar` grava sempre no mesmo diretório. No treino de 2026-08-09 o pico
    # de recall@1 foi 0,461 no passo ~38.000, e o modelo entregue no fim media
    # 0,441 — o melhor tinha sido SOBRESCRITO. Naquele caso a diferença era
    # ruído (±0,031 de erro padrão), então não se perdeu nada de verdade. Foi
    # sorte, não projeto.
    #
    # ## Por que o critério é MRR e não recall@1
    #
    # `recall@1` é uma proporção binária: com 256 candidatos, cada acerto vale
    # 0,004 e a métrica pula em degraus grosseiros. MRR usa a POSIÇÃO de cada
    # acerto, então varia continuamente e é muito menos ruidosa com a mesma
    # amostra. Escolher o melhor por recall@1 seria escolher pelo maior ruído.
    #
    # ## Por que uma CÓPIA e não um link
    #
    # O melhor tem de sobreviver ao fim do treino, e o diretório principal segue
    # sendo sobrescrito. Custa ~440 MB de disco, contra perder o único artefato
    # que passou o portão.

    def _talvez_melhor(self, saida: Path, m: Metricas, passo: int,
                       r1: float, r10: float, mrr: float, ndcg: float) -> None:
        """Guarda o checkpoint de pico. ⚠️ O critério é nDCG@10, não MRR.

        Era MRR, e o portão G1 usa nDCG@10 — as duas divergem. Medido em
        2026-08-16 no treino com 511 negativos: recall@1 subiu e nDCG@10 caiu ao
        mesmo tempo, e o `campeao()` do avaliador, que também usava recall@1,
        elegeu o pior dos nossos na métrica que decide.

        Escolher checkpoint por uma métrica e julgar o portão por outra é o mesmo
        erro em dois lugares. Corrigido aqui antes de gastar 15 h de treino, em vez
        de descobrir depois que o pico guardado não era o pico que importa.
        """
        if ndcg <= self._melhor_ndcg:
            return
        self._melhor_ndcg = ndcg
        destino = saida.parent / f"{saida.name}-melhor"
        destino.mkdir(parents=True, exist_ok=True)
        self.mod.save_pretrained(destino, state_dict=self._para_cpu(self.mod.state_dict()))
        self.tok.save_pretrained(destino)
        (destino / "melhor.json").write_text(
            json.dumps({"passo": passo, "recall_1": r1, "recall_10": r10, "mrr": mrr,
                        "ndcg_10": ndcg,
                        "n_candidatos": self.cfg.n_candidatos, "base": self.cfg.base,
                        "criterio": "nDCG@10 — a métrica do portão G1 (DOC-00 §5)"},
                       indent=2, ensure_ascii=False),
            encoding="utf-8")
        # ⚠️ Anuncia a métrica que DECIDIU, não outra. A linha dizia
        # "(MRR %.3f)" enquanto a escolha era por nDCG@10 — e neste projeto a
        # confusão entre as duas já elegeu o campeão errado e produziu um veredito
        # de G1.2 a −0,014 em vez de −0,005. Um log que nomeia a métrica errada é
        # o mesmo defeito num lugar mais barato de errar, e mais fácil de crer.
        log.info("  ★ melhor até agora (nDCG@10 %.3f) → %s", ndcg, destino.name)

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

        # Marcador de progresso legível DE FORA, e a razão é o supervisor.
        #
        # O passo vive dentro do `.pt`, que o PowerShell não sabe abrir. O
        # `phiemb.json` — que o supervisor lê — só aparece no diretório de saída
        # ao FIM do treino: durante a corrida ele é gravado em `<saida>-melhor`,
        # e só quando o nDCG melhora. Resultado: o progresso lido durante um
        # treino era sempre -1, a guarda de "morre sempre no mesmo ponto" ficava
        # desligada, e o supervisor relançaria 40 vezes um treino que morre
        # sempre no passo 150 — laço infinito com aparência de resiliência.
        #
        # Este arquivo é o único progresso durável e INCONDICIONAL do treino:
        # mesma cadência do estado (`passos_estado`), sem depender de melhora.
        (saida / "progresso.json").write_text(
            json.dumps({"passo": passo, "ts": _agora()}), encoding="utf-8")

    def retomar(self, saida: Path) -> int:
        """Devolve o passo de onde continuar, ou 0 se não houver estado."""
        # ⚠️ O melhor tem de ser reconstruído ANTES de qualquer avaliação. Sem
        # isto, `_melhor_ndcg` começaria em -1 numa retomada e a PRIMEIRA avaliação
        # sobrescreveria o melhor checkpoint com um pior — e o treino deste projeto
        # foi retomado dez vezes num único dia, então isso não é hipótese remota.
        #
        # `ndcg_10` com recuo para `mrr`: checkpoints gravados antes de 2026-08-16
        # só têm MRR. Recuar para ele é melhor que começar em -1, mas a primeira
        # avaliação vai comparar nDCG contra um MRR — então o log AVISA, para
        # ninguém ler a substituição do melhor como regressão do modelo.
        anterior = saida.parent / f"{saida.name}-melhor" / "melhor.json"
        if anterior.exists():
            try:
                d = json.loads(anterior.read_text(encoding="utf-8"))
                if "ndcg_10" in d:
                    self._melhor_ndcg = float(d["ndcg_10"])
                    log.info("melhor anterior preservado: nDCG@10 %.3f", self._melhor_ndcg)
                else:
                    self._melhor_ndcg = -1.0
                    log.warning("checkpoint anterior tem só MRR (critério antigo) — "
                                "a busca do melhor recomeça em nDCG@10")
            except Exception as exc:
                log.warning("melhor.json ilegível (%s) — recomeçando a busca do melhor", exc)

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
               concluido: bool = False, passo: int = 0) -> None:
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
        # `actual_count` é a MESMA chave que o supervisor lê nos manifestos de
        # coleta para decidir se o processo AVANÇOU entre duas mortes.
        #
        # Isto só cobre o treino CONCLUÍDO. Durante a corrida este arquivo não
        # existe no diretório de saída, e o progresso vivo vem do
        # `progresso.json` que `salvar_estado` grava — ver o comentário lá.
        meta["actual_count"] = m.passo if m else passo
        # `completed_at` é a MESMA chave que o supervisor já procura nos
        # manifestos de coleta. Reusar o nome evita um segundo mecanismo de
        # "acabou" — e um mecanismo a menos é um ramo a menos sem teste.
        if concluido:
            meta["completed_at"] = _agora()
        if m:
            meta["metricas"] = dict(asdict(m).items())
        (saida / "phiemb.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
