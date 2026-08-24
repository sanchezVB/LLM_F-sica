"""ΦRank — cross-encoder que reordena o top-100 do ΦEmb (DOC-07 §4).

O ΦEmb pontua consulta e documento **separadamente** e compara vetores: barato, e
por isso serve para varrer 667 mil candidatos. O cross-encoder concatena os dois e
os pontua **juntos**, então a atenção vê as duas metades ao mesmo tempo — muito mais
preciso, e caro o bastante para só valer sobre um top-K já reduzido.

    consulta ──► ΦEmb ──► top-100 de 667 mil ──► ΦRank ──► top-10 reordenado
                (denso, ~ms)                    (cross, ~s)

## Desvio de especificação, declarado

O DOC-07 §4 pede o cross-encoder **inicializado do ΦEnc**. O ΦEnc não existe — ele
exige 15–30 B tokens de texto completo (DOC-07 §2) e o Sprint S3 entregou 13,15 B,
mas o treino dele não foi feito. Então a inicialização é do `all-MiniLM-L6-v2`, a
mesma base do ΦEmb campeão.

Consequência a registrar: um ΦRank sobre MiniLM não é o ΦRank do documento. Ele
testa a mecânica do reranking e entrega o T1b; a versão especificada precisa do
ΦEnc.

## O rótulo, e o cuidado que ele exige

Positivo = o paper que a âncora cita. Negativo = os minerados pelo ΦEmb campeão,
**depois de remover os co-citados**.

⚠️ Sem essa remoção, 15,4% dos negativos são documentos que a literatura cita junto
com o positivo (medido em 2026-08-18, contra 0,1% num controle aleatório). Treinar
um reranker com eles ensinaria a rebaixar o que é relevante — e o efeito seria
invisível no treino, porque a perda desce normalmente.

## Por que `num_labels=1` e não duas classes

Reranking é ordenação, não classificação. Com uma saída escalar e a perda de
entropia cruzada sobre o grupo `(1 positivo + N negativos)`, o modelo é otimizado
para a POSIÇÃO do positivo dentro do grupo — que é o que o nDCG mede. Com duas
classes e perda binária, ele seria otimizado para acertar rótulos
independentemente, e dois documentos poderiam receber 0,9 sem que nada os ordenasse.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from phifm.training.amostragem import amostrar_por_documento
from phifm.training.embedding import _agora, _vram_mb, escolher_dispositivo

log = logging.getLogger(__name__)

BASE_PADRAO = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class ConfigRank:
    base: str = BASE_PADRAO
    # 384 e não 192: o cross-encoder recebe consulta E documento no mesmo fluxo.
    # Com 192 o par seria truncado ao meio e o modelo veria menos do documento do
    # que o ΦEmb vê — o reranqueador julgaria com menos informação que o
    # recuperador, o que inverte o propósito.
    max_tokens: int = 384
    # Grupos por passo. Cada grupo é 1 positivo + `n_negativos` negativos, então o
    # lote real em textos é `grupos x (1 + n_negativos)`.
    #
    # Medido nesta máquina: 32,9 exemplos/s a 384 tokens no lote 32. Com 4 grupos de
    # 8 são 32 exemplos por passo — o teto confortável dos 8 GB.
    grupos: int = 4
    n_negativos: int = 7
    lr: float = 2e-5
    max_grupos: int | None = None
    checkpointing: bool = True
    amp: bool = True
    passos_log: int = 50
    passos_estado: int = 200
    passos_aval: int = 500
    # Grupos de validação por avaliação. 500 grupos x 8 = 4.000 pares para
    # pontuar, ~2 min — barato o suficiente para avaliar com frequência.
    grupos_aval: int = 500
    semente: int = 17
    dispositivo: str = "auto"


@dataclass
class MetricasRank:
    passo: int = 0
    perda: float = 0.0
    # Acurácia no topo: em que fração dos grupos o positivo ficou em primeiro.
    # É a métrica natural do reranking, e é comparável entre execuções porque o
    # tamanho do grupo é fixo — ao contrário do recall@1 do ΦEmb, que depende de
    # `n_candidatos`.
    acerto_top1: float = 0.0
    mrr_grupo: float = 0.0
    # ⚠️ DOCUMENTOS distintos, nao linhas. Linhas do mesmo documento citado nao
    # sao observacoes independentes, e chamar isto de "grupos" foi o que escondeu
    # que `head(500)` media 35 documentos. Ver `TreinadorRank.avaliar`.
    n_documentos_aval: int = 0
    exemplos_por_s: float = 0.0
    historico: list[dict] = field(default_factory=list)


class GruposDataset(Dataset):
    """Um item = (consulta, [positivo, negativos…]). O primeiro é sempre o certo.

    ⚠️ A posição do positivo é FIXA no índice 0, e o embaralhamento acontece na
    perda, não aqui: `cross_entropy` recebe o alvo 0 e os logits na ordem do grupo.
    Embaralhar aqui exigiria carregar o índice do positivo junto e nada garantiria
    que os dois ficassem em sincronia — o mesmo tipo de desalinhamento que envenenou
    o cache de vetores.
    """

    def __init__(self, df: pl.DataFrame, n_negativos: int, semente: int = 17):
        self.consulta = df["ancora"]
        self.positivo = df["positivo"]
        self.negativos = df["negativos"]
        self.n = n_negativos
        self.rng = __import__("random").Random(semente)

    def __len__(self) -> int:
        return len(self.consulta)

    def __getitem__(self, i: int) -> tuple[str, list[str]]:
        negs = list(self.negativos[i])
        if len(negs) >= self.n:
            negs = self.rng.sample(negs, self.n)
        elif negs:
            # Repete para completar o grupo. Grupos de tamanho variável dariam
            # logits de formas diferentes por passo, e o `cross_entropy` sobre grupo
            # exige forma fixa. Repetir um negativo é mais honesto que descartar a
            # linha: o par continua no conjunto.
            negs = [negs[j % len(negs)] for j in range(self.n)]
        else:
            # Nenhum negativo (0,06% dos pares depois do filtro de co-citação): o
            # grupo fica só com o positivo repetido, e o alvo 0 é trivialmente
            # certo. Contribui ~zero de gradiente em vez de quebrar o lote.
            negs = [self.positivo[i]] * self.n
        return self.consulta[i], [self.positivo[i], *negs]


def colar_grupos(lote: list) -> tuple[list[str], list[list[str]]]:
    consultas, docs = zip(*lote, strict=True)
    return list(consultas), list(docs)


class TreinadorRank:
    def __init__(self, cfg: ConfigRank):
        self.cfg = cfg
        torch.manual_seed(cfg.semente)
        self.dev = escolher_dispositivo(cfg.dispositivo)
        self.tok = AutoTokenizer.from_pretrained(cfg.base)
        atencao = "sdpa" if self.dev.type == "cuda" else "eager"
        self.mod = AutoModelForSequenceClassification.from_pretrained(
            cfg.base, num_labels=1, attn_implementation=atencao).to(self.dev)
        if cfg.checkpointing:
            self.mod.gradient_checkpointing_enable()
            self.mod.config.use_cache = False
        self.opt = torch.optim.AdamW(self.mod.parameters(), lr=cfg.lr)
        self.amp = self.dev.type == "cuda" and cfg.amp
        self.escala = torch.amp.GradScaler("cuda") if self.amp else None
        self._melhor = -1.0
        log.info("ΦRank sobre %s · atenção %s · AMP %s · grupos de %d",
                 cfg.base, atencao, self.amp, 1 + cfg.n_negativos)

    def _pontuar(self, consultas: list[str], docs: list[list[str]]) -> torch.Tensor:
        """Devolve `(grupos, 1 + n_negativos)` de logits.

        Achata o grupo num lote só: o cross-encoder não sabe que existem grupos, e
        a estrutura volta pelo `view` — um forward de `g*(1+n)` pares é muito mais
        rápido que `g` forwards de `(1+n)`.
        """
        g, k = len(consultas), len(docs[0])
        pares_a = [c for c, ds in zip(consultas, docs, strict=True) for _ in ds]
        pares_b = [d for ds in docs for d in ds]
        b = self.tok(pares_a, pares_b, padding="max_length", truncation=True,
                     max_length=self.cfg.max_tokens, return_tensors="pt")
        b = {k_: v.to(self.dev) for k_, v in b.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            logits = self.mod(**b).logits
        # fp32 na saída: a ordenação é decidida em diferenças pequenas, e o portão
        # G1 é decidido na terceira casa decimal. Ver o mesmo cuidado em `_codificar`.
        return logits.float().view(g, k)

    def _perda(self, escores: torch.Tensor) -> torch.Tensor:
        """Entropia cruzada sobre o GRUPO, alvo 0 (o positivo).

        Otimiza a POSIÇÃO do positivo dentro do grupo — que é o que o nDCG mede. Uma
        perda binária por documento otimizaria acertar rótulos independentemente, e
        dois documentos poderiam receber 0,9 sem que nada os ordenasse.
        """
        alvo = torch.zeros(escores.size(0), dtype=torch.long, device=escores.device)
        return F.cross_entropy(escores, alvo)

    @torch.no_grad()
    def avaliar(self, val: pl.DataFrame) -> tuple[float, float, int]:
        """(acerto no topo, MRR no grupo, DOCUMENTOS distintos avaliados).

        ⚠️ Amostra ALEATÓRIA, não `head`. Medido em 2026-08-24: o parquet vem
        agrupado por documento citado, então `val.head(500)` são 500 linhas mas
        apenas **35 documentos**, cada um repetido ~14 vezes. Linhas do mesmo
        documento não são observações independentes — o n efetivo da métrica é o
        número de DOCUMENTOS, e era 35.

        O estrago: com n efetivo 35, o acerto@1 de 0,364 carrega intervalo de 95%
        de ±0,159 — [0,205, 0,523] contra uma base de 0,198. O ganho inteiro cabia
        dentro do ruído da própria medida. Foi por isso que a divisão contaminada e
        a divisão honesta reportaram o mesmo número: as duas mediam as mesmas três
        dezenas de documentos, e nenhuma das duas media generalização.

        Com `sample`, 1.500 grupos cobrem 457 documentos e o intervalo cai para
        ±0,045 — aí dá para decidir alguma coisa.

        O terceiro valor devolvido passa a ser DOCUMENTOS distintos, não linhas,
        porque é ele que dimensiona o intervalo de confiança.
        """
        self.mod.eval()
        amostra, n_doc = amostrar_por_documento(val, self.cfg.grupos_aval,
                                                self.cfg.semente)
        ds = GruposDataset(amostra, self.cfg.n_negativos, self.cfg.semente)
        carregador = DataLoader(ds, batch_size=self.cfg.grupos, shuffle=False,
                                collate_fn=colar_grupos)
        top1 = rr = n = 0.0
        for consultas, docs in carregador:
            e = self._pontuar(consultas, docs)
            pos = (e.argsort(dim=1, descending=True) == 0).float().argmax(dim=1)
            top1 += float((pos == 0).sum())
            rr += float((1.0 / (pos + 1)).sum())
            n += len(consultas)
        self.mod.train()
        return top1 / max(n, 1), rr / max(n, 1), int(n_doc)

    @staticmethod
    def _ic95(p: float, n_doc: int) -> float:
        """Meia-largura do intervalo de 95% sobre o n EFETIVO (documentos).

        Impressa junto da métrica de propósito: foi a ausência dela que deixou
        `head(500)` passar por 500 observações durante toda uma semana.
        """
        return 1.96 * (max(p * (1.0 - p), 0.0) / max(n_doc, 1)) ** 0.5

    def salvar(self, saida: Path, m: MetricasRank | None = None,
               concluido: bool = False) -> None:
        saida.mkdir(parents=True, exist_ok=True)
        pesos = {k: v.detach().to("cpu") for k, v in self.mod.state_dict().items()}
        self.mod.save_pretrained(saida, state_dict=pesos)
        self.tok.save_pretrained(saida)
        meta = {"config": asdict(self.cfg), "base": self.cfg.base,
                "desvio_de_especificacao": (
                    "DOC-07 §4 pede inicialização do ΦEnc, que não existe; "
                    "inicializado do MiniLM, a mesma base do ΦEmb campeão"),
                "actual_count": m.passo if m else 0}
        if concluido:
            meta["completed_at"] = _agora()
        if m:
            meta["metricas"] = asdict(m)
        (saida / "phirank.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def salvar_estado(self, saida: Path, passo: int) -> None:
        saida.mkdir(parents=True, exist_ok=True)
        tmp = saida / "estado_rank.pt.tmp"
        torch.save({"passo": passo, "modelo": self.mod.state_dict(),
                    "otimizador": self.opt.state_dict(), "cfg": asdict(self.cfg)}, tmp)
        tmp.replace(saida / "estado_rank.pt")
        (saida / "progresso.json").write_text(
            json.dumps({"passo": passo, "ts": _agora()}), encoding="utf-8")

    def retomar(self, saida: Path) -> int:
        melhor = saida.parent / f"{saida.name}-melhor" / "melhor.json"
        if melhor.exists():
            try:
                self._melhor = float(json.loads(
                    melhor.read_text(encoding="utf-8"))["acerto_top1"])
                log.info("melhor anterior preservado: acerto@1 %.3f", self._melhor)
            except Exception as exc:
                log.warning("melhor.json ilegível (%s) — busca reiniciada", exc)
        p = saida / "estado_rank.pt"
        if not p.exists():
            return 0
        est = torch.load(p, map_location="cpu", weights_only=False)
        self.mod.load_state_dict(est["modelo"])
        self.mod.to(self.dev)
        self.opt.load_state_dict(est["otimizador"])
        log.info("retomando do passo %s", f"{est['passo']:,}")
        return int(est["passo"])

    def _talvez_melhor(self, saida: Path, m: MetricasRank, top1: float) -> None:
        if top1 <= self._melhor:
            return
        self._melhor = top1
        destino = saida.parent / f"{saida.name}-melhor"
        self.salvar(destino, m)
        (destino / "melhor.json").write_text(
            json.dumps({"passo": m.passo, "acerto_top1": top1,
                        "mrr_grupo": m.mrr_grupo,
                        "grupo": 1 + self.cfg.n_negativos,
                        "base": self.cfg.base,
                        "criterio": "acerto@1 no grupo"},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("  melhor ate agora (acerto@1 %.3f) -> %s", top1, destino.name)

    def treinar(self, treino: pl.DataFrame, val: pl.DataFrame,
                saida: Path) -> MetricasRank:
        if self.cfg.max_grupos and self.cfg.max_grupos < len(treino):
            # ⚠️ `sample`, nao `head` — pelo mesmo motivo que em `avaliar`, e aqui
            # o preco e maior porque afeta o TREINO e nao so a medida.
            #
            # O parquet vem agrupado por documento citado. `head(12.500)` sao 12.500
            # linhas cobrindo ~700 documentos distintos, cada um repetido ~18 vezes:
            # o modelo ve o mesmo punhado de papers como positivo o treino inteiro.
            # `sample(12.500)` cobre ~9.000 documentos pelo MESMO custo de GPU.
            _, antes = amostrar_por_documento(treino, 0, self.cfg.semente)
            treino, n_doc_tr = amostrar_por_documento(
                treino, self.cfg.max_grupos, self.cfg.semente)
            log.info("treino: %s grupos sorteados · %s documentos distintos "
                     "(de %s disponiveis)", f"{len(treino):,}",
                     f"{n_doc_tr:,}", f"{antes:,}")
        g = torch.Generator().manual_seed(self.cfg.semente)
        carregador = DataLoader(
            GruposDataset(treino, self.cfg.n_negativos, self.cfg.semente),
            batch_size=self.cfg.grupos, shuffle=True, drop_last=True,
            generator=g, collate_fn=colar_grupos)

        m = MetricasRank()
        inicio = self.retomar(saida)

        top1, mrr, n = self.avaliar(val)
        log.info("%s | grupos de %d: acerto@1 %.3f ±%.3f · MRR %.3f (%d documentos)",
                 "base" if not inicio else f"ponto de partida (passo {inicio:,})",
                 1 + self.cfg.n_negativos, top1, self._ic95(top1, n), mrr, n)
        # ⚠️ O acerto ao acaso é 1/(1+n_negativos) = 0,125 com grupo de 8. Um modelo
        # não treinado fica em torno disso, e um número perto de 0,125 no fim
        # significaria que ele não aprendeu — não que a métrica é ruim.
        log.info("  (acaso = %.3f)", 1.0 / (1 + self.cfg.n_negativos))
        m.n_documentos_aval = n
        m.historico.append({"passo": inicio, "acerto_top1": top1, "mrr_grupo": mrr,
                            "nota": "antes do treino" if not inicio
                                    else f"retomado do passo {inicio}"})

        t0, vistos, soma, nl = time.perf_counter(), 0, 0.0, 0
        for passo, (consultas, docs) in enumerate(carregador, start=1):
            if passo <= inicio:
                continue
            escores = self._pontuar(consultas, docs)
            perda = self._perda(escores)
            if self.escala is not None:
                self.escala.scale(perda).backward()
                self.escala.step(self.opt)
                self.escala.update()
            else:
                perda.backward()
                self.opt.step()
            self.opt.zero_grad()

            valor = perda.item()
            if valor != valor or valor in (float("inf"), float("-inf")):
                raise RuntimeError(
                    f"perda não finita ({valor}) no passo {passo}. Para aqui de "
                    "propósito: continuar produziria pesos sem sentido e o defeito "
                    "só apareceria horas depois.")
            soma += valor
            nl += 1
            vistos += len(consultas) * len(docs[0])
            m.passo = passo

            if passo % self.cfg.passos_log == 0:
                m.exemplos_por_s = vistos / (time.perf_counter() - t0)
                v = _vram_mb()
                log.info("passo %d | perda %.4f | %.1f exemplos/s%s", passo,
                         soma / nl, m.exemplos_por_s,
                         "" if v is None else f" | VRAM {v:,.0f} MB")
                soma, nl = 0.0, 0

            if passo % self.cfg.passos_aval == 0:
                top1, mrr, n = self.avaliar(val)
                m.acerto_top1, m.mrr_grupo, m.n_documentos_aval = top1, mrr, n
                log.info("  aval: acerto@1 %.3f ±%.3f · MRR %.3f (%d documentos)",
                         top1, self._ic95(top1, n), mrr, n)
                m.historico.append({"passo": passo, "acerto_top1": top1,
                                    "mrr_grupo": mrr, "perda": valor})
                self.salvar(saida, m)
                self._talvez_melhor(saida, m, top1)

            if passo % self.cfg.passos_estado == 0:
                self.salvar_estado(saida, passo)

        top1, mrr, n = self.avaliar(val)
        m.acerto_top1, m.mrr_grupo, m.n_documentos_aval = top1, mrr, n
        self.salvar_estado(saida, m.passo)
        self.salvar(saida, m, concluido=True)
        self._talvez_melhor(saida, m, top1)
        log.info("passo %d | CONCLUIDO · acerto@1 %.3f ±%.3f · MRR %.3f (%d documentos)",
                 m.passo, top1, self._ic95(top1, n), mrr, n)
        return m
