"""Classificador de Física — Sprint S2 (DOC-02 §6).

A ideia que torna isto gratuito: **os ~2,7 M resumos do arXiv já vêm rotulados
pelos próprios autores**, com exatamente a taxonomia que queremos. É dado
supervisionado de graça, com a única fonte de rótulo que é autoritativa
(o autor sabe em que subárea escreveu; um classificador de terceiros, não).

Duas tarefas, com maturidades diferentes:

  ``subfield``  — 14 subáreas de Física. Treinável AGORA, com o que já está
                  em disco. Serve à mistura estratificada (DOC-06 §2.3) e à
                  perda de validação por subárea (DOC-08 §9.2), que é a
                  métrica de maior valor por unidade de instrumentação.

  ``is_physics`` — binária, para filtrar peS2o / OpenWebMath / The Stack.
                  Exige exemplos NEGATIVOS (cs, q-bio, econ…), que serão
                  coletados quando a coleta de Física terminar. Rodar dois
                  coletores contra o arXiv ao mesmo tempo violaria o
                  princípio A5 (cortesia) do DOC-02.

**Desvio registrado em relação ao DOC-02 §6.** Aquele documento especifica
fastText. Usamos um modelo linear do scikit-learn sobre n-gramas. A razão do
DOC-02 era operacional — "treina em minutos na CPU, infere a ~10⁵ doc/s" — e
o sklearn satisfaz as duas, sem exigir a compilação C++ do fastText, que é
frágil e prende o pipeline a um binário. A qualidade em classificação de
texto por tópico é equivalente para modelos lineares sobre n-gramas.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Aparece nos avisos do módulo; declarado aqui para não repetir a lista.
NEGATIVO_AUSENTE = "math"

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)

Task = Literal["subfield", "is_physics"]


@dataclass
class CalibrationResult:
    """Limiares por classe que atingem a precisão alvo.

    DOC-02 §6: **não usar 0,5**. O corpus é abundante e o custo de um
    documento irrelevante contaminando o treino é maior que o de perder um
    documento relevante — a assimetria pede alta precisão, revocação menor.
    """

    target_precision: float
    thresholds: dict[str, float]
    achieved_precision: dict[str, float]
    achieved_recall: dict[str, float]
    coverage: float  # fração de itens que passam algum limiar

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)


@dataclass
class PhysicsClassifier:
    task: Task
    pipeline: Pipeline
    classes: list[str] = field(default_factory=list)
    calibration: CalibrationResult | None = None
    version: str = "0.1.0"

    # ── inferência ────────────────────────────────────────────────────────

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    def predict(self, texts: list[str], use_thresholds: bool = True) -> list[str | None]:
        """Devolve a classe, ou ``None`` quando nenhuma atinge o limiar.

        ``None`` é resposta legítima, não falha: é o análogo do veredito
        ``INCONCLUSIVE`` do barramento de verificação (DOC-10 §2.1). Forçar
        um rótulo em documento ambíguo injeta ruído no corpus.
        """
        proba = self.predict_proba(texts)
        out: list[str | None] = []
        for row in proba:
            i = int(row.argmax())
            cls = self.classes[i]
            if (
                use_thresholds
                and self.calibration
                and row[i] < self.calibration.thresholds.get(cls, 0.5)
            ):
                out.append(None)
                continue
            out.append(cls)
        return out


def montar_binario(
    spine: Path,
    negativos: Path,
    max_por_classe: int = 400_000,
    seed: int = 17,
) -> pl.DataFrame:
    """Dataset rotulado para `is_physics`, com a regra autoritativa.

    ## O rótulo negativo não é "veio de cs/econ/q-bio"

    Coletamos negativos dos conjuntos `cs`, `econ` e `q-bio`. Mas um paper de
    `cs.LG` com cross-list em `quant-ph` **é** Física, e usá-lo como negativo
    ensina o classificador a rejeitar Física. Medido em 2026-08-11 nos negativos
    coletados:

        cs      988.244   5,7% com cross-list de Física
        econ     16.984   5,5%
        q-bio    56.142  32,8%   ← um terço

    ## Por que a regra é «fora do spine» e não uma lista de prefixos

    Testei as duas. Dos 1.041.652 negativos únicos, 72.919 têm cross-list de
    Física — e **exatamente** esses 72.919 estão no spine, com **zero** fora.
    Os conjuntos OAI-PMH do arXiv são consistentes com as listas de categoria, o
    que valida as duas coletas de uma vez.

    Então a pertinência ao spine é o rótulo autoritativo, e usá-la dispensa
    manter uma lista de prefixos — que é justamente onde eu erraria: a nossa
    `PHYSICS_PREFIXES` não inclui os arquivos legados (`adap-org`, `chao-dyn`,
    `patt-sol`, `solv-int`, `acc-phys`, `atom-ph`, `chem-ph`, `plasm-ph`,
    `supr-con`…). Nesse caso específico não custou nada — o arXiv retroagiu
    cross-lists atuais em todos os 5.5 mil papers legados, e nenhum ficou sem
    prefixo atual — mas a lista continua sendo dívida esperando um caso novo.

    ## ⚠️ Limite de domínio, declarado

    Os negativos são resumos do arXiv de cs/econ/q-bio. O classificador vai ser
    aplicado ao peS2o e ao OpenWebMath, cuja distribuição negativa é muito mais
    ampla — química, biologia, medicina, humanidades, texto de web.

    Pior: **`math` não está nos negativos**, e matemática é a vizinha mais
    confundível da Física. O OpenWebMath é cheio dela. Esperar alta precisão ali
    sem negativos de `math` é otimismo, não medição.
    """
    ids_fisica = pl.scan_parquet(spine).select("arxiv_id")

    # ── negativos: fora do spine, deduplicados ────────────────────────────
    # `**` e não `*`: os sets grandes são coletados em fatias anuais
    # (`math/2019/part-*.parquet`, ver `SETS_FATIADOS` em `harvest_negativos.py`),
    # e um glob de um nível só os deixaria de fora — silenciosamente, treinando
    # sem o negativo mais importante.
    neg = (
        pl.scan_parquet(str(negativos / "**" / "*.parquet"))
        .unique(subset=["arxiv_id"])
        .join(ids_fisica.with_columns(pl.lit(True).alias("_fis")),
              on="arxiv_id", how="anti")
        .select("arxiv_id", "title", "abstract")
        .with_columns(pl.lit("nao_fisica").alias("is_physics"))
    )
    pos = (
        pl.scan_parquet(spine)
        .select("arxiv_id", "title", "abstract")
        .with_columns(pl.lit("fisica").alias("is_physics"))
    )

    # Amostragem determinística por hash. `sample` não existe em plano lazy, e
    # materializar 2,6 M de títulos+resumos para poder sortear é exatamente o
    # erro que já custou quatro travamentos de memória neste projeto.
    def cortar(lf: pl.LazyFrame) -> pl.LazyFrame:
        return (lf.with_columns(pl.col("arxiv_id").hash(seed=seed).alias("_h"))
                  .sort("_h").head(max_por_classe).drop("_h"))

    df = pl.concat([cortar(pos), cortar(neg)]).collect(engine="streaming")
    log.info("is_physics: %s física · %s não-física",
             f"{df.filter(pl.col('is_physics') == 'fisica').height:,}",
             f"{df.filter(pl.col('is_physics') == 'nao_fisica').height:,}")
    return df


@dataclass
class Transferencia:
    """Quanto o classificador perde num domínio negativo que nunca viu."""

    dominio_omitido: str
    n_treino_pos: int
    n_treino_neg: int
    n_teste_neg: int
    precisao_dentro: float
    precisao_fora: float
    fp_dentro: float          # fração dos negativos aceitos como Física
    fp_fora: float
    revocacao: float
    # (limiar, precisão, revocação, fp) no domínio de fora. O limiar é o botão
    # que se tem em produção, então a curva dele é o que informa a decisão.
    curva: list[tuple[float, float, float, float]] = field(default_factory=list)

    @property
    def degradacao(self) -> float:
        """Quantas vezes o falso positivo piora fora do domínio."""
        return self.fp_fora / self.fp_dentro if self.fp_dentro else float("inf")


def avaliar_transferencia(
    spine: Path,
    negativos: Path,
    omitir: str,
    n_por_classe: int = 120_000,
    limiares: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999),
    seed: int = 17,
) -> Transferencia:
    """Deixa-um-domínio-de-fora: treina sem `omitir`, testa nele.

    ## Por que esta medição existe

    O `is_physics` dá F1 0,972 na validação — que é arXiv contra arXiv, mesma
    distribuição. Mas ele vai ser aplicado ao peS2o e ao OpenWebMath, cuja
    distribuição negativa é muito mais ampla. O número de validação não responde
    a pergunta que importa, e é o número que qualquer relatório mostraria.

    Omitir um domínio inteiro do treino e testar só nele é a aproximação
    disponível sem dado novo: mede o que acontece quando o negativo é de um tipo
    que o modelo nunca viu.

    Medido em 2026-08-13, omitindo `q-bio`:

        dentro do domínio (cs novos)     falso positivo   1,9%
        domínio nunca visto (q-bio)      falso positivo  32,9%

    Dezessete vezes pior. E a curva de limiar mostra que subir a exigência não
    resolve: de 0,5 a 0,999 o falso positivo cai de 32,9% para 10,0% e estanca —
    o `modified_huber` satura as probabilidades, então o limiar perde resolução
    justamente onde se precisaria dele.

    ## O que este número NÃO é

    Não é a taxa esperada no peS2o. `q-bio` é o vizinho mais difícil possível
    (biofísica é a zona de sobreposição por excelência), então serve de cota
    pessimista para vizinhos próximos — e diz pouco sobre texto de web.
    """
    from sklearn.metrics import precision_recall_fscore_support

    ids_fisica = pl.scan_parquet(spine).select("arxiv_id")
    dominios = sorted(p.name for p in negativos.iterdir() if p.is_dir())
    if omitir not in dominios:
        raise ValueError(f"domínio {omitir!r} não está em {dominios}")
    treino_dom = [d for d in dominios if d != omitir]

    def carregar(doms: list[str], n: int) -> pl.DataFrame:
        return (
            pl.scan_parquet([str(negativos / d / "**" / "*.parquet") for d in doms])
            .unique(subset=["arxiv_id"])
            .join(ids_fisica, on="arxiv_id", how="anti")
            .select("arxiv_id", "title", "abstract")
            .with_columns(pl.col("arxiv_id").hash(seed=seed).alias("_h"))
            .sort("_h").head(n).drop("_h")
            .collect(engine="streaming")
        )

    pos = (pl.scan_parquet(spine).select("arxiv_id", "title", "abstract")
             .with_columns(pl.col("arxiv_id").hash(seed=seed).alias("_h"))
             .sort("_h").head(2 * n_por_classe).drop("_h")
             .collect(engine="streaming"))
    pos_tr, pos_te = pos.head(n_por_classe), pos.tail(n_por_classe)
    neg_tr = carregar(treino_dom, n_por_classe)
    neg_te = carregar([omitir], n_por_classe)

    # Negativos DENTRO do domínio de treino que não foram usados no treino: é a
    # comparação honesta. Usar os próprios do treino mediria memorização.
    usados = set(neg_tr["arxiv_id"])
    neg_dentro = (
        pl.scan_parquet([str(negativos / d / "**" / "*.parquet") for d in treino_dom])
        .unique(subset=["arxiv_id"])
        .join(ids_fisica, on="arxiv_id", how="anti")
        .filter(~pl.col("arxiv_id").is_in(pl.Series(list(usados)).implode()))
        .select("arxiv_id", "title", "abstract")
        .head(n_por_classe).collect(engine="streaming")
    )

    log.info("omitindo %s · treino: %s física + %s não-física de %s",
             omitir, f"{pos_tr.height:,}", f"{neg_tr.height:,}", ", ".join(treino_dom))
    pipe = make_pipeline()
    pipe.fit(build_text(pos_tr) + build_text(neg_tr),
             ["fisica"] * pos_tr.height + ["nao_fisica"] * neg_tr.height)
    i_fis = list(pipe.classes_).index("fisica")

    Xf = build_text(pos_te)
    def medir(Xn):
        y = ["fisica"] * len(Xf) + ["nao_fisica"] * len(Xn)
        p = pipe.predict(Xf + Xn)
        pr, rc, _, _ = precision_recall_fscore_support(
            y, p, labels=["fisica"], zero_division=0)
        fp = sum(1 for a, b in zip(y, p) if a == "nao_fisica" and b == "fisica")
        return float(pr[0]), float(rc[0]), fp / max(len(Xn), 1)

    p_in, rc, fp_in = medir(build_text(neg_dentro))
    p_out, _, fp_out = medir(build_text(neg_te))

    sf = pipe.predict_proba(Xf)[:, i_fis]
    sn = pipe.predict_proba(build_text(neg_te))[:, i_fis]
    curva = []
    for t in limiares:
        tp, fp = int((sf >= t).sum()), int((sn >= t).sum())
        curva.append((t, tp / (tp + fp) if tp + fp else 0.0,
                      tp / len(sf), fp / len(sn)))

    return Transferencia(omitir, pos_tr.height, neg_tr.height, neg_te.height,
                         p_in, p_out, fp_in, fp_out, rc, curva)


def build_text(df: pl.DataFrame) -> list[str]:
    """Título + resumo. O título carrega sinal desproporcional ao tamanho."""
    return (
        df.select(
            (pl.col("title").fill_null("") + "\n" + pl.col("abstract").fill_null(""))
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )
        .to_series()
        .to_list()
    )


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    sublinear_tf=True,
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=400_000,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                # `modified_huber` dá `predict_proba`, que a calibração exige.
                # `class_weight="balanced"` é obrigatório: o desbalanceamento
                # entre subáreas é de ~29:1, e sem isso o modelo aprenderia a
                # prever cond-mat quase sempre.
                SGDClassifier(
                    loss="modified_huber",
                    alpha=1e-6,
                    max_iter=15,
                    tol=1e-4,
                    class_weight="balanced",
                    random_state=17,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def calibrate(
    clf: PhysicsClassifier, X_val: list[str], y_val: list[str], target_precision: float = 0.95
) -> CalibrationResult:
    """Encontra, por classe, o menor limiar que atinge a precisão alvo."""
    proba = clf.predict_proba(X_val)
    y = np.asarray(y_val)
    thresholds, prec, rec = {}, {}, {}

    for i, cls in enumerate(clf.classes):
        scores = proba[:, i]
        truth = y == cls
        best_t, best_r = 1.01, 0.0  # 1.01 = classe nunca atinge a precisão alvo
        for t in np.arange(0.05, 1.0, 0.01):
            sel = scores >= t
            if sel.sum() == 0:
                continue
            p = truth[sel].mean()
            if p >= target_precision:
                best_t = float(t)
                best_r = float(truth[sel].sum() / max(truth.sum(), 1))
                break
        thresholds[cls] = best_t
        sel = scores >= best_t
        prec[cls] = float(truth[sel].mean()) if sel.sum() else 0.0
        rec[cls] = best_r

    argmax = proba.argmax(axis=1)
    passa = np.array(
        [proba[j, argmax[j]] >= thresholds[clf.classes[argmax[j]]] for j in range(len(y))]
    )
    return CalibrationResult(target_precision, thresholds, prec, rec, float(passa.mean()))


def train(
    df: pl.DataFrame,
    task: Task = "subfield",
    label_col: str = "subfield",
    val_frac: float = 0.15,
    target_precision: float = 0.95,
) -> tuple[PhysicsClassifier, str]:
    """Treina e calibra. Devolve o classificador e o relatório."""
    df = df.filter(pl.col(label_col).is_not_null() & (pl.col(label_col) != "Outro"))
    df = df.sample(fraction=1.0, shuffle=True, seed=17)

    n_val = int(df.height * val_frac)
    val, tr = df.head(n_val), df.tail(df.height - n_val)
    Xtr, ytr = build_text(tr), tr[label_col].to_list()
    Xva, yva = build_text(val), val[label_col].to_list()
    log.info("treino: %s | validação: %s | classes: %d",
             f"{len(Xtr):,}", f"{len(Xva):,}", len(set(ytr)))

    pipe = make_pipeline()
    pipe.fit(Xtr, ytr)
    clf = PhysicsClassifier(task=task, pipeline=pipe, classes=list(pipe.classes_))
    clf.calibration = calibrate(clf, Xva, yva, target_precision)

    rep = classification_report(yva, pipe.predict(Xva), zero_division=0, digits=3)
    return clf, rep


def save(clf: PhysicsClassifier, out_dir: Path) -> None:
    import pickle

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(clf.pipeline, f)
    meta = {
        "task": clf.task,
        "version": clf.version,
        "classes": clf.classes,
        "calibration": clf.calibration.__dict__ if clf.calibration else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    log.info("→ %s", out_dir)
