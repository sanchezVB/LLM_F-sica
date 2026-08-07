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
            if use_thresholds and self.calibration:
                if row[i] < self.calibration.thresholds.get(cls, 0.5):
                    out.append(None)
                    continue
            out.append(cls)
        return out


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
