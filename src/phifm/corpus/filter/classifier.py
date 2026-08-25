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

# ⚠️ Fica DEPOIS dos imports. Entre eles, os seis imports de terceiros contavam
# como "import fora do topo do arquivo" (E402) e o ruff barrava o CI — que nunca
# passou uma vez desde que foi criado, em 2026-08-09.
# Aparece nos avisos do módulo; declarado aqui para não repetir a lista.
NEGATIVO_AUSENTE = "math"

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

    # Amostragem determinística por hash. `sample` não existe em plano lazy, e
    # materializar 2,6 M de títulos+resumos para poder sortear é exatamente o
    # erro que já custou quatro travamentos de memória neste projeto.
    def cortar(lf: pl.LazyFrame, n: int) -> pl.LazyFrame:
        return (lf.with_columns(pl.col("arxiv_id").hash(seed=seed).alias("_h"))
                  .sort("_h").head(n).drop("_h"))

    # ── negativos ESTRATIFICADOS por domínio ──────────────────────────────
    #
    # Amostrar uniformemente sobre todos os negativos dá a proporção do arXiv, e
    # ela é desequilibrada: `cs` tem 932 mil utilizáveis, `econ` tem 16 mil. Com
    # isso `q-bio` fica com 3% do treino e `econ` com 1% — os domínios seguem
    # quase NÃO VISTOS mesmo estando em disco.
    #
    # Medido em 2026-08-13, falso positivo por domínio, 80 mil negativos:
    #
    #                      cs    econ   q-bio    math   revocacao
    #   proporcional     2,2%    4,9%   18,9%    5,1%       0,952
    #   estratificado    2,8%    1,3%    4,8%    5,8%       0,944
    #
    # O pior caso cai de 18,9% para 5,8% — 3,3x — por 0,8 ponto de revocação.
    # Para um FILTRO é o pior domínio que determina a contaminação, então o pior
    # caso é o critério, não a média.
    #
    # `econ` não alcança a cota igual (só tem 16 mil), e por isso a cota é um
    # TETO por domínio, não uma exigência: quem tem menos contribui menos, e o
    # resto não é redistribuído para não recriar o desequilíbrio.
    dominios = sorted(p.name for p in negativos.iterdir() if p.is_dir())
    if not dominios:
        raise ValueError(f"nenhum domínio de negativos em {negativos}")
    cota = max_por_classe // len(dominios)

    partes = []
    for d in dominios:
        # `**` e não `*`: os sets grandes são coletados em fatias anuais
        # (`math/2019/part-*.parquet`, ver `SETS_FATIADOS` em
        # `harvest_negativos.py`), e um glob de um nível os deixaria de fora —
        # silenciosamente, treinando sem o negativo mais importante.
        lf = (
            pl.scan_parquet(str(negativos / d / "**" / "*.parquet"))
            .unique(subset=["arxiv_id"])
            .join(ids_fisica, on="arxiv_id", how="anti")
            .select("arxiv_id", "title", "abstract")
        )
        partes.append(cortar(lf, cota))

    # Um paper cross-listado aparece em dois sets (`cs.IT`+`math.IT`), então o
    # dedupe entre domínios é necessário. `keep="first"` com `maintain_order`
    # porque o padrão não garante QUAL linha sobrevive nem em que ordem — e sem
    # isso duas montagens com a mesma semente dão conjuntos diferentes, o que
    # quebra a comparabilidade entre rodadas. Achado por teste.
    neg = (pl.concat(partes)
             .unique(subset=["arxiv_id"], keep="first", maintain_order=True)
             .with_columns(pl.lit("nao_fisica").alias("is_physics")))
    pos = cortar(
        pl.scan_parquet(spine).select("arxiv_id", "title", "abstract"),
        max_por_classe,
    ).with_columns(pl.lit("fisica").alias("is_physics"))

    df = pl.concat([pos, neg]).collect(engine="streaming")
    log.info("negativos por domínio (cota %s): %s", f"{cota:,}", ", ".join(dominios))
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
        fp = sum(1 for a, b in zip(y, p, strict=True) if a == "nao_fisica" and b == "fisica")
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


def recuperar_legado(df: pl.DataFrame, label_col: str = "subfield") -> pl.DataFrame:
    """Dá subárea aos papers de arquivo LEGADO que ficaram em "Outro".

    ⚠️ Isto conserta um artefato velho, não um defeito de código.
    `normalize/spine.py` já aplica `_LEGADO` desde 2026-08-13; a `spine.parquet` no
    disco foi construída antes disso. São **4.042 papers** (0,25% da espinha) com
    primária pré-1998 — `chao-dyn`, `solv-int`, `patt-sol`, `mtrl-th`, `supr-con` —
    que `train()` descartaria, porque ela descarta "Outro".

    Por que aqui e não reconstruindo a espinha: ela é entrada dos pares de citação
    (6,5 M linhas), do próprio classificador e da fatia do RedPajama — refazê-la
    exigiria rebaixar 81 GB e re-derivar 22 GB de corpus, para 0,25%. Consertar no
    CONSUMIDOR não cascateia nada.

    ⚠️ Não é normalização geral: só sobe de "Outro" para uma subárea conhecida.
    Papers cuja primária é `math.AP`, `cs.LG` ou `q-bio.PE` continuam em "Outro" —
    para eles esse é o rótulo CERTO, e dar-lhes subárea de Física seria inventar.
    Medido: dos 72.872 em "Outro", só 6,2% são legado; os outros 68.328 são papers
    de outra área com cross-list de Física.

    Quando a espinha for reconstruída por outro motivo (o bulk pago do arXiv, por
    exemplo), esta função para de encontrar o que consertar — e o teste que a cobre
    passa a valer como verificação de que a reconstrução funcionou.
    """
    from phifm.corpus.normalize.spine import _LEGADO, SUBFIELD_MAP

    if "primary_category" not in df.columns:
        return df
    recuperado = (pl.col("primary_category")
                  .str.split(".").list.first()
                  .replace(_LEGADO)
                  .replace_strict(SUBFIELD_MAP, default="Outro"))
    antes = int((pl.select(df[label_col] == "Outro").to_series()).sum())
    df = df.with_columns(
        pl.when((pl.col(label_col) == "Outro") & (recuperado != "Outro"))
        .then(recuperado).otherwise(pl.col(label_col)).alias(label_col))
    depois = int((pl.select(df[label_col] == "Outro").to_series()).sum())
    if antes != depois:
        log.info("legado recuperado: %s papers saíram de \"Outro\" e entram no treino "
                 "de subárea (artefato anterior a 2026-08-13)", f"{antes - depois:,}")
    return df


def train(
    df: pl.DataFrame,
    task: Task = "subfield",
    label_col: str = "subfield",
    val_frac: float = 0.15,
    target_precision: float = 0.95,
) -> tuple[PhysicsClassifier, str]:
    """Treina e calibra. Devolve o classificador e o relatório."""
    if label_col == "subfield":
        df = recuperar_legado(df, label_col)
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
