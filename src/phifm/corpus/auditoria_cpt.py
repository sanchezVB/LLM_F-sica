"""Triagem explícita de um corpus CPT contra um banco de referências fornecido.

Não abre benchmarks por conta própria. Esta triagem literal não detecta paráfrases nem
substitui sondas de memorização. Um manifesto autoafirmado nunca certifica cobertura G2:
o chamador deve fornecer o hash e as contagens ancorados no protocolo congelado.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

SCHEMA = "phifm.cpt-audit.v1"
PARTICOES = frozenset({"train_open", "train_only"})
BENCHMARKS_G2 = frozenset({"ugphysics", "olympiadbench", "phybench", "scibench",
                          "mmlu", "evalplus", "ifeval"})


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with Path(caminho).open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _normalizar(texto: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", texto).casefold().split())


def _palavras(texto: str) -> list[str]:
    return re.findall(r"\w+", _normalizar(texto))


def _ngrams(palavras: list[str], n: int) -> Iterator[tuple[str, ...]]:
    for i in range(len(palavras) - n + 1):
        yield tuple(palavras[i:i + n])


def _jsonl(caminho: Path) -> Iterator[dict]:
    with caminho.open(encoding="utf-8") as f:
        for linha, texto in enumerate(f, 1):
            if not texto.strip():
                continue
            try:
                doc = json.loads(texto)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido em {caminho.name}:{linha}") from exc
            if not isinstance(doc, dict):
                raise ValueError(f"objeto JSON esperado em {caminho.name}:{linha}")
            for campo in ("id", "text"):
                if not isinstance(doc.get(campo), str) or not doc[campo].strip():
                    raise ValueError(f"{campo} ausente/inválido em {caminho.name}:{linha}")
            yield doc


def _referencias(caminho: Path) -> tuple[dict, set, set, dict]:
    contagens: Counter = Counter()
    ids = set()
    exatos, trezes = set(), set()
    respostas: dict[int, set[tuple[str, ...]]] = {}
    for item in _jsonl(caminho):
        bench = item.get("bench")
        if not isinstance(bench, str) or not bench.strip():
            raise ValueError("referência sem benchmark identificável")
        chave = (bench, item["id"])
        if chave in ids:
            raise ValueError("id de referência duplicado no mesmo benchmark")
        ids.add(chave)
        contagens[bench] += 1
        exatos.add(_normalizar(item["text"]))
        trezes.update(_ngrams(_palavras(item["text"]), 13))
        resposta = item.get("answer", "")
        if not isinstance(resposta, str):
            raise ValueError("answer da referência precisa ser texto")
        palavras = _palavras(resposta)
        if len(palavras) >= 8:
            respostas.setdefault(len(palavras), set()).add(tuple(palavras))
    if not contagens:
        raise ValueError("banco de referências vazio: triagem não executada")
    return dict(sorted(contagens.items())), exatos, trezes, respostas


def _contaminacao(texto: str, exatos: set, trezes: set, respostas: dict) -> str | None:
    if _normalizar(texto) in exatos:
        return "referencia_exata"
    palavras = _palavras(texto)
    if any(g in trezes for g in _ngrams(palavras, 13)):
        return "referencia_13grama"
    for n, conhecidas in respostas.items():
        if any(g in conhecidas for g in _ngrams(palavras, n)):
            return "resposta_referencia"
    return None


def _artefato(caminho: Path) -> dict:
    return {"path": str(caminho.resolve()), "sha256": sha256(caminho)}


def auditar_corpus(treino: Path, dev: Path, referencias: Path, saida: Path) -> dict:
    """Escreve train/dev.jsonl e manifesto apenas numa saída nova, após todas as guardas.

    Documentos exigem id/text/source. Partição ausente ou fora de train_open/train_only
    exclui o documento. Splits sobrepostos são erro, mesmo antes das exclusões.
    Duplicatas textuais dentro de um split são removidas; ids repetidos são ambíguos e
    recusados. A pasta temporária fica ao lado da saída, no mesmo volume (HD).
    """
    treino, dev, referencias = (Path(p).resolve(strict=True)
                                for p in (treino, dev, referencias))
    saida = Path(saida).absolute()
    if saida.exists():
        raise FileExistsError(f"saída já existe; preserve a auditoria anterior: {saida}")
    if len({treino, dev, referencias}) != 3:
        raise ValueError("treino, dev e referências precisam ser arquivos distintos")
    entradas = {"train": _artefato(treino), "dev": _artefato(dev)}
    banco = _artefato(referencias)
    counts, exatos, trezes, respostas = _referencias(referencias)
    saida.parent.mkdir(parents=True, exist_ok=True)
    temporario = Path(tempfile.mkdtemp(prefix=f".{saida.name}-", dir=saida.parent))
    vistos_id: dict[str, str] = {}
    vistos_texto: dict[str, str] = {}
    resultados = {}
    try:
        for split, entrada in (("train", treino), ("dev", dev)):
            cont: Counter = Counter()
            rejeitados: Counter = Counter()
            destino = temporario / f"{split}.jsonl"
            with destino.open("w", encoding="utf-8", newline="\n") as f:
                for doc in _jsonl(entrada):
                    cont["input"] += 1
                    if not isinstance(doc.get("source"), str) or not doc["source"].strip():
                        raise ValueError(f"documento sem source/proveniência no split {split}")
                    if doc["id"] in vistos_id:
                        raise ValueError("id repetido no corpus ou compartilhado entre splits")
                    vistos_id[doc["id"]] = split
                    digest = hashlib.sha256(_normalizar(doc["text"]).encode()).hexdigest()
                    anterior = vistos_texto.get(digest)
                    if anterior is not None and anterior != split:
                        raise ValueError("texto compartilhado entre treino e dev")
                    vistos_texto[digest] = split
                    if doc.get("partition") not in PARTICOES:
                        rejeitados["particao_nao_treinavel"] += 1
                        continue
                    if anterior is not None:
                        rejeitados["duplicata_textual"] += 1
                        continue
                    motivo = _contaminacao(doc["text"], exatos, trezes, respostas)
                    if motivo:
                        rejeitados[motivo] += 1
                        continue
                    f.write(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n")
                    cont["accepted"] += 1
            if not cont["accepted"]:
                raise ValueError(f"split {split} vazio após triagem; corpus não aprovado")
            resultados[split] = {
                "path": f"{split}.jsonl", "sha256": sha256(destino),
                "documents": cont["accepted"], "input_documents": cont["input"],
                "rejected": dict(sorted(rejeitados.items())),
            }
        for info in [*entradas.values(), banco]:
            if sha256(Path(info["path"])) != info["sha256"]:
                raise ValueError("arquivo de entrada mudou durante a auditoria")
        manifesto = {
            "schema": SCHEMA, "status": "aprovado", "scope": "triagem_literal_referencias_fornecidas",
            "inputs": entradas, "outputs": resultados,
            "references": {**banco, "counts": counts, "verified_for_g2": False},
            "code": {"path": "src/phifm/corpus/auditoria_cpt.py", "sha256": sha256(Path(__file__))},
            "policy": {"partitions": sorted(PARTICOES), "ngram_words": 13,
                       "answer_min_words": 8, "disjoint_by": ["id", "normalized_text_sha256"],
                       "limits": ["não detecta paráfrases", "não substitui sondas de memorização",
                                  "cobertura G2 exige âncora no protocolo congelado"]},
        }
        (temporario / "manifesto.json").write_text(
            json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporario.rename(saida)
        return manifesto
    finally:
        if temporario.exists():
            shutil.rmtree(temporario)


def validar_manifesto(caminho: Path, exigir_referencias_g2: bool = False,
                      *, referencias_g2: dict | None = None) -> dict:
    """Confere arquivos e proveniência; não confia no status ou em verified_for_g2 sozinho.

    `referencias_g2` deve vir do protocolo congelado já conferido pelo chamador e conter
    sha256 do banco + counts exatas dos sete benchmarks. Não é um campo autoafirmado do
    manifesto. Sem essa âncora a triagem só é válida para desenvolvimento.
    """
    caminho = Path(caminho).resolve(strict=True)
    m = json.loads(caminho.read_text(encoding="utf-8"))
    if m.get("schema") != SCHEMA or m.get("status") != "aprovado":
        raise ValueError("manifesto CPT ausente, não aprovado ou de versão desconhecida")
    if m.get("code", {}).get("sha256") != sha256(Path(__file__)):
        raise ValueError("código de auditoria mudou; refaça a triagem numa saída nova")
    try:
        for info in [*m["inputs"].values(), m["references"]]:
            if sha256(Path(info["path"])) != info["sha256"]:
                raise ValueError("hash da entrada/referência diverge do manifesto")
        counts, _, _, _ = _referencias(Path(m["references"]["path"]))
        if counts != m["references"]["counts"]:
            raise ValueError("contagens de referências divergem do manifesto")
        ids, textos = set(), set()
        for split in ("train", "dev"):
            info = m["outputs"][split]
            relativo = Path(info["path"])
            arquivo = (caminho.parent / relativo).resolve(strict=True)
            if relativo.is_absolute() or not arquivo.is_relative_to(caminho.parent):
                raise ValueError("saída do manifesto escapa da pasta auditada")
            if sha256(arquivo) != info["sha256"]:
                raise ValueError("hash da saída diverge do manifesto")
            n = 0
            for doc in _jsonl(arquivo):
                if doc.get("partition") not in PARTICOES or not doc.get("source"):
                    raise ValueError("saída sem política treinável/proveniência")
                normalizado = _normalizar(doc["text"])
                if doc["id"] in ids or normalizado in textos:
                    raise ValueError("saídas duplicadas ou splits sobrepostos")
                ids.add(doc["id"])
                textos.add(normalizado)
                n += 1
            if n == 0 or n != info["documents"]:
                raise ValueError("contagem de documentos inconsistente")
    except (KeyError, TypeError) as exc:
        raise ValueError("manifesto CPT incompleto") from exc
    if exigir_referencias_g2:
        if not referencias_g2:
            raise ValueError("cobertura G2 não verificada: falta banco ancorado no protocolo congelado")
        esperadas = referencias_g2.get("counts", {})
        if (set(esperadas) != BENCHMARKS_G2 or any(type(n) is not int or n <= 0
                                                for n in esperadas.values())
                or esperadas != counts
                or referencias_g2.get("sha256") != m["references"]["sha256"]):
            raise ValueError("banco de referências não corresponde à âncora completa do G2")
    return m
