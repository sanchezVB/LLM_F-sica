"""Triagem CPT com documentos artificiais, sem abrir benchmarks reais."""
from __future__ import annotations

import json

import pytest

from phifm.corpus.auditoria_cpt import (
    BENCHMARKS_G2,
    auditar_corpus,
    sha256,
    validar_manifesto,
)


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _doc(id, text, partition="train_open", **extra):
    return {"id": id, "text": text, "partition": partition, "source": "sintetico", **extra}


def _inputs(tmp_path):
    tr = _jsonl(tmp_path / "entrada_train.jsonl", [_doc("a", "O corpo livre conserva seu momento.")])
    dev = _jsonl(tmp_path / "entrada_dev.jsonl", [_doc("b", "Um oscilador tem frequência própria.")])
    refs = _jsonl(tmp_path / "refs.jsonl", [{"id": "r1", "bench": "ugphysics",
                                           "text": "Referência artificial sem exposição."}])
    return tr, dev, refs, tmp_path / "auditado"


def test_manifesto_confere_hashes_contagens_e_proveniencia(tmp_path):
    args = _inputs(tmp_path)
    m = auditar_corpus(*args)
    assert m["status"] == "aprovado"
    assert m["outputs"]["train"]["documents"] == 1
    assert not m["references"]["verified_for_g2"]
    assert validar_manifesto(args[3] / "manifesto.json") == m


def test_particoes_proibidas_nao_chegam_ao_treino(tmp_path):
    tr, dev, refs, out = _inputs(tmp_path)
    rows = [_doc("a", "Documento aberto permitido."),
            _doc("b2", "Outro documento permitido.", "train_only")]
    for n, part in enumerate(["eval_only", "excluded", "inventada", None]):
        rows.append(_doc(f"proibido{n}", f"Texto reservado número {n}.", part))
    _jsonl(tr, rows)
    m = auditar_corpus(tr, dev, refs, out)
    assert m["outputs"]["train"]["documents"] == 2
    assert m["outputs"]["train"]["rejected"] == {"particao_nao_treinavel": 4}
    assert [r["id"] for r in map(json.loads, (out / "train.jsonl").read_text().splitlines())] == ["a", "b2"]


@pytest.mark.parametrize("mudanca", ["id", "text"])
def test_recusa_splits_sobrepostos_sem_deixar_saida(tmp_path, mudanca):
    tr, dev, refs, out = _inputs(tmp_path)
    d = _doc("a" if mudanca == "id" else "b", "outra coisa" if mudanca == "id"
             else "  O CORPO LIVRE CONSERVA seu momento. ")
    _jsonl(dev, [d])
    with pytest.raises(ValueError, match="compartilhado|repetido"):
        auditar_corpus(tr, dev, refs, out)
    assert not out.exists()
    assert not list(tmp_path.glob(".auditado-*"))


def test_exclui_13grama_exato_resposta_e_duplicata(tmp_path):
    tr, dev, refs, out = _inputs(tmp_path)
    enunciado = "um dois tres quatro cinco seis sete oito nove dez onze doze treze"
    resposta = "resposta artificial longa com oito palavras distintas aqui"
    _jsonl(refs, [{"id": "r", "bench": "ugphysics", "text": enunciado, "answer": resposta}])
    _jsonl(tr, [_doc("bom", "Texto de Física aceitável."),
                _doc("dup", "Texto de FÍSICA aceitável. "),
                _doc("exato", enunciado), _doc("ng", "prefixo " + enunciado + " fim"),
                _doc("answer", "explicação: " + resposta + " termina.")])
    m = auditar_corpus(tr, dev, refs, out)
    assert m["outputs"]["train"]["rejected"] == {
        "duplicata_textual": 1, "referencia_exata": 1,
        "referencia_13grama": 1, "resposta_referencia": 1}


def test_dados_sem_source_sao_recusados(tmp_path):
    tr, dev, refs, out = _inputs(tmp_path)
    _jsonl(tr, [{"id": "x", "text": "Sem origem", "partition": "train_open"}])
    with pytest.raises(ValueError, match="proveniência"):
        auditar_corpus(tr, dev, refs, out)
    assert not out.exists()


def test_banco_vazio_nao_produz_falso_atestado(tmp_path):
    tr, dev, refs, out = _inputs(tmp_path)
    refs.write_text("")
    with pytest.raises(ValueError, match="vazio"):
        auditar_corpus(tr, dev, refs, out)
    assert not out.exists()


@pytest.mark.parametrize("alvo", ["input", "references", "output"])
def test_validacao_recusa_alteracao_posterior(tmp_path, alvo):
    args = _inputs(tmp_path)
    auditar_corpus(*args)
    path = {"input": args[0], "references": args[2], "output": args[3] / "train.jsonl"}[alvo]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        validar_manifesto(args[3] / "manifesto.json")


def test_nao_sobrescreve_auditoria_existente(tmp_path):
    args = _inputs(tmp_path)
    auditar_corpus(*args)
    before = (args[3] / "manifesto.json").read_bytes()
    with pytest.raises(FileExistsError):
        auditar_corpus(*args)
    assert (args[3] / "manifesto.json").read_bytes() == before


def test_autoafirmacao_g2_nao_substitui_ancora_externa(tmp_path):
    args = _inputs(tmp_path)
    auditar_corpus(*args)
    path = args[3] / "manifesto.json"
    m = json.loads(path.read_text())
    m["references"]["verified_for_g2"] = True
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="protocolo congelado"):
        validar_manifesto(path, exigir_referencias_g2=True)


def test_ancora_g2_exige_hash_e_contagens_dos_sete_benchmarks(tmp_path):
    tr, dev, refs, out = _inputs(tmp_path)
    _jsonl(refs, [{"id": f"r{n}", "bench": b, "text": f"Referência artificial {n}."}
                  for n, b in enumerate(sorted(BENCHMARKS_G2))])
    m = auditar_corpus(tr, dev, refs, out)
    anchor = {"sha256": sha256(refs), "counts": m["references"]["counts"]}
    assert validar_manifesto(out / "manifesto.json", True, referencias_g2=anchor) == m
    with pytest.raises(ValueError, match="âncora"):
        validar_manifesto(out / "manifesto.json", True,
                          referencias_g2={**anchor, "sha256": "0" * 64})


def test_validacao_recusa_path_escape(tmp_path):
    args = _inputs(tmp_path)
    auditar_corpus(*args)
    path = args[3] / "manifesto.json"
    m = json.loads(path.read_text())
    m["outputs"]["train"]["path"] = str(args[0])
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="escapa"):
        validar_manifesto(path)


def test_ids_de_referencia_repetidos_sao_recusados(tmp_path):
    tr, dev, refs, out = _inputs(tmp_path)
    row = {"id": "r", "bench": "ugphysics", "text": "Exemplo artificial"}
    _jsonl(refs, [row, row])
    with pytest.raises(ValueError, match="duplicado"):
        auditar_corpus(tr, dev, refs, out)
