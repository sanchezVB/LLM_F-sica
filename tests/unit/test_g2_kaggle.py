"""O executor do G2 no Kaggle: a linha cortada, o prazo e o estado entre sessões."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2_kaggle as k  # noqa: E402

FIM = b"\n"


def test_aparar_tira_so_a_linha_cortada(tmp_path):
    a = tmp_path / "x.jsonl"
    a.write_bytes(b'{"j": 0}\n{"j": 1}\n{"j": 2, "tex')
    assert k.aparar_jsonl(a) == len(b'{"j": 2, "tex')
    assert a.read_bytes() == b'{"j": 0}\n{"j": 1}\n'
    assert k.aparar_jsonl(a) == 0
    a.write_bytes(b'{"j": 0}\n{"j": 1}')                 # inteira, só sem o "\n": fica
    assert k.aparar_jsonl(a) == 0 and a.read_bytes().endswith(b"}" + FIM)   # ganha o fim
    a.write_bytes(b'{"j": 0}\n\x00\x00\x00')             # lixo de queda de energia
    assert k.aparar_jsonl(a) == 3
    assert k.aparar_jsonl(tmp_path / "nao_existe.jsonl") == 0


def test_prazo_mata_e_ok_termina(tmp_path):
    log = tmp_path / "log.txt"
    t0 = time.time()
    r = k.rodar_com_prazo([sys.executable, "-c", "import time; time.sleep(60)"],
                          prazo=time.time() + 3, log=log, carencia_s=5)
    assert r == "prazo" and time.time() - t0 < 30
    assert k.rodar_com_prazo([sys.executable, "-c", "print('oi')"], prazo=time.time() + 60,
                             log=log) == "ok"
    assert k.rodar_com_prazo([sys.executable, "-c", "raise SystemExit(3)"],
                             prazo=time.time() + 60, log=log) == "falhou"
    assert k.rodar_com_prazo([sys.executable, "-c", "pass"], prazo=time.time() - 1,
                             log=log) == "prazo"
    assert "oi" in log.read_text(encoding="utf-8")


def _repo(raiz: Path, livro: bytes) -> Path:
    av = raiz / "data/processed/avaliacao"
    (av / "g2_saidas").mkdir(parents=True)
    (av / "g2_acessos.jsonl").write_bytes(livro)
    (av / "g2_particao.json").write_text("{}", encoding="utf-8")       # congelado: não vai
    return raiz


def test_estado_vai_e_volta_conferido(tmp_path):
    livro = b'{"tamanho": "1.7B"}' + FIM
    sessao = _repo(tmp_path / "s1", livro)
    av = sessao / "data/processed/avaliacao"
    (av / "g2_saidas/teste_8B_base_veredito.jsonl").write_bytes(b'{"j": 0}' + FIM + b'{"j": 1, "x')
    (av / "g2_dev_8B_base_it1.json").write_text("{}", encoding="utf-8")
    with (av / "g2_acessos.jsonl").open("ab") as f:
        f.write(b'{"tamanho": "8B"}' + FIM)
    m = k.salvar_estado(sessao, tmp_path / "saida", sessao=1)
    assert m["sessao"] == 1 and (tmp_path / "saida" / m["zip"]).is_file()
    # salvar roda com a medida no ar: a linha pela metade do arquivo vivo fica como está
    assert (av / "g2_saidas/teste_8B_base_veredito.jsonl").read_bytes().endswith(b'"x')
    assert m["zip"].startswith("estado_") and m["zip"].endswith(".zip.bin")
    m2 = k.salvar_estado(sessao, tmp_path / "saida", sessao=1)       # o zip velho sai
    assert [p.name for p in (tmp_path / "saida").glob("estado_*")] == [m2["zip"]]
    assert set(m["arquivos"]) == {"data/processed/avaliacao/g2_saidas/teste_8B_base_veredito.jsonl",
                                  "data/processed/avaliacao/g2_dev_8B_base_it1.json",
                                  "data/processed/avaliacao/g2_acessos.jsonl"}
    nova = _repo(tmp_path / "s2", livro)
    k.restaurar_estado(tmp_path / "saida", nova, sessao_esperada=1)
    jsonl = nova / "data/processed/avaliacao/g2_saidas/teste_8B_base_veredito.jsonl"
    assert jsonl.read_bytes() == b'{"j": 0}' + FIM                # a linha cortada saiu
    assert (nova / k.LIVRO).read_bytes() == livro + b'{"tamanho": "8B"}' + FIM


def test_estado_velho_alterado_ou_livro_que_nao_continua_e_recusado(tmp_path):
    livro = b'{"tamanho": "1.7B"}' + FIM
    k.salvar_estado(_repo(tmp_path / "s1", livro), tmp_path / "saida", sessao=2)
    with pytest.raises(SystemExit, match="sessão 2"):
        k.restaurar_estado(tmp_path / "saida", _repo(tmp_path / "s0", livro), sessao_esperada=3)
    with pytest.raises(SystemExit, match="não é da 8B"):
        k.restaurar_estado(tmp_path / "saida", _repo(tmp_path / "s2", b'{"t": "outro"}' + FIM),
                           sessao_esperada=2)
    nome = json.loads((tmp_path / "saida/ESTADO.json").read_text(encoding="utf-8"))["zip"]
    (tmp_path / "saida" / nome).write_bytes(b"adulterado")
    with pytest.raises(SystemExit, match="não confere"):
        k.restaurar_estado(tmp_path / "saida", _repo(tmp_path / "s3", livro), sessao_esperada=2)
    manifesto = json.loads((tmp_path / "saida/ESTADO.json").read_text(encoding="utf-8"))
    assert manifesto["estado"] == "g2_8b"


def test_mapas_de_gguf_separam_metadado_de_tensor():
    a = {"metadados": {"general.name": "x", "tok": "1"}, "tensores": {"w": "a", "v": "b"}}
    b = {"metadados": {"general.name": "x", "tok": "2"}, "tensores": {"w": "a", "v": "b"}}
    assert k.comparar_mapas(a, b) == {"metadados_diferentes": ["tok"], "tensores_diferentes": []}
    b["tensores"]["v"] = "c"
    assert k.comparar_mapas(a, b)["tensores_diferentes"] == ["v"]


def test_livro_junta_por_registro_sem_perder_acesso():
    base = b'{"tamanho": "1.7B", "n": 1}' + FIM
    pacote_novo = base + b'{"tamanho": "1.7B", "n": 2}' + FIM          # o daqui cresceu
    estado = base + b'{"tamanho": "8B", "n": 3}' + FIM
    junto = k.juntar_livros(pacote_novo, estado)
    assert [r["n"] for r in k.registros(junto)] == [1, 2, 3]
    assert k.juntar_livros(junto, estado) == junto                       # idempotente
    with pytest.raises(SystemExit, match="não é da 8B"):
        k.juntar_livros(b"", estado)                                     # o 1.7B sumiria


def test_conferir_etapa_pelos_contadores(tmp_path):
    av = tmp_path / "data/processed/avaliacao"
    av.mkdir(parents=True)
    assert k.conferir_etapa("mmlu", tmp_path) == (False, "sem o agregado")
    (av / "g2_mmlu_teste_8B_base.json").write_text(json.dumps({"sem_previsao": 3}), encoding="utf-8")
    assert not k.conferir_etapa("mmlu", tmp_path)[0]
    ref = {"por_benchmark": {"ugphysics": {"amostras": 6}, "phybench": {"amostras": 4}}}
    (av / "g2_teste_1.7B_base_veredito.json").write_text(json.dumps(ref), encoding="utf-8")
    d = {"por_benchmark": {"ugphysics": {"amostras": 6, "fracoes": {"pass": 1.0}},
                           "phybench": {"amostras": 4, "fracoes": {"fail": 0.5, "error": 0.5}}}}
    (av / "g2_teste_8B_base_veredito.json").write_text(json.dumps(d), encoding="utf-8")
    assert k.conferir_etapa("g2_1", tmp_path) == (False, "phybench: amostras com ERROR")
    d["por_benchmark"]["phybench"]["fracoes"] = {"fail": 1.0}
    (av / "g2_teste_8B_base_veredito.json").write_text(json.dumps(d), encoding="utf-8")
    assert k.conferir_etapa("g2_1", tmp_path) == (True, "completa")
    i = {"amostras": {"n": 541 * 16}, "gulosa": {"n": 541}, "crua": {"n": 540}}
    (av / "g2_ifeval_teste_8B_base.json").write_text(json.dumps(i), encoding="utf-8")
    assert k.conferir_etapa("ifeval", tmp_path)[1].startswith("crua: 540")


def test_trazer_funde_so_a_8B_no_livro_daqui(tmp_path):
    sys.path.insert(0, str(RAIZ / "scripts"))
    import trazer_g2_8b as t

    base = b'{"tamanho": "1.7B", "n": 1}' + FIM
    kaggle = _repo(tmp_path / "kaggle", base + b'{"tamanho": "8B", "n": 9}' + FIM)
    av = kaggle / "data/processed/avaliacao"
    (av / "g2_saidas/teste_8B_base_veredito.jsonl").write_bytes(b'{"j": 0}' + FIM)
    (av / "g2_mmlu_teste_8B_base.json").write_text("{}", encoding="utf-8")
    ep = kaggle / "data/processed/kaggle_g2_evalplus/amostras"
    ep.mkdir(parents=True)
    (ep / "teste_8B_base_gulosa_mbpp.jsonl").write_bytes(b"{}" + FIM)
    k.salvar_estado(kaggle, tmp_path / "estado", sessao=3)
    aqui = _repo(tmp_path / "aqui", base + b'{"tamanho": "1.7B", "n": 2}' + FIM)   # cresceu
    r = t.trazer(tmp_path / "estado", aqui)
    assert r["linhas_novas_no_livro"] == 1 and r["sessao"] == 3
    assert [x["n"] for x in k.registros((aqui / k.LIVRO).read_bytes())] == [1, 2, 9]
    assert (aqui / "data/processed/kaggle_g2_evalplus_8B/amostras/teste_8B_base_gulosa_mbpp.jsonl").exists()
    assert not (aqui / "data/processed/kaggle_g2_evalplus").exists()
