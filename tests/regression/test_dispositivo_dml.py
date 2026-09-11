"""`--dispositivo dml` quebrava o avaliador de encoders, e é o caminho rápido daqui.

`torch.device("dml")` LEVANTA: "dml" não é um tipo de device do torch, é um backend
que vem de `torch_directml.device()`. Quem resolve a string é
`phifm.training.embedding.escolher_dispositivo`, e `eval/encoders.py` não a usava.

Medido em 2026-09-11 num ensaio do T2a: `avaliar_encoders.py --dispositivo dml`
morria no primeiro modelo, depois de carregar 133 mil pares de validação e sortear
o pool. A máquina do dono do projeto é uma RX 7600 — `dml` é a única bandeira que
valia a pena usar ali, e era a que não funcionava.

⚠️ Por AST e não por import: a suíte rápida roda sem torch.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

ENCODERS = (RAIZ / "src/phifm/eval/encoders.py").read_text(encoding="utf-8")
CLI = (RAIZ / "scripts/avaliar_encoders.py").read_text(encoding="utf-8")


def _corpo(fonte: str, nome: str) -> ast.FunctionDef:
    for no in ast.walk(ast.parse(fonte)):
        if isinstance(no, ast.FunctionDef) and no.name == nome:
            return no
    raise AssertionError(f"{nome} não existe mais em encoders.py")


def test_avaliar_um_resolve_o_dispositivo_pelo_RESOLVEDOR():
    """`torch.device(dispositivo)` cru recusa "dml" e derruba a medição."""
    fn = _corpo(ENCODERS, "avaliar_um")
    chamadas = {ast.unparse(n.func) for n in ast.walk(fn)
                if isinstance(n, ast.Call)}
    assert "escolher_dispositivo" in chamadas, (
        "avaliar_um voltou a construir o device na mão; `dml` quebra assim")
    assert "torch.device" not in chamadas, (
        "`torch.device(<string da CLI>)` não aceita 'dml'")


def test_a_CLI_oferece_dml_e_por_isso_ele_tem_de_funcionar():
    """A bandeira e o resolvedor têm de concordar: uma escolha que a CLI oferece
    e o código recusa é pior que uma que não existe — ela falha depois de o
    trabalho caro já ter sido feito."""
    assert '"dml"' in CLI
    arvore = ast.parse(CLI)
    escolhas = None
    for no in ast.walk(arvore):
        if not (isinstance(no, ast.Call)
                and ast.unparse(no.func).endswith("add_argument")):
            continue
        if no.args and ast.unparse(no.args[0]) != "'--dispositivo'":
            continue
        for kw in no.keywords:
            if kw.arg == "choices":
                escolhas = ast.literal_eval(kw.value)
    assert escolhas is not None, "--dispositivo perdeu as `choices`"
    assert set(escolhas) <= {"cpu", "dml", "cuda", "auto"}, escolhas
