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


def test_NENHUMA_avaliacao_constroi_o_device_de_uma_variavel():
    """⚠️ O teste acima fixava o NOME `avaliar_um`, e por isso não viu a regressão
    voltar. Na junção com o branch do Mac (2026-09-16), `avaliar_carregado` — o
    novo caminho único da métrica — e as três avaliações do §11.2 nasceram de uma
    base anterior a `c35eebb` e traziam `torch.device(dispositivo)` em CINCO
    lugares, com a suíte verde.

    Agora a guarda é sobre o comportamento, em todo `eval/`: `torch.device(...)`
    só com literal. Uma string vinda da CLI passa pelo resolvedor."""
    ofensores = []
    for arq in sorted((RAIZ / "src/phifm/eval").rglob("*.py")):
        for no in ast.walk(ast.parse(arq.read_text(encoding="utf-8"))):
            if (isinstance(no, ast.Call) and ast.unparse(no.func) == "torch.device"
                    and no.args and not isinstance(no.args[0], ast.Constant)):
                ofensores.append(f"{arq.relative_to(RAIZ).as_posix()}:{no.lineno}")
    assert not ofensores, (
        f"torch.device(<variável>) em {ofensores}: `dml` quebra assim. Use "
        "`phifm.training.embedding.escolher_dispositivo`.")


def test_ha_UMA_binomial_exata_so():
    """Os dois lados da junção limparam a mesma duplicação em lugares diferentes —
    `main` no `mcnemar_em`, o Mac em `statistics.proporcao` —, e juntos deixariam
    duas cópias. A conta mora em `proporcao`."""
    copias = []
    for arq in sorted((RAIZ / "src").rglob("*.py")):
        if "math.comb(" in arq.read_text(encoding="utf-8"):
            copias.append(arq.relative_to(RAIZ).as_posix())
    assert copias == ["src/phifm/eval/statistics/proporcao.py"], copias
