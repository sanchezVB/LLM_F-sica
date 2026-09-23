"""T1g: o GTE-base@1M é a receita do T1f com UM número mudado — o volume.

O que este arquivo tranca:

1. **Os hiperparâmetros são os do T1f**, lidos pela AST das duas células. Se o lote
   ou a semente divergissem, o GTE@1M deixaria de continuar a curva do GTE@200k e
   @400k, e o diagnóstico contra o MiniLM@1,5M mediria receita junto com base.
2. **O volume é 1 M e CHEGA ao treino** (`--max-pares`). Sem a bandeira, o treino
   usaria os 1,5 M do dataset: ~9,6 h, que não cabem na cota da semana.
3. **O dataset é o do T1a 1,5 M**, conferido pela célula antes de treinar.
4. **A regra está escrita antes**, com os três desfechos, o alvo de 0,6223 e o custo
   de serviço de 4,4× — e declara que NÃO é ablação de uma variável.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "kaggle"))

import t1f_base_gte  # noqa: E402
import t1g_gte_1m  # noqa: E402

from phifm.core.kaggle import obter  # noqa: E402

CELULA = t1g_gte_1m.CELULA
NOMES = ("BASE", "LOTE", "MAX_TOKENS", "SEMENTE", "N_CANDIDATOS", "PASSOS_AVAL")


def _constantes(celula: str, nomes: tuple[str, ...]) -> dict:
    out = {}
    for n in ast.walk(ast.parse(celula)):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in nomes:
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out


def test_a_celula_e_python_valido():
    ast.parse(CELULA)


def test_os_hiperparametros_sao_os_do_T1f():
    do_t1g = _constantes(CELULA, NOMES)
    assert len(do_t1g) == len(NOMES), do_t1g
    assert do_t1g == _constantes(t1f_base_gte.CELULA, NOMES)


def test_o_volume_e_1M_e_CHEGA_ao_treino():
    """Sem `--max-pares`, o dataset de 1,5 M iria inteiro: ~9,6 h, fora da cota."""
    assert _constantes(CELULA, ("MAX_PARES",)) == {"MAX_PARES": 1_000_000}
    assert '"--max-pares", MAX_PARES' in CELULA
    assert obter("t1g").max_pares == 1_000_000


def test_o_dataset_e_o_do_T1a_15M_e_e_conferido():
    g, fonte = obter("t1g"), obter("t1a15")
    assert g.reusa_dados_de == "t1a15"
    for campo in ("slug_dados", "titulo_dados", "pacote", "arquivos"):
        assert getattr(g, campo) == getattr(fonte, campo), campo
    assert 'man["experimento"] == "t1a15"' in CELULA
    g.conferir()


def test_os_hashes_e_a_assinatura_sao_conferidos_ANTES_de_treinar():
    treino = CELULA.index("_rodar([")
    assert CELULA.index('esperado["blake3"]') < treino
    assert CELULA.index("ASSINATURA_ESPERADA") < treino


def test_a_regra_esta_escrita_ANTES_com_o_alvo_e_o_custo():
    r = " ".join(t1g_gte_1m.CELULA.split())
    for exigido in ("A REGRA, escrita ANTES", "GTE À FRENTE", "EMPATE",
                    "SISTEMA À FRENTE", "0,6223", "4,4x", "Não é ablação de uma variável",
                    "bootstrap pareado por ITEM"):
        assert exigido in r, exigido
