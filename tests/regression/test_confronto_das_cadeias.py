"""O confronto que a regra pre-registrada nomeia tem de ser CALCULADO.

O T1b2 de 2026-09-08 pre-registrou: "se PhiEmb+PhiRank VENCER PhiEmb+BM25+PhiRank
no pareado, o BM25 sai da composicao; se EMPATAR, sai tambem". A rodada mediu as
duas cadeias -- e pareou as duas CONTRA A FUSAO, que nao e o confronto da regra.

Duas comparacoes contra um terceiro sistema nao sao a comparacao entre elas. Na
rodada, as cadeias deram p=0,086 e p=0,149 contra a fusao, e nenhum desses dois
numeros e o veredito que a regra pedia. Deu para decidir por aritmetica (ver o
ultimo teste), mas isso foi sorte do tamanho do efeito, nao desenho.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval.hibrido import mcnemar_em  # noqa: E402

FONTE = (RAIZ / "scripts" / "avaliar_t1b.py").read_text(encoding="utf-8")


def _chamadas_de_mcnemar() -> list[ast.Call]:
    return [n for n in ast.walk(ast.parse(FONTE))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "mcnemar_em"]


def _nomes(c: ast.Call) -> tuple[str | None, str | None]:
    def nome(x: ast.expr) -> str | None:
        return x.id if isinstance(x, ast.Name) else None
    return (nome(c.args[0]) if c.args else None,
            nome(c.args[1]) if len(c.args) > 1 else None)


def test_as_duas_cadeias_sao_pareadas_UMA_CONTRA_A_OUTRA():
    """A assercao central: existe um McNemar cujos dois lados sao as duas cadeias.

    Sem ele, a regra do BM25 nao tem veredito -- so dois vereditos contra um
    terceiro sistema, que e o que a rodada de 2026-09-08 produziu.
    """
    pares = {frozenset(_nomes(c)) for c in _chamadas_de_mcnemar()}
    assert frozenset({"pos_rank_denso", "pos_rank"}) in pares, (
        "nenhum pareado confronta a cadeia SEM a fusao com a cadeia COM ela; "
        f"os pares presentes sao {sorted(map(sorted, pares))}")


def test_o_confronto_nao_passa_pela_fusao():
    """Se um dos lados fosse `pos_rrf`, seria de novo a comparacao antiga com
    outro nome -- e o nome novo esconderia que a regra segue sem veredito."""
    confrontos = [c for c in _chamadas_de_mcnemar()
                  if frozenset(_nomes(c)) == frozenset({"pos_rank_denso",
                                                        "pos_rank"})]
    assert confrontos
    for c in confrontos:
        assert "pos_rrf" not in _nomes(c)


def test_o_confronto_sai_no_artefato_fora_da_chave_da_fusao():
    """`pareado_contra_a_fusao` promete que a referencia e a fusao. Um par sem
    ela dentro dessa chave faria o nome mentir para quem le o JSON depois."""
    assert '"confronto_das_cadeias": confronto' in FONTE
    i = FONTE.index('"pareado_contra_a_fusao": pareados')
    j = FONTE.index('"confronto_das_cadeias": confronto')
    assert i != j


def test_um_liquido_de_5_em_2000_NAO_alcanca_significancia():
    """Por que a decisao de 2026-09-08 foi segura mesmo com o teste faltando.

    A cadeia com fusao acertou 594 das 2.000 no top-10 e a sem fusao 589: um
    liquido de 5. O McNemar exato so olha os discordantes, e o menor `p`
    possivel para um liquido de 5 e o caso extremo 5 a 0 -- que da 0,0625.
    Qualquer outra estrutura da mais. Logo a fusao NAO podia vencer o pareado
    sob NENHUMA estrutura de discordantes, e a regra ("so fica se vencer")
    resolvia sem o numero.

    O teste guarda o argumento em forma executavel: se um dia alguem afirmar que
    um liquido pequeno "provavelmente venceria", isto responde.
    """
    liquido = 594 - 589
    ps = []
    for perde in range(0, 300):
        vence = perde + liquido
        # `a` acerta em `perde` consultas exclusivas, `b` em `vence`.
        a = [0] * perde + [99] * vence + [0] * (2000 - perde - vence)
        b = [99] * perde + [0] * vence + [0] * (2000 - perde - vence)
        ps.append(mcnemar_em(a, b, 10, "sem fusao", "com fusao")["p"])
    assert min(ps) > 0.05, f"minimo {min(ps):.4f}"
    assert abs(min(ps) - 0.0625) < 1e-9, (
        f"o caso extremo 5 a 0 deveria dar 0,0625; deu {min(ps)}")
