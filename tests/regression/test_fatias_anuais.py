"""Set grande do OAI-PMH tem de ser fatiado, e a fatiagem não pode perder registro.

Regressão de 2026-08-13. A coleta de negativos de `math` não baixou UM registro em
dez tentativas. Medido, com o endpoint saudável (`Identify` em 0,3 s):

    set inteiro       503 após 183 s
    fatia de 5 anos   503 após 183 s
    fatia de 1 ano    200 após  56 s, 1.300 registros e resumptionToken
    fatia de 1 mês    200 após  40 s

O `math` é grande demais para o arXiv montar o conjunto de resultados dentro do
timeout dele. Fatiar por ano resolve.

## O risco que estes testes cobrem

Fatiar troca «uma requisição que falha» por «22 requisições que precisam cobrir
exatamente o mesmo conjunto». Duas formas de errar em silêncio:

**Lacuna.** Uma fatia terminando em 31/12 e a seguinte começando em 02/01 perde um
dia por ano. Vinte e dois dias de metadados a menos, e nada no log diria.

**Sobreposição.** Fatias que se cruzam trazem o mesmo registro duas vezes. A
dedup por `arxiv_id` salvaria o dado, mas o número de "registros coletados" ficaria
inflado e as contagens do manifesto deixariam de fechar.

⚠️ `from`/`until` filtram por DATESTAMP (quando o metadado foi criado ou
alterado), não pela data de submissão. Isso não abre lacuna: cada registro tem
exatamente um datestamp, então fatias contíguas cobrindo
[earliestDatestamp, hoje] particionam o set. Um paper de 1995 revisado em 2024
cai na fatia de 2024, e cai uma vez só.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))


def carregar():
    """`harvest_negativos.py` é script, não módulo do pacote."""
    spec = importlib.util.spec_from_file_location(
        "harvest_negativos", RAIZ / "scripts" / "harvest_negativos.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hn = carregar()


# ─── cobertura: sem lacuna, sem sobreposição ─────────────────────────────────

def test_comeca_no_primeiro_datestamp_do_repositorio():
    """Antes disso são fatias vazias; depois, registros perdidos.

    `2005-09-16` vem do verbo `Identify` do arXiv, não de palpite.
    """
    assert hn.fatias_anuais()[0][0] == hn.PRIMEIRO_DATESTAMP == "2005-09-16"


def test_termina_hoje():
    assert hn.fatias_anuais()[-1][1] == date.today().isoformat()


def test_fatias_sao_contiguas():
    """O fim de uma e o começo da seguinte diferem em exatamente um dia."""
    fs = hn.fatias_anuais()
    for (_, fim), (inicio, _) in zip(fs, fs[1:], strict=False):  # comprimentos diferem por construcao
        d_fim, d_ini = date.fromisoformat(fim), date.fromisoformat(inicio)
        assert d_ini - d_fim == timedelta(days=1), (
            f"lacuna ou sobreposição entre {fim} e {inicio}")


def test_nenhuma_fatia_invertida():
    for de, ate in hn.fatias_anuais():
        assert date.fromisoformat(de) <= date.fromisoformat(ate)


def test_uma_fatia_por_ano_civil():
    fs = hn.fatias_anuais()
    anos = [f[0][:4] for f in fs]
    assert len(anos) == len(set(anos)), "ano repetido em duas fatias"
    assert anos == sorted(anos)


def test_fatia_de_um_ano_e_o_tamanho_medido():
    """A medição diz que 1 ano passa e 5 anos dão 503. Fatia anual, não plurianual."""
    for de, ate in hn.fatias_anuais()[1:-1]:      # ignora a primeira e a última, parciais
        assert de[:4] == ate[:4], f"fatia cruzando anos: {de}..{ate}"


def test_inicio_customizado_e_respeitado():
    fs = hn.fatias_anuais("2019-03-05")
    assert fs[0] == ("2019-03-05", "2019-12-31")
    assert len(fs) == date.today().year - 2019 + 1


# ─── quais sets são fatiados ─────────────────────────────────────────────────

def test_math_e_fatiado_e_os_outros_nao():
    """Fatiar quem não precisa custa 22 requisições extras por set, de graça.

    Os três sets originais foram coletados inteiros com sucesso em 2026-08-07 —
    não há motivo para fatiá-los, e fatiar mudaria o caminho dos shards já em
    disco.
    """
    assert {"math"} == hn.SETS_FATIADOS
    for s in ("cs", "q-bio", "econ"):
        assert s not in hn.SETS_FATIADOS


# ─── o leitor tem de ver as subpastas por ano ────────────────────────────────

def test_resumir_conta_shards_em_subpasta_de_ano(tmp_path):
    """`glob` de um nível reportaria ZERO para um set fatiado.

    Seria o pior tipo de falha: o relatório diria que a coleta não trouxe nada
    quando ela trouxe tudo, e a reação natural seria coletar de novo.
    """
    for ano, prim in (("2019", "math.AG"), ("2020", "math-ph")):
        d = tmp_path / ano
        d.mkdir()
        pl.DataFrame({"primary_category": [prim] * 5}).write_parquet(d / "part-000.parquet")

    baixados, uteis = hn.resumir(tmp_path)
    assert baixados == 10, "não achou os shards nas subpastas de ano"
    # `math-ph` é Física: 5 dos 10 não servem como negativo.
    assert uteis == 5


def test_resumir_ainda_conta_layout_plano(tmp_path):
    """Os sets antigos guardam os shards direto na pasta do set."""
    pl.DataFrame({"primary_category": ["cs.LG"] * 4}).write_parquet(
        tmp_path / "part-000.parquet")
    assert hn.resumir(tmp_path) == (4, 4)


def test_montar_binario_le_subpastas_de_ano(tmp_path):
    """O glob do dataset de treino tem de ser recursivo.

    Se não for, o classificador treina sem o negativo de `math` — que é
    exatamente o negativo que motivou toda esta coleta — e nada avisa.
    """
    from phifm.corpus.filter.classifier import montar_binario

    spine = tmp_path / "spine.parquet"
    pl.DataFrame({"arxiv_id": ["fis/1"], "title": ["T"],
                  "abstract": ["a" * 40]}).write_parquet(spine)

    negs = tmp_path / "negativos"
    (negs / "cs").mkdir(parents=True)                  # layout plano
    pl.DataFrame({"arxiv_id": ["cs/1"], "title": ["Transformers"],
                  "abstract": ["b" * 40]}).write_parquet(negs / "cs" / "000.parquet")
    (negs / "math" / "2019").mkdir(parents=True)       # layout fatiado
    pl.DataFrame({"arxiv_id": ["math/1"], "title": ["Sheaf cohomology"],
                  "abstract": ["c" * 40]}).write_parquet(negs / "math" / "2019" / "000.parquet")

    df = montar_binario(spine, negs, max_por_classe=100)
    negativos = set(df.filter(pl.col("is_physics") == "nao_fisica")["arxiv_id"])
    assert negativos == {"cs/1", "math/1"}, f"faltou o fatiado: {negativos}"
