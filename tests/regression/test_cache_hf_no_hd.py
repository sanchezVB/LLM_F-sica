"""O cache do HuggingFace vai para o HD, e isso depende da ORDEM dos imports.

Medido em 2026-09-17: 6,8 GB de modelos no SSD, em `C:\\Users\\User\\.cache\\huggingface`.
O `.env` declarava `HF_HOME` no HD e nada o carregava. Agora `phifm/__init__.py` o
carrega — mas o `huggingface_hub` lê o caminho no import, então `phifm` tem de vir
ANTES de `transformers` em todo script. Oito não vinham.

Por AST: a suíte rápida roda sem torch nem transformers.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm import hf_home_do_env  # noqa: E402

HF = ("transformers", "sentence_transformers", "huggingface_hub", "datasets")


def test_todo_script_CHAMA_o_cache_hf_ANTES_do_huggingface():
    """⚠️ A CHAMADA, e não só o import: o `ruff --fix` ordena imports em grupos e empurrou
    `import phifm` para depois do `transformers` na primeira versão deste conserto, com a
    suíte verde. Uma chamada entre os imports quebra o bloco que o isort reordena."""
    ruins = []
    for p in sorted((RAIZ / "scripts").glob("*.py")):
        prim_hf = prim_chamada = None
        for no in ast.parse(p.read_text(encoding="utf-8")).body:
            mods = ([a.name for a in no.names] if isinstance(no, ast.Import)
                    else [no.module] if isinstance(no, ast.ImportFrom) and no.module
                    else [])
            if prim_hf is None and any(m.split(".")[0] in HF for m in mods):
                prim_hf = no.lineno
            if (prim_chamada is None and isinstance(no, ast.Expr)
                    and isinstance(no.value, ast.Call)
                    and ast.unparse(no.value.func).endswith("cache_hf_no_hd")):
                prim_chamada = no.lineno
        if prim_hf is not None and not (prim_chamada is not None and prim_chamada < prim_hf):
            ruins.append(p.name)
    assert not ruins, (
        f"{ruins} importam o HuggingFace sem chamar `cache_hf_no_hd()` antes: o cache "
        "iria para o padrão do usuário, no SSD")


def test_le_HF_HOME_do_env(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comentário\nOUTRA=1\nHF_HOME=D:/x/cache\n", encoding="utf-8")
    assert hf_home_do_env(env) == "D:/x/cache"


def test_aceita_export_e_aspas(tmp_path):
    env = tmp_path / ".env"
    env.write_text('export HF_HOME="D:/y"\n', encoding="utf-8")
    assert hf_home_do_env(env) == "D:/y"


def test_sem_env_ou_sem_chave_nao_define_nada(tmp_path):
    assert hf_home_do_env(tmp_path / "nao_existe") is None
    env = tmp_path / ".env"
    env.write_text("OUTRA=1\n", encoding="utf-8")
    assert hf_home_do_env(env) is None


def test_o_pacote_usa_setdefault_e_nao_sobrescreve():
    """Uma variável já definida no ambiente manda — no Kaggle, num CI, ou à mão."""
    fonte = (RAIZ / "src" / "phifm" / "__init__.py").read_text(encoding="utf-8")
    assert 'os.environ.setdefault("HF_HOME"' in fonte
    assert 'os.environ["HF_HOME"] =' not in fonte
