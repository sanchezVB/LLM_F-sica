"""Qual checkpoint é "o do sistema" mora num lugar só.

Até 2026-09-08 o recuperador era a string `models/phiemb-minilm-melhor` copiada
como default em três scripts, e o reordenador era outra copiada em dois. Trocar o
recuperador significava editar todos e acertar todos.

E há uma armadilha específica: **um quarto lugar NÃO deve mudar.** O
`avaliar_encoders.py` lista o `phiemb-minilm-melhor` entre os candidatos do G1 —
lá ele é um ponto da curva de volume, evidência de uma medição passada, e não "o
recuperador do sistema". Trocar aquele caminho junto com os outros apagaria uma
comparação.

Estes testes fixam a distinção, e a guarda que impede instalar um modelo cujo
número do portão veio de um protocolo com teto abaixo de 1,0 — que é como o G1.2
"passou" por +0,003 em 2026-08-27.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from conftest import so_codigo_de  # noqa: E402
from phifm.core.modelos import RECUPERADOR, REORDENADOR  # noqa: E402

# Os scripts que consomem "o modelo do sistema" e devem importar a constante.
CONSUMIDORES = (
    "scripts/avaliar_t1b.py",
    "scripts/minerar_do_recuperador.py",
    "scripts/minerar_negativos.py",
)


def test_os_consumidores_importam_a_constante_e_nao_o_caminho():
    """⚠️ A asserção central. Ver a docstring do módulo.

    Com a string copiada, trocar o recuperador é uma edição em três arquivos que
    precisam concordar — e o sintoma de discordarem é uma cadeia medida com dois
    recuperadores diferentes, sem nada avisando.
    """
    for caminho in CONSUMIDORES:
        fonte = so_codigo_de(RAIZ / caminho)
        assert "RECUPERADOR" in fonte, f"{caminho} não usa a constante"
        assert RECUPERADOR not in fonte, (
            f"{caminho} ainda traz o caminho literal {RECUPERADOR!r}; ele tem de "
            "vir de `phifm.core.modelos`")


def test_o_reordenador_do_sistema_tambem_vem_da_constante():
    fonte = so_codigo_de(RAIZ / "scripts/avaliar_t1b.py")
    assert "REORDENADOR" in fonte
    assert REORDENADOR not in fonte


def test_a_comparacao_do_G1_mantem_o_caminho_LITERAL():
    """⚠️ O lugar que NÃO deve seguir a constante.

    No `avaliar_encoders.py` este checkpoint é um ponto da curva de volume —
    evidência de que um treino de 400 mil pares sobre MiniLM deu nDCG@10 0,5246.
    Se ele passasse a apontar para "o recuperador do sistema", instalar um modelo
    novo mudaria retroativamente o que a tabela do G1 compara, e uma linha da
    curva desapareceria.
    """
    fonte = so_codigo_de(RAIZ / "scripts/avaliar_encoders.py")
    # ⚠️ Afirma a PROPRIEDADE, não o valor atual da constante. A primeira versão
    # deste teste checava `RECUPERADOR in fonte` — e isso reprovaria assim que o
    # recuperador do sistema mudasse, que é justamente o evento que ele existe
    # para tornar seguro.
    assert "RECUPERADOR" not in fonte, (
        "o ponto da curva virou referência à constante; instalar um modelo novo "
        "passaria a apagar uma linha da comparação do G1")
    assert "models/phiemb-minilm-melhor" in fonte, (
        "o checkpoint de 400 mil pares saiu da comparação do G1; ele é a "
        "evidência de que aquele volume dá nDCG@10 0,5246")


def _instalar(tmp_path: Path, *extra: str, de: Path | None = None):
    return subprocess.run(
        [sys.executable, str(RAIZ / "scripts/instalar_phiemb.py"),
         "--de", str(de or tmp_path / "de"), "--para", str(tmp_path / "para"),
         "--nota", "teste", *extra],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=RAIZ, env={**os.environ, "PYTHONUTF8": "1",
                       "PYTHONPATH": str(RAIZ / "src")})


def _modelo_falso(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for n in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (d / n).write_text("{}", encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"\x00" * 16)
    (d / "melhor.json").write_text(json.dumps({
        "passo": 23400, "ndcg_10": 0.6804, "recall_1": 0.498,
        "base": "sentence-transformers/all-MiniLM-L6-v2",
        "criterio": "nDCG@10 — a métrica do portão G1"}), encoding="utf-8")
    return d


def test_instalar_recusa_teto_abaixo_de_um(tmp_path):
    """⚠️ É a guarda que o protocolo quebrado de agosto teria disparado.

    Naquele pool o teto de um modelo perfeito era nDCG@10 0,7562: 62% dos itens
    tinham alvo repetido, colunas byte-idênticas empatam no cosseno e o desempate
    do `argsort` é arbitrário. Um nDCG@10 gravado sem o teto não diz se o limite
    é o modelo ou a régua.
    """
    de = _modelo_falso(tmp_path / "de")
    r = _instalar(tmp_path, "--copiar", "--ndcg-do-portao", "0.4657",
                  "--teto-do-portao", "0.7562", de=de)
    assert r.returncode != 0
    saida = r.stdout + r.stderr
    assert "0.7562" in saida and "desempate" in saida


def test_instalar_exige_o_melhor_json(tmp_path):
    """Sem ele não se sabe em que passo o checkpoint foi eleito nem por qual
    critério — e o critério já divergiu neste projeto: o `-gc-melhor` foi eleito
    por MRR quando o portão pede nDCG@10."""
    de = _modelo_falso(tmp_path / "de")
    (de / "melhor.json").unlink()
    r = _instalar(tmp_path, "--copiar", "--ndcg-do-portao", "0.6",
                  "--teto-do-portao", "1.0", de=de)
    assert r.returncode != 0
    assert "melhor.json" in (r.stdout + r.stderr)


def test_instalar_grava_o_teto_e_avisa_que_a_constante_nao_mudou(tmp_path):
    """Instalar os pesos NÃO troca o recuperador do sistema.

    A constante é uma edição deliberada, porque trocar exige remedir a cadeia do
    T1b/T1c — cuja referência (nDCG 0,1666) foi medida com o recuperador antigo.
    Um script que trocasse os dois de uma vez mediria duas coisas.
    """
    de = _modelo_falso(tmp_path / "de")
    r = _instalar(tmp_path, "--copiar", "--ndcg-do-portao", "0.6026",
                  "--teto-do-portao", "1.0", de=de)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RECUPERADOR" in r.stdout and RECUPERADOR in r.stdout

    man = next((tmp_path / "para").glob("*manifesto*"), None) or next(
        (tmp_path / "para").glob("*.json"), None)
    assert man is not None
    # O manifesto de etapa é gravado por `gravar_manifesto_etapa`; o que importa
    # aqui é que o teto e a lacuna de registro estejam nele.
    textos = "\n".join(p.read_text(encoding="utf-8")
                       for p in (tmp_path / "para").rglob("*.json"))
    assert "teto_do_protocolo" in textos
    assert "o_que_a_medicao_NAO_diz" in textos
    assert "lacuna_de_registro" in textos
