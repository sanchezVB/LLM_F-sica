"""O estrato "corpo": as guardas que conferem que a resposta está na passagem e não no resumo.

A leitura de quem escreve é a conferência principal; estas guardas pegam o que é mecânico —
um número que já está no resumo, ou um número que não está na passagem.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

import perguntas_corpo as pc  # noqa: E402


def contar(textos: list[str]) -> list[int]:
    return [len(t.split()) for t in textos]


def test_numeros_normaliza_a_virgula_e_ignora_digito_sozinho():
    assert pc.numeros("cerca de 3,2 meV a 150 K, em 2 amostras") == {"3.2", "150"}


def test_numero_que_ja_esta_no_resumo_derruba_o_item():
    assert pc.guarda_numerica("Cerca de 3,2 meV.", "a gap of 3.2 meV", "we find 3.2 meV") == "no_resumo"


def test_numero_que_nao_esta_na_passagem_derruba_o_item():
    assert pc.guarda_numerica("Cerca de 4,5 meV.", "a gap opens", "we find 3.2 meV") == "fora_da_passagem"


def test_numero_so_na_passagem_passa():
    assert pc.guarda_numerica("3,2 meV a 150 K.", "measured at 150 K", "3.2 meV at 150 K") == "ok"


def test_gabarito_sem_numero_fica_registrado_como_tal():
    assert pc.guarda_numerica("Ela diminui.", "x", "y") == "sem_numero"


def _doc() -> str:
    def par(marca: str) -> str:
        return " ".join(f"{marca}{i}" for i in range(120)) + "."

    return (r"\section{Introduction}" + par("i") + "\n\n" + par("j") + "\n\n" + par("k")
            + r"\section{Setup}" + par("a") + "\n\n" + par("b") + "\n\n" + par("c") + "\n\n" + par("d")
            + r"\section{Conclusions}" + par("z") + "\n\n" + par("y") + "\n\n" + par("w"))


def test_a_passagem_sai_do_corpo_e_de_uma_secao_so():
    p = pc.passagem_de("1234.5678", _doc(), contar)
    assert p["secao"] == "Setup" and p["n_tokens"] >= pc.MIN_TOKENS_PASSAGEM
    assert "i0" not in p["passagem"] and "z0" not in p["passagem"]


def test_a_passagem_e_a_mesma_a_cada_sorteio_e_presa_ao_artigo():
    a = [pc.passagem_de("1234.5678", _doc(), contar)["primeiro_trecho"] for _ in range(3)]
    assert len(set(a)) == 1
    outros = {pc.passagem_de(f"2000.{i:05d}", _doc(), contar)["primeiro_trecho"] for i in range(30)}
    assert len(outros) == 2          # as duas janelas da seção, cada artigo com a sua


def test_artigo_sem_corpo_bastante_nao_tem_passagem():
    assert pc.passagem_de("1", r"\section{Introduction} curto demais.", contar) is None
