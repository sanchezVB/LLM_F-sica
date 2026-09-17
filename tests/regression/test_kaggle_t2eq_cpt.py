"""Caminho B do ADR-0003: os dois braços têm de diferir SÓ em `p_equacao`, e usar as DUAS T4.

Três coisas que este arquivo tranca, e cada uma já custou caro em outro lugar do
projeto:

1. **Uma variável.** Se qualquer constante de orçamento pudesse divergir entre os
   braços, o resultado sairia com uma diferença sem nome dentro dele.
2. **As duas placas em TODOS os braços.** Conferido em 2026-09-17: a sessão sempre foi
   "T4 x2" e o projeto usava uma. Rodar um braço em duas e o outro em uma daria o mesmo
   modelo em expectativa e uma diferença não declarada entre eles — a lição do T1b2.
3. **A regra escrita antes, e a regra CERTA.** A primeira versão da regra do §2.3 usava
   a perda em tokens de equação como primária; token de equação é +0,1286 mais fácil
   que prosa SEM tratamento nenhum, então aquilo lia o formato do LaTeX como evidência.
   A primária é a diferença das diferenças.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "kaggle"))

import t2eq_cpt  # noqa: E402

CELULA = t2eq_cpt.CELULA
ORCAMENTO = ("CONTEXTO", "SEQUENCIAS", "ACUMULACAO", "TOKENS", "LR", "SEMENTE",
             "PROCESSOS", "LIMITE_HORAS", "BASE")


def _arvore(variante: str) -> ast.Module:
    return ast.parse(CELULA.replace("__VARIANTE__", variante))


def _constantes(variante: str, nomes: tuple[str, ...]) -> dict:
    out = {}
    for n in ast.walk(_arvore(variante)):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in nomes:
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out


def test_a_celula_e_python_valido():
    for variante in ("controle", "tratado"):
        _arvore(variante)


def test_os_dois_bracos_diferem_SO_em_p_equacao():
    assert _constantes("controle", ORCAMENTO) == _constantes("tratado", ORCAMENTO)
    assert len(_constantes("controle", ORCAMENTO)) == len(ORCAMENTO)
    p = _constantes("controle", ("P_EQUACAO",))["P_EQUACAO"]
    assert p == {"controle": "0.0", "tratado": "0.6"}


def test_o_treino_roda_nas_DUAS_placas_e_ABORTA_se_nao_houver():
    assert "torch.distributed.run" in CELULA and '"--standalone"' in CELULA
    assert 'f"--nproc_per_node={PROCESSOS}"' in CELULA
    assert "assert N_GPUS >= PROCESSOS" in CELULA, (
        "sem esta guarda um braço poderia rodar em uma placa e o outro em duas")
    assert _constantes("controle", ("PROCESSOS",)) == {"PROCESSOS": 2}


def test_o_lote_logico_e_o_dos_proxies_e_nao_depende_dos_processos():
    """`SEQUENCIAS * ACUMULACAO` = 64 sequências por passo, como no §2.3 de 48 M."""
    c = _constantes("controle", ORCAMENTO)
    assert c["SEQUENCIAS"] * c["ACUMULACAO"] == 64
    assert c["ACUMULACAO"] % c["PROCESSOS"] == 0, (
        "a acumulação do passo inteiro tem de se dividir pelos processos")
    assert "PASSOS = TOKENS // (CONTEXTO * SEQUENCIAS * ACUMULACAO)" in CELULA


def test_o_orcamento_cabe_na_sessao_e_a_guarda_de_horas_esta_ligada():
    c = _constantes("controle", ORCAMENTO)
    assert c["TOKENS"] == 400_000_000, (
        "0,6 B não cabe: ~10,7 h projetadas contra o teto de 9 h da sessão")
    assert 0 < c["LIMITE_HORAS"] <= 8.5
    assert '"--limite-horas", str(LIMITE_HORAS)' in CELULA


def test_a_LR_e_de_pre_treino_CONTINUADO():
    """1e-3 é a do treino do zero; num modelo convergido ela apaga o que ele sabe."""
    assert _constantes("controle", ("LR",))["LR"] <= 2e-4


def test_a_base_e_o_ModernBERT_e_o_treino_a_recebe():
    c = _constantes("controle", ("BASE",))
    assert c["BASE"] == "answerdotai/ModernBERT-base"
    assert '"--base", BASE' in CELULA


def test_a_fatia_TEM_de_declarar_o_id_de_mascara_do_tokenizer_da_base():
    """Mascarar com o id de outro tokenizer não levanta: a perda desce igual."""
    assert "especiais_do_tokenizer" in CELULA and "id_mascara" in CELULA
    assert "assert _man_fatia.get(\"especiais_do_tokenizer\")" in CELULA


def test_a_regra_esta_escrita_e_a_primaria_e_a_DIFERENCA_DAS_DIFERENCAS():
    assert "DIFERENÇA DAS DIFERENÇAS" in CELULA
    assert "+0,1286" in CELULA, "a linha de base sem tratamento tem de estar ao lado"
    for desfecho in ("POSITIVA", "NEGATIVA", "IC CRUZANDO ZERO"):
        assert desfecho in CELULA, f"o desfecho {desfecho} não está escrito"
    assert "fracao_tratada" in CELULA and "INVÁLIDO" in CELULA
    assert "print(REGRA)" in CELULA, "a regra tem de sair no log do run"
