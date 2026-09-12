"""O Kaggle aceita 2 sessões de GPU, e recusa a terceira SAINDO COM CÓDIGO 0.

Medido em 2026-09-11: com o T1f e o braço A do T2a rodando, o push do braço E
imprimiu `Kernel push error: Maximum batch GPU session count of 2 reached.` e o
`publicar_kaggle.py` respondeu `✅ enviado` logo abaixo. O notebook existe, a versão
foi criada, e a EXECUÇÃO não foi enfileirada — que é a única parte que importa.

É a pior forma da falha silenciosa: quem lê "enviado" espera um resultado que nunca
vem, e só descobre horas depois ao procurar a saída.

Este arquivo fixa as duas metades do conserto: o publicador RECUSA chamar isso de
enviado, e a fila espera uma vaga pela CAUSA (status dos kernels) em vez do relógio.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

PUBLICAR = (RAIZ / "scripts/publicar_kaggle.py").read_text(encoding="utf-8")
FILA = (RAIZ / "scripts/enfileirar_kaggle.py").read_text(encoding="utf-8")


def test_sessao_esgotada_NAO_passa_por_enviado():
    """⚠️ A CLI sai com 0 e imprime o erro. Sem a string na lista, o script
    anuncia sucesso sobre um experimento que não vai rodar."""
    arvore = ast.parse(PUBLICAR)
    falhas = next(
        ast.literal_eval(n.value) for n in ast.walk(arvore)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "FALHAS_SILENCIOSAS"
                for t in n.targets))
    linha_real = "Kernel push error: Maximum batch GPU session count of 2 reached."
    assert any(f in linha_real for f in falhas), (
        f"nenhuma das {len(falhas)} entradas casa com a mensagem medida: "
        f"{linha_real!r}")


def test_a_lista_pega_qualquer_erro_de_push():
    """A mensagem exata pode mudar de forma; o prefixo que a CLI usa para erro de
    push não deve passar por sucesso em nenhuma variante."""
    arvore = ast.parse(PUBLICAR)
    falhas = next(
        ast.literal_eval(n.value) for n in ast.walk(arvore)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "FALHAS_SILENCIOSAS"
                for t in n.targets))
    assert "Kernel push error" in falhas


def test_a_fila_espera_pela_CAUSA_e_nao_pelo_relogio():
    """Esperar "N horas" seria adivinhar. E um kernel que morre aos 2 min — como
    o T1f morreu na primeira tentativa — libera a vaga na hora."""
    assert "--esperar" in FILA
    arvore = ast.parse(FILA)
    fn = next(n for n in ast.walk(arvore)
              if isinstance(n, ast.FunctionDef) and n.name == "ocupado")
    corpo = ast.unparse(fn)
    assert "kernels" in corpo and "status" in corpo, (
        "a fila não consulta o status dos kernels")


def test_consulta_que_FALHA_nao_conta_como_vaga_livre():
    """⚠️ `None` e `False` são coisas diferentes. Uma consulta que falhou (rede,
    API fora) não é vaga: empurrar nesse estado gastaria a tentativa e receberia
    exatamente a recusa que esta fila existe para evitar."""
    import enfileirar_kaggle as fila

    arvore = ast.parse(FILA)
    main = next(n for n in ast.walk(arvore)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    corpo = ast.unparse(main)
    assert "v is False" in corpo, (
        "a condição de vaga aceita valores que não são `False` — uma consulta "
        "falhada passaria por vaga livre")
    assert "QUEUED" in fila.OCUPADOS and "RUNNING" in fila.OCUPADOS


def test_a_fila_NAO_tenta_de_novo_apos_falha_de_conteudo():
    """Um push que falha por conteúdo falharia igual daqui a uma hora, e repetir
    só esconderia a causa."""
    assert "tentar de novo" in FILA
    assert "falharia igual daqui" in FILA


def test_a_fila_DESISTE_em_vez_de_esperar_para_sempre():
    assert "--limite-horas" in FILA
    assert "Não vou esperar indefinidamente" in FILA
