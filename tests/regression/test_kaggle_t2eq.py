"""O braço TRATADO do §2.3 tem de ser o controle com UMA variável trocada.

O controle — o braço E do T2a — já treinou. Reaproveitá-lo só é honesto se a célula
do tratado for a dele com trocas declaradas, e nenhuma outra. Estes testes travam
isso por construção: a célula publicada é `derivar(célula do T2a)`, byte a byte, e
nenhuma troca toca orçamento, semente, configuração, fatia ou conferência de dados.

⚠️ O teste central é `test_o_comando_de_treino_difere_SO_no_p_equacao`: ele lê as
duas chamadas ao `train_phienc.py` pela AST e compara argumento a argumento.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "kaggle"))

import t2a_tokenizer  # noqa: E402
import t2eq_tratado  # noqa: E402

from phifm.core.kaggle import obter  # noqa: E402

CONTROLE = t2a_tokenizer.CELULA
TRATADO = t2eq_tratado.CELULA

# O que não pode aparecer em NENHUM texto removido pelas trocas.
INTOCAVEIS = ("TOKENS =", "CONTEXTO =", "SEQUENCIAS =", "ACUMULACAO =",
              "PASSOS =", "LIMITE_H =", "--semente", "--config", "--total-passos",
              "--dados", "--contexto", "--sequencias", "--acumulacao",
              "--limite-horas", "blake3", "ASSINATURA", "FATIA =", "SHA =",
              "--tokenizer", "--mesmo-assim")


def _chamada(celula: str, script: str) -> list[str]:
    """Os argumentos da chamada `_rodar([... script ...])`, como texto."""
    for no in ast.walk(ast.parse(celula)):
        if (isinstance(no, ast.Call) and ast.unparse(no.func) == "_rodar"
                and script in ast.unparse(no.args[0])):
            return [ast.unparse(e) for e in no.args[0].elts]
    raise AssertionError(f"nenhuma chamada a {script}")


def _removido_pelos_blocos() -> list[str]:
    fora = []
    for inicio, fim, _ in t2eq_tratado.BLOCOS:
        a = CONTROLE.index(inicio)
        fora.append(CONTROLE[a:CONTROLE.index(fim, a) + len(fim)])
    return fora


# ── a derivação ─────────────────────────────────────────────────────────────

def test_a_celula_publicada_E_a_derivacao_do_controle():
    """⚠️ Se alguém editar a célula do tratado à mão, ou a do T2a mudar, isto
    reprova — e a correção é regerar, não ajustar o esperado."""
    assert t2eq_tratado.derivar(CONTROLE) == TRATADO


def test_a_celula_e_python_valido():
    ast.parse(TRATADO.replace("__VARIANTE__", "E"))


def test_derivar_LEVANTA_quando_uma_ancora_some():
    antigo, _ = t2eq_tratado.TROCAS[2]
    with pytest.raises(ValueError, match="aparece 0 vezes"):
        t2eq_tratado.derivar(CONTROLE.replace(antigo, "", 1))
    inicio, _, _ = t2eq_tratado.BLOCOS[0]
    with pytest.raises(ValueError, match="aparece 0 vezes"):
        t2eq_tratado.derivar(CONTROLE.replace(inicio, "", 1))


def test_nenhuma_troca_toca_orcamento_semente_fatia_ou_dados():
    removidos = [a for a, _ in t2eq_tratado.TROCAS] + _removido_pelos_blocos()
    for texto in removidos:
        for proibido in INTOCAVEIS:
            if proibido == "--semente" and '"--p-equacao", 0.0, "--semente", 17,' in texto:
                continue  # a troca do p_equacao carrega a semente como ÂNCORA, intacta
            assert proibido not in texto, (
                f"uma troca remove {proibido!r} — o tratado deixaria de ser o controle "
                f"com uma variável trocada: {texto[:80]!r}")


# ── o que difere, e o que não pode diferir ──────────────────────────────────

def test_o_comando_de_treino_difere_SO_no_p_equacao():
    c = _chamada(CONTROLE, "train_phienc.py")
    t = _chamada(TRATADO, "train_phienc.py")
    assert len(c) == len(t)
    diferentes = [(x, y) for x, y in zip(c, t, strict=True) if x != y]
    # `RUN` é o mesmo NOME nas duas; o diretório difere na atribuição, que o
    # teste dos artefatos confere. Na chamada, só o p_equacao pode mudar.
    assert diferentes == [("0.0", "P_EQUACAO")], diferentes


def test_o_p_equacao_e_0_6_e_decidido_na_celula():
    arvore = ast.parse(TRATADO.replace("__VARIANTE__", "E"))
    valores = [ast.literal_eval(n.value) for n in ast.walk(arvore)
               if isinstance(n, ast.Assign)
               and any(isinstance(a, ast.Name) and a.id == "P_EQUACAO" for a in n.targets)]
    assert valores == [0.6]


def test_o_orcamento_e_IDENTICO_ao_do_controle():
    def constantes(celula):
        out = {}
        for n in ast.walk(ast.parse(celula)):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                    and isinstance(n.targets[0], ast.Name) \
                    and n.targets[0].id in ("TOKENS", "CONTEXTO", "SEQUENCIAS",
                                            "ACUMULACAO", "PASSOS", "LIMITE_H"):
                out[n.targets[0].id] = ast.unparse(n.value)
        return out
    c, t = constantes(CONTROLE), constantes(TRATADO)
    assert len(c) == 6
    assert c == t


def test_a_exportacao_difere_SO_no_destino_e_na_nota():
    c = _chamada(CONTROLE, "exportar_phienc.py")
    t = _chamada(TRATADO, "exportar_phienc.py")
    assert len(c) == len(t)
    for x, y in zip(c, t, strict=True):
        if x != y:
            assert x.startswith("f'T2a braço") or x.startswith('f"T2a braço'), (x, y)


def test_a_variante_e_TRAVADA_em_E():
    """O controle é o braço E. Publicar este braço com A daria um tratado sem
    controle, e 8 h de T4 para nada."""
    assert 'assert VARIANTE == "E"' in TRATADO
    assert obter("t2eq_tratado").variante == "E"


def test_os_artefatos_NAO_colidem_com_os_do_controle():
    """O download cai em casa ao lado do controle; um `phienc-E/` sobrescreveria
    o checkpoint que a comparação precisa."""
    for nome in ('f"phienc_{VARIANTE}_tratado"', 'f"phienc-{VARIANTE}-tratado"',
                 'f"treino_{VARIANTE}_tratado.log"', 'f"exportar_{VARIANTE}_tratado.log"',
                 'f"t2eq_tratado_{VARIANTE}.json"'):
        assert nome in TRATADO, nome
    assert 'f"t2a_{VARIANTE}.json"' not in TRATADO


def test_a_fracao_tratada_e_CONFERIDA_no_fim():
    assert '"fracao_tratada"' in TRATADO
    assert "_ft < 0.50" in TRATADO


# ── o registro ──────────────────────────────────────────────────────────────

def test_o_registro_reusa_DADOS_e_CODIGO_do_controle():
    """Tudo que identifica o que treina e sobre o quê é igual ao `t2a_e`."""
    t, c = obter("t2eq_tratado"), obter("t2a_e")
    for campo in ("reusa_dados_de", "titulo_dados", "slug_dados", "pacote",
                  "arquivos", "scripts", "modelos", "variante", "repo"):
        assert getattr(t, campo) == getattr(c, campo), campo
    assert t.slug_notebook != c.slug_notebook
    assert t.fonte_celula == "kaggle/t2eq_tratado.py"
    t.conferir()


# ── a regra ─────────────────────────────────────────────────────────────────

def test_a_regra_esta_escrita_ANTES_e_tem_o_que_decide():
    r = t2eq_tratado.REGRA_ABLACAO
    for exigido in ("CHECAGENS DE MANIPULAÇÃO", "0,50", "2.000", "contexto 1.024",
                    "bootstrap pareado por SEQUÊNCIA", "NÃO o McNemar por token",
                    "DIFERENÇA DAS DIFERENÇAS", "mlm_regiao", "UNIFORME",
                    "O viés da prova aponta para o CONTROLE",
                    "NÃO é o negativo do DOC-07", "0,6 B"):
        assert exigido in r, exigido
    assert r in TRATADO, "a regra do módulo não é a que a célula imprime"


def test_a_regra_do_T2a_NAO_vazou_para_o_tratado():
    assert "a §8 vale alguma coisa" not in TRATADO
    assert "MEDIDA PRIMÁRIA: bits por byte" not in TRATADO


# ── o publicador, que quebrou no lançamento deste braço ─────────────────────

def test_toda_saida_de_subprocesso_do_publicador_e_decodificada_em_UTF8():
    """⚠️ Medido em 2026-09-16, no primeiro `--enviar` deste braço: `datasets list`
    devolveu o título "PhiFM T2a — fatias…", o Windows decodificou em cp1252, a
    thread de leitura levantou no travessão, `stdout` veio None, e a publicação
    caiu ANTES de enviar. Seis das sete chamadas com `text=True` não declaravam
    encoding; só a do envio declarava."""
    fonte = (RAIZ / "scripts" / "publicar_kaggle.py").read_text(encoding="utf-8")
    sem = []
    for no in ast.walk(ast.parse(fonte)):
        if not (isinstance(no, ast.Call) and ast.unparse(no.func).startswith("subprocess.")):
            continue
        kws = {k.arg: ast.unparse(k.value) for k in no.keywords}
        if kws.get("text") == "True" and kws.get("encoding") != "'utf-8'":
            sem.append(no.lineno)
    assert not sem, f"subprocess com text=True sem encoding utf-8 nas linhas {sem}"


def test_a_regra_declara_a_CORRECAO_e_por_que():
    """⚠️ A primeira versão tinha a perda nos tokens de equação como primária. Token
    de equação já é muito mais fácil que prosa SEM tratamento (+0,1286 no
    ModernBERT-base, `mlm_regiao`), e um tratado que melhorasse tudo por igual sairia
    lido como evidência. A correção entrou antes de qualquer número do tratado, e
    tem de ficar DITA na regra — não só no histórico do git."""
    r = t2eq_tratado.REGRA_ABLACAO
    assert "CORRIGIDA em 2026-09-16" in r
    assert "ANTES de qualquer" in " ".join(r.split())
    assert "+0,1286" in r
    # e a primária antiga não pode ter sobrado como primária
    assert "MEDIDA PRIMÁRIA: a célula de MECANISMO" not in r
