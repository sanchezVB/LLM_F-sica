"""Regressão da deduplicação (DOC-04 §5).

Dois riscos assimétricos, e os testes refletem essa assimetria:

  falso NEGATIVO → um documento duplicado no corpus. Custo baixo.
  falso POSITIVO → conteúdo genuíno removido, para sempre. Custo alto.

Por isso o limiar é 0,85 e não os 0,8 de corpus web: papers de Física
legitimamente compartilham trechos longos — descrições do mesmo detector,
seções de método padronizadas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.corpus.dedup import (  # noqa: E402
    Candidato,
    MinHasher,
    agrupar,
    escolher,
    ganho_de_licenca,
    hash_exato,
    jaccard,
    normalizar,
)

CC_BY = "http://creativecommons.org/licenses/by/4.0/"
ARXIV = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"
NC = "http://creativecommons.org/licenses/by-nc-sa/4.0/"


@pytest.fixture(scope="module")
def mh() -> MinHasher:
    return MinHasher()


class TestDedupExata:
    def test_identico_colide(self):
        assert hash_exato("Energia cinética é mv²/2") == hash_exato("Energia cinética é mv²/2")

    def test_caixa_e_pontuacao_nao_importam(self):
        assert hash_exato("O eletron tem carga -e.") == hash_exato("o  eletron   tem carga  e")

    def test_nfc_une_as_duas_codificacoes_do_mesmo_caractere(self):
        """`é` como ponto de código único vs `e`+acento combinante são
        canonicamente equivalentes e renderizam idêntico. Pipelines diferentes
        produzem uma ou outra — sem NFC o mesmo documento vindo de duas fontes
        teria hashes distintos, e a dedup falharia justamente no caso que ela
        existe para pegar."""
        composto = "el\u00e9tron"        # é  = U+00E9
        decomposto = "ele\u0301tron"     # e + U+0301
        assert composto != decomposto, "as strings precisam ser distintas na entrada"
        assert hash_exato(composto) == hash_exato(decomposto)

    def test_acento_nao_e_removido(self):
        """NFC é equivalência canônica; remover acento é outra coisa, mais
        agressiva, e mudaria palavras em vez de codificação."""
        assert hash_exato("elétron") != hash_exato("eletron")

    def test_conteudo_diferente_nao_colide(self):
        assert hash_exato("mecânica quântica") != hash_exato("mecânica clássica")

    def test_normalizar_nao_esvazia_texto_util(self):
        assert normalizar("E = mc^2").split() == ["e", "mc", "2"]


class TestMinHash:
    def test_identico_da_jaccard_um(self, mh):
        t = "a teoria quântica de campos descreve partículas como excitações de campos"
        assert jaccard(mh.assinatura(t), mh.assinatura(t)) == 1.0

    def test_disjunto_da_jaccard_baixo(self, mh):
        a = "cromodinâmica quântica descreve a interação forte entre quarks e glúons"
        b = "a expansão acelerada do universo é atribuída à energia escura observada"
        assert jaccard(mh.assinatura(a), mh.assinatura(b)) < 0.1

    def test_estimativa_bate_com_jaccard_real(self, mh):
        """O erro-padrão do MinHash é ~1/√128 ≈ 2,8 pp. Tolerância de 12 pp
        cobre com folga e ainda detecta um estimador quebrado."""
        base = [f"palavra{i}" for i in range(100)]
        a = " ".join(base)
        b = " ".join(base[:80] + [f"outra{i}" for i in range(20)])

        def shingles(t, k=5):
            p = t.split()
            return {" ".join(p[i : i + k]) for i in range(len(p) - k + 1)}

        real = len(shingles(a) & shingles(b)) / len(shingles(a) | shingles(b))
        est = jaccard(mh.assinatura(a), mh.assinatura(b))
        assert abs(est - real) < 0.12, f"estimado {est:.3f} vs real {real:.3f}"

    def test_documento_curto_nao_colide_com_todo_mundo(self, mh):
        """Texto menor que o shingle não pode gerar assinatura vazia — seria
        falso positivo em massa, que é o erro caro."""
        a, b = mh.assinatura("oi"), mh.assinatura("tchau")
        assert jaccard(a, b) < 0.5

    def test_assinatura_e_deterministica(self, mh):
        t = "a equação de Schrödinger governa a evolução temporal do estado"
        assert np.array_equal(mh.assinatura(t), mh.assinatura(t))


class TestAgrupamento:
    def test_agrupa_quase_duplicatas(self, mh):
        base = "Medimos a seção de choque de produção de pares top-antitop em colisões próton-próton a 13 TeV com o detector ATLAS no LHC durante a Run 2"
        textos = [base, base + " Os resultados concordam com o Modelo Padrão.",
                  "Um estudo de matéria condensada sobre supercondutividade em cupratos a alta temperatura crítica"]
        sig = np.stack([mh.assinatura(t) for t in textos])
        clusters, _ = agrupar(sig, limiar=0.7)
        agrupados = {i for m in clusters.values() for i in m}
        assert 0 in agrupados and 1 in agrupados, "não agrupou as quase-duplicatas"
        assert 2 not in agrupados, "agrupou documento não relacionado — falso positivo"

    def test_lsh_reduz_drasticamente_os_pares(self, mh):
        """A justificativa do LSH existir. Sem ele seriam n²/2 comparações."""
        textos = [f"documento independente número {i} sobre um tema completamente distinto dos demais" for i in range(300)]
        sig = np.stack([mh.assinatura(t) for t in textos])
        _, avaliados = agrupar(sig)
        assert avaliados < 300 * 299 / 2 * 0.05, f"{avaliados} pares — LSH não filtrou"

    def test_nao_agrupa_abaixo_do_limiar(self, mh):
        a = "a lei de Gauss relaciona o fluxo elétrico à carga interna à superfície"
        b = "a lei de Faraday relaciona a força eletromotriz à variação do fluxo magnético"
        sig = np.stack([mh.assinatura(a), mh.assinatura(b)])
        clusters, _ = agrupar(sig, limiar=0.85)
        assert not clusters


class TestRepresentante:
    def test_licenca_permissiva_vence(self):
        """★ O critério não óbvio do DOC-04 §5.3. Manter o CC BY aumenta o
        PhysCorpus-Open de graça, sem perder uma linha de conteúdo."""
        assert escolher([Candidato(0, licenca=ARXIV), Candidato(1, licenca=CC_BY)]).indice == 1

    def test_licenca_vence_ate_sobre_revisao_por_pares(self):
        """Deliberado: o critério 1 precede o 2. Um CC BY sem journal-ref
        vence um arXiv-padrão publicado."""
        vencedor = escolher([
            Candidato(0, licenca=ARXIV, journal_ref="Phys. Rev. D 106, 063007"),
            Candidato(1, licenca=CC_BY),
        ])
        assert vencedor.indice == 1

    def test_entre_mesma_licenca_vence_o_revisado(self):
        v = escolher([Candidato(0, licenca=CC_BY), Candidato(1, licenca=CC_BY, doi="10.1103/x")])
        assert v.indice == 1

    def test_entre_iguais_vence_a_versao_mais_recente(self):
        v = escolher([Candidato(0, licenca=CC_BY, arxiv_id="2401.00001v1"),
                      Candidato(1, licenca=CC_BY, arxiv_id="2401.00001v3")])
        assert v.indice == 1

    def test_nc_so_sobrevive_se_for_o_unico(self):
        """NC é EVAL_ONLY sob Q3 (ADR-0001 §4) e perde para qualquer treinável."""
        assert escolher([Candidato(0, licenca=NC), Candidato(1, licenca=ARXIV)]).indice == 1
        assert escolher([Candidato(0, licenca=NC)]).indice == 0

    def test_deterministico_em_empate_total(self):
        """Sem desempate estável, dois runs escolheriam representantes
        diferentes e o corpus deixaria de ser reconstruível (G1.5)."""
        c = [Candidato(5, licenca=CC_BY), Candidato(2, licenca=CC_BY), Candidato(9, licenca=CC_BY)]
        assert len({escolher(list(reversed(c))).indice, escolher(c).indice}) == 1

    def test_cluster_vazio_levanta(self):
        with pytest.raises(ValueError):
            escolher([])

    def test_ganho_de_licenca_so_reporta_ganho_real(self):
        assert ganho_de_licenca([Candidato(0, licenca=ARXIV), Candidato(1, licenca=CC_BY)])
        assert not ganho_de_licenca([Candidato(0, licenca=CC_BY), Candidato(1, licenca=CC_BY)])
        assert not ganho_de_licenca([Candidato(0, licenca=CC_BY)])
