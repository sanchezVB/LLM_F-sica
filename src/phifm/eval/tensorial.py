"""Sonda de estrutura tensorial — a terceira das três do DOC-05 §11.2.

    PYTHONPATH=src python scripts/avaliar_tensorial.py \\
        --controle models/phienc-controle --tratado models/phienc-tratado

## O que ela mede, e por que este projeto precisa dela

O DOC-00 nomeia **colapso de notação** como modo de falha F5: confundir índice
covariante com contravariante, `∇×` com `∇·`, a ordem dos índices de um tensor.
E o repositório já pagou por isso no próprio parser — `\\rho_{xy}` e `\\rho_{yx}`
viravam o **mesmo símbolo**, porque produto comuta e nome não. A resistividade
Hall é antissimétrica: ρ_xy = −ρ_yx. O DOC-03 §3.2 lista "normalizar posição de
índices" entre as transformações **rejeitadas**, de propósito.

A sonda pergunta o análogo disso para a representação aprendida: **o encoder
distingue expressões que diferem só na estrutura de índices?**

## ⚠️ Sem classificador treinado, e isso é a decisão central

O jeito convencional seria uma sonda supervisionada: congela o encoder, treina um
classificador sobre as representações, reporta acurácia. Não foi o escolhido, por
dois motivos que se somam:

1. **Confusão de capacidade.** Uma sonda forte extrai quase qualquer coisa de
   quase qualquer representação, e a acurácia passa a medir a sonda. A literatura
   responde a isso com tarefa de controle e seletividade; aqui dá para evitar o
   problema em vez de corrigi-lo.
2. **Não há dado rotulado**, e construí-lo seria o trabalho inteiro.

A alternativa é **par mínimo**: duas expressões que diferem no mínimo possível, e
a pergunta é se a representação as separa. Nada é treinado, então nada pode ser
confundido com capacidade da sonda.

## O par de controle é o que torna a medida legível

Distância entre representações de duas cadeias quase idênticas é pequena para
**qualquer** modelo — a medida absoluta não diz nada. O que diz é a comparação
contra um par de controle:

| par | exemplo | a Física muda? |
|---|---|---|
| **significativo** | `\\partial_\\mu A^\\mu` → `\\partial_\\mu A^\\nu` | sim: contração virou índice livre |
| **controle** | `\\partial_\\mu A^\\mu` → `\\partial_\\nu A^\\nu` | **não**: renomear índice mudo é no-op |

Um modelo sem estrutura tensorial responde ao número de caracteres trocados, e os
dois pares o movem de forma parecida. Um modelo **com** estrutura move-se muito
mais no significativo. É o mesmo raciocínio de tarefa de controle da literatura
de sondagem, mas sem nada treinado no meio.

Os dois pares saem da **mesma expressão base**, então a comparação é pareada por
construção e o teste é o mesmo McNemar exato do resto do repositório.

## ⚠️ A distância é NORMALIZADA pela edição, e a razão é medida

Os dois arms de um par não trocam o mesmo número de caracteres, e a diferença é
grande: quebrar uma contração mexe em **um** caractere, enquanto renomear o índice
mudo mexe em **todas** as ocorrências dele — medido no inventário atual, 1 contra
18 no par `posicao_tensor`.

Comparar as distâncias cruas com essa disparidade não mede estrutura tensorial,
mede quantos caracteres mudaram. A primeira execução desta sonda mostrou isso sem
ambiguidade: com pesos ALEATÓRIOS, todos os 8 pares se moveram mais no controle.

Por isso a estatística que decide é a distância **por caractere editado**
(`dist / edicao`). Um modelo de superfície move-se proporcionalmente ao número de
caracteres e tem a mesma razão nos dois arms; um modelo com estrutura tensorial
tem razão maior no arm significativo. A distância crua continua no artefato, ao
lado, para a normalização ser auditável em vez de acreditada.

## ⚠️ A agregação por média é fraca para a categoria `ordem`

Média mascarada sobre os tokens é **quase invariante a permutação**: trocar dois
índices de lugar produz quase o mesmo conjunto de tokens, e o vetor médio mal se
move. Medido no par `hall` com um modelo aleatório: distância **exatamente
0,00000** para a troca de ordem.

Isso é limitação do READOUT, não do encoder — a informação de ordem está nos
estados por posição, e a média a descarta. A consequência prática é que a
categoria `ordem` é a mais difícil das três por construção, e um resultado nulo
nela não deve ser lido como "o modelo não sabe ordem de índices".

A agregação continua sendo a média porque é a MESMA das outras duas avaliações, e
trocá-la só aqui tornaria os três números incomparáveis. O caminho para medir
ordem de verdade é outro readout, e fica declarado como trabalho futuro.

## O que a sonda NÃO estabelece

As expressões são **sintetizadas** a partir de moldes, não mineradas do corpus.
Isso compra controle e custa validade externa: um modelo pode distinguir estes
moldes e falhar em notação real, ou o contrário. É a mesma troca de qualquer
suíte de par mínimo, e ela fica declarada no artefato.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from phifm.eval.encoders import _codificar
from phifm.eval.statistics.proporcao import binomial_exata_bicaudal

log = logging.getLogger(__name__)

# Moldura de prosa. O encoder viu equações DENTRO de texto, então medir a
# expressão nua mede uma distribuição que ele não conhece. `--sem-moldura` existe
# para checar que a conclusão não depende dela.
MOLDURA = "In the derivation above we obtain the expression ${}$ for the field."


@dataclass(frozen=True)
class ParMinimo:
    """Uma base e as duas edições que a sonda compara.

    `significativo` muda a Física; `controle` não muda nada. As duas saem da
    MESMA base, e é isso que torna a comparação pareada.
    """

    nome: str
    categoria: str
    base: str
    significativo: str
    controle: str
    porque: str


# ── O inventário, explícito de propósito ────────────────────────────────────
#
# Fica como dado legível e não como gerador: alguém que discorde da Física de um
# par tem de poder apontar a linha. Cada `porque` é a justificativa do par, e um
# par sem justificativa defensável não deve estar aqui.
#
# ⚠️ A antissimetria é a condição que torna a troca de ordem significativa. Num
# tensor SIMÉTRICO trocar os índices é no-op, e o par entraria invertido — é por
# isso que `g_{\mu\nu}` (métrica, simétrica) NÃO aparece como par de ordem,
# apesar de ser o exemplo mais citado no ESTADO.md.
PARES: tuple[ParMinimo, ...] = (
    # ⚠️ `\rho_{xy}` sozinho — o exemplo mais citado no ESTADO.md — NÃO entra, e a
    # razão é instrutiva: `x` e `y` são direções nomeadas, não índices mudos, então
    # **não existe edição no-op** para servir de controle. A primeira versão deste
    # inventário pôs a própria base como controle, o que dá distância zero e
    # inflaria a sonda em todas as bases. Um teste pegou.
    #
    # A forma com índices somados preserva a Física de Hall e admite controle.
    ParMinimo(
        nome="hall", categoria="ordem",
        base=r"j_i = \sigma_{ij} E_j",
        significativo=r"j_i = \sigma_{ji} E_j",
        controle=r"j_i = \sigma_{ik} E_k",
        porque=("a condutividade de Hall é antissimétrica (σ_xy = −σ_yx), então "
                "trocar a ordem muda a Física; renomear o índice somado não")),
    ParMinimo(
        nome="faraday", categoria="ordem",
        base=r"F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu",
        significativo=r"F_{\nu\mu} = \partial_\mu A_\nu - \partial_\nu A_\mu",
        controle=r"F_{\alpha\beta} = \partial_\alpha A_\beta - \partial_\beta A_\alpha",
        porque="o tensor de Faraday é antissimétrico; renomear os dois índices é no-op"),
    ParMinimo(
        nome="levi_civita", categoria="ordem",
        base=r"\epsilon_{ijk} a_j b_k",
        significativo=r"\epsilon_{jik} a_j b_k",
        controle=r"\epsilon_{ilm} a_l b_m",
        porque="Levi-Civita troca de sinal ao permutar dois índices"),
    ParMinimo(
        nome="posicao_quadrivetor", categoria="posicao",
        base=r"A^\mu = \eta^{\mu\nu} A_\nu",
        significativo=r"A_\mu = \eta^{\mu\nu} A_\nu",
        controle=r"A^\alpha = \eta^{\alpha\nu} A_\nu",
        porque="baixar o índice livre muda a variância do objeto"),
    ParMinimo(
        nome="posicao_tensor", categoria="posicao",
        base=r"T^{\mu\nu} U_{\mu\nu}",
        significativo=r"T_{\mu\nu} U_{\mu\nu}",
        controle=r"T^{\alpha\beta} U_{\alpha\beta}",
        porque="com os dois covariantes não há contração: os índices não se casam"),
    ParMinimo(
        nome="contracao_lorenz", categoria="contracao",
        base=r"\partial_\mu A^\mu = 0",
        significativo=r"\partial_\mu A^\nu = 0",
        controle=r"\partial_\nu A^\nu = 0",
        porque="a condição de Lorenz é um escalar; com índice livre vira outra coisa"),
    ParMinimo(
        nome="contracao_traco", categoria="contracao",
        base=r"T^\mu{}_\mu",
        significativo=r"T^\mu{}_\nu",
        controle=r"T^\nu{}_\nu",
        porque="o traço é escalar; sem a contração sobra um tensor de posto 2"),
    ParMinimo(
        nome="soma_muda", categoria="contracao",
        base=r"\sum_i c_i x_i",
        significativo=r"\sum_i c_i x_j",
        controle=r"\sum_k c_k x_k",
        porque="o índice do somatório tem de casar com o do termo"),
)


@dataclass
class Sondagem:
    """O resultado de um modelo na sonda."""

    nome: str
    # Distância média POR CARACTERE EDITADO. É a estatística que decide: a crua
    # mede quantos caracteres mudaram, não se a Física mudou. A crua fica em
    # `por_par`, ao lado, para a normalização ser auditável.
    dist_significativo: float
    dist_controle: float
    # Quantas bases moveram mais no par significativo que no de controle. É o que
    # o teste pareado consome.
    bases_a_favor: int
    bases_contra: int
    p: float
    # Distância de edição média de cada arm, para a confusão de superfície ser
    # LIDA em vez de suposta. Ver a docstring do módulo.
    edicao_significativo: float
    edicao_controle: float
    por_par: list[dict] = field(default_factory=list)
    # Pares que o tokenizer tornou indistinguíveis. Eles NAO entram na contagem, e
    # o nome deles vai para o artefato: «o modelo nao distingue» e «a entrada nunca
    # carregou a distincao» sao conclusoes diferentes.
    colapsados: list[str] = field(default_factory=list)

    @property
    def por_categoria(self) -> dict[str, dict]:
        """Quebra por tipo de edição.

        ⚠️ `ordem` é a mais difícil das três POR CONSTRUÇÃO: a média mascarada é
        quase invariante a permutação, então trocar dois índices de lugar mal move
        o vetor. Um nulo aqui é do readout, não do encoder — ver a docstring.
        """
        saida: dict[str, dict] = {}
        for linha in self.por_par:
            c = saida.setdefault(linha["categoria"], {"pares": 0, "a_favor": 0})
            c["pares"] += 1
            c["a_favor"] += int(linha["a_favor"])
        return saida

    @property
    def indice(self) -> float:
        """Quanto o par significativo move a mais, em proporção.

        Normalizado pela soma para ficar em [−1, 1] e não explodir quando as duas
        distâncias são minúsculas — que é o regime normal desta sonda.
        """
        s = self.dist_significativo + self.dist_controle
        return (self.dist_significativo - self.dist_controle) / s if s else 0.0


def _edicao(a: str, b: str) -> int:
    """Levenshtein. Pequeno o bastante para não justificar dependência nova."""
    if a == b:
        return 0
    anterior = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        atual = [i]
        for j, cb in enumerate(b, 1):
            atual.append(min(anterior[j] + 1, atual[j - 1] + 1,
                             anterior[j - 1] + (ca != cb)))
        anterior = atual
    return anterior[-1]


def textos_da_sonda(pares=PARES, com_moldura: bool = True) -> list[str]:
    """Todas as cadeias a codificar, na ordem que `sondar` espera."""
    def _f(e: str) -> str:
        return MOLDURA.format(e) if com_moldura else e

    saida = []
    for p in pares:
        saida += [_f(p.base), _f(p.significativo), _f(p.controle)]
    return saida


def colapsados_pelo_tokenizer(tok, pares=PARES, com_moldura: bool = True) -> list[str]:
    """Pares cujas duas cadeias viram a MESMA sequência de ids.

    ⚠️ Esta é a distinção que a sonda existe para não borrar: **"o modelo não
    distingue"** e **"a entrada nunca carregou a distinção"** são conclusões
    diferentes, e sem esta checagem as duas produzem o mesmo número — distância
    zero.

    Se o tokenizer colapsa `\\rho_{xy}` e `\\rho_{yx}` em ids iguais, o modelo não
    tem como separá-los nem em princípio, e atribuir isso à representação seria
    culpar a camada errada. É justamente o defeito que o DOC-05 §8 existe para
    evitar, então medi-lo aqui também testa o tokenizer.

    Um par colapsado sai da contagem e vai para o artefato pelo nome.
    """
    def _ids(e: str) -> tuple[int, ...]:
        texto = MOLDURA.format(e) if com_moldura else e
        return tuple(tok(texto, add_special_tokens=False)["input_ids"])

    return [p.nome for p in pares
            if _ids(p.base) == _ids(p.significativo)
            or _ids(p.base) == _ids(p.controle)]


@torch.no_grad()
def sondar(mod, tok, nome: str, *, pares=PARES, com_moldura: bool = True,
           max_tokens: int = 64, lote: int = 24, dispositivo: str = "cpu") -> Sondagem:
    """Codifica os pares e compara quanto cada edição move a representação.

    A codificação passa por `_codificar`, de `phifm.eval.encoders` — a mesma
    agregação por média mascarada, a mesma normalização e a mesma restrição de
    entrada das outras duas avaliações. Um caminho de codificação só.

    Pares que o tokenizer colapsa são EXCLUÍDOS e nomeados: ver
    `colapsados_pelo_tokenizer`.
    """
    dev = torch.device(dispositivo)
    colapsados = colapsados_pelo_tokenizer(tok, pares, com_moldura)
    if colapsados:
        log.warning("%s · %d par(es) colapsado(s) pelo tokenizer e excluído(s): %s",
                    nome, len(colapsados), colapsados)
    usaveis = tuple(p for p in pares if p.nome not in colapsados)
    if not usaveis:
        return Sondagem(nome=nome, dist_significativo=float("nan"),
                        dist_controle=float("nan"), bases_a_favor=0, bases_contra=0,
                        p=1.0, edicao_significativo=0.0, edicao_controle=0.0,
                        colapsados=colapsados)

    vetores = _codificar(mod, tok, textos_da_sonda(usaveis, com_moldura), dev,
                         max_tokens, lote)

    por_par, a_favor, contra = [], 0, 0
    soma_sig = soma_ctl = ed_sig = ed_ctl = 0.0
    for i, p in enumerate(usaveis):
        vb, vs, vc = vetores[3 * i], vetores[3 * i + 1], vetores[3 * i + 2]
        # Vetores já vêm normalizados de `_codificar`, então `1 − cos` é a distância.
        d_sig = float(1.0 - torch.dot(vb, vs))
        d_ctl = float(1.0 - torch.dot(vb, vc))
        e_sig, e_ctl = _edicao(p.base, p.significativo), _edicao(p.base, p.controle)

        # ⚠️ A comparação é por CARACTERE EDITADO. Crua, ela mede quantos
        # caracteres mudaram — com pesos aleatórios os 8 pares se moveram mais no
        # controle, que troca até 18× mais caracteres. Ver a docstring do módulo.
        r_sig = d_sig / e_sig if e_sig else 0.0
        r_ctl = d_ctl / e_ctl if e_ctl else 0.0
        if r_sig > r_ctl:
            a_favor += 1
        elif r_ctl > r_sig:
            contra += 1
        soma_sig += r_sig
        soma_ctl += r_ctl
        ed_sig += e_sig
        ed_ctl += e_ctl
        por_par.append({"nome": p.nome, "categoria": p.categoria,
                        "dist_significativo": d_sig, "dist_controle": d_ctl,
                        "edicao_significativo": e_sig, "edicao_controle": e_ctl,
                        "razao_significativo": r_sig, "razao_controle": r_ctl,
                        "a_favor": r_sig > r_ctl, "porque": p.porque})

    n = len(usaveis)
    teste = binomial_exata_bicaudal(a_favor, contra)
    return Sondagem(nome=nome, dist_significativo=soma_sig / n,
                    dist_controle=soma_ctl / n, bases_a_favor=a_favor,
                    bases_contra=contra, p=teste["p"],
                    edicao_significativo=ed_sig / n, edicao_controle=ed_ctl / n,
                    por_par=por_par, colapsados=colapsados)


def _leitura(s: Sondagem) -> str:
    if s.colapsados and not s.por_par:
        return (f"INCONCLUSIVO — o TOKENIZER colapsou todos os {len(s.colapsados)} "
                "pares: as duas cadeias viram os mesmos ids, e o modelo não poderia "
                "distingui-las nem em princípio. O defeito é da tokenização, não da "
                "representação")
    if s.bases_a_favor + s.bases_contra == 0:
        return "INCONCLUSIVO — nenhuma base separou os dois pares"
    if s.p >= 0.05:
        return (f"sem estrutura tensorial detectável: {s.bases_a_favor} de "
                f"{s.bases_a_favor + s.bases_contra} bases a favor (p={s.p:.3f})")
    if s.bases_a_favor > s.bases_contra:
        return (f"distingue estrutura de índices: {s.bases_a_favor} de "
                f"{s.bases_a_favor + s.bases_contra} bases (p={s.p:.4f}), "
                f"índice {s.indice:+.3f}")
    return (f"⚠️ move-se MAIS no par de controle (p={s.p:.4f}) — a representação "
            "responde ao número de caracteres, não à Física")


def comparar_bracos(sondagens: dict[str, Sondagem]) -> dict:
    """Junta as sondagens e compara os braços da ablação, quando há dois."""
    d = {
        "tarefa": "sonda_de_estrutura_tensorial",
        "pares": len(PARES),
        "colapsados_pelo_tokenizer": {n: s.colapsados for n, s in sondagens.items()},
        "categorias": sorted({p.categoria for p in PARES}),
        "modelos": {n: s.__dict__ | {"indice": s.indice,
                                     "por_categoria": s.por_categoria,
                                     "leitura": _leitura(s)}
                    for n, s in sondagens.items()},
        "ressalva": (
            "Pares SINTETIZADOS a partir de moldes, não minerados do corpus: compra "
            "controle e custa validade externa. E a distância absoluta entre cadeias "
            "quase idênticas é pequena para qualquer modelo — o que informa é a "
            "comparação com o par de controle, não o valor."),
    }

    if {"controle", "tratado"} <= set(sondagens):
        c, t = sondagens["controle"], sondagens["tratado"]
        # Pareado POR BASE: em quantas o tratado separou melhor que o controle.
        if not c.por_par or not t.por_par:
            d["ablacao"] = {"erro": (
                "não há par utilizável: o tokenizer colapsou todos. Comparar os "
                "braços aqui compararia dois vazios")}
            return d
        if c.colapsados != t.colapsados:
            d["ablacao"] = {"erro": (
                "os dois braços excluíram pares diferentes "
                f"({c.colapsados} contra {t.colapsados}) — não há como parear")}
            return d
        ganha_t = ganha_c = 0
        for pc, pt in zip(c.por_par, t.por_par, strict=True):
            mc = pc["razao_significativo"] - pc["razao_controle"]
            mt = pt["razao_significativo"] - pt["razao_controle"]
            if mt > mc:
                ganha_t += 1
            elif mc > mt:
                ganha_c += 1
        teste = binomial_exata_bicaudal(ganha_c, ganha_t)
        d["ablacao"] = teste | {
            "controle_melhor": ganha_c, "tratado_melhor": ganha_t,
            "delta_indice": t.indice - c.indice,
            "leitura": _leitura_ablacao(teste, ganha_c, ganha_t, len(PARES)),
        }
    return d


def _leitura_ablacao(teste: dict, ganha_c: int, ganha_t: int, n_pares: int) -> str:
    # ⚠️ Com 8 pares o teste exato NÃO pode dar p < 0,05 nem no extremo: 2/2^8 é
    # 0,0078, mas só com 8 a 0 — e qualquer discordância o mata. Dizer isso é
    # melhor que reportar "empate" como se a amostra bastasse.
    if teste["discordantes"] == 0:
        return "os dois braços separam os pares igualmente — nada a decidir"
    if teste["p"] < 0.05:
        quem = "tratado" if ganha_t > ganha_c else "controle"
        return f"o {quem} separa melhor a estrutura de índices (p={teste['p']:.4f})"
    return (f"sem diferença detectável entre os braços (p={teste['p']:.3f}); com "
            f"{n_pares} pares o teste exato mal tem poder — ler como ausência de "
            "evidência, não evidência de ausência")
