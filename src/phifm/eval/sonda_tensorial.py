"""Sonda de ESTRUTURA TENSORIAL: a terceira medida do DOC-05 §11.2. Sem torch.

O §11.2 nomeia "uma sonda de estrutura tensorial" e não a especifica. Este módulo
é a especificação, e ela é desenhada para testar exatamente o que a **§8** afirma
ser "a decisão mais consequente e menos visível" do documento: que `^{` e `_{`
precisam ser pré-tokens atômicos, junto com as sequências de controle.

Se a §8 estiver certa, um modelo cujo tokenizer estilhaça `^{` em `^` + `{` tem
mais dificuldade de representar o **papel** de um índice. E é o que a variante E
(sem as regras da §8) existe para medir contra a A.

## O desenho: uma sonda de DOIS LADOS

Cada item é um trio sobre o mesmo carregador de texto:

    base         T^{\\mu\\nu}        o objeto
    estrutural   T_{\\mu\\nu}        MESMOS símbolos, estrutura DIFERENTE
    renomeado    T^{\\alpha\\beta}   estrutura IGUAL, símbolos diferentes

Uma boa representação tem de fazer as duas coisas ao mesmo tempo: **separar** a
base do estrutural (são objetos diferentes — covariante contra contravariante) e
**identificar** a base com o renomeado (renomear índice não muda o objeto).

O item passa quando `sim(base, renomeado) > sim(base, estrutural)`.

Medir só um dos lados seria fácil de satisfazer pelo motivo errado: um modelo que
colapsa tudo passa no lado da invariância e falha na separação; um que decora
forma de superfície faz o contrário.

## ⚠️ E a superfície empurra na direção CONTRÁRIA, o que torna a sonda conservadora

`T^{\\mu\\nu}` → `T_{\\mu\\nu}` é **um caractere** de diferença. `T^{\\mu\\nu}` →
`T^{\\alpha\\beta}` muda dois símbolos inteiros. Então, por semelhança de
superfície, o **estrutural** é o mais parecido — exatamente o oposto do que o item
pede.

Um modelo que só mede forma de string **reprova** esta sonda. Por isso
`similaridade_de_superficie` está aqui: o piso da sonda é computável sem GPU
nenhuma, e um resultado só é sobre estrutura se estiver acima dele.

E o piso é forte. Medido em 2026-09-08 sobre os 72 itens, semelhança de trigramas
de caractere acerta **0 de 72** — margem média −0,078, negativa em todas as quatro
famílias. Isso dá uma escala de leitura que a maioria das sondas não tem:

    taxa ≈ 0,0    representa forma de superfície
    taxa ≈ 0,5    não representa nem uma nem outra (moeda)
    taxa >  0,5   representa estrutura, contra a superfície

O 0,5 do acaso fica **entre** os dois regimes, então o sinal da distância até 0,5
já diz de que lado o modelo está. Um resultado abaixo de 0,5 não é "fraco": é
evidência de que o modelo segue a string.

## ⚠️ Calibrada em 2026-09-08: NENHUM encoder existente chega a 0,5

| modelo | taxa | margem |
|---|---|---|
| piso (só superfície) | 0,0000 | −0,0784 |
| ModernBERT-base (MLM) | **0,2222** | −0,0017 |
| SciBERT (MLM puro) | 0,2083 | −0,0068 |
| PhysBERT (MLM, Física) | 0,0278 | −0,0271 |
| MiniLM-L6 (contrastivo) | 0,0000 | −0,1794 |

Duas leituras saem daí, e as duas importam para o §11.2.

**A sonda discrimina.** De 0,0000 a 0,2222 entre quatro encoders, e por família até
0,50 de faixa. Ela não está saturada.

**Mas todos estão do lado da superfície.** Então o que a sonda mede, no regime em
que os modelos de hoje vivem, é *quanto o modelo resiste ao empurrão da string* —
e isso é exatamente a pergunta da §8, que afirma que `^{` e `_{` atômicos ajudam o
modelo a ver o PAPEL do índice em vez dos caracteres.

E o padrão tem mecanismo, não é ruído: o **único modelo treinado
contrastivamente** é o mais preso à superfície de todos (0,0000, margem −0,179),
enquanto os de MLM puro são os menos presos. Treino contrastivo de sentença empurra
para semelhança de superfície; MLM não.

## ⚠️ A expressão vai num CARREGADOR, e o carregador é o mesmo nos três

Uma expressão de seis tokens, mascarada por média, é dominada por ela mesma e o
número viraria ruído de comprimento. Com carregador idêntico nos três, a única
diferença entre eles continua sendo a expressão.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

# Os carregadores. Texto de artigo, não frase de laboratório: a sonda mede a
# representação de uma expressão EM CONTEXTO, que é como o corpus a apresenta.
CARREGADORES = (
    "Consider the field equation where the quantity {} appears in the action.",
    "Substituting {} into the previous expression yields the desired result.",
    "The term {} transforms according to the symmetry of the underlying space.",
)


@dataclass(frozen=True)
class Item:
    """Um trio. `familia` é o eixo estrutural que ele isola."""

    familia: str
    base: str
    estrutural: str
    renomeado: str
    carregador: str

    def textos(self) -> tuple[str, str, str]:
        return (self.carregador.format(self.base),
                self.carregador.format(self.estrutural),
                self.carregador.format(self.renomeado))


# ⚠️ Cada família é um eixo em que a diferença é INEQUÍVOCA em física. Foram
# deliberadamente deixadas de fora as que parecem boas e não são: `g_{\mu\nu}`
# contra `g_{\nu\mu}` (o tensor métrico é simétrico, então são o MESMO objeto) e
# trocas de ordem de índices de mesma valência, onde a convenção varia. Um item
# cuja resposta certa é discutível vira ruído com cara de medida.
FAMILIAS = {
    # Covariante contra contravariante: objetos diferentes, um caractere de
    # diferença na superfície. É o item que a §8 prevê mais diretamente.
    "posicao_do_indice": [
        (r"T^{\mu\nu}", r"T_{\mu\nu}", r"T^{\alpha\beta}"),
        (r"F^{\mu\nu}", r"F_{\mu\nu}", r"F^{\rho\sigma}"),
        (r"R^{\mu\nu}", r"R_{\mu\nu}", r"R^{\alpha\beta}"),
        (r"j^{\mu}", r"j_{\mu}", r"j^{\alpha}"),
        (r"p^{\mu}", r"p_{\mu}", r"p^{\nu}"),
        (r"A^{\mu}", r"A_{\mu}", r"A^{\beta}"),
    ],
    # Índice repetido é contração (um escalar); índices livres distintos não.
    "contracao": [
        (r"T^{\mu}{}_{\mu}", r"T^{\mu}{}_{\nu}", r"T^{\alpha}{}_{\alpha}"),
        (r"R^{\mu}{}_{\mu}", r"R^{\mu}{}_{\nu}", r"R^{\beta}{}_{\beta}"),
        (r"F^{\mu\nu}F_{\mu\nu}", r"F^{\mu\nu}F_{\rho\sigma}",
         r"F^{\alpha\beta}F_{\alpha\beta}"),
        (r"A^{\mu}A_{\mu}", r"A^{\mu}A_{\nu}", r"A^{\rho}A_{\rho}"),
        (r"\partial^{\mu}\partial_{\mu}", r"\partial^{\mu}\partial_{\nu}",
         r"\partial^{\alpha}\partial_{\alpha}"),
        (r"g^{\mu\nu}g_{\mu\nu}", r"g^{\mu\nu}g_{\rho\sigma}",
         r"g^{\alpha\beta}g_{\alpha\beta}"),
    ],
    # Valência: número de índices, que é o rank do objeto.
    "valencia": [
        (r"F_{\mu\nu}", r"F_{\mu\nu\rho}", r"F_{\alpha\beta}"),
        (r"T^{\mu\nu}", r"T^{\mu\nu\rho}", r"T^{\alpha\beta}"),
        (r"R_{\mu\nu}", r"R_{\mu\nu\rho\sigma}", r"R_{\alpha\beta}"),
        (r"\Gamma^{\mu}{}_{\nu}", r"\Gamma^{\mu}{}_{\nu\rho}",
         r"\Gamma^{\alpha}{}_{\beta}"),
        (r"C_{\mu\nu}", r"C_{\mu\nu\rho\sigma}", r"C_{\rho\sigma}"),
        (r"h_{\mu\nu}", r"h_{\mu\nu\rho}", r"h_{\alpha\beta}"),
    ],
    # Qual índice está no operador e qual no campo — troca que a superfície
    # quase não vê e que muda a expressão por completo.
    "operador_contra_campo": [
        (r"\nabla_{\mu} V^{\nu}", r"\nabla^{\nu} V_{\mu}",
         r"\nabla_{\alpha} V^{\beta}"),
        (r"\partial_{\mu} A^{\nu}", r"\partial^{\nu} A_{\mu}",
         r"\partial_{\alpha} A^{\beta}"),
        (r"\nabla_{\mu} T^{\nu\rho}", r"\nabla^{\nu} T_{\mu}{}^{\rho}",
         r"\nabla_{\alpha} T^{\beta\gamma}"),
        (r"\partial_{\mu} \phi^{\nu}", r"\partial^{\nu} \phi_{\mu}",
         r"\partial_{\rho} \phi^{\sigma}"),
        (r"\nabla_{\nu} F^{\mu\nu}", r"\nabla^{\mu} F_{\nu}{}^{\nu}",
         r"\nabla_{\beta} F^{\alpha\beta}"),
        (r"D_{\mu} \psi^{\nu}", r"D^{\nu} \psi_{\mu}", r"D_{\alpha} \psi^{\beta}"),
    ],
}


def itens() -> list[Item]:
    """Todos os trios × todos os carregadores.

    O carregador entra no produto de propósito: um efeito que só aparece num
    carregador é efeito do carregador, e com três dá para ver isso na quebra por
    família em vez de descobrir depois.
    """
    saida = []
    for familia, trios in FAMILIAS.items():
        for base, estrutural, renomeado in trios:
            for c in CARREGADORES:
                saida.append(Item(familia=familia, base=base,
                                  estrutural=estrutural, renomeado=renomeado,
                                  carregador=c))
    return saida


def margem_de_superficie_por_familia() -> dict[str, float]:
    """A DIFICULDADE de cada família, computável sem modelo nenhum.

    ## ⚠️ Por que este número acompanha o resultado em vez de virar um corte

    Calibrada em 2026-09-08 sobre quatro encoders que já existiam (ver
    `sonda_tensorial_calibracao.json`), a sonda deu:

        família                  margem de superfície   faixa entre os 4 modelos
        posicao_do_indice              −0,039              0,00 – 0,50
        valencia                       −0,046              0,00 – 0,44
        operador_contra_campo          −0,078              0,00 – 0,00
        contracao                      −0,152              0,00 – 0,00

    A ordem é exata: as duas famílias com margem de superfície branda separam os
    quatro encoders, e as duas com margem forte ficam presas no zero para todos.
    A dificuldade da família é a margem, e ela sai das strings.

    Escrevi primeiro um `familia_tem_faixa()` com critério estrutural — "o
    estrutural não é anagrama" e "renomear não custa mais símbolos". Ele reprovava
    `posicao_do_indice`, que é justamente a família que discrimina melhor. O
    critério estava errado, e um limiar ajustado a quatro pontos seria pior: seria
    escolher família pelo resultado com cara de critério.

    Então nada é cortado. A margem sai ao lado de cada taxa, e quem lê vê qual
    família estava presa no piso — o que também é informação sobre o modelo, não
    só sobre o item.
    """
    saida = {}
    for familia, trios in FAMILIAS.items():
        ms = []
        for base, estrutural, renomeado in trios:
            for c in CARREGADORES:
                b, e, r = (c.format(base), c.format(estrutural),
                           c.format(renomeado))
                ms.append(similaridade_de_superficie(b, r)
                          - similaridade_de_superficie(b, e))
        saida[familia] = round(sum(ms) / len(ms), 4)
    return saida


def _trigramas(s: str) -> Counter:
    s = f"  {s}  "
    return Counter(s[i:i + 3] for i in range(len(s) - 2))


def similaridade_de_superficie(a: str, b: str) -> float:
    """Cosseno de trigramas de caractere: o PISO da sonda, sem modelo nenhum.

    Ver o §"a superfície empurra na direção contrária". Este número existe para
    que ninguém leia como estrutura o que é forma de string — se um encoder não
    supera este piso, o resultado dele não é sobre estrutura.
    """
    ta, tb = _trigramas(a), _trigramas(b)
    if not ta or not tb:
        return 0.0
    num = sum(v * tb.get(k, 0) for k, v in ta.items())
    na = math.sqrt(sum(v * v for v in ta.values()))
    nb = math.sqrt(sum(v * v for v in tb.values()))
    return num / (na * nb) if na and nb else 0.0


@dataclass
class Resultado:
    """Acertos e margens, com o vetor por item para o pareado entre braços."""

    acertos: list[int] = field(default_factory=list)
    margens: list[float] = field(default_factory=list)
    familias: list[str] = field(default_factory=list)

    def somar(self, familia: str, sim_renomeado: float,
              sim_estrutural: float) -> None:
        # ⚠️ Estritamente maior. Empate exato NÃO é acerto: acontece quando o
        # modelo dá o mesmo vetor para os três, que é o colapso que a sonda de
        # dois lados existe para pegar.
        self.acertos.append(int(sim_renomeado > sim_estrutural))
        self.margens.append(sim_renomeado - sim_estrutural)
        self.familias.append(familia)

    def como_dict(self) -> dict:
        n = len(self.acertos)
        if not n:
            return {"itens": 0, "erro": "nenhum item avaliado"}
        # ⚠️ A DIFICULDADE da família sai ao lado da taxa dela. Sem isso, uma
        # família presa no zero é lida como "o modelo não sabe isso" quando pode
        # ser "a superfície empurra forte demais nesta família" — ver
        # `margem_de_superficie_por_familia`.
        dificuldade = margem_de_superficie_por_familia()
        por_familia = {}
        for f in sorted(set(self.familias)):
            idx = [i for i, x in enumerate(self.familias) if x == f]
            por_familia[f] = {
                "itens": len(idx),
                "taxa": round(sum(self.acertos[i] for i in idx) / len(idx), 4),
                "margem_media": round(
                    sum(self.margens[i] for i in idx) / len(idx), 4),
                "margem_de_superficie": dificuldade.get(f),
            }
        return {
            "itens": n,
            "taxa": round(sum(self.acertos) / n, 4),
            "margem_media": round(sum(self.margens) / n, 4),
            "por_familia": por_familia,
            "nota": (
                "Sonda de DOIS LADOS: o item passa quando sim(base, renomeado) > "
                "sim(base, estrutural) — separar objetos diferentes E identificar "
                "renomeação de índice. Por semelhança de SUPERFÍCIE o estrutural é "
                "o mais parecido (um caractere de diferença), então um modelo que "
                "só mede forma de string REPROVA. Comparar sempre com o piso de "
                "`similaridade_de_superficie`."),
        }


def piso_de_superficie() -> dict:
    """O que a sonda dá para um "modelo" que é só semelhança de trigramas.

    Roda em milissegundos, sem GPU, e é o número contra o qual todo resultado da
    sonda tem de ser lido.
    """
    r = Resultado()
    for it in itens():
        base, estrutural, renomeado = it.textos()
        r.somar(it.familia,
                similaridade_de_superficie(base, renomeado),
                similaridade_de_superficie(base, estrutural))
    d = r.como_dict()
    d["nota"] = ("PISO: semelhança de trigramas de caractere, sem modelo. Um "
                 "encoder que não supere isto não está medindo estrutura.")
    return d
