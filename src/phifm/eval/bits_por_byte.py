"""Bits por byte: a única das três medidas do §11.2 que compara VOCABULÁRIOS.

A pergunta do T2a é qual tokenizer produz o melhor modelo. Medir isso exige uma
quantidade que não dependa de como o texto foi partido — e acurácia de MLM
depende.

## ⚠️ Por que acurácia de MLM não compara tokenizers, e por que o viés é traiçoeiro

As variantes A e E partem o MESMO texto de jeitos diferentes. Um token de E é mais
curto (5,14 bytes/token contra 7,07 de A), então prevê-lo a partir da vizinhança é
mais fácil: sobra mais redundância local depois de mascarar. Mascarar `\\fr`, `ac`
e `{` separadamente é mais fácil que mascarar `\\frac` de uma vez.

**E marcaria acurácia maior sem ser modelo melhor.** O viés aponta contra a
hipótese sob teste, que é o que torna o erro difícil de perceber: um resultado
"E empata com A" pareceria uma refutação honesta da §8 quando é o instrumento.

Eu escolhi acurácia como medida primária deste experimento pelo PODER — 154 mil
tokens pareados — antes de perguntar se ela media a coisa certa. É a mesma falha
do T1b2, onde a regra pré-registrada nomeava McNemar sobre pertinência ao top-k
para julgar um reordenador, que reordena DENTRO do top-k. Pré-registrar não
salva de escolher o instrumento errado.

## A quantidade que vale

    bits_por_byte = (soma de -log2 p(token mascarado)) / (bytes que eles cobriam)

O numerador é informação, o denominador é TEXTO. Um modelo que precisa de menos
bits para reconstruir o mesmo texto é melhor, e a conta não pergunta em quantos
pedaços o texto foi cortado. É a forma padrão de comparar modelos de vocabulários
diferentes, e é a única das três medidas do §11.2 que precisa de cuidado — a sonda
e a recuperação comparam cossenos entre textos e nunca olham para tokens.

## ⚠️ O viés RESIDUAL, que não desaparece e aponta para o mesmo lado

Mascarando 15% dos tokens de cada modelo, A esconde pedaços MAIORES de texto de
uma vez. Reconstruir `\\frac` inteiro é mais difícil que reconstruir `ac` sabendo
que antes vinha `\\fr`. Então o teste continua conservador para A.

A consequência para a leitura, e ela tem de estar escrita antes do número:

  - **A vence** — robusto. Venceu apesar de o instrumento favorecer E.
  - **empate**  — ambíguo, e não é evidência contra a §8.
  - **E vence** — ambíguo pelo mesmo motivo, e pede o teste canônico
    (pseudo-verossimilhança de Salazar et al., um passe por token) antes de virar
    conclusão. É caro: O(n) passes por sequência em vez de um.

Medir os bytes REALMENTE escondidos por cada modelo, em vez de supor que 15% dos
tokens é 15% do texto, é o que mantém o denominador honesto sob esse viés.

## O pareamento é por DOCUMENTO, e não por posição

Entre tokenizers diferentes não existe "a mesma posição": os fluxos têm
comprimentos diferentes. O que existe é o mesmo DOCUMENTO, e é nele que o teste
pareado se apoia — um bits/byte por documento em cada braço, e a diferença entre
os dois vetores. Comparar médias globais jogaria fora o pareamento e perderia
poder sem necessidade.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# `-log2 p` a partir de `-ln p`. O laço e o `transformers` trabalham em nats.
BITS_POR_NAT = 1.0 / math.log(2.0)


@dataclass
class PorDocumento:
    """O que um documento contribui. Guardado inteiro para o teste pareado."""

    bits: float
    bytes_escondidos: int
    tokens_escondidos: int
    acertos: int

    def bits_por_byte(self) -> float:
        return self.bits / self.bytes_escondidos if self.bytes_escondidos else 0.0


@dataclass
class Acumulador:
    """Junta documentos. Um por sequência avaliada, na ORDEM em que entraram."""

    docs: list[PorDocumento] = field(default_factory=list)

    def somar(self, log_probs_nats: np.ndarray, bytes_do_token: np.ndarray,
              certo: np.ndarray) -> None:
        """Um documento: os log-probs dos mascarados, em nats e NEGATIVOS-para-baixo.

        `log_probs_nats[i]` é `ln p(token_verdadeiro_i)`, que é ≤ 0.
        `bytes_do_token[i]` é quantos bytes UTF-8 aquele token cobria.
        `certo[i]` diz se o argmax bateu — diagnóstico, não decide nada.
        """
        if not (len(log_probs_nats) == len(bytes_do_token) == len(certo)):
            raise ValueError(
                f"formas diferentes: {len(log_probs_nats)} log-probs, "
                f"{len(bytes_do_token)} tamanhos e {len(certo)} acertos. Um "
                "desalinhamento aqui atribuiria os bits ao token errado.")
        if len(log_probs_nats) and float(np.max(log_probs_nats)) > 1e-6:
            raise ValueError(
                f"log-prob positivo ({float(np.max(log_probs_nats)):.4g}): a "
                "entrada tem de ser `ln p` em nats, não `-ln p` nem probabilidade. "
                "Com o sinal trocado o melhor modelo sairia como o pior.")
        self.docs.append(PorDocumento(
            bits=float(-np.sum(log_probs_nats) * BITS_POR_NAT),
            bytes_escondidos=int(np.sum(bytes_do_token)),
            tokens_escondidos=int(len(log_probs_nats)),
            acertos=int(np.sum(certo)),
        ))

    def por_documento(self) -> np.ndarray:
        """O vetor que o teste pareado consome, na ordem dos documentos."""
        return np.array([d.bits_por_byte() for d in self.docs], dtype=np.float64)

    def como_dict(self) -> dict:
        bits = sum(d.bits for d in self.docs)
        by = sum(d.bytes_escondidos for d in self.docs)
        tk = sum(d.tokens_escondidos for d in self.docs)
        ac = sum(d.acertos for d in self.docs)
        return {
            # ⚠️ A quantidade que decide. Agregada sobre TODOS os bytes, e não a
            # média dos por-documento: documentos longos escondem mais texto e
            # devem pesar mais numa média de bits por byte.
            "bits_por_byte": round(bits / by, 5) if by else None,
            "documentos": len(self.docs),
            "bytes_escondidos": by,
            "tokens_escondidos": tk,
            "bytes_por_token_escondido": round(by / tk, 3) if tk else None,
            # ⚠️ DIAGNÓSTICO, e ela não decide nada. Ver o § sobre o viés: um
            # tokenizer de tokens curtos marca acurácia maior sem ser melhor.
            "acuracia_diagnostica": round(ac / tk, 4) if tk else None,
            "bits_por_token": round(bits / tk, 4) if tk else None,
            "nota": (
                "⚠️ `acuracia_diagnostica` NÃO compara tokenizers: quem parte em "
                "pedaços menores acerta mais sem ser melhor. Quem decide é "
                "`bits_por_byte`, cujo denominador é TEXTO. E ele ainda é "
                "conservador para o tokenizer de tokens longos, que esconde "
                "pedaços maiores por máscara — ver o módulo."),
        }


def bootstrap_pareado(a: np.ndarray, b: np.ndarray, semente: int = 17,
                      n: int = 10_000) -> dict:
    """`a - b` por documento, com IC de 95% por reamostragem dos DOCUMENTOS.

    ⚠️ Reamostra documentos, não posições. As posições dentro de um documento não
    são independentes — compartilham contexto, tópico e notação —, e reamostrá-las
    daria um intervalo estreito demais por contar como independente o que não é.

    Pareado: a mesma reamostra de índices vale para os dois braços, porque os dois
    viram os MESMOS documentos. Reamostrar cada braço por conta própria jogaria
    fora o pareamento, que é de onde vem o poder aqui.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"{a.shape} contra {b.shape}: os braços não viram os mesmos "
            "documentos, e sem isso não há o que parear.")
    if a.size == 0:
        raise ValueError("nenhum documento: não há o que comparar")
    d = a - b
    rng = np.random.default_rng(semente)
    idx = rng.integers(0, d.size, size=(n, d.size))
    medias = d[idx].mean(axis=1)
    lo, hi = np.percentile(medias, [2.5, 97.5])
    return {
        "diferenca_media": round(float(d.mean()), 6),
        "ic95": [round(float(lo), 6), round(float(hi), 6)],
        # ⚠️ O sinal do IC é o que decide, e não a diferença sozinha: um IC que
        # cruza zero é empate, por maior que seja a diferença pontual.
        "cruza_zero": bool(lo <= 0.0 <= hi),
        "documentos": int(d.size),
        "vence": None if bool(lo <= 0.0 <= hi) else ("b" if d.mean() > 0 else "a"),
        "reamostras": n,
        "nota": ("Em bits por byte, MENOR é melhor: `vence` nomeia o braço com "
                 "menos bits. `cruza_zero` verdadeiro é EMPATE, e num run "
                 "subdimensionado empate não é evidência contra a hipótese."),
    }


def confronto(rotulo_a: str, acum_a: Acumulador,
              rotulo_b: str, acum_b: Acumulador, semente: int = 17) -> dict:
    """O confronto entre dois braços, pareado por documento.

    ⚠️ Exige o mesmo número de documentos, na mesma ordem. O avaliador tem de
    percorrer a MESMA lista nos dois braços; se um pular um documento por erro de
    tokenização, o pareamento passa a comparar textos diferentes e nada acusaria.
    """
    va, vb = acum_a.por_documento(), acum_b.por_documento()
    if va.size != vb.size:
        raise ValueError(
            f"{rotulo_a} tem {va.size} documentos e {rotulo_b} tem {vb.size}. O "
            "pareamento compararia textos diferentes.")
    r = bootstrap_pareado(va, vb, semente=semente)
    r["braços"] = {"a": rotulo_a, "b": rotulo_b}
    if r["vence"] in ("a", "b"):
        r["vencedor"] = rotulo_a if r["vence"] == "a" else rotulo_b
    else:
        r["vencedor"] = None
    r["bits_por_byte"] = {rotulo_a: acum_a.como_dict()["bits_por_byte"],
                          rotulo_b: acum_b.como_dict()["bits_por_byte"]}
    return r
