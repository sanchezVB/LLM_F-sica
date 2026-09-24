"""O assistente: responde perguntas de Física citando os artigos que a busca encontra.

É a geração ancorada do DOC-13 §6, na versão que cabe numa máquina: um modelo aberto
(Qwen3-8B, local) que responde SÓ com as fontes recuperadas e as cita como [n], e um
portão que confere as citações antes de a resposta sair.

## Os quatro passos

1. **A consulta hipotética (HyDE).** O recuperador aprendeu com textos `título. resumo`
   em inglês; uma pergunta curta, talvez em português, está fora dessa distribuição. O
   modelo escreve primeiro o título e o resumo de um artigo HIPOTÉTICO que responderia à
   pergunta, e a busca usa esse texto — que tem a forma exata do treino do encoder.
2. **A busca**, com os resumos completos dos `k` primeiros.
3. **A resposta**, com as fontes numeradas e a instrução de citar e de dizer quando elas
   não bastam.
4. **O portão de citações** (`verificar_citacoes`): citação para uma fonte que não foi
   fornecida é REMOVIDA, e frase sem citação válida é marcada. Nada disso impede o modelo
   de citar uma fonte fornecida que não sustenta a frase — conferir isso (o *entailment*
   do DOC-13 §6.2) fica para a medição, e está declarado como limite.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

SISTEMA_HYDE = (
    "You write the title and abstract of a physics research paper. Given a question, write "
    "a plausible, specific title and a ~120-word abstract, in English, of a paper that would "
    "answer it. Use standard physics terminology. Output only: the title, a period, then "
    "the abstract. No preamble.")

SISTEMA_RESPOSTA = (
    "You are a physics research assistant. Answer the question using ONLY the numbered "
    "sources below. After each factual claim, cite the source(s) that support it, like [1] "
    "or [2][4]. Do not cite a source that does not support the claim, and never invent "
    "sources. If the sources do not contain enough information to answer, say so plainly "
    "instead of guessing. Answer clearly and concisely.")
_PORTUGUES = re.compile(r"[ãõçáéíóúâêôà]|\b(que|qual|quais|como|por|porque|são|não|é|um|uma"
                        r"|dos|das|na|no|em)\b", re.IGNORECASE)


def instrucao_de_idioma(pergunta: str) -> str:
    """O idioma NOMEADO. "Responda no idioma da pergunta" não bastou: com seis resumos em
    inglês no prompt, a recusa de uma pergunta em português saiu em inglês duas vezes."""
    if len(_PORTUGUES.findall(pergunta)) >= 2:
        return "Write your entire answer in Brazilian Portuguese."
    return "Write your answer in the same language as the question above."

_CITACAO = re.compile(r"\[(\d+(?:\s*[-–,;]\s*\d+)*)\]")
_FRASE = re.compile(r"(?<=[.!?])\s+")
# Frase que admite que as fontes não bastam não precisa de citação — é o comportamento
# pedido. Estreito de propósito: "as fontes mostram que neutrinos não oscilam" é uma
# afirmação, e tem de continuar marcada se vier sem citação.
_FONTES = r"\b(fontes?|sources?|documentos?|documents?|textos?|texts?|artigos?|papers?)\b"
_ADMITE_FALTA = re.compile(
    rf"(?i){_FONTES}[^.]{{0,80}}\b(não|do not|don't|does not|doesn't|lack)\b[^.]{{0,20}}"
    r"\b(contain|mention|cover|address|provide|include|discuss|answer|cont[êé]m|menciona"
    r"|trata|aborda|inclu|respond|traz)"
    rf"|\b(não|not)\b[^.]{{0,20}}\b(abordad|tratad|mencionad|covered|addressed|mentioned)"
    rf"[^.]{{0,40}}{_FONTES}"
    rf"|\b(cannot|can't|unable to|não posso|não é possível|não consigo)\b[^.]{{0,80}}{_FONTES}"
    rf"|{_FONTES}[^.]{{0,40}}\b(insuficientes?|insufficient|not enough|não bastam)")


def _numeros(dentro: str) -> list[int]:
    """`"1, 3-5"` → `[1, 3, 4, 5]`. Intervalo invertido ou com mais de 20 vira só as pontas."""
    nums: list[int] = []
    for parte in re.split(r"\s*[,;]\s*", dentro):
        pontas = [int(x) for x in re.split(r"\s*[-–]\s*", parte)]
        if len(pontas) == 2 and 0 <= pontas[1] - pontas[0] <= 20:
            nums += range(pontas[0], pontas[1] + 1)
        else:
            nums += pontas
    return nums


@dataclass
class Fonte:
    numero: int
    arxiv_id: str
    titulo: str
    ano: int | None
    resumo: str
    escore: float

    @property
    def link(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


@dataclass
class Resposta:
    pergunta: str
    consulta: str
    texto: str
    fontes: list[Fonte]
    citadas: list[int] = field(default_factory=list)
    removidas: list[int] = field(default_factory=list)
    frases_sem_fonte: list[str] = field(default_factory=list)


def verificar_citacoes(texto: str, n_fontes: int) -> tuple[str, list[int], list[int], list[str]]:
    """O portão: `(texto limpo, citadas, removidas, frases sem citação válida)`.

    Citação fora de `1..n_fontes` é alucinação de fonte e SAI do texto — nunca aparece
    como se tivesse origem. Frases sem nenhuma citação válida voltam para o chamador
    marcar; frases curtas de ligação ("Em resumo:") não contam.
    """
    citadas: set[int] = set()
    removidas: set[int] = set()

    def trocar(m: re.Match) -> str:
        nums = _numeros(m.group(1))
        validos = list(dict.fromkeys(n for n in nums if 1 <= n <= n_fontes))
        removidas.update(n for n in nums if not 1 <= n <= n_fontes)
        citadas.update(validos)
        return "".join(f"[{n}]" for n in validos)

    limpo = _CITACAO.sub(trocar, texto)
    limpo = re.sub(r"[ \t]+([.,;:])", r"\1", limpo)
    limpo = re.sub(r"[ \t]{2,}", " ", limpo)
    sem_fonte = [f.strip() for f in _FRASE.split(limpo)
                 if len(f.strip()) > 40 and not _CITACAO.search(f)
                 and not _ADMITE_FALTA.search(f)]
    return limpo, sorted(citadas), sorted(removidas), sem_fonte


def bloco_de_fontes(fontes: list[Fonte]) -> str:
    return "\n\n".join(
        f"[{f.numero}] {f.titulo} ({f.ano or 'n.d.'}, arXiv:{f.arxiv_id})\n{f.resumo}"
        for f in fontes)


class Assistente:
    def __init__(self, busca, modelo, spine: Path, k: int = 6):
        self.busca, self.modelo, self.k = busca, modelo, k
        self._spine = pl.scan_parquet(spine)
        # O primeiro acesso ao parquet lê do HD frio (~16 s medidos); aquecer aqui tira
        # esse custo da primeira pergunta.
        self._resumos(["0704.0001"])

    def _resumos(self, ids: list[str]) -> dict[str, dict]:
        d = (self._spine.filter(pl.col("arxiv_id").is_in(ids))
             .select("arxiv_id", "title", "abstract", "year").collect())
        return {r["arxiv_id"]: r for r in d.iter_rows(named=True)}

    def consulta_hipotetica(self, pergunta: str) -> str:
        texto = self.modelo.gerar(SISTEMA_HYDE, pergunta, max_tokens=260, temperatura=0.3)
        texto = " ".join(texto.split())
        return texto or pergunta

    def recuperar(self, consulta: str) -> list[Fonte]:
        resultados = self.busca.buscar(consulta, k=self.k)
        resumos = self._resumos([r.arxiv_id for r in resultados])
        fontes = []
        for r in resultados:
            meta = resumos.get(r.arxiv_id, {})
            fontes.append(Fonte(len(fontes) + 1, r.arxiv_id, r.titulo, r.ano,
                                " ".join((meta.get("abstract") or "").split()), r.escore))
        return fontes

    def responder(self, pergunta: str) -> Resposta:
        consulta = self.consulta_hipotetica(pergunta)
        fontes = self.recuperar(consulta)
        usuario = (f"Sources:\n\n{bloco_de_fontes(fontes)}\n\nQuestion: {pergunta}\n\n"
                   f"{instrucao_de_idioma(pergunta)}")
        bruto = self.modelo.gerar(SISTEMA_RESPOSTA, usuario)
        texto, citadas, removidas, sem_fonte = verificar_citacoes(bruto, len(fontes))
        return Resposta(pergunta, consulta, texto, fontes, citadas, removidas, sem_fonte)
