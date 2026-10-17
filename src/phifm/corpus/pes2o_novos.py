"""Quanto do peS2o de Física já está no arXiv — e quanto seria acervo novo para a busca.

O índice tem só o arXiv (1,59 M artigos). O peS2o filtrado pelo `is_physics` tem 3,6 M
resumos (`s2ag`), mas o filtro NÃO guardou o identificador do artigo — só o texto, a
origem (`s2ag/train`, ...) e o escore (`hf_filtrado._gravar`). Sem id, separar o que já
está no arXiv tem de ser pelo texto. Este módulo faz isso, e a conta vem antes de
investir em baixar de novo para recuperar os ids.

## Como um resumo do peS2o casa com um artigo do arXiv

Por **qualquer** de duas chaves:

1. **o título normalizado**, se tiver pelo menos `MIN_PALAVRAS_TITULO` palavras — título
   curto ("Introduction", "Reply") casaria artigos diferentes;
2. **o começo do resumo normalizado** (as `PALAVRAS_DO_RESUMO` primeiras palavras), se o
   resumo tiver esse tanto — porque a revista às vezes troca o título e mantém o resumo.

A normalização junta as grafias que o arXiv e o Semantic Scholar usam para a mesma coisa:
LaTeX (`$\\alpha$`, `$^{3}$He`, `\\mathrm{K}`) contra Unicode (`α`, `³He`, `K`), acentos,
caixa, pontuação e espaços.

## ⚠️ O que a conta superestima e subestima

- **Superestima os novos:** um artigo do arXiv com título E resumo trocados na revista
  conta como novo. Por isso o script grava exemplos dos dois lados, para conferir à mão.
- **Subestima pouco:** dois artigos diferentes com o mesmo título de quatro palavras ou
  mais e o mesmo começo de resumo são raros.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

MIN_PALAVRAS_TITULO = 4
PALAVRAS_DO_RESUMO = 30
MAX_CARACTERES_TITULO = 400
MIN_CARACTERES_RESUMO = 50
# Abaixo disto de casamento com o arXiv, a separação de título e resumo provavelmente leu
# errado o formato do texto — e a conta inteira estaria errada sem dar erro.
MIN_TAXA_PLAUSIVEL = 0.05

# Comandos de formatação do LaTeX: somem, com o argumento ficando.
_FORMATACAO = re.compile(
    r"\\(?:mathrm|mathbf|mathcal|mathit|mathsf|mathbb|mathfrak|boldsymbol|operatorname"
    r"|textrm|textbf|textit|textsf|texttt|text|emph|rm|bf|it|left|right|big|Big)(?![A-Za-z])")
_COMANDO = re.compile(r"\\([A-Za-z]+)")
# O Unicode chama a letra λ de LAMDA; o LaTeX, de \lambda. E \varepsilon é ε.
_NOMES = {"lamda": "lambda", "varepsilon": "epsilon", "vartheta": "theta", "varphi": "phi",
          "varrho": "rho", "varsigma": "sigma", "varpi": "pi"}


def _grego(ch: str) -> str | None:
    nome = unicodedata.name(ch, "")
    if not nome.startswith("GREEK") or not ch.isalpha():
        return None
    letra = nome.split(" WITH ")[0].split()[-1].lower()
    return _NOMES.get(letra, letra)


def normalizar(texto: str | None) -> str:
    """Palavras em minúsculas, só letras e dígitos, com LaTeX e Unicode na mesma grafia."""
    t = _FORMATACAO.sub(" ", texto or "")
    t = _COMANDO.sub(lambda m: f" {_NOMES.get(m.group(1), m.group(1))} ", t)
    partes = []
    for ch in unicodedata.normalize("NFKD", t):
        if unicodedata.combining(ch):
            continue
        g = _grego(ch)
        partes.append(f" {g} " if g else ch)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", "".join(partes).lower()).split())


def _hash(chave: str) -> int:
    return int.from_bytes(hashlib.blake2b(chave.encode("utf-8"), digest_size=8).digest(),
                          "little")


def chave_do_titulo(titulo: str | None) -> int | None:
    """A chave sem espaços (`$^{3}$He` e `³He` dão `3he`), ou None se o título for curto."""
    palavras = normalizar(titulo).split()
    if len(palavras) < MIN_PALAVRAS_TITULO:
        return None
    return _hash("".join(palavras))


def chave_do_resumo(resumo: str | None) -> int | None:
    palavras = normalizar(resumo).split()
    if len(palavras) < PALAVRAS_DO_RESUMO:
        return None
    return _hash("".join(palavras[:PALAVRAS_DO_RESUMO]))


def separar(texto: str | None) -> tuple[str, str] | None:
    """`(título, resumo)` de um registro `s2ag`: o título é a primeira linha. None se o
    texto não tiver esse formato — sem quebra de linha, título vazio ou longo demais,
    resumo curto demais."""
    texto = (texto or "").strip()
    if "\n" not in texto:
        return None
    titulo, resumo = texto.split("\n", 1)
    titulo, resumo = titulo.strip(), resumo.strip()
    if not titulo or len(titulo) > MAX_CARACTERES_TITULO or len(resumo) < MIN_CARACTERES_RESUMO:
        return None
    return titulo, resumo


def sha_do_texto(texto: str) -> str:
    """O hash do texto como o filtro gravou (cru, byte a byte): é por ele que o passo
    seguinte reencontra o registro no peS2o original e recupera o id."""
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


class Arxiv:
    """As duas chaves de cada artigo do arXiv, e o id de quem tem cada chave."""

    def __init__(self) -> None:
        self.por_titulo: dict[int, str] = {}
        self.por_resumo: dict[int, str] = {}
        self.artigos = 0

    def acrescentar(self, arxiv_id: str, titulo: str | None, resumo: str | None) -> None:
        self.artigos += 1
        if (k := chave_do_titulo(titulo)) is not None:
            self.por_titulo.setdefault(k, arxiv_id)
        if (k := chave_do_resumo(resumo)) is not None:
            self.por_resumo.setdefault(k, arxiv_id)

    def casar(self, titulo: str, resumo: str) -> tuple[str | None, str | None]:
        """`(arxiv_id, por qual chave)` ou `(None, None)`."""
        kt, kr = chave_do_titulo(titulo), chave_do_resumo(resumo)
        if kt is not None and kt in self.por_titulo:
            return self.por_titulo[kt], "titulo"
        if kr is not None and kr in self.por_resumo:
            return self.por_resumo[kr], "resumo"
        return None, None


N_EXEMPLOS = 50


class _Reservatorio:
    """Uma amostra uniforme de tamanho fixo de um fluxo (algoritmo R), com semente."""

    def __init__(self, n: int, semente: int) -> None:
        import random

        self.n, self.vistos, self.itens = n, 0, []
        self._rng = random.Random(semente)

    def ver(self, item: dict) -> None:
        self.vistos += 1
        if len(self.itens) < self.n:
            self.itens.append(item)
        elif (j := self._rng.randrange(self.vistos)) < self.n:
            self.itens[j] = item


class Contagem:
    """A conta, registro a registro. Os novos saem por `tirar_novos` (o script grava a
    cada parte, e a memória não cresce com 3 milhões de registros); dos dois lados fica
    uma amostra de `N_EXEMPLOS` para conferir à mão."""

    def __init__(self, arxiv: Arxiv, semente: int = 20261017) -> None:
        self.arxiv = arxiv
        self.n = {"registros": 0, "fora_do_formato": 0, "no_arxiv_pelo_titulo": 0,
                  "no_arxiv_so_pelo_resumo": 0, "novos": 0, "novos_com_titulo_curto": 0,
                  "repetidos_entre_os_novos": 0}
        self._vistos_t: set[int] = set()
        self._vistos_r: set[int] = set()
        self.novos: list[dict] = []
        self.exemplos_novos = _Reservatorio(N_EXEMPLOS, semente)
        self.exemplos_casados = _Reservatorio(N_EXEMPLOS, semente + 1)

    def tirar_novos(self) -> list[dict]:
        saida, self.novos = self.novos, []
        return saida

    def ver(self, texto: str, parte: str, linha: int) -> None:
        self.n["registros"] += 1
        sep = separar(texto)
        if sep is None:
            self.n["fora_do_formato"] += 1
            return
        titulo, resumo = sep
        arxiv_id, por = self.arxiv.casar(titulo, resumo)
        if arxiv_id is not None:
            self.n["no_arxiv_pelo_titulo" if por == "titulo" else "no_arxiv_so_pelo_resumo"] += 1
            self.exemplos_casados.ver({"titulo": titulo, "arxiv_id": arxiv_id, "por": por,
                                       "inicio_do_resumo": resumo[:300]})
            return
        kt, kr = chave_do_titulo(titulo), chave_do_resumo(resumo)
        if (kt is not None and kt in self._vistos_t) or (kr is not None and kr in self._vistos_r):
            self.n["repetidos_entre_os_novos"] += 1
            return
        if kt is not None:
            self._vistos_t.add(kt)
        if kr is not None:
            self._vistos_r.add(kr)
        self.n["novos"] += 1
        self.n["novos_com_titulo_curto"] += kt is None
        self.novos.append({"parte": parte, "linha": linha, "sha256": sha_do_texto(texto),
                           "titulo": titulo, "caracteres": len(texto)})
        self.exemplos_novos.ver({"titulo": titulo, "inicio_do_resumo": resumo[:300]})

    def resumo(self) -> dict:
        n = self.n
        no_formato = n["registros"] - n["fora_do_formato"]
        casados = n["no_arxiv_pelo_titulo"] + n["no_arxiv_so_pelo_resumo"]
        taxa = casados / no_formato if no_formato else 0.0
        return {**n, "no_arxiv": casados, "taxa_no_arxiv": round(taxa, 4),
                "plausivel": taxa >= MIN_TAXA_PLAUSIVEL,
                "aviso": None if taxa >= MIN_TAXA_PLAUSIVEL else
                f"só {taxa:.1%} casou com o arXiv: a separação de título e resumo pode ter "
                "lido errado o formato do texto. Confira os exemplos antes de usar a conta."}
