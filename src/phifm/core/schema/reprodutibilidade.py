"""G1.5 — o corpus inteiro atestado por **um** hash (DOC-00 §5).

O critério do portão é: "Construção completa do corpus reprodutível ponta a ponta
a partir de um único hash de manifesto".

## O que já existia e o que faltava

A **aquisição** já era auditável: `AcquisitionManifest` (DOC-02 §8.1) grava
`manifest_id`, `checksum_index` por arquivo, `pipeline_git_sha`,
`license_resolution` e `failures`. Cinco coletas têm o seu.

O que **não** existia era manifesto dos **derivados** — a espinha, os pares de
citação, o classificador, as fatias filtradas do HuggingFace. São 19,7 GB de
artefatos que o corpus final é, e nenhum deles declarava de que entradas saiu,
com que parâmetros, nem que bytes tem. Sem isso, "um único hash" não existe: há
cinco hashes de coleta e um monte de arquivo órfão.

## A cadeia, e por que ela é de Merkle

    ManifestoRaiz.hash_raiz
      └── hash canônico de cada manifesto de etapa
            └── checksum_index: BLAKE3 de cada arquivo da etapa

Cada nível atesta o de baixo. Trocar um byte num parquet muda o `checksum_index`,
que muda o hash do manifesto da etapa, que muda o `hash_raiz`. **Um** número na
capa de um paper cobre 22 GB em 30 arquivos.

Isto também é o que permite verificação **barata**: conferir a cadeia de
manifestos custa milissegundos e pega manifesto adulterado. Só a verificação
`--profundo` relê os 22 GB, e é ela que pega parquet adulterado.

## ⚠️ O que este hash prova, e o que NÃO prova

**Prova:** que o corpus neste disco é exatamente o que o manifesto descreve, e de
que entradas e parâmetros cada etapa saiu.

**Não prova** que rodar o pipeline de novo produz os mesmos bytes. Duas razões
medidas, não hipotéticas:

1. **O arXiv OAI-PMH é mutável por projeto.** `from`/`until` filtram por
   *datestamp*, e o datestamp muda quando um autor publica uma versão nova. Uma
   coleta refeita amanhã traz registros que a de hoje não tinha. Isto não é
   defeito do coletor, é a semântica da fonte.
2. **As fatias do HuggingFace baixavam de `resolve/main/`** — alvo móvel. A API
   entrega o `sha` da revisão (o OpenWebMath está em `fde8ef8d…` desde
   2023-10-17), e passar a fixá-lo é o que torna a fatia refazível. Antes disso, a
   reprodutibilidade da fatia era sorte: o dataset não mudou.

Então a garantia declarada é a mesma família da que o DOC-01 §8 declara para o
treino — lá "reprodutibilidade estatística, não bit a bit". Aqui:
**verificabilidade bit a bit do que existe, e refazibilidade por fonte, com as
fontes mutáveis nomeadas.** Alegar mais que isso seria falso, e um critério de
portão satisfeito por alegação falsa é pior que um critério aberto.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from blake3 import blake3
from pydantic import BaseModel, Field

from phifm.core.schema.manifest import FailureRecord, canonical_hash, utcnow

ESQUEMA = "0.1.0"

# Lidos em blocos para não materializar um parquet de 500 MB na memória.
BLOCO = 4 * 1024 * 1024

# Arquivos que NÃO entram no índice, com o motivo de cada um. Lista explícita e
# curta de propósito: ignorar por padrão amplo é como uma lacuna se esconde.
#
#   *.tmp        gravação atômica em curso (ver `salvar_estado`)
#   _temp/       o arquivo baixado que a fatia apaga depois de filtrar
#   *.log        registro de execução, não é conteúdo do corpus
#   progresso.json  marcador de progresso vivo, muda a cada 100 passos
#
# ⚠️ Os MANIFESTOS também ficam fora, e não é conveniência — é necessidade
# aritmética. O manifesto de uma etapa vive dentro do diretório dela e contém o
# índice dela; incluí-lo no próprio índice é autorreferência impossível. Pior: sem
# esta exclusão, uma SEGUNDA construção indexaria o manifesto da primeira e o
# hash de um corpus intacto mudaria. Pego por teste antes de eu declarar o G1.5
# pronto, o que é o único motivo de estar escrito aqui e não descoberto em três
# meses.
#
# A cobertura deles não se perde: o `hash_manifesto` na raiz atesta cada um.
NOME_MANIFESTO_ETAPA = "_manifesto_etapa.json"
IGNORADOS = ("*.tmp", "*.log", "progresso.json",
             "_manifest.json", NOME_MANIFESTO_ETAPA, f"*{NOME_MANIFESTO_ETAPA}",
             "MANIFESTO-RAIZ.json")
#   _cache_vetores/  embeddings recomputáveis do minerador, ~1 GB
DIRS_IGNORADOS = ("_temp", "__pycache__", "_cache_vetores")


def hash_arquivo(caminho: Path) -> str:
    """BLAKE3 de um arquivo, em blocos.

    Mesma função de hash que o `canonical_hash` usa para objetos (DOC-01 P4), para
    não haver duas noções de identidade no mesmo projeto.
    """
    h = blake3()
    with open(caminho, "rb") as f:
        while bloco := f.read(BLOCO):
            h.update(bloco)
    return h.hexdigest()


def indexar(raiz: Path, padrao: str = "**/*") -> dict[str, str]:
    """`caminho relativo → BLAKE3` de tudo sob `raiz`, exceto os ignorados.

    Caminhos com `/` sempre, mesmo no Windows: um índice com `\\` não confere
    contra outro com `/`, e o corpus tem de ser verificável nos dois sistemas.
    """
    fora = set()
    for p in IGNORADOS:
        fora |= set(raiz.glob(f"**/{p}"))
    idx: dict[str, str] = {}
    for f in sorted(raiz.glob(padrao)):
        if not f.is_file() or f in fora:
            continue
        if any(d in f.relative_to(raiz).parts for d in DIRS_IGNORADOS):
            continue
        idx[f.relative_to(raiz).as_posix()] = hash_arquivo(f)
    return idx


class Entrada(BaseModel):
    """De onde uma etapa saiu.

    `manifesto_id` quando a entrada é outra etapa — é o que forma a cadeia. Um
    caminho solto sem `manifesto_id` é entrada externa ao pipeline, e o campo
    `nota` tem de dizer o que é, senão a proveniência quebra ali.
    """

    caminho: str
    manifesto_id: str | None = None
    nota: str | None = None


class ManifestoEtapa(BaseModel):
    """Uma etapa de PROCESSAMENTO — o que faltava para o G1.5.

    O par de `AcquisitionManifest` para o outro lado do pipeline: aquele atesta o
    que veio da rede, este atesta o que foi derivado dele.
    """

    manifesto_id: str = ""
    schema_version: str = ESQUEMA
    etapa: str                        # "spine", "pares_citacao", "openwebmath_fisica"
    descricao: str
    entradas: list[Entrada] = Field(default_factory=list)
    parametros: dict[str, Any] = Field(default_factory=dict)
    # ⚠️ `parametros_reconstruidos=True` quando a etapa rodou ANTES deste
    # manifesto existir e os parâmetros foram lidos do código e dos logs, não
    # capturados na execução. É retroaterro honesto, e quem lê precisa saber a
    # diferença: um parâmetro reconstruído pode estar errado sem que nada acuse.
    parametros_reconstruidos: bool = False
    raiz: str                         # diretório (ou arquivo) da saída
    registros: int | None = None
    checksum_index: dict[str, str] = Field(default_factory=dict)
    bytes_saida: int = 0
    falhas: list[FailureRecord] = Field(default_factory=list)
    # ⚠️ NÃO há `gerado_em` aqui, e a ausência é deliberada — duas razões.
    #
    # 1. Non-determinismo por caminho lateral. A raiz guarda o hash do ARQUIVO de
    #    manifesto, então um horário dentro dele fazia o hash raiz mudar a cada
    #    construção de um corpus intacto. Eu havia excluído `gerado_em` da
    #    `identidade()` e esquecido que o arquivo é hasheado inteiro. Pego pelo
    #    teste de idempotência, não pela leitura do código.
    #
    # 2. Provenência enganosa. O campo dizia "gerado em", e quem lê entende "a
    #    etapa rodou em" — mas era quando o MANIFESTO foi escrito, que aqui é dias
    #    depois. Um campo que responde outra pergunta é pior que campo ausente.
    #
    # Quando a etapa passar a gravar o próprio manifesto ao terminar, o horário
    # certo é o da EXECUÇÃO, e entra em `parametros`, onde não afeta a identidade.
    git_sha: str = ""
    tool_version: str = f"phifm.core.schema.reprodutibilidade {ESQUEMA}"

    def identidade(self) -> str:
        """Hash canônico de TUDO que descreve a etapa, menos os campos voláteis.

        `manifesto_id` sai do cálculo porque é o próprio resultado. Não há mais
        campo volátil a excluir: ver o comentário de por que `gerado_em` não existe.
        """
        d = self.model_dump(mode="json", exclude={"manifesto_id"})
        return canonical_hash(d)

    def selar(self) -> ManifestoEtapa:
        self.manifesto_id = self.identidade()
        return self


class EtapaRef(BaseModel):
    tipo: Literal["aquisicao", "processamento"]
    caminho: str            # onde o manifesto da etapa vive
    etapa: str
    manifesto_id: str
    hash_manifesto: str     # hash canônico do ARQUIVO de manifesto como está


class ManifestoRaiz(BaseModel):
    """O "único hash" do G1.5.

    `hash_raiz` é o hash canônico da lista ordenada de etapas. Ordenada por
    `caminho` e não por ordem de descoberta: a mesma árvore em duas máquinas tem
    de dar o mesmo hash, e `glob` não promete ordem entre sistemas de arquivos.
    """

    hash_raiz: str = ""
    schema_version: str = ESQUEMA
    gerado_em: datetime = Field(default_factory=utcnow)
    git_sha: str = ""
    etapas: list[EtapaRef] = Field(default_factory=list)
    bytes_totais: int = 0
    arquivos_totais: int = 0
    # Fontes que uma refeitura NÃO reproduz byte a byte, com o motivo. Ver a
    # docstring do módulo: nomear isto é parte da garantia, não ressalva de rodapé.
    fontes_mutaveis: list[str] = Field(default_factory=list)
    # E as que SÃO reprodutíveis, com o mecanismo. "Reprodutível" sem dizer por que
    # é alegação, e o G1.5 existe para o corpus não depender de alegações.
    fontes_imutaveis: list[str] = Field(default_factory=list)

    def selar(self) -> ManifestoRaiz:
        self.etapas.sort(key=lambda e: e.caminho)
        self.hash_raiz = canonical_hash(
            [e.model_dump(mode="json") for e in self.etapas])
        return self


def gravar_manifesto_etapa(
    *, etapa: str, descricao: str, raiz: Path, base: Path | None = None,
    entradas: list[Entrada] | None = None, parametros: dict | None = None,
    registros: int | None = None, falhas: list[FailureRecord] | None = None,
) -> ManifestoEtapa:
    """Grava o manifesto de uma etapa NO FIM DA EXECUÇÃO dela.

    É o que fecha a metade aberta do G1.5. Hoje os manifestos das etapas já
    executadas carregam `parametros_reconstruidos=True`: os parâmetros foram lidos
    do código pelo construtor do manifesto raiz, não capturados quando a etapa
    rodou. A diferença não é formal — um parâmetro reconstruído pode estar errado
    sem que nada acuse, porque não há nada com que confrontá-lo.

    Chamado daqui, os parâmetros são os ARGUMENTOS DE VERDADE daquela execução, e
    `parametros_reconstruidos` fica `False`. O construtor do raiz preserva
    manifestos assim em vez de sobrescrevê-los — ver `preservavel`.

    Também grava `executado_em`, que é o horário da EXECUÇÃO. Note a diferença com
    o `gerado_em` que foi removido do modelo: aquele dizia quando o manifesto foi
    escrito e quem lia entendia "quando a etapa rodou". Aqui, escrito no fim da
    etapa, os dois coincidem — e por isso o campo pode existir. Ele vai em
    `parametros`, fora da identidade, para não quebrar a idempotência do hash.
    """
    base = base or Path(".")
    idx = indexar(raiz) if raiz.is_dir() else {raiz.name: hash_arquivo(raiz)}
    b, _ = tamanho_de(raiz)
    params = dict(parametros or {})
    params["executado_em"] = _agora()
    # ⚠️ Caminho relativo à base QUANDO possível, absoluto quando não.
    # `relative_to` LEVANTA se a saída estiver fora da base — e uma saída fora da
    # base é o caso normal num teste (`tmp_path`) ou numa execução apontada para
    # outro disco. Derrubar a etapa inteira no fim, depois do trabalho feito, por
    # causa do formato de um caminho no manifesto, seria perder a corrida pelo
    # registro dela.
    try:
        rel = raiz.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        rel = raiz.resolve().as_posix()
    me = ManifestoEtapa(
        etapa=etapa, descricao=descricao,
        raiz=rel,
        entradas=entradas or [], parametros=params,
        parametros_reconstruidos=False,
        registros=registros, checksum_index=idx, bytes_saida=b,
        falhas=falhas or [], git_sha=git_sha_curto()).selar()
    destino = (raiz / NOME_MANIFESTO_ETAPA if raiz.is_dir()
               else raiz.parent / f"{raiz.name}{NOME_MANIFESTO_ETAPA}")
    # ⚠️ Duas etapas que gravam na MESMA `raiz` de diretório apontam para o
    # mesmo arquivo, e a segunda apagava a proveniência da primeira em silêncio.
    # Aconteceu em 2026-08-24: um teste de fumaça de `minerar_do_recuperador.py`
    # (400 âncoras) sobrescreveu o manifesto que estava em
    # `data/processed/negativos_dificeis/`, e não havia como recuperar — o
    # diretório não é versionado.
    #
    # Perder proveniência sem aviso é exatamente o que este módulo existe para
    # impedir, então aqui ele para. Reescrever o manifesto da PRÓPRIA etapa
    # continua livre: reexecutar uma etapa e regravar o manifesto dela é o fluxo
    # normal, e é idempotente.
    if destino.exists():
        try:
            anterior = json.loads(destino.read_text(encoding="utf-8")).get("etapa")
        except Exception:
            anterior = None
        if anterior and anterior != etapa:
            raise RuntimeError(
                f"{destino} já descreve a etapa '{anterior}' e esta é '{etapa}'. "
                "Duas etapas gravando na mesma raiz apagariam a proveniência uma da "
                "outra. Passe `raiz` como o ARQUIVO de saída desta etapa (o "
                "manifesto vira '<arquivo>_manifesto_etapa.json') em vez do "
                "diretório compartilhado.")
    destino.write_text(me.model_dump_json(indent=2), encoding="utf-8")
    return me


def _agora() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def preservavel(destino: Path, idx_atual: dict[str, str]) -> ManifestoEtapa | None:
    """O manifesto existente foi capturado na execução E ainda descreve o disco?

    Se sim, o construtor do raiz tem de PRESERVÁ-LO. Sobrescrever um manifesto com
    parâmetros de verdade por um com parâmetros reconstruídos seria trocar
    proveniência boa por proveniência adivinhada — e o construtor roda muito mais
    vezes que as etapas.

    Se o índice divergir, a etapa rodou de novo sem gravar o seu manifesto (ou
    alguém mexeu nos arquivos), e aí o reconstruído é o melhor disponível.
    """
    if not destino.exists():
        return None
    try:
        me = ManifestoEtapa.model_validate_json(destino.read_text(encoding="utf-8"))
    except Exception:
        return None
    if me.parametros_reconstruidos:
        return None
    return me if me.checksum_index == idx_atual else None


class Divergencia(BaseModel):
    tipo: Literal["ausente", "alterado", "extra", "manifesto_alterado",
                  "etapa_ausente", "raiz_alterada"]
    onde: str
    detalhe: str = ""


class Relatorio(BaseModel):
    ok: bool
    profundo: bool
    hash_raiz_esperado: str
    hash_raiz_obtido: str
    etapas_conferidas: int = 0
    arquivos_conferidos: int = 0
    divergencias: list[Divergencia] = Field(default_factory=list)

    def resumo(self) -> str:
        if self.ok:
            modo = "profunda" if self.profundo else "rasa"
            return (f"✅ verificação {modo} OK · hash raiz {self.hash_raiz_obtido[:16]}… · "
                    f"{self.etapas_conferidas} etapas, {self.arquivos_conferidos} arquivos")
        linhas = [f"❌ {len(self.divergencias)} divergência(s):"]
        for d in self.divergencias[:20]:
            linhas.append(f"   [{d.tipo}] {d.onde} {d.detalhe}".rstrip())
        if len(self.divergencias) > 20:
            linhas.append(f"   … e mais {len(self.divergencias) - 20}")
        return "\n".join(linhas)


def verificar(raiz_json: Path, *, profundo: bool = False,
              base: Path | None = None) -> Relatorio:
    """Confere o corpus contra o manifesto raiz.

    Dois níveis, e a diferença entre eles é o que cada um pega:

    - **rasa**: os manifestos de etapa existem e não foram alterados, e o
      `hash_raiz` recalculado bate. Custa milissegundos. Pega manifesto mexido.
    - **profunda**: relê cada arquivo de cada `checksum_index`. Custa a leitura
      dos 22 GB. É a única que pega **parquet** mexido.

    Uma verificação rasa que passa não diz que o corpus está íntegro — diz que os
    manifestos estão. Chamar isso de "corpus verificado" seria o defeito de sempre:
    ausência de erro lida como sucesso.
    """
    base = base or raiz_json.parent
    import json

    raiz = ManifestoRaiz.model_validate_json(raiz_json.read_text(encoding="utf-8"))
    esperado = raiz.hash_raiz
    div: list[Divergencia] = []
    n_arq = 0

    # 1. O hash raiz é derivável da própria lista? Pega raiz editada à mão.
    recalc = canonical_hash([e.model_dump(mode="json") for e in raiz.etapas])
    if recalc != esperado:
        div.append(Divergencia(
            tipo="raiz_alterada", onde=raiz_json.name,
            detalhe=f"lista de etapas dá {recalc[:16]}…, o arquivo diz {esperado[:16]}…"))

    for ref in raiz.etapas:
        p = base / ref.caminho
        if not p.exists():
            div.append(Divergencia(tipo="etapa_ausente", onde=ref.caminho))
            continue
        if hash_arquivo(p) != ref.hash_manifesto:
            div.append(Divergencia(
                tipo="manifesto_alterado", onde=ref.caminho,
                detalhe="o arquivo de manifesto não bate com o hash registrado na raiz"))
            continue

        if not profundo:
            continue

        d = json.loads(p.read_text(encoding="utf-8"))
        idx = d.get("checksum_index") or {}
        # A aquisição grava o índice relativo à pasta dela; o processamento grava
        # `raiz` explícito. Os dois casos resolvem para o mesmo lugar.
        dir_etapa = base / d["raiz"] if "raiz" in d else p.parent
        alvo = dir_etapa if dir_etapa.is_dir() else dir_etapa.parent

        for rel, h in idx.items():
            f = alvo / rel
            n_arq += 1
            if not f.exists():
                div.append(Divergencia(tipo="ausente", onde=f"{ref.etapa}:{rel}"))
            elif hash_arquivo(f) != h:
                div.append(Divergencia(tipo="alterado", onde=f"{ref.etapa}:{rel}",
                                       detalhe="conteúdo difere do hash registrado"))

        # Arquivo a mais é divergência: um corpus com um parquet extra não é o
        # corpus manifestado, e um treino que o leia mede outra coisa.
        #
        # ⚠️ Só quando a raiz da etapa é DIRETÓRIO. Quando é um arquivo
        # (`spine.parquet`), o índice tem exatamente uma entrada e não há nada a
        # enumerar — varrer o diretório pai acusava o repositório inteiro como
        # extra, o que um teste pegou de imediato.
        if dir_etapa.is_dir() and idx:
            vistos = set(idx)
            for f in indexar(alvo):
                if f not in vistos:
                    div.append(Divergencia(tipo="extra", onde=f"{ref.etapa}:{f}",
                                           detalhe="não está no índice do manifesto"))

    return Relatorio(ok=not div, profundo=profundo, hash_raiz_esperado=esperado,
                     hash_raiz_obtido=recalc, etapas_conferidas=len(raiz.etapas),
                     arquivos_conferidos=n_arq, divergencias=div)


def git_sha_curto() -> str:
    """SHA do commit atual, ou "sujo"/"desconhecido" — nunca string vazia calada.

    Um manifesto sem proveniência de código é um manifesto que não diz de que
    versão do pipeline aquele corpus saiu, e é justamente o que o G1.5 pede.
    """
    import subprocess

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=30,
                             check=True).stdout.strip()
        sujo = subprocess.run(["git", "status", "--porcelain"],
                              capture_output=True, text=True, timeout=30,
                              check=True).stdout.strip()
        return f"{sha}-sujo" if sujo else sha
    except Exception:
        return "desconhecido"


def tamanho_de(raiz: Path) -> tuple[int, int]:
    """(bytes, arquivos) de uma árvore, pelos mesmos critérios do índice."""
    if raiz.is_file():
        return raiz.stat().st_size, 1
    b = n = 0
    for rel in indexar(raiz):
        b += (raiz / rel).stat().st_size
        n += 1
    return b, n
