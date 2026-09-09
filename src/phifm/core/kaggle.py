"""Registro dos experimentos que rodam na cota gratuita do Kaggle.

Existe porque `empacotar_kaggle.py` e `publicar_kaggle.py` precisam concordar sobre
os mesmos nomes, e a primeira versão dos dois tinha os slugs como constantes de
módulo em CADA script. Acrescentar um segundo experimento por cópia duplicaria
também a lição de cada erro já pago lá — `machine_shape`, `.zip.bin`, o pin da
imagem docker — e a cópia divergiria em silêncio.

Este módulo é de propósito sem torch, sem polars e sem rede: ele é só o contrato de
nomes, e por isso entra na suíte rápida.

## O invariante que este módulo faz valer, e só onde ele vale

No `kernels push` o Kaggle **deriva o slug do TÍTULO** e derruba tudo que não for
`[a-z0-9]`. Medido em 2026-08-24: o título "PhiFM T1a - PhiEmb" virou
`phifm-t1a-emb` — o Φ e o hífen solto sumiram —, o slug derivado deixou de casar com
o `id` declarado (`phifm-t1a-phiemb`) e a CLI devolveu 409 depois do upload.
`conferir()` compara `slug_derivado(titulo)` com o slug declarado e levanta ANTES de
subir nada.

⚠️ **Isto NÃO vale para o dataset**, e eu quase gravei a regra errada. A primeira
versão deste módulo checava os dois, e a suíte reprovou em segundos: o dataset do
T1a está publicado desde 2026-08-24 com o título "PhiFM T1a — pares de citação
arXiv" e o slug `phifm-t1a-pares-citacao`, que aquele título não deriva — travessão,
cedilha e a palavra "arXiv" a mais. Para `datasets create` quem manda é o campo
`id`; só o `kernels push` deriva do título.

Generalizar uma lição para além do que ela mediu tem custo: aqui teria proibido um
título legível em português por um problema que o dataset não tem.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# Como o Kaggle transforma título em slug: baixa a caixa, e tudo que não é
# alfanumérico ASCII colapsa num hífen. Não é adivinhação — é o que reproduz o
# `phifm-t1a-emb` observado.
_NAO_SLUG = re.compile(r"[^a-z0-9]+")


def slug_derivado(titulo: str) -> str:
    return _NAO_SLUG.sub("-", titulo.lower()).strip("-")


def assinatura_do_manifesto(arquivos: dict[str, dict]) -> str:
    """Uma impressão digital do bundle inteiro, a partir dos hashes declarados.

    ## ⚠️ Por que ela não é redundante com a conferência de blake3

    A célula já confere cada arquivo contra o `MANIFESTO.json` — mas contra o
    manifesto que veio **no mesmo dataset**. Isso pega upload truncado e arquivo
    trocado, e **não** pega bundle da versão errada, porque um bundle velho é
    internamente consistente: os hashes dele batem com os arquivos dele.

    O Kaggle FIXA a versão do dataset no momento em que ela é anexada ao kernel, e
    `kernels push` não re-resolve para a mais recente. Medido em 2026-09-03 na T1c:
    uma versão nova subiu, `datasets status` disse `ready`, e o notebook rodou 15
    min sobre o conteúdo ANTIGO. Só foi percebido porque a saída imprimia o
    `git_sha` e alguém leu — e depender de leitura humana não é uma guarda.

    Só um valor vindo de FORA do dataset distingue os dois. Este é injetado na
    célula no momento da publicação, quando esse valor existe.

    Cobre TODOS os arquivos declarados, e não um escolhido pela ordem da tupla.
    """
    if not arquivos:
        raise ValueError("manifesto sem arquivos: não há bundle para assinar")
    corpo = "".join(f"{nome}:{arquivos[nome]['blake3']}\n"
                    for nome in sorted(arquivos))
    return hashlib.sha256(corpo.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Experimento:
    """Um experimento de GPU: o que empacotar, com que nome, e qual célula roda."""

    nome: str
    titulo_dados: str
    slug_dados: str
    titulo_notebook: str
    slug_notebook: str
    # Caminhos RELATIVOS à raiz do repositório. Absolutos aqui fariam o registro
    # depender da máquina, e ele é lido tanto pelos scripts quanto pelos testes.
    pacote: str
    fonte_celula: str
    # ⚠️ Lista DECLARADA do que o pacote contém. Montar o manifesto de `iterdir()`
    # já atestou 175 KB de código obsoleto como se fizesse parte do pacote
    # (2026-08-24). O que não está aqui é removido do diretório de saída.
    arquivos: tuple[str, ...]
    # Scripts do repositório que entram no `phifm_src.zip.bin`, para o notebook
    # chamar exatamente o que roda na máquina local em vez de reimplementar.
    scripts: tuple[str, ...]
    # Diretórios de modelo (relativos à raiz) que sobem junto. Vazio quando os
    # pesos são públicos e o Kaggle os baixa do HuggingFace — subir 90 MB de
    # `all-MiniLM-L6-v2` seria pagar banda por algo que já está lá.
    modelos: tuple[str, ...] = ()
    # ⚠️ Volume de pares do experimento, e não uma bandeira de linha de comando.
    #
    # Ele é a VARIÁVEL de um experimento de volume, então pertence à identidade
    # dele: com `--max-pares` solto, rodar `empacotar_kaggle.py --experimento
    # t1a15` sem a bandeira montaria 400 mil pares sob o nome do de 1,5 M, e o
    # manifesto atestaria o número errado com a cara certa.
    max_pares: int = 400_000
    # ⚠️ O arquivo de NEGATIVOS pertence à identidade, pelo mesmo motivo do
    # `max_pares` acima.
    #
    # O `--negativos` do empacotador tinha default fixo apontando para os
    # negativos da FUSÃO. Montar o `t1d` sem a bandeira empacotaria os antigos
    # sob o nome do experimento novo, e o manifesto atestaria o arquivo errado
    # com a cara certa.
    #
    # No T1d isso seria fatal e invisível: a hipótese sob teste É a distribuição
    # dos negativos. O pacote diria "densos" e conteria os da fusão, o resultado
    # sairia igual ao do T1c, e a leitura seria "a distribuição não importa".
    negativos: str | None = None
    # `owner/repo` do GitHub. Quando preenchido, o código NÃO viaja no dataset: o
    # notebook baixa o tarball do commit exato.
    #
    # ⚠️ Isto existe por uma falha medida em 2026-09-03. O Kaggle **fixa a versão do
    # dataset** no momento em que ela é anexada ao kernel, e `kernels push` não
    # re-resolve para a mais recente — o `dataset_sources` do metadado nem carrega
    # número de versão. Resultado: o conserto do fp16 subiu numa versão nova, o
    # `datasets status` disse `ready`, e o notebook rodou 15 min sobre o código
    # ANTIGO. Ele delatou na saída (`git_sha: 73088dc` contra o conserto em
    # `68fe86e`), que foi a única razão de eu perceber.
    #
    # Com o código vindo do GitHub num SHA, não há versão a fixar: o notebook é
    # reempurrado a cada publicação. E os dados podem ficar no dataset justamente
    # porque não mudam — era o zip de 188 KB que mudava dentro dos 457 MB.
    repo: str | None = None

    def conferir(self) -> None:
        """Só o notebook. Ver a ressalva na docstring do módulo."""
        obtido = slug_derivado(self.titulo_notebook)
        if obtido != self.slug_notebook:
            raise ValueError(
                f"{self.nome}: o título do notebook {self.titulo_notebook!r} deriva "
                f"o slug {obtido!r}, e o declarado é {self.slug_notebook!r}. O "
                f"`kernels push` usa o derivado e rejeita o id com 409 — depois de "
                f"receber os arquivos. Ajuste o título ou o slug para coincidirem.")


T1A = Experimento(
    nome="t1a",
    # ⚠️ Slug NOVO em 2026-09-06, e o motivo é a versão fixada no anexo.
    #
    # O `pares_treino.parquet` mudou: deixou de ser `head(400.000)` e passou a ser
    # sorteio — 191.300 documentos citados distintos em vez de 17.844. O dataset
    # antigo (`phifm-t1a-pares-citacao`, publicado em 2026-08-24) continua anexado
    # ao kernel na versão dele, e `kernels push` não re-resolve para a mais
    # recente: subir uma versão nova ali produziria um retreino silencioso sobre os
    # dados velhos. Um dataset recém-criado tem uma versão só, e não há versão
    # velha para o kernel fixar.
    titulo_dados="PhiFM T1a — pares de citação arXiv sorteados",
    slug_dados="phifm-t1a-pares-sorteados",
    titulo_notebook="PhiFM T1a Gpu",
    slug_notebook="phifm-t1a-gpu",
    pacote="data/processed/kaggle_t1a",
    fonte_celula="kaggle/t1a_phiemb.py",
    # Sem `phifm_src.zip.bin`: o código vem do GitHub (ver `repo`).
    arquivos=("pares_treino.parquet", "pares_validacao.parquet"),
    scripts=("train_embedding.py",),
    # 400 mil: o volume do campeao do G1.1.
    #
    # A justificativa original era "mais que isso a medicao diz que nao compra
    # nada (p=0,950)", e ela esta EM DUVIDA desde 2026-09-07. Ela vinha do run de
    # 1,5 M interrompido em 38% por plato -- e o plato era artefato do prefixo:
    # os 256 mil pares novos saiam de 67 mil documentos, entao era exaustao de
    # DOCUMENTOS, nao de dados. O run sorteado de 400 mil tambem nao platoou
    # (nDCG@10 subindo ate o passo 2.800 de 3.125).
    #
    # E por isso que o `t1a15` existe: para remedir com 390.966 documentos.
    max_pares=400_000,
    repo="sanchezVB/LLM_F-sica",
)

# ⚠️ Mesma célula, mesmo código, mesmo tudo — o volume é a única variável.
#
# O run de 400 mil sorteados NÃO platôou: nDCG@10 subiu até o passo 2.800 de 3.125
# (0,5515 → 0,6147). E o "platô medido" que interrompeu o run histórico de 1,5 M em
# 38% era artefato do prefixo: os 256 mil pares novos saíam de 67 mil documentos,
# então era exaustão de DOCUMENTOS, não de dados. Sorteados, 1,5 M dão **390.966**
# documentos citados distintos.
#
# Slug de dataset e de notebook próprios, e não uma versão nova dos do T1a: o
# Kaggle fixa a versão do dataset no anexo e `kernels push` não re-resolve. Subir
# 1,5 M como versão nova faria a assinatura do bundle LEVANTAR na célula — a guarda
# funcionando, e o run bloqueado. Notebook separado também preserva a execução de
# 400 mil como registro, em vez de sobrescrevê-la.
T1A15 = Experimento(
    nome="t1a15",
    titulo_dados="PhiFM T1a 1,5 M — pares de citação arXiv sorteados",
    slug_dados="phifm-t1a-pares-15m",
    titulo_notebook="PhiFM T1a 15m Gpu",
    slug_notebook="phifm-t1a-15m-gpu",
    pacote="data/processed/kaggle_t1a15",
    fonte_celula="kaggle/t1a_phiemb.py",
    arquivos=("pares_treino.parquet", "pares_validacao.parquet"),
    scripts=("train_embedding.py",),
    max_pares=1_500_000,
    repo="sanchezVB/LLM_F-sica",
)

# Terceiro ponto da curva de volume. O run de 1,5 M empatou com o GTE-large
# (nDCG@10 0,5780 contra 0,5788) e o pico ficou no passo 11.400 de 11.718 -- 97%
# do caminho, sem plato. O corpus tem 6.564.111 pares e 667.304 documentos
# citados distintos, entao 3 M ainda ficam bem abaixo do teto de dado.
#
# ~4h45 de T4 a 196,2 pares/s: cabe numa sessao de 9 h e na cota de 30 h/semana.
T1A3M = Experimento(
    nome="t1a3m",
    titulo_dados="PhiFM T1a 3 M — pares de citação arXiv sorteados",
    slug_dados="phifm-t1a-pares-3m",
    titulo_notebook="PhiFM T1a 3m Gpu",
    slug_notebook="phifm-t1a-3m-gpu",
    pacote="data/processed/kaggle_t1a3m",
    fonte_celula="kaggle/t1a_phiemb.py",
    arquivos=("pares_treino.parquet", "pares_validacao.parquet"),
    scripts=("train_embedding.py",),
    max_pares=3_000_000,
    repo="sanchezVB/LLM_F-sica",
)

# Quarto ponto da curva, e o ultimo que cabe numa sessao.
#
# Medido no run de 3 M: 197,8 pares/s ponta a ponta, ja com as avaliacoes. Entao
#
#     5,4 M -> 7h35    630.173 documentos (94,4%)
#     6,0 M -> 8h25    650.162 documentos (97,4%)   <- este
#     6,56 M -> 9h13   667.304 documentos (100%)    <- ESTOURA a sessao de 9 h
#
# 6 M e o maior volume com margem real (~35 min) e entrega 97,4% dos documentos.
# O corpus inteiro precisaria de retomada entre sessoes, e a retomada do Kaggle
# depende de `/kaggle/working` que NAO sobrevive a um relancamento -- seria outro
# problema de engenharia, nao mais dado.
T1A6M = Experimento(
    nome="t1a6m",
    titulo_dados="PhiFM T1a 6 M — pares de citação arXiv sorteados",
    slug_dados="phifm-t1a-pares-6m",
    titulo_notebook="PhiFM T1a 6m Gpu",
    slug_notebook="phifm-t1a-6m-gpu",
    pacote="data/processed/kaggle_t1a6m",
    fonte_celula="kaggle/t1a_phiemb.py",
    arquivos=("pares_treino.parquet", "pares_validacao.parquet"),
    scripts=("train_embedding.py",),
    max_pares=6_000_000,
    repo="sanchezVB/LLM_F-sica",
)

# Remedicao da CADEIA depois da troca do recuperador em 2026-09-08.
#
# O nDCG 0,1666 do PhiRank (p=0,0062) foi medido sobre a fusao RRF do recuperador
# ANTIGO, e esta obsoleto. Os DOIS recuperadores vao no bundle porque a celula
# mede os dois na MESMA sessao: comparar contra o numero de agosto seria invalido
# -- entre agosto e hoje mudaram o protocolo do G1, o pool de candidatos e quatro
# versoes do codigo, e a diferenca nao seria atribuivel a troca.
T1B2 = Experimento(
    nome="t1b2",
    titulo_dados="PhiFM T1b2 — cadeia remedida",
    # ⚠️ Slug DIFERENTE do notebook. Na Kaggle os dois vivem em namespaces
    # separados (`/datasets/<dono>/<slug>` e `/code/<dono>/<slug>`), então
    # compartilhar pareceria inofensivo — mas a suite recusa, e com razao: com o
    # mesmo slug nos dois, qualquer erro de digitacao num comando aponta para o
    # objeto errado sem avisar, e a mensagem de erro da CLI nao distingue.
    slug_dados="phifm-t1b2-pares-e-modelos",
    titulo_notebook="PhiFM T1b2 Cadeia",
    slug_notebook="phifm-t1b2-cadeia",
    pacote="data/processed/kaggle_t1b2",
    fonte_celula="kaggle/t1b2_cadeia.py",
    arquivos=("pares_validacao.parquet", "modelos.zip.bin"),
    scripts=("avaliar_t1b.py",),
    modelos=("models/phiemb-do-sistema", "models/phiemb-minilm-melhor",
             "models/phirank-physbert-melhor"),
    repo="sanchezVB/LLM_F-sica",
)

T1C = Experimento(
    nome="t1c",
    titulo_dados="PhiFM T1c — ΦRank de base diferente",
    slug_dados="phifm-t1c-rerank-bases",
    titulo_notebook="PhiFM T1c Rerank",
    slug_notebook="phifm-t1c-rerank",
    pacote="data/processed/kaggle_t1c",
    fonte_celula="kaggle/t1c_phirank.py",
    # Sem `phifm_src.zip.bin`: o código vem do GitHub (ver `repo`).
    arquivos=("pares_do_recuperador_limpos.parquet", "pares_validacao.parquet",
              "modelos.zip.bin"),
    scripts=("train_rerank.py", "avaliar_t1b.py"),
    repo="sanchezVB/LLM_F-sica",
    # ⚠️ O ΦEmb vai junto e é o `phiemb-minilm-melhor`, NÃO o `-t4-melhor`.
    # O resultado de referência do T1b (nDCG 0,1584) foi medido com este, e trocar
    # o recuperador ao mesmo tempo que o reranqueador mediria duas coisas.
    #
    # O `phirank-rrf-melhor` é o CONTROLE: é o reranqueador que empatou com a fusão
    # (p=0,118), e ele precisa ser reavaliado no MESMO número de consultas que os
    # novos, senão a comparação de poder estatístico fica torta.
    modelos=("models/phiemb-minilm-melhor", "models/phirank-rrf-melhor"),
    negativos=("data/processed/negativos_dificeis/"
               "pares_do_recuperador_limpos.parquet"),
)

T1D = Experimento(
    nome="t1d",
    titulo_dados="PhiFM T1d — ΦRank nos negativos do recuperador novo",
    slug_dados="phifm-t1d-negativos-densos",
    titulo_notebook="PhiFM T1d Rerank Denso",
    slug_notebook="phifm-t1d-rerank-denso",
    pacote="data/processed/kaggle_t1d",
    fonte_celula="kaggle/t1d_phirank_denso.py",
    # ⚠️ Os negativos DENSOS e limpos: minerados do top-50 do ΦEmb, que é a
    # composição decidida no T1b2, e passados pelo filtro de co-citação.
    #
    # O ΦRank instalado foi treinado com negativos minerados pela FUSÃO RRF, que
    # deixou de ser a composição — a distribuição de treino dele não é mais a
    # distribuição que ele vê. É a causa mecânica provável de o ganho marginal
    # dele ter encolhido de p=0,0081 para p=0,086.
    arquivos=("pares_do_recuperador_denso_limpos.parquet",
              "pares_validacao.parquet", "modelos.zip.bin"),
    scripts=("train_rerank.py", "avaliar_t1b.py"),
    # ⚠️ O ΦRank ANTIGO vai junto, e é o ponto do experimento.
    #
    # A pergunta é "quanto mudou", e essa exige o braço de referência medido na
    # MESMA sessão. Em 2026-09-08 o T1b2 mediu o recuperador antigo junto e
    # descobriu que o número histórico (0,1666) era 0,1685 no protocolo de hoje:
    # comparar contra o histórico teria reportado SETE VEZES o efeito real.
    modelos=("models/phiemb-do-sistema", "models/phirank-physbert-melhor"),
    negativos=("data/processed/negativos_dificeis/"
               "pares_do_recuperador_denso_limpos.parquet"),
    repo="sanchezVB/LLM_F-sica",
)

# ⚠️ Os experimentos de VOLUME da T1a. A lista existe para o teste conferir que
# eles só diferem no volume e nos slugs — três entradas quase idênticas divergem
# em silêncio, e foi para não duplicar lição paga que este módulo nasceu.
VARIANTES_DE_VOLUME = ("t1a", "t1a15", "t1a3m", "t1a6m")

EXPERIMENTOS: dict[str, Experimento] = {
    e.nome: e for e in (T1A, T1A15, T1A3M, T1A6M, T1B2, T1C, T1D)}


def obter(nome: str) -> Experimento:
    try:
        exp = EXPERIMENTOS[nome]
    except KeyError:
        raise SystemExit(
            f"experimento {nome!r} não existe. Conhecidos: "
            f"{sorted(EXPERIMENTOS)}") from None
    exp.conferir()
    return exp
