"""§2.3 — o braço TRATADO da ablação de mascaramento de equações.

Este arquivo é o CONTEÚDO de uma célula do Kaggle, como `t2a_tokenizer.py`. A
diferença é que esta célula **não é escrita: é DERIVADA** da célula do T2a.

## A pergunta

O DOC-07 §2.3 chama o mascaramento consciente de equações de *"a única adição
específica de Física ao objetivo de pré-treino"*: em parte dos exemplos, mascarar
uma equação INTEIRA, para o modelo reconstruí-la a partir da prosa. *"Se não ajudar,
é descartado e o negativo é publicado."*

## ⚠️ O controle já existe, e isto é a premissa, não economia

O braço **E do T2a** treinou com `--p-equacao 0.0` — o controle desta ablação — no
tokenizer que venceu o bake-off (DOC-05 §11.2-medido). Rodar controle e tratado de
novo custaria 16,5 h de T4, e a cota tem 10 h 25. Reaproveitar só é honesto se o
tratado for o controle com UMA variável trocada, e é por isso que esta célula não é
escrita à mão:

    CELULA = derivar(CELULA do T2a)

`derivar` aplica as trocas declaradas em `BLOCOS` e `TROCAS`, cada uma exigida
EXATAMENTE uma vez. `tests/regression/test_kaggle_t2eq.py` recalcula a derivação
e confere que a célula publicada é ela, byte a byte, e que nenhuma troca toca o
orçamento, a semente, a configuração, a fatia ou a conferência de dados. O que
difere, portanto, é só:

- `p_equacao` 0,0 → **0,6** (decidido antes, ver `scripts/train_phienc.py`);
- a variante travada em E;
- os NOMES dos artefatos, com `_tratado`, para o download não sobrescrever o
  controle em casa;
- a REGRA e o cabeçalho, que são de outro experimento.

⚠️ E uma diferença de CÓDIGO, fora da célula: o controle rodou em `fc1523a`, e este
braço roda no commit publicado, que inclui a correção do corte no fim da janela
(`dados.desempacotar`, a167bfd). Com `p_equacao=0` ela não muda nenhuma máscara — há
teste (`test_o_CONTROLE_nao_muda_com_a_correcao`). Todo o resto do caminho de
treino é idêntico a `fc1523a`, conferido por `git diff`.

## O que a avaliação ainda NÃO tem, e tem de ter antes de medir

As três avaliações do §11.2 estão no branch do Mac (`claude/artigo-modelo-ia-
fisica-ey4lbe`), não em `main`. Lidas contra esta REGRA, faltam três coisas no
`eval/mlm.py` de lá:

1. **A célula de mecanismo não tem teste.** `avaliar_ablacao` reporta só a
   diferença das médias; o teste pareado de lá é de sinais sobre a perda GERAL da
   sequência, que mistura prosa. A REGRA pede bootstrap pareado por sequência da
   perda nos tokens de equação.
2. **O default é 64 sequências.** A REGRA fixa 2.000 — a lição do T1f, onde um
   default pequeno quase registrou um empate que era vitória.
3. **`medir` usa `torch.device(dispositivo)`**, a regressão do `--dispositivo dml`
   que `main` consertou em `c35eebb`.
"""
from __future__ import annotations

# ── a REGRA, escrita antes de o braço existir ───────────────────────────────
REGRA_ABLACAO = r"""
  ── A REGRA, escrita ANTES ────────────────────────────────────────────────

  A PERGUNTA (DOC-07 §2.3): mascarar uma equação INTEIRA, em parte dos
  exemplos, ensina a relação entre a prosa e a expressão formal?

  CONTROLE: o braço E do T2a, já treinado (kernel phifm-t2a-tokenizer-e,
  código fc1523a, p_equacao 0,0). TRATADO: esta célula, p_equacao 0,6. Mesma
  fatia, mesmo tokenizer, mesmo orçamento, mesma semente.

  ⚠️ ESTE RUN NÃO É A ABLAÇÃO COMO PLANEJADA. Ela foi preparada a 2 B tokens
     (21 tokens/parâmetro); isto roda 0,6 B, 3,3x menos, porque o controle
     existe a 0,6 B e reaproveitá-lo é o que faz caber na cota.

  CHECAGENS DE MANIPULAÇÃO — reprovada qualquer uma, o run não testa o §2.3:
    1. fração tratada no treino >= 0,50. Medida antes nesta fatia: 0,549.
    2. no regime de EQUAÇÃO, o tratado tem perda menor que o controle. É a
       célula tautológica: se nem nela ele ganha, o tratamento não pegou.

  MEDIDA PRIMÁRIA: a célula de MECANISMO — perda nos tokens de equação em
  display (inteira na janela) sob mascaramento ALEATÓRIO, a tarefa do controle.
    · conjunto DISJUNTO: partes que nem A nem E usaram no treino;
    · 2.000 sequências, as MESMAS máscaras aplicadas aos dois braços;
    · bootstrap pareado por SEQUÊNCIA da perda média nos tokens de equação,
      IC de 95%. NÃO o teste de sinais sobre a perda geral, que mistura prosa.

  ⚠️ O viés da primária aponta para o CONTROLE: o regime aleatório é o
     objetivo em que ele treinou, e o tratado gastou 1/3 dos exemplos noutra
     coisa.

  Os desfechos:

    TRATADO à frente  ->  robusto: venceu no objetivo do outro. A hipótese do
                          §2.3 fica sustentada a 0,6 B.

    CONTROLE à frente ->  ambíguo entre "não há mecanismo" e "custo geral de
                          distribuição". Decide a diferença das diferenças,
                          mesmo bootstrap: (tratado − controle) nos tokens de
                          equação MENOS (tratado − controle) nos de prosa.
                          Prejuízo em equação MAIOR que em prosa (IC exclui
                          zero) -> é o negativo que o DOC-07 manda publicar.
                          Senão -> custo geral, não refutação: não decidido.

    EMPATE            ->  ⚠️ NÃO é o negativo do DOC-07. É "a 0,6 B não dá
                          para ver", registrado como não decidido.

  Secundárias: recuperação de Física e sonda tensorial, controle × tratado.
  Não derrubam a primária; se discordarem, a discordância é o resultado.

  ⚠️ O empate é o desfecho mais provável de um run subdimensionado, e o mais
     fácil de escrever como "tentamos, não funciona". O DOC-07 promete publicar
     o negativo — e um empate a 0,6 B não é esse negativo.
"""

CABECALHO_ABLACAO = r"""{'=' * 74}
§2.3 — mascarar equações INTEIRAS ensina alguma coisa?  ·  braço TRATADO

  tokenizer {VARIANTE} (o vencedor do T2a) · p_equacao {P_EQUACAO}
  O controle NÃO roda aqui: é o braço E do T2a, já treinado com p_equacao 0,0.
  Esta célula é a do controle com trocas DECLARADAS em kaggle/t2eq_tratado.py,
  e um teste confere que não há nenhuma outra.

  orçamento IDÊNTICO ao do controle: {TOKENS:,} tokens · contexto {CONTEXTO} ·
                      {SEQUENCIAS}x{ACUMULACAO} · {PASSOS:,} passos
                      teto de {LIMITE_H} h, conferido contra a vazão MEDIDA
  medido antes, nesta fatia, com o código corrigido: fração tratada 0,549,
  equação escolhida com mediana de 76 tokens, 0 atravessando o fim da janela.

  ⚠️ A COMPARAÇÃO NÃO ACONTECE AQUI. Ela roda LOCAL, com os dois checkpoints
     no mesmo processo e as MESMAS máscaras.
"""

# ── as trocas, e NADA além delas ────────────────────────────────────────────
#
# Cada bloco é `(começa com, termina com, texto novo)` e cada troca pontual é
# `(antigo, novo)`. `derivar` exige que cada âncora apareça EXATAMENTE uma vez na
# célula do controle: uma âncora que some por edição da célula do T2a quebra a
# derivação alto, em vez de deixar passar uma célula que diverge em silêncio.
BLOCOS: tuple[tuple[str, str, str], ...] = (
    ('REGRA = r"""\n',
     '{REGRA}\n{\'=\' * 74}\n""", flush=True)',
     'REGRA = r"""' + REGRA_ABLACAO + '"""\n\nprint(f"""\n' + CABECALHO_ABLACAO
     + '{REGRA}\n{\'=\' * 74}\n""", flush=True)'),
)

TROCAS: tuple[tuple[str, str], ...] = (
    ('assert VARIANTE in ("A", "E"), VARIANTE\n',
     'assert VARIANTE == "E", (\n'
     '    f"o braço tratado do §2.3 usa o tokenizer E, o vencedor do T2a e o do "\n'
     '    f"controle já treinado; recebi {VARIANTE}. Com outro tokenizer o controle "\n'
     '    f"não serve.")\n'
     '\n'
     '# ⚠️ A ÚNICA variável do experimento. Decidida em 2026-09-16, antes de este braço\n'
     '# existir — a conta está no docstring de `scripts/train_phienc.py`.\n'
     'P_EQUACAO = 0.6\n'),
    ('RUN = TRABALHO / f"phienc_{VARIANTE}"\n',
     'RUN = TRABALHO / f"phienc_{VARIANTE}_tratado"\n'),
    ('"--p-equacao", 0.0, "--semente", 17,',
     '"--p-equacao", P_EQUACAO, "--semente", 17,'),
    ('TRABALHO / f"treino_{VARIANTE}.log")',
     'TRABALHO / f"treino_{VARIANTE}_tratado.log")'),
    ('EXPORTADO = TRABALHO / f"phienc-{VARIANTE}"\n',
     'EXPORTADO = TRABALHO / f"phienc-{VARIANTE}-tratado"\n'),
    ('"--nota", f"T2a braço {VARIANTE} · {TOKENS:,} tokens · contexto {CONTEXTO}"],',
     '"--nota", f"§2.3 braço TRATADO · p_equacao {P_EQUACAO} · tokenizer {VARIANTE} '
     '· {TOKENS:,} tokens · contexto {CONTEXTO}"],'),
    ('TRABALHO / f"exportar_{VARIANTE}.log")',
     'TRABALHO / f"exportar_{VARIANTE}_tratado.log")'),
    ('(TRABALHO / f"t2a_{VARIANTE}.json").write_text(json.dumps({\n'
     '    "variante": VARIANTE,\n',
     '(TRABALHO / f"t2eq_tratado_{VARIANTE}.json").write_text(json.dumps({\n'
     '    "variante": VARIANTE,\n'
     '    "p_equacao": P_EQUACAO,\n'
     '    "controle": ("braço E do T2a, p_equacao 0,0, kernel "\n'
     '                 "phifm-t2a-tokenizer-e, código fc1523a"),\n'),
    ('    "nota": ("A comparação A×E NÃO está aqui. Cada braço treina numa sessão e as "\n'
     '             "três medidas do §11.2 rodam local, com os dois checkpoints no mesmo "\n'
     '             "processo — a variável é o treino, a medição é única."),\n',
     '    "nota": ("A comparação controle × tratado NÃO está aqui. Ela roda local, com "\n'
     '             "os dois checkpoints no mesmo processo e as MESMAS máscaras, pela "\n'
     '             "REGRA impressa no início desta célula."),\n'),
    ('print(f"  T2a braço {VARIANTE} · {PASSOS:,} passos · {custo_treino/3600:.2f} h")',
     'print(f"  §2.3 braço TRATADO · tokenizer {VARIANTE} · p_equacao {P_EQUACAO} · "\n'
     '      f"{PASSOS:,} passos · {custo_treino/3600:.2f} h")'),
    ("print(f\"  spikes: {(metricas.get('spike') or {}).get('n_spikes')}\")\n",
     "print(f\"  spikes: {(metricas.get('spike') or {}).get('n_spikes')}\")\n"
     '_ft = (metricas.get("mascaramento") or {}).get("fracao_tratada")\n'
     'print(f"  fração tratada: {_ft}  (medida antes nesta fatia: 0,549)")\n'
     'if _ft is None or _ft < 0.50:\n'
     '    print("  ⚠️ CHECAGEM DE MANIPULAÇÃO 1 REPROVADA: abaixo de 0,50 o braço "\n'
     '          "tratado recaiu em MLM aleatório, e este run NÃO testa o §2.3.")\n'),
    ('print(f"\\n⚠️ Baixe `phienc-{VARIANTE}/` inteiro, `t2a_{VARIANTE}.json` e os logs. "\n'
     '      "Rode o outro braço, e só então meça os dois juntos, LOCAL.")',
     'print(f"\\n⚠️ Baixe `phienc-{VARIANTE}-tratado/` inteiro, "\n'
     '      f"`t2eq_tratado_{VARIANTE}.json` e os logs. O controle já está em casa; "\n'
     '      "meça os dois juntos, LOCAL, pela REGRA.")'),
)


def derivar(controle: str) -> str:
    """A célula do tratado, a partir da célula do controle. Levanta se uma âncora
    não aparecer exatamente uma vez."""
    celula = controle
    for inicio, fim, novo in BLOCOS:
        for ancora in (inicio, fim):
            n = celula.count(ancora)
            if n != 1:
                raise ValueError(f"âncora de bloco aparece {n} vezes: {ancora[:60]!r}")
        a = celula.index(inicio)
        b = celula.index(fim, a) + len(fim)
        celula = celula[:a] + novo + celula[b:]
    for antigo, novo in TROCAS:
        n = celula.count(antigo)
        if n != 1:
            raise ValueError(f"troca aparece {n} vezes: {antigo[:60]!r}")
        celula = celula.replace(antigo, novo)
    return celula


CELULA = r'''
# ─── T2a / tokenizer A contra E — cole isto numa célula do Kaggle ───────────
import hashlib, io, json, os, subprocess, sys, tarfile, time, urllib.request
from pathlib import Path

ENTRADA = Path("/kaggle/input")
TRABALHO = Path("/kaggle/working")

VARIANTE = "__VARIANTE__"          # "A" ou "E", injetado na publicação
assert VARIANTE == "E", (
    f"o braço tratado do §2.3 usa o tokenizer E, o vencedor do T2a e o do "
    f"controle já treinado; recebi {VARIANTE}. Com outro tokenizer o controle "
    f"não serve.")

# ⚠️ A ÚNICA variável do experimento. Decidida em 2026-09-16, antes de este braço
# existir — a conta está no docstring de `scripts/train_phienc.py`.
P_EQUACAO = 0.6

# ── 1. Achar o dataset ──────────────────────────────────────────────────────
candidatos = []
for profundidade in range(0, 5):
    padrao = "/".join(["*"] * profundidade + ["MANIFESTO.json"])
    candidatos = sorted({m.parent for m in ENTRADA.glob(padrao)})
    if candidatos:
        break
assert candidatos, (
    f"nenhum MANIFESTO.json em {ENTRADA} até 4 níveis. Presentes na raiz: "
    f"{[d.name for d in ENTRADA.iterdir()]}")
DADOS = candidatos[0]
man = json.loads((DADOS / "MANIFESTO.json").read_text())
print(f"dataset: {DADOS} · git {man['git_sha']} · variante {VARIANTE}")

# ── 2. Conferir os hashes ANTES de treinar ──────────────────────────────────
try:
    from blake3 import blake3 as _h
    algo = "blake3"
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "blake3"], check=False)
    try:
        from blake3 import blake3 as _h
        algo = "blake3"
    except ImportError:
        _h, algo = hashlib.sha256, "sha256"
        print("⚠️ blake3 indisponível — conferência DESATIVADA, o manifesto é blake3")

if algo == "blake3":
    for nome, esperado in man["arquivos"].items():
        caminho = DADOS / nome
        assert caminho.exists(), f"{nome} não está no dataset"
        h = _h()
        with open(caminho, "rb") as f:
            while b := f.read(1 << 22):
                h.update(b)
        assert h.hexdigest() == esperado["blake3"], (
            f"{nome}: hash difere. Upload truncado ou dataset trocado — treinar "
            f"sobre isto daria um braço incomparável com o outro.")
    print(f"✅ {len(man['arquivos'])} arquivos conferidos por blake3")

# ── 3. Remontar a fatia desta variante ──────────────────────────────────────
# O dataset traz as DUAS fatias achatadas (`tokens_A.u16.bin`, `tokens_E.u16.bin`,
# …) porque o manifesto indexa por nome e não por caminho. O laço espera um
# diretório com os nomes canônicos, então aqui os links são refeitos.
FATIA = TRABALHO / f"fatia_{VARIANTE}"
FATIA.mkdir(parents=True, exist_ok=True)
import shutil
for canonico, no_dataset in (("tokens.u16.bin", f"tokens_{VARIANTE}.u16.bin"),
                             ("marcas.u8.bin", f"marcas_{VARIANTE}.u8.bin"),
                             ("MANIFESTO_DADOS.json", f"MANIFESTO_{VARIANTE}.json")):
    origem = DADOS / no_dataset
    assert origem.exists(), f"{no_dataset} não está no dataset"
    destino = FATIA / canonico
    if not destino.exists():
        try:
            os.symlink(origem, destino)
        except OSError:
            shutil.copy2(origem, destino)
man_fatia = json.loads((FATIA / "MANIFESTO_DADOS.json").read_text())
print(f"fatia {VARIANTE}: {man_fatia['tokens']:,} tokens · "
      f"{man_fatia['documentos_nesta_execucao']:,} docs · "
      f"tokenizer {man_fatia['tokenizer'].split('/')[-1]}")

# ⚠️ A fatia TEM de ser a da variante anunciada. O manifesto grava o caminho do
# tokenizer; se ele não casar com a variante, os dois braços treinariam no mesmo
# dado e o experimento daria empate por construção.
assert f"variante_{VARIANTE}" in man_fatia["tokenizer"], (
    f"a fatia montada como {VARIANTE} foi tokenizada por "
    f"{man_fatia['tokenizer']}. Os dois braços treinariam no mesmo dado.")

# ── 4. Código do GitHub ─────────────────────────────────────────────────────
SHA = "__SHA__"
REPO = "__REPO__"
url = f"https://codeload.github.com/{REPO}/tar.gz/{SHA}"
print(f"baixando o código de {url}", flush=True)
with urllib.request.urlopen(url, timeout=180) as r:
    bruto = r.read()
with tarfile.open(fileobj=io.BytesIO(bruto)) as tf:
    raizes = {m.name.split("/")[0] for m in tf.getmembers()}
    assert len(raizes) == 1, f"tarball com {len(raizes)} raízes: {sorted(raizes)}"
    tf.extractall(TRABALHO / "codigo", filter="data")
CODIGO = TRABALHO / "codigo" / raizes.pop()
FONTE = CODIGO / "src"
for exigido in ("scripts/train_phienc.py", "scripts/exportar_phienc.py",
                "src/phifm/training/pretrain/laco.py"):
    assert (CODIGO / exigido).exists(), (
        f"{exigido} não está no tarball de {SHA[:7]} — SHA errado ou renomeado")
sys.path.insert(0, str(FONTE))
print(f"código em {CODIGO} · {len(bruto)/1e3:.0f} KB · SHA {SHA[:7]}")

from phifm.core.kaggle import assinatura_do_manifesto

ASSINATURA_ESPERADA = "__ASSINATURA_DADOS__"
_obtida = assinatura_do_manifesto(man["arquivos"])
assert _obtida == ASSINATURA_ESPERADA, (
    f"o dataset anexado tem assinatura {_obtida} e esta célula foi publicada para "
    f"{ASSINATURA_ESPERADA}. O Kaggle fixou uma versão ANTIGA do dataset no anexo.")
print(f"✅ assinatura do bundle confere: {_obtida}")

import torch
print(f"torch {torch.__version__} · CUDA {torch.cuda.is_available()} · "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sem GPU'}")
assert torch.cuda.is_available(), "sem GPU. Settings → Accelerator → GPU."

# ── 5. O orçamento, IDÊNTICO nos dois braços ────────────────────────────────
# ⚠️ Os quatro números abaixo são constantes da célula, e a célula é a MESMA nos
# dois braços. É isso que garante tokens iguais — calcular passos separadamente
# por braço seria o jeito de eles divergirem sem ninguém ver.
# ⚠️ 0,6 B com 8x8, e o número de tokens mudou depois de uma MEDIÇÃO — 2026-09-11.
#
# A primeira tentativa foi 0,8 B. A guarda de sessão mediu 22.530 tok/s na segunda
# janela de log e projetou **9,9 h** contra as 8,5 h do teto: a estimativa por
# FLOPs errava por 32%, e a MFU real é 10,0% (contra uns 13-15% que a conta
# supunha). O run abortou aos 2,5 min em vez de morrer a 85% com a sessão esgotada.
#
# **0,6 B cabe com a vazão JÁ MEDIDA**: 600 M ÷ 22.530 tok/s = 7,40 h, dentro das
# 8,5 h. Não depende de nenhum ganho.
#
# ## ⚠️ 8 sequências por micro-passo é o TETO desta GPU, medido
#
# A segunda tentativa trocou 8x8 por 16x4 para ganhar vazão — mantendo o produto
# em 64, é o mesmo treino (o fluxo indexa por `micro_global * sequencias + i` e o
# laço faz `micro_global = passo * acumulacao + micro`, então o passo 0 consome as
# sequências 0..63 nos dois casos, na mesma ordem).
#
# Deu **OutOfMemory no primeiro backward**: tentou alocar 2,50 GiB com 270 MB
# livres numa T4 de 14,56 GiB. O que estoura não é o modelo de 48 M — são os
# LOGITS de MLM, que a 16 sequências são 16 × 1.024 × 40.960 em fp16, **1,34 GB**,
# mais a cópia do cross-entropy e o backward.
#
# ⚠️ E a lição é sobre o meu processo, não sobre a GPU: eu tinha escrito acima que
# 0,6 B cabe **sem ganho nenhum**, e mesmo assim mudei o lote na mesma rodada. Uma
# mudança que não era necessária, não era medida, e custou outra sessão. Uma
# variável por vez vale também quando a variável não é do experimento.
TOKENS = 600_000_000
CONTEXTO = 1024
# ⚠️ NÃO aumente sem medir: 16 estoura a memória da T4 (ver acima). O produto
# `SEQUENCIAS * ACUMULACAO` é o lote lógico e tem de continuar 64.
SEQUENCIAS = 8
ACUMULACAO = 8
PASSOS = TOKENS // (CONTEXTO * SEQUENCIAS * ACUMULACAO)

# ⚠️ O teto da sessão, conferido contra a vazão MEDIDA e não contra a estimada.
#
# O dimensionamento saiu de FLOPs: 2,56e17 com a atenção contada. Converter isso
# em horas exige supor a MFU, e a 48 M de parâmetros a suposição é frouxa — 15%
# contra 25% de MFU é 8,9 h contra 5,9 h, os dois lados de uma sessão de 9 h.
#
# O laço projeta na segunda janela de log e aborta se não couber: custa minutos
# em vez de descobrir aos 85%, com a sessão inteira gasta e nada exportado.
LIMITE_H = 8.5

# ── a REGRA de leitura, escrita antes de qualquer número existir ────────────
REGRA = r"""
  ── A REGRA, escrita ANTES ────────────────────────────────────────────────

  A PERGUNTA (DOC-07 §2.3): mascarar uma equação INTEIRA, em parte dos
  exemplos, ensina a relação entre a prosa e a expressão formal?

  CONTROLE: o braço E do T2a, já treinado (kernel phifm-t2a-tokenizer-e,
  código fc1523a, p_equacao 0,0). TRATADO: esta célula, p_equacao 0,6. Mesma
  fatia, mesmo tokenizer, mesmo orçamento, mesma semente.

  ⚠️ ESTE RUN NÃO É A ABLAÇÃO COMO PLANEJADA. Ela foi preparada a 2 B tokens
     (21 tokens/parâmetro); isto roda 0,6 B, 3,3x menos, porque o controle
     existe a 0,6 B e reaproveitá-lo é o que faz caber na cota.

  CHECAGENS DE MANIPULAÇÃO — reprovada qualquer uma, o run não testa o §2.3:
    1. fração tratada no treino >= 0,50. Medida antes nesta fatia: 0,549.
    2. no regime de EQUAÇÃO, o tratado tem perda menor que o controle. É a
       célula tautológica: se nem nela ele ganha, o tratamento não pegou.

  MEDIDA PRIMÁRIA: a célula de MECANISMO — perda nos tokens de equação em
  display (inteira na janela) sob mascaramento ALEATÓRIO, a tarefa do controle.
    · conjunto DISJUNTO: partes que nem A nem E usaram no treino;
    · 2.000 sequências, as MESMAS máscaras aplicadas aos dois braços;
    · bootstrap pareado por SEQUÊNCIA da perda média nos tokens de equação,
      IC de 95%. NÃO o teste de sinais sobre a perda geral, que mistura prosa.

  ⚠️ O viés da primária aponta para o CONTROLE: o regime aleatório é o
     objetivo em que ele treinou, e o tratado gastou 1/3 dos exemplos noutra
     coisa.

  Os desfechos:

    TRATADO à frente  ->  robusto: venceu no objetivo do outro. A hipótese do
                          §2.3 fica sustentada a 0,6 B.

    CONTROLE à frente ->  ambíguo entre "não há mecanismo" e "custo geral de
                          distribuição". Decide a diferença das diferenças,
                          mesmo bootstrap: (tratado − controle) nos tokens de
                          equação MENOS (tratado − controle) nos de prosa.
                          Prejuízo em equação MAIOR que em prosa (IC exclui
                          zero) -> é o negativo que o DOC-07 manda publicar.
                          Senão -> custo geral, não refutação: não decidido.

    EMPATE            ->  ⚠️ NÃO é o negativo do DOC-07. É "a 0,6 B não dá
                          para ver", registrado como não decidido.

  Secundárias: recuperação de Física e sonda tensorial, controle × tratado.
  Não derrubam a primária; se discordarem, a discordância é o resultado.

  ⚠️ O empate é o desfecho mais provável de um run subdimensionado, e o mais
     fácil de escrever como "tentamos, não funciona". O DOC-07 promete publicar
     o negativo — e um empate a 0,6 B não é esse negativo.
"""

print(f"""
{'=' * 74}
§2.3 — mascarar equações INTEIRAS ensina alguma coisa?  ·  braço TRATADO

  tokenizer {VARIANTE} (o vencedor do T2a) · p_equacao {P_EQUACAO}
  O controle NÃO roda aqui: é o braço E do T2a, já treinado com p_equacao 0,0.
  Esta célula é a do controle com trocas DECLARADAS em kaggle/t2eq_tratado.py,
  e um teste confere que não há nenhuma outra.

  orçamento IDÊNTICO ao do controle: {TOKENS:,} tokens · contexto {CONTEXTO} ·
                      {SEQUENCIAS}x{ACUMULACAO} · {PASSOS:,} passos
                      teto de {LIMITE_H} h, conferido contra a vazão MEDIDA
  medido antes, nesta fatia, com o código corrigido: fração tratada 0,549,
  equação escolhida com mediana de 76 tokens, 0 atravessando o fim da janela.

  ⚠️ A COMPARAÇÃO NÃO ACONTECE AQUI. Ela roda LOCAL, com os dois checkpoints
     no mesmo processo e as MESMAS máscaras.
{REGRA}
{'=' * 74}
""", flush=True)

AMBIENTE = {**os.environ, "PYTHONPATH": str(FONTE)}


def _rodar(cmd, log_em):
    """Roda com a saída nos DOIS destinos, e LEVANTA se falhou."""
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    with open(log_em, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                encoding="utf-8", errors="replace", env=AMBIENTE)
        for linha in proc.stdout:
            fh.write(linha)
            fh.flush()
            print(linha, end="", flush=True)
        codigo = proc.wait()
    if codigo != 0:
        raise SystemExit(f"{cmd[2]} saiu com {codigo}; log em {log_em}")


# ── 6. Treinar ──────────────────────────────────────────────────────────────
t0 = time.perf_counter()
RUN = TRABALHO / f"phienc_{VARIANTE}_tratado"
_rodar([sys.executable, "-u", CODIGO / "scripts/train_phienc.py",
        "--config", "proxy-bakeoff", "--dados", FATIA, "--out", RUN,
        "--total-passos", PASSOS, "--contexto", CONTEXTO,
        "--sequencias", SEQUENCIAS, "--acumulacao", ACUMULACAO,
        "--p-equacao", P_EQUACAO, "--semente", 17,
        "--limite-horas", LIMITE_H],
       TRABALHO / f"treino_{VARIANTE}_tratado.log")
custo_treino = round(time.perf_counter() - t0, 1)

# ── 7. Exportar, para as três medidas poderem abrir ─────────────────────────
# ⚠️ O laço grava `state_dict` cru; `AutoModel.from_pretrained` precisa de
# `config.json` ao lado. Sem esta etapa o checkpoint só o próprio laço entende.
#
# ## ⚠️ `--mesmo-assim`, e a razão é o custo de NÃO exportar
#
# O exportador recusa checkpoint com spike de perda, com esta mensagem:
#
#     num bake-off isso pode ser lido como 'este tokenizer é pior' quando o que
#     houve foi treino instável
#
# Ela está certa, e foi por isso que o braço A terminou em ERROR em 2026-09-11
# depois de treinar os 9.155 passos: 1 spike no passo 3.798, com 1 rollback. O
# detector fez o que o DOC-08 §6.1 manda — voltou ao checkpoint, pulou a janela
# suspeita, reduziu a LR até o passo 4.298 — e a perda final (1,5558) veio de uma
# curva estável.
#
# Recusar exportar não protege a comparação: ela é feita LOCAL, e o que protege é
# a ressalva estar no manifesto do artefato, onde quem compara tem de olhar. O que
# a recusa produz é **8,3 h de T4 num formato que só o laço abre** — e depois
# alguém exporta à mão com esta mesma bandeira, sem nenhuma informação a mais.
#
# ⚠️ O que a comparação local TEM de fazer: ler `spike.n_spikes` dos dois braços.
# Um braço com spike e outro sem é uma assimetria real, e ela entra na leitura —
# não como "empate" nem como "A é pior", mas como ressalva nomeada.
EXPORTADO = TRABALHO / f"phienc-{VARIANTE}-tratado"
_rodar([sys.executable, "-u", CODIGO / "scripts/exportar_phienc.py",
        "--run", RUN, "--para", EXPORTADO, "--mesmo-assim",
        "--tokenizer", DADOS / f"variante_{VARIANTE}.json",
        "--nota", f"§2.3 braço TRATADO · p_equacao {P_EQUACAO} · tokenizer {VARIANTE} · {TOKENS:,} tokens · contexto {CONTEXTO}"],
       TRABALHO / f"exportar_{VARIANTE}_tratado.log")

metricas = json.loads((RUN / "phienc.json").read_text(encoding="utf-8"))
(TRABALHO / f"t2eq_tratado_{VARIANTE}.json").write_text(json.dumps({
    "variante": VARIANTE,
    "p_equacao": P_EQUACAO,
    "controle": ("braço E do T2a, p_equacao 0,0, kernel "
                 "phifm-t2a-tokenizer-e, código fc1523a"),
    "git_sha_dados": man["git_sha"], "git_sha_codigo": SHA,
    "orcamento": {"tokens": TOKENS, "contexto": CONTEXTO,
                  "sequencias": SEQUENCIAS, "acumulacao": ACUMULACAO,
                  "passos": PASSOS},
    "fatia": {k: man_fatia.get(k) for k in
              ("tokens", "documentos_nesta_execucao", "tokenizer",
               "tokenizer_sha", "partes_usadas")},
    "custo_segundos": {"treino_s": custo_treino},
    "metricas": metricas.get("metricas"),
    "mascaramento": metricas.get("mascaramento"),
    "spike": metricas.get("spike"),
    "concluido": metricas.get("concluido"),
    "nota": ("A comparação controle × tratado NÃO está aqui. Ela roda local, com "
             "os dois checkpoints no mesmo processo e as MESMAS máscaras, pela "
             "REGRA impressa no início desta célula."),
}, indent=2, ensure_ascii=False), encoding="utf-8")

print()
print("=" * 74)
print(f"  §2.3 braço TRATADO · tokenizer {VARIANTE} · p_equacao {P_EQUACAO} · "
      f"{PASSOS:,} passos · {custo_treino/3600:.2f} h")
m = metricas.get("metricas") or {}
print(f"  perda final {m.get('perda')} · {m.get('tokens'):,} tokens · "
      f"{m.get('tokens_por_s', 0):.0f} tok/s")
print(f"  spikes: {(metricas.get('spike') or {}).get('n_spikes')}")
_ft = (metricas.get("mascaramento") or {}).get("fracao_tratada")
print(f"  fração tratada: {_ft}  (medida antes nesta fatia: 0,549)")
if _ft is None or _ft < 0.50:
    print("  ⚠️ CHECAGEM DE MANIPULAÇÃO 1 REPROVADA: abaixo de 0,50 o braço "
          "tratado recaiu em MLM aleatório, e este run NÃO testa o §2.3.")
print(f"  -> {EXPORTADO}")
print("=" * 74)
print(f"\n⚠️ Baixe `phienc-{VARIANTE}-tratado/` inteiro, "
      f"`t2eq_tratado_{VARIANTE}.json` e os logs. O controle já está em casa; "
      "meça os dois juntos, LOCAL, pela REGRA.")
'''
