"""T2a — A contra E: as regras de pré-tokenização da §8 valem alguma coisa?

Este arquivo é o CONTEÚDO de uma célula do Kaggle, mantido aqui como `.py` para
ficar sob controle de versão e ser testável. A **mesma** célula roda os dois braços;
o que muda é `VARIANTE`, injetado na publicação.

## A pergunta

O DOC-05 §8 acrescenta um regex de pré-tokenização que torna `\\frac`, `\\begin{…}`
e `^{`/`_{` **pré-tokens atômicos**, e o documento chama isso de *"a decisão mais
consequente e menos visível"* dele. A variante **E** é a **A** sem esse regex.

O §11.2 diz que E é a comparação mais importante das seis: *"se E empatar com A, a
§8 está errada e o documento precisa ser revisada"*.

## ⚠️ Por que A contra E é o ÚNICO par limpo das seis

As variantes diferem em tamanho de vocabulário — C tem 32.768, A/B/E têm 40.960, D
tem 65.536 — e num modelo de 48 M a tabela de embedding é **43,7% dos parâmetros**.
Então C e D têm **contagens de parâmetro diferentes**, e compará-las com A carrega um
confundidor de capacidade que o `config.py` já registrava antes deste experimento.

A e E compartilham vocabulário, arquitetura e contagem. A única diferença é o regex.

## O efeito disponível, medido ANTES de treinar

| | fertilidade | em equações | bytes/token |
|---|---|---|---|
| A | 0,962 | 7,31 | 7,074 |
| E | 1,325 | 9,46 | 5,138 |

E nas fatias de treino de verdade, com o mesmo orçamento de tokens:

    A: 2.001.270.262 tokens de 143.810 documentos  ->  13.916 tokens/doc
    E: 2.003.418.640 tokens de 127.828 documentos  ->  15.673 tokens/doc

**E gasta 12,6% mais tokens por documento**, então A vê **12,5% mais texto** pelo
mesmo custo. É essa a vantagem sob teste, e é por isso que o protocolo iguala
**tokens** e não texto — igualar o texto apagaria o efeito.

⚠️ A métrica intrínseca superestima por 3×: 37,7% contra 12,6% reais. Ela foi medida
em resumos do arXiv e o corpus de treino é fonte LaTeX íntegro. Registrado porque é
o tamanho do efeito que a expectativa deve usar.

## ⚠️ Mascaramento PADRÃO nos dois braços, e isso não é preguiça

`--p-equacao 0.0`. O tratamento de equação inteira do DOC-07 §2.3 **depende de o
tokenizer marcar equações** — com E, `\\frac` está estilhaçado, então o mascaramento
por span se comportaria de outro jeito. Ligá-lo faria este experimento medir
"tokenizer + interação dele com o mascaramento" em vez do tokenizer.

A ablação do §2.3 é outro experimento, e precisa deste resolvido antes.
"""

CELULA = r'''
# ─── T2a / tokenizer A contra E — cole isto numa célula do Kaggle ───────────
import hashlib, io, json, os, subprocess, sys, tarfile, time, urllib.request
from pathlib import Path

ENTRADA = Path("/kaggle/input")
TRABALHO = Path("/kaggle/working")

VARIANTE = "__VARIANTE__"          # "A" ou "E", injetado na publicação
assert VARIANTE in ("A", "E"), VARIANTE

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
TOKENS = 800_000_000
CONTEXTO = 1024
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

  ⚠️ ESTE RUN NÃO É O §11.2 COMO ESCRITO. O protocolo de lá pede 5 B tokens por
     variante; isto roda 0,8 B — 6,25x menos. 5 B numa T4 seriam ~47 h por
     braço e a cota semanal inteira é 30 h. A redução não foi escolhida por
     conveniência de análise: ela é o que existe.

  ⚠️ MEDIDA PRIMÁRIA: bits por byte. NÃO acurácia de MLM.

     Acurácia de MLM não é comparável entre tokenizers diferentes, e o viés dela
     aponta CONTRA a hipótese. A e E partem o mesmo texto de jeitos diferentes:
     um token de E é mais curto, então prevê-lo a partir da vizinhança é mais
     fácil — sobra mais redundância local. E tende a marcar acurácia maior sem
     ser modelo melhor. Bits por byte normaliza pelo TEXTO em vez do token, que
     é a forma padrão de comparar vocabulários diferentes.

     A acurácia por região continua sendo calculada, como DIAGNÓSTICO e com o
     viés nomeado. Ela não decide nada.

  Secundárias, as duas agnósticas ao tokenizer por construção: a sonda de
  estrutura tensorial (cosseno entre textos, 72 itens) e a recuperação de
  Física (ranking por cosseno). Nenhuma das duas olha para tokens.

  Os desfechos:

    A VENCE em bits por byte  ->  a §8 compra desempenho, e já a 0,8 B. As
                                  secundárias entram como corroboração; elas
                                  não podem derrubar a primária, e se
                                  contrariarem isso é o resultado a reportar.

    EMPATE                    ->  ⚠️ NÃO refuta a §8. O protocolo pré-registrado
                                  era 5 B e este run tem 6,25x menos; o que um
                                  empate diz é "a 0,8 B não dá para ver". A §8
                                  FICA, e o teste do §11.2 segue em aberto —
                                  registrado como não decidido, não como nulo.

    E VENCE                   ->  notícia, e contra duas expectativas: a §11.1
                                  mediu E 27% pior em fertilidade, e A ainda vê
                                  12,5% mais texto pelo mesmo orçamento. E
                                  vencer APESAR disso seria evidência de que o
                                  regex custa, e aí a §8 é que precisa cair.

  ⚠️ A assimetria entre "empate" e as outras duas é deliberada e é a parte que
     mais precisa estar escrita antes. Um empate é o desfecho MAIS PROVÁVEL de
     um run subdimensionado por 6x, e é também o que se escreve com mais
     facilidade como "tentamos, não funciona" — fechando a linha com um nulo
     que o experimento não tinha poder para produzir.
"""

print(f"""
{'=' * 74}
T2a — a §8 vale alguma coisa?  ·  braço {VARIANTE}

  A variante E é a A SEM o regex de pré-tokenização da §8, que torna `\\frac`,
  `\\begin{{…}}` e `^{{`/`_{{` pré-tokens atômicos. O DOC-05 chama esse regex de
  "a decisão mais consequente e menos visível" do documento, e o §11.2 diz:
  "se E empatar com A, a §8 está errada".

  ⚠️ É o ÚNICO par limpo das seis variantes. C (V=32.768) e D (V=65.536) têm
     contagens de parâmetro diferentes de A — a 48 M a embedding é 43,7% —, e
     compará-las carregaria um confundidor de capacidade. A e E compartilham
     vocabulário, arquitetura e contagem; a única diferença é o regex.

  efeito disponível, medido antes: E gasta 12,6% mais tokens por documento, então
  A vê 12,5% mais texto pelo mesmo orçamento. (A métrica intrínseca dizia 37,7%;
  ela foi medida em resumos e superestima por 3x no corpus de verdade.)

  orçamento IDÊNTICO: {TOKENS:,} tokens · contexto {CONTEXTO} ·
                      {SEQUENCIAS}x{ACUMULACAO} · {PASSOS:,} passos
                      teto de {LIMITE_H} h, conferido contra a vazão MEDIDA na
                      segunda janela de log — se não couber, aborta em minutos
  mascaramento PADRÃO (p_equacao=0,0) nos dois: o tratamento do §2.3 depende de o
  tokenizer marcar equações, e ligá-lo mediria tokenizer + interação.

  ⚠️ A COMPARAÇÃO NÃO ACONTECE AQUI. Cada braço treina numa sessão, e as três
     medidas do §11.2 rodam LOCAL, com os dois checkpoints no mesmo processo —
     é assim que a lição do T1b2 fica respeitada mesmo com treinos separados.
     Esta célula produz um checkpoint exportado e nada mais.
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
RUN = TRABALHO / f"phienc_{VARIANTE}"
_rodar([sys.executable, "-u", CODIGO / "scripts/train_phienc.py",
        "--config", "proxy-bakeoff", "--dados", FATIA, "--out", RUN,
        "--total-passos", PASSOS, "--contexto", CONTEXTO,
        "--sequencias", SEQUENCIAS, "--acumulacao", ACUMULACAO,
        "--p-equacao", 0.0, "--semente", 17,
        "--limite-horas", LIMITE_H],
       TRABALHO / f"treino_{VARIANTE}.log")
custo_treino = round(time.perf_counter() - t0, 1)

# ── 7. Exportar, para as três medidas poderem abrir ─────────────────────────
# ⚠️ O laço grava `state_dict` cru; `AutoModel.from_pretrained` precisa de
# `config.json` ao lado. Sem esta etapa o checkpoint só o próprio laço entende.
EXPORTADO = TRABALHO / f"phienc-{VARIANTE}"
_rodar([sys.executable, "-u", CODIGO / "scripts/exportar_phienc.py",
        "--run", RUN, "--para", EXPORTADO,
        "--tokenizer", DADOS / f"variante_{VARIANTE}.json",
        "--nota", f"T2a braço {VARIANTE} · {TOKENS:,} tokens · contexto {CONTEXTO}"],
       TRABALHO / f"exportar_{VARIANTE}.log")

metricas = json.loads((RUN / "phienc.json").read_text(encoding="utf-8"))
(TRABALHO / f"t2a_{VARIANTE}.json").write_text(json.dumps({
    "variante": VARIANTE,
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
    "nota": ("A comparação A×E NÃO está aqui. Cada braço treina numa sessão e as "
             "três medidas do §11.2 rodam local, com os dois checkpoints no mesmo "
             "processo — a variável é o treino, a medição é única."),
}, indent=2, ensure_ascii=False), encoding="utf-8")

print()
print("=" * 74)
print(f"  T2a braço {VARIANTE} · {PASSOS:,} passos · {custo_treino/3600:.2f} h")
m = metricas.get("metricas") or {}
print(f"  perda final {m.get('perda')} · {m.get('tokens'):,} tokens · "
      f"{m.get('tokens_por_s', 0):.0f} tok/s")
print(f"  spikes: {(metricas.get('spike') or {}).get('n_spikes')}")
print(f"  -> {EXPORTADO}")
print("=" * 74)
print(f"\n⚠️ Baixe `phienc-{VARIANTE}/` inteiro, `t2a_{VARIANTE}.json` e os logs. "
      "Rode o outro braço, e só então meça os dois juntos, LOCAL.")
'''
