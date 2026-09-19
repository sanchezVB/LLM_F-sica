"""Caminho B do ADR-0003 — pré-treino CONTINUADO do ModernBERT-base, em DUAS T4.

Este arquivo é o CONTEÚDO de uma célula do Kaggle, mantido aqui como `.py` para ficar
sob controle de versão e ser testável. A MESMA célula roda os dois braços; o que muda
é `VARIANTE` ("controle" ou "tratado"), injetada na publicação.

## A pergunta

O §2.3 mediu, em proxies de 48 M treinados do zero, que mascarar equações inteiras
**não** melhora a previsão de token de equação (diferença das diferenças −0,0040) e
**melhora** a recuperação depois do ajuste contrastivo (+0,084 de nDCG@10). O ADR-0003
concluiu que isso não é argumento para treinar do zero: as marcas de equação saem de
offsets de CARACTERE e servem a qualquer base. O caminho B pergunta se o efeito
aparece onde ele importa — um encoder de 150 M que já sabe ler, adaptado ao corpus de
Física com o objetivo do §2.3.

Uma variável entre os braços: `p_equacao` (0,0 contra 0,6). Base, dados, orçamento,
semente e hiperparâmetros idênticos.

## ⚠️ Duas GPUs, e por que isso não muda o experimento

A sessão do Kaggle é "GPU T4 x2" e conta horas de SESSÃO — conferido em 2026-09-17:
**todo run anterior do projeto recebeu duas placas e usou uma**. Aqui o treino roda em
`torchrun --nproc_per_node 2`, e `tests/regression/test_laco_ddp.py` prova pelos pesos
que dois processos treinam o mesmo modelo que um: `--acumulacao` é a do passo inteiro e
se divide entre os processos, e a máscara sai de `(semente, micro-passo)`.

A célula ABORTA se não houver duas GPUs. Rodar um braço em duas e o outro em uma daria
o mesmo modelo em expectativa, mas deixaria uma diferença não declarada entre eles — e
este projeto já pagou por comparar coisas medidas de jeitos diferentes (T1b2).

## ⚠️ 0,4 B tokens, e não os 0,6 B do §2.3

A conta, com a vazão MEDIDA do proxy (20.478 tok/s numa T4) e o custo medido do
ModernBERT-base contra ele (2,64× por passo): ~7,8 mil tok/s numa T4, ~15,6 mil em
duas. A 0,6 B seriam ~10,7 h — acima do teto de 9 h da sessão, e o laço não retoma
entre sessões. A 0,4 B são ~7,1 h, com margem.

`--limite-horas 8.0` confere isso contra a vazão REAL na segunda janela de log e aborta
em minutos se a estimativa estiver errada, em vez de descobrir aos 85%.

⚠️ O orçamento é o MESMO nos dois braços; se um for reduzido, o outro tem de ser
refeito. E 0,4 B não se compara com os 0,6 B dos proxies de 48 M: outra base, outro
tokenizer, outro volume.

## ⚠️ O que esta célula NÃO decide

Ela produz os dois encoders. O veredito do §2.3 nesta escala sai da medição posterior,
pela regra abaixo, que é a mesma do braço de 48 M (`kaggle/t2eq_tratado.py`) — primária
pela diferença das diferenças de `mlm_regiao`, não pela perda.
"""
from __future__ import annotations

CELULA = r'''
# ─── Caminho B do ADR-0003 · CPT do ModernBERT-base em 2 T4 — cole numa célula ───
import hashlib, io, json, os, shutil, subprocess, sys, tarfile, time, urllib.request
from pathlib import Path

ENTRADA = Path("/kaggle/input")
TRABALHO = Path("/kaggle/working")

VARIANTE = "__VARIANTE__"          # "controle" ou "tratado"
P_EQUACAO = {"controle": "0.0", "tratado": "0.6"}
assert VARIANTE in P_EQUACAO, VARIANTE

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
print(f"dataset: {DADOS.name} · git {man['git_sha']} · braço {VARIANTE}")
assert man["experimento"] == "t2eq_cpt_controle", (
    f"o dataset anexado é do experimento {man['experimento']!r}; os dois braços "
    "têm de treinar sobre o MESMO pacote")

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
            f"{nome}: hash difere. Upload truncado ou dataset trocado — o braço "
            "deixaria de ser comparável ao outro.")
    print(f"✅ {len(man['arquivos'])} arquivos conferidos por blake3")

# ── 3. Código do GitHub ─────────────────────────────────────────────────────
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
for s in ("scripts/train_phienc.py", "scripts/exportar_phienc.py"):
    assert (CODIGO / s).exists(), f"{s} não está no tarball de {SHA[:7]}"
sys.path.insert(0, str(FONTE))
print(f"código em {CODIGO} · {len(bruto)/1e3:.0f} KB · SHA {SHA[:7]}")

from phifm.core.kaggle import assinatura_do_manifesto

ASSINATURA_ESPERADA = "__ASSINATURA_DADOS__"
_obtida = assinatura_do_manifesto(man["arquivos"])
assert _obtida == ASSINATURA_ESPERADA, (
    f"o dataset anexado tem assinatura {_obtida} e esta célula foi publicada para "
    f"{ASSINATURA_ESPERADA}. O Kaggle fixou uma versão ANTIGA do dataset no anexo.")
print(f"✅ assinatura do bundle confere: {_obtida}")

# ── 4. A fatia, com os nomes que o laço espera ──────────────────────────────
FATIA = TRABALHO / "fatia"
FATIA.mkdir(exist_ok=True)
for origem, destino in (("tokens_MB.u16.bin", "tokens.u16.bin"),
                        ("marcas_MB.u8.bin", "marcas.u8.bin"),
                        ("MANIFESTO_MB.json", "MANIFESTO_DADOS.json")):
    if not (FATIA / destino).exists():
        shutil.copy(DADOS / origem, FATIA / destino)
_man_fatia = json.loads((FATIA / "MANIFESTO_DADOS.json").read_text())
# ⚠️ A fatia TEM de ser a tokenizada com o tokenizer da base. O `train_phienc.py`
# confere o id de máscara contra o tokenizer do Hub e RECUSA se divergir — treinar
# com o id errado mascara com um token qualquer e a perda desce igual.
assert _man_fatia.get("especiais_do_tokenizer") and "id_mascara" in _man_fatia, (
    "a fatia não declara os especiais do tokenizer da base")
print(f"fatia: {_man_fatia['tokens']:,} tokens · máscara id {_man_fatia['id_mascara']} "
      f"· disjunta do treino: {_man_fatia.get('disjunto_do_treino')}")

import torch
N_GPUS = torch.cuda.device_count()
print(f"torch {torch.__version__} · {N_GPUS} GPU(s): "
      f"{[torch.cuda.get_device_name(i) for i in range(N_GPUS)]}")

# ── 5. Os parâmetros, e o orçamento ────────────────────────────────────────
BASE = "answerdotai/ModernBERT-base"
PROCESSOS = 2
# ⚠️ ABORTA em vez de cair para uma placa: um braço em duas GPUs e o outro em uma
# daria o mesmo modelo em expectativa, com uma diferença não declarada entre eles.
assert N_GPUS >= PROCESSOS, (
    f"{N_GPUS} GPU(s) e este experimento pede {PROCESSOS}. Settings → Accelerator "
    "→ GPU T4 x2. Rodar em uma placa mudaria o run sem mudar o registro.")

CONTEXTO = 1024
# ⚠️ Por PROCESSO. O produto `SEQUENCIAS * ACUMULACAO` é o lote lógico do passo
# inteiro (64 sequências, como nos proxies do §2.3) e não depende do número de
# processos: cada um faz `ACUMULACAO / PROCESSOS` micro-passos.
SEQUENCIAS = 4
ACUMULACAO = 16
TOKENS = 400_000_000
PASSOS = TOKENS // (CONTEXTO * SEQUENCIAS * ACUMULACAO)
# ⚠️ LR de pré-treino CONTINUADO, e não de treino do zero: 1e-3 num modelo que já
# convergiu apaga o que ele sabe nos primeiros passos. Igual nos dois braços.
LR = 1e-4
SEMENTE = 17
# Teto da sessão conferido contra a vazão MEDIDA, na segunda janela de log.
LIMITE_HORAS = 8.0

REGRA = r"""
  ── A REGRA, escrita ANTES ────────────────────────────────────────────────

  A PERGUNTA (ADR-0003, caminho B): mascarar equações inteiras durante o
  pré-treino CONTINUADO de uma base forte melhora o encoder de Física?

  MEDIDA PRIMÁRIA: a DIFERENÇA DAS DIFERENÇAS de `mlm_regiao` — (equação −
  prosa) do tratado menos (equação − prosa) do controle —, com máscara
  UNIFORME de 15% nas MESMAS posições, fatia disjunta tokenizada com o
  tokenizer da base, 2.000 sequências, contexto 1.024, IC por bootstrap
  pareado por SEQUÊNCIA.

  ⚠️ NÃO é a perda de treino, e nem a acurácia em equação sozinha: token de
  equação é +0,1286 mais fácil que prosa no ModernBERT-base SEM tratamento
  nenhum (medido em 2026-09-10). Um tratado que melhorasse tudo por igual
  sairia lido como evidência.

  Os desfechos:

    POSITIVA (IC exclui zero, acima) -> o §2.3 se sustenta na escala que
      importa. O ΦEnc de Física passa a ser este, e o DOC-07 §2.3 é confirmado.

    NEGATIVA (IC exclui zero, abaixo) -> o negativo do §2.3 se repete fora do
      proxy, e agora com uma base forte. O objetivo do §2.3 sai do programa.

    IC CRUZANDO ZERO -> NÃO DECIDIDO. Não é o negativo, e a escala fica
      declarada ao lado.

  CHECAGENS DE MANIPULAÇÃO (não são o desfecho; se falharem, o run é INVÁLIDO):
    1. `fracao_tratada` >= 0,50 no braço tratado — abaixo disso ele recaiu em
       MLM aleatório e não há tratamento a medir;
    2. com a prova de EQUAÇÃO INTEIRA mascarada, o tratado tem de vencer: é o
       que prova que o treinamento pegou.

  SECUNDÁRIA (não derruba a primária): recuperação no protocolo do G1 depois
  do ajuste contrastivo, contra a BARRA do passo 1 — o ModernBERT-base SEM
  pré-treino continuado, ajustado nos mesmos 200 mil pares. Foi nessa medida
  que o §2.3 deu positivo a 48 M (+0,084), então ela é a razão de este run
  existir; mas a primária é a do objetivo, e ela manda.
"""
print(REGRA)
print(f"  base {BASE} · {PROCESSOS} processos · contexto {CONTEXTO}")
print(f"  {PASSOS:,} passos × {CONTEXTO * SEQUENCIAS * ACUMULACAO:,} tokens = "
      f"{PASSOS * CONTEXTO * SEQUENCIAS * ACUMULACAO / 1e9:.3f} B tokens")
print(f"  p_equacao {P_EQUACAO[VARIANTE]} · lr {LR:.0e} · semente {SEMENTE} · "
      f"teto {LIMITE_HORAS} h")

# ── 6. Treinar ─────────────────────────────────────────────────────────────
SAIDA = TRABALHO / f"phienc-cpt-{VARIANTE}"
LOG = TRABALHO / f"treino_{VARIANTE}.log"
env = dict(os.environ, PYTHONPATH=str(FONTE), PYTHONIOENCODING="utf-8",
           TOKENIZERS_PARALLELISM="false")
cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
       f"--nproc_per_node={PROCESSOS}", str(CODIGO / "scripts/train_phienc.py"),
       "--base", BASE, "--dados", str(FATIA), "--contexto", str(CONTEXTO),
       "--sequencias", str(SEQUENCIAS), "--acumulacao", str(ACUMULACAO),
       "--total-passos", str(PASSOS), "--p-equacao", P_EQUACAO[VARIANTE],
       "--lr-pico", str(LR), "--semente", str(SEMENTE),
       "--passos-log", "50", "--passos-estado", "250",
       "--limite-horas", str(LIMITE_HORAS), "--out", str(SAIDA)]
print("\n" + " ".join(cmd), flush=True)
t0 = time.perf_counter()
with open(LOG, "w", encoding="utf-8") as flog:
    p = subprocess.Popen(cmd, cwd=str(CODIGO), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", bufsize=1)
    for linha in p.stdout:
        flog.write(linha)
        if any(x in linha for x in ("passo ", "SPIKE", "rollback", "Error", "Traceback",
                                    "WSD", "CPT de", "limite")):
            print(linha.rstrip(), flush=True)
    codigo_saida = p.wait()
minutos = (time.perf_counter() - t0) / 60
print(f"\ntreino terminou em {minutos:.1f} min · código {codigo_saida}")
assert codigo_saida == 0, f"o treino falhou; veja {LOG}"

# ── 7. Exportar, para o avaliador conseguir abrir ──────────────────────────
# ⚠️ `--mesmo-assim`, e a lição já estava paga: o braço A do T2a terminou em ERROR
# em 2026-09-11 depois de treinar inteiro, e as células do T2a e do §2.3 passaram a
# levar a bandeira (ver `kaggle/t2eq_tratado.py`). Esta célula nasceu sem ela, e o
# controle do caminho B repetiu o ERROR em 2026-09-19 — 441 min de treino completo,
# 2 spikes de norma com rollback, e a exportação recusada no fim.
#
# A ressalva vai para o manifesto do artefato, que é onde a comparação local tem de
# ler `spike.n_spikes` dos dois braços. Recusar aqui só produz um `.pt` que alguém
# exporta à mão depois, com a mesma bandeira e nenhuma informação a mais.
EXPORT = TRABALHO / f"phienc-cpt-{VARIANTE}-exportado"
r = subprocess.run([sys.executable, str(CODIGO / "scripts/exportar_phienc.py"),
                    "--run", str(SAIDA), "--para", str(EXPORT), "--mesmo-assim",
                    "--nota", f"caminho B · braço {VARIANTE} · p_equacao "
                              f"{P_EQUACAO[VARIANTE]} · {PASSOS:,} passos"],
                   cwd=str(CODIGO), env=env, capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
print(r.stdout[-2000:] if r.stdout else "", r.stderr[-2000:] if r.returncode else "")
assert r.returncode == 0, "a exportação falhou"

metricas = json.loads((SAIDA / "phienc.json").read_text(encoding="utf-8"))
resumo = {
    "experimento": "t2eq_cpt",
    "variante": VARIANTE,
    "base": BASE,
    "codigo": SHA,
    "assinatura_dados": ASSINATURA_ESPERADA,
    "orcamento": {"tokens": TOKENS, "passos": PASSOS, "contexto": CONTEXTO,
                  "sequencias_por_processo": SEQUENCIAS, "acumulacao": ACUMULACAO,
                  "processos": PROCESSOS, "lr_pico": LR, "semente": SEMENTE},
    "gpus": [torch.cuda.get_device_name(i) for i in range(N_GPUS)],
    "minutos": round(minutos, 1),
    "metricas": metricas.get("metricas"),
    "mascaramento": metricas.get("mascaramento"),
    "spike": metricas.get("spike"),
    "distribuicao": metricas.get("distribuicao"),
    "regra": REGRA,
}
(TRABALHO / f"t2eq_cpt_{VARIANTE}.json").write_text(
    json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: resumo[k] for k in ("variante", "minutos", "mascaramento",
                                         "distribuicao")}, indent=2,
                 ensure_ascii=False)[:1500])
print(f"\n✅ {VARIANTE}: pesos em {SAIDA} e exportação em {EXPORT}")
'''
