"""T1f — a BASE do recuperador: GTE-base ajustado contra o nosso MiniLM ajustado.

Este arquivo é o CONTEÚDO de uma célula do Kaggle, mantido aqui como `.py` para
ficar sob controle de versão e ser testável.

## A pergunta, e por que ela ficou em aberto

Medido em 2026-09-10: o **GTE-base zero-shot EMPATA** com o nosso ΦEmb ajustado no
top-10 — r@10 0,2755 contra 0,2810, p=0,545 — **sem ter visto uma linha do nosso
dado**. Um modelo de prateleira alcançando o nosso ajustado é um resultado que pede
explicação, e a hipótese natural é que a base importa mais que o ajuste.

Este experimento ajusta o GTE-base nos MESMOS 400 mil pares sorteados, com os
MESMOS hiperparâmetros. Uma variável: a base.

## ⚠️ É uma SONDA, e a resposta final custa 15× mais

O recuperador do sistema é o `phiemb-do-sistema`, treinado em **6 M** de pares. Um
GTE-base@6M custa **~39 h de T4** — 4,7× o custo por par vezes 15× os pares — e não
cabe na cota. Comparar GTE-base@400k contra o ΦEmb@6M mudaria base *e* volume.

Então o par é **GTE-base@400k contra MiniLM@400k**, e o que ele decide é se as 39 h
valem a pena, não qual recuperador o sistema usa.

## ⚠️ A métrica primária é nDCG@10, e ela MUDOU de significado em 2026-09-10

O GTE-base zero-shot empata no top-10 e **perde fundo**: r@200 de 0,645 contra
0,730. Até a semana passada isso seria decisivo — a cadeia era `ΦEmb → RRF → ΦRank`
sobre um pool profundo, e quem perde fundo entrega menos candidatos ao
reranqueador.

O T1e tirou o ΦRank do sistema: a cadeia é `ΦEmb → top-10`. **Nada lê o fundo
hoje.** Então o r@200 vira diagnóstico, e volta a decidir só se um reranqueador
voltar ao sistema. Registrar isto antes é o que impede de mudar a métrica primária
depois de ver o resultado.

## ⚠️ E o custo de serviço entra na decisão, porque é permanente

O GTE-base levou **954 s** para embutir o mesmo universo contra **217 s** do nosso:
**4,4×**. Uma vitória do GTE não é automaticamente uma troca de base — a margem tem
de pagar 4,4× de inferência para sempre. Essa parte da decisão é do dono do
projeto, e esta célula não a toma; ela produz o número.

## O braço de referência NÃO é retreinado, e isso é deliberado

`phiemb-minilm-t4-sorteado-melhor` já existe: mesma T4, mesmo sorteio, mesmos 400
mil pares. Retreiná-lo gastaria 36 min de cota para reproduzir o que está no disco.

⚠️ A lição do T1d — "comparar contra o número histórico teria reportado SETE VEZES
o efeito real" — era sobre o **protocolo de avaliação** ter mudado entre as
medições, não sobre os pesos. Aqui a avaliação dos dois checkpoints roda LOCAL, na
mesma sessão, com o mesmo `avaliar_encoders.py`. O que difere é só o treino, que é
a variável.
"""

CELULA = r'''
# ─── T1f / base do recuperador — cole isto numa célula do Kaggle ────────────
import hashlib, io, json, os, subprocess, sys, tarfile, time, urllib.request
from pathlib import Path

ENTRADA = Path("/kaggle/input")
TRABALHO = Path("/kaggle/working")

# ── 1. Achar o dataset ──────────────────────────────────────────────────────
# Busca em PROFUNDIDADE: a imagem nova do Kaggle monta em
# `/kaggle/input/datasets/<dono>/<slug>/` e a antiga em `/kaggle/input/<slug>/`.
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
print(f"dataset: {DADOS.name} · git {man['git_sha']} · "
      f"{man['linhas_treino']:,} pares · "
      f"{man.get('documentos_distintos', 0):,} documentos citados distintos")

# ⚠️ Este experimento REUSA o dataset do T1a, e é isso que dá a comparabilidade:
# o braço de referência treinou nestes mesmos bytes. Um dataset novo, mesmo
# montado com a mesma semente, deixaria de provar isso.
assert man["experimento"] == "t1a", (
    f"o dataset anexado é do experimento {man['experimento']!r}. O T1f compara "
    "contra um checkpoint treinado nos pares do T1a; com outro pacote a "
    "comparação mudaria base E dado.")

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
            f"{nome}: hash difere. Treinar sobre isto daria um braço que não é "
            "comparável ao MiniLM que treinou nos pares do T1a.")
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
assert (CODIGO / "scripts/train_embedding.py").exists(), (
    f"scripts/train_embedding.py não está no tarball de {SHA[:7]}")
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
_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sem GPU"
print(f"torch {torch.__version__} · {_gpu}")
assert torch.cuda.is_available(), "sem GPU. Settings → Accelerator → GPU."

# ── 4. Os hiperparâmetros, que são os do T1a e NÃO se mexem ─────────────────
# ⚠️ Cada um destes é uma variável que não pode mudar junto com a base.
#
# O `--lote 128` é o que define quantos negativos o InfoNCE vê (127), e mudá-lo
# mudaria a dificuldade da tarefa. O `--max-tokens 192` é o do campeão, e o T1e
# mediu que ler 256 ou 384 não muda o recall — então 192 não está deixando nada
# na mesa. A semente 17 governa o sorteio dos pares E o pool de avaliação.
BASE = "thenlper/gte-base"
LOTE = 128
MAX_TOKENS = 192
SEMENTE = 17
N_CANDIDATOS = 1000
PASSOS_AVAL = 200

REGRA = r"""
  ── A REGRA, escrita ANTES ────────────────────────────────────────────────

  MEDIDA PRIMÁRIA: nDCG@10, no protocolo do G1 (`avaliar_encoders.py`, pool
  desduplicado, teto 1,0), com os DOIS checkpoints medidos LOCAL na mesma sessão.

  ⚠️ E ela mudou de significado na semana passada. O GTE-base zero-shot empata no
     top-10 e perde fundo (r@200 0,645 contra 0,730). Até o T1e isso seria
     decisivo — a cadeia era ΦEmb -> RRF -> ΦRank sobre um pool profundo. O T1e
     tirou o ΦRank do sistema e a cadeia virou ΦEmb -> top-10: NADA lê o fundo
     hoje. O r@200 fica como diagnóstico e só volta a decidir se um reranqueador
     voltar ao sistema.

  Os desfechos:

    GTE VENCE      ->  a base importa mais que o ajuste, e a pergunta passa a ser
                       se as ~39 h de um GTE-base@6M valem. ⚠️ Vitória aqui NÃO é
                       troca de base: o GTE custa 4,4x para embutir (954 s contra
                       217 s no mesmo universo), e essa conta é permanente. A
                       decisão de pagá-la é do dono do projeto, não deste número.

    EMPATE         ->  ajustar o GTE não o tira do empate que ele já tinha DE
                       GRAÇA. A hipótese "a base é o gargalo" perde força, e as
                       39 h não se justificam.

    MiniLM VENCE   ->  fecha: a base atual está certa e ainda economiza 4,4x de
                       inferência. É o desfecho que mais simplifica o sistema.

  ⚠️ Esta é uma SONDA a 400 mil pares. O recuperador do sistema treinou em 6 M, e
     a ordem entre duas bases a 400 mil pode não sobreviver a 6 M — bases maiores
     costumam precisar de mais dado para se separar. Um empate aqui não prova que
     elas empatam a 6 M; prova que a 400 mil não dá para ver.
"""

print(f"""
{'=' * 74}
T1f — a BASE do recuperador importa mais que o ajuste?

  O GTE-base ZERO-SHOT já empata com o nosso ajustado no top-10 (r@10 0,2755
  contra 0,2810, p=0,545) sem ter visto uma linha do nosso dado. Aqui ele é
  ajustado nos MESMOS 400 mil pares sorteados, com os MESMOS hiperparâmetros.

  base {BASE} · lote {LOTE} ({LOTE - 1} negativos) ·
  max_tokens {MAX_TOKENS} · semente {SEMENTE}

  referência (NÃO retreinada aqui): models/phiemb-minilm-t4-sorteado-melhor,
  mesma T4, mesmo sorteio, mesmos 400 mil pares. Retreinar gastaria 36 min de
  cota para reproduzir o que já está no disco — e a lição do T1d era sobre o
  PROTOCOLO DE AVALIAÇÃO mudar entre medições, não sobre os pesos. A avaliação
  dos dois roda local, na mesma sessão, com o mesmo avaliador.

  ⚠️ A COMPARAÇÃO NÃO ACONTECE AQUI. Esta célula treina um braço e para.
{REGRA}
{'=' * 74}
""", flush=True)

AMBIENTE = {**os.environ, "PYTHONPATH": str(FONTE)}
SAIDA = TRABALHO / "phiemb-gte-base-400k"


def _rodar(cmd, log_em):
    """Roda com a saída nos DOIS destinos, e LEVANTA se falhou.

    ⚠️ Mandar só para o arquivo custou 33 h de cegueira em 2026-09-07.
    """
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


# ── 5. Treinar ──────────────────────────────────────────────────────────────
# ⚠️ Se estourar a memória, o plano B está escrito e é `--sub-lote 64 --sem-amp`:
# o GradCache preserva o lote LÓGICO, então continuam sendo 127 negativos, e o que
# muda é fp16 -> fp32. Isso é numérico, não é desenho. Reduzir o `--lote` em vez
# disso mudaria a dificuldade da tarefa junto com a base, e o resultado deixaria
# de ter dono. O GTE-base tem 109 M contra os 23 M do MiniLM.
t0 = time.perf_counter()
_rodar([sys.executable, "-u", CODIGO / "scripts/train_embedding.py",
        "--pares", DADOS, "--out", SAIDA, "--base", BASE,
        "--lote", LOTE, "--max-tokens", MAX_TOKENS,
        "--passos-aval", PASSOS_AVAL, "--n-candidatos", N_CANDIDATOS,
        "--dispositivo", "cuda", "--semente", SEMENTE],
       TRABALHO / "treino_t1f.log")
custo = round(time.perf_counter() - t0, 1)

melhor = SAIDA.parent / f"{SAIDA.name}-melhor"
assert melhor.exists(), (
    f"{melhor} não existe — o treino não elegeu checkpoint. Sem ele não há o que "
    "baixar, e a sessão foi gasta.")
m = json.loads((melhor / "melhor.json").read_text(encoding="utf-8"))

(TRABALHO / "t1f.json").write_text(json.dumps({
    "experimento": "t1f",
    "pergunta": "a base do recuperador importa mais que o ajuste?",
    "base": BASE,
    "git_sha_dados": man["git_sha"], "git_sha_codigo": SHA,
    "hiperparametros": {"lote": LOTE, "negativos_infonce": LOTE - 1,
                        "max_tokens": MAX_TOKENS, "semente": SEMENTE,
                        "n_candidatos_aval": N_CANDIDATOS,
                        "passos_aval": PASSOS_AVAL},
    "pares": man["linhas_treino"],
    "documentos_distintos": man.get("documentos_distintos"),
    "custo_segundos": custo,
    "melhor": m,
    "referencia": {
        "checkpoint": "models/phiemb-minilm-t4-sorteado-melhor",
        "nota": ("NÃO retreinado aqui: mesma T4, mesmo sorteio, mesmos 400 mil "
                 "pares, já no disco. A avaliação dos dois roda LOCAL na mesma "
                 "sessão, com o mesmo avaliar_encoders.py."),
    },
    "custo_de_servico": {
        "gte_base_s": 954, "nosso_s": 217, "razao": 4.4,
        "nota": ("⚠️ Permanente. Uma vitória do GTE não é automaticamente troca "
                 "de base: a margem tem de pagar 4,4x de inferência para sempre, "
                 "e essa decisão é do dono do projeto."),
    },
    "sonda_de": {
        "alvo": "GTE-base@6M", "custo_estimado_h_t4": 39,
        "nota": ("⚠️ A 400 mil pares a ordem entre duas bases pode não sobreviver "
                 "a 6 M — bases maiores costumam precisar de mais dado para se "
                 "separar. Empate aqui prova que a 400 mil não dá para ver, não "
                 "que elas empatam a 6 M."),
    },
    "metrica_primaria": {
        "nome": "nDCG@10",
        "por_que": ("O T1e tirou o ΦRank do sistema e a cadeia virou ΦEmb -> "
                    "top-10. NADA lê o fundo hoje, então o r@200 — em que o GTE "
                    "zero-shot perde por 0,085 — é diagnóstico, e só volta a "
                    "decidir se um reranqueador voltar ao sistema."),
    },
}, indent=2, ensure_ascii=False), encoding="utf-8")

print()
print("=" * 74)
print(f"  T1f · GTE-base@400k · {custo/3600:.2f} h")
print(f"  melhor no pool interno (n={N_CANDIDATOS}): "
      f"nDCG@10 {m.get('ndcg_10')} · recall@10 {m.get('recall_10')} · "
      f"passo {m.get('passo')}")
print("  ⚠️ Este nDCG é do pool INTERNO do treino e NÃO é o número do G1: o "
      "protocolo do")
print("     G1 usa pool desduplicado com teto 1,0. Comparar este com o do MiniLM "
      "seria")
print("     comparar duas réguas. A comparação é local, com o avaliador do G1.")
print(f"  -> {melhor}")
print("=" * 74)
print(f"\n⚠️ Baixe `{melhor.name}/` inteiro, `t1f.json` e o log. Depois, LOCAL:\n"
      "   avaliar_encoders.py --modelo 'MiniLM@400k=models/phiemb-minilm-t4-"
      "sorteado-melhor' \\\n"
      f"                       --modelo 'GTE-base@400k=models/{melhor.name}'")
'''
