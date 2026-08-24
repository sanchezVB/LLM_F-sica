#!/usr/bin/env python3
"""Prepara (e opcionalmente envia) o Dataset e o Notebook do Kaggle para o T1a.

No PowerShell, que e o shell desta maquina:

    .venv\\Scripts\\python.exe -m kaggle auth login
    .venv\\Scripts\\python.exe scripts\\publicar_kaggle.py --usuario SEU_USUARIO
    .venv\\Scripts\\python.exe scripts\\publicar_kaggle.py --usuario SEU_USUARIO --enviar

⚠️ SEM `PYTHONPATH=src`. Este script nao importa nada de `phifm` — so a stdlib —,
e o prefixo veio por habito dos outros comandos do projeto. Pior: `VAR=valor cmd`
e sintaxe de bash e o PowerShell nao a tem, entao o comando morria em
`CommandNotFoundException` antes de rodar qualquer coisa. No PowerShell seria
`$env:PYTHONPATH = 'src'`, mas aqui nao e necessario.

## Por que um script e não arquivos escritos à mão

Os dois `*-metadata.json` do Kaggle carregam o slug do dataset em dois lugares — o
`id` do dataset e o `dataset_sources` do notebook. Escritos à mão eles divergem, e o
sintoma é um notebook que sobe, roda e falha no `assert` de que não achou o
`MANIFESTO.json` — depois de consumir cota de GPU.

Aqui o slug é derivado de um lugar só.

## O `.ipynb` é GERADO, não versionado

`kaggle/t1a_phiemb.py` é a fonte da célula e é o que está sob controle de versão,
porque um `.ipynb` é JSON com saídas embutidas: diff ilegível, merge impossível, e
o executável misturado com o resultado da última execução. O `.ipynb` sai daqui e é
descartável.

## ⚠️ PRIVADO, e sem opção de não ser

Não há bandeira `--publico`. O dataset são 400 mil pares de citação derivados do
arXiv mais o código do projeto, e publicá-los é uma decisão de divulgação — não um
detalhe de configuração que um script deveria tornar fácil. Quem quiser publicar
muda a visibilidade na interface do Kaggle, deliberadamente.

A licença declarada é `other`, não CC0: os resumos do arXiv seguem a licença de cada
submissão, e declarar CC0 sobre conteúdo de terceiros seria uma afirmação que não
temos como sustentar.

## ⚠️ O notebook precisa de INTERNET

Não pelo `blake3` (esse tem fallback para sha256, registrado). É pelo
`all-MiniLM-L6-v2`, baixado do HuggingFace, que não tem fallback nenhum. No Kaggle,
habilitar internet exige conta verificada por telefone — se não estiver verificada,
o notebook sobe, roda e morre no download.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PACOTE = RAIZ / "data/processed/kaggle_t1a"
SAIDA_NB = RAIZ / "data/processed/kaggle_notebook"
FONTE_CELULA = RAIZ / "kaggle/t1a_phiemb.py"

SLUG_DADOS = "phifm-t1a-pares-citacao"
SLUG_NB = "phifm-t1a-emb"
# ⚠️ ASCII puro. O Kaggle deriva o slug do TITULO, e derruba o que nao
# for [a-z0-9]: "PhiFM T1a - PhiEmb" com Φ virou `phifm-t1a-emb`, que nao
# casava com o `id` `phifm-t1a-phiemb` — a CLI avisou e devolveu 409.
TITULO_NB = "PhiFM T1a Emb"


def _celula() -> str:
    """Extrai a `CELULA` de `kaggle/t1a_phiemb.py` sem importar o módulo.

    Sem importar porque o arquivo vive em `kaggle/`, que não é um pacote, e um
    `sys.path` remendado para ler uma constante seria mais frágil que um regex
    sobre um arquivo que nós mesmos escrevemos.
    """
    texto = FONTE_CELULA.read_text(encoding="utf-8")
    m = re.search(r"CELULA = r'''\n(.*?)'''", texto, re.S)
    if not m:
        raise SystemExit(
            f"não achei `CELULA = r'''…'''` em {FONTE_CELULA}. Se o formato mudou, "
            "este extrator precisa mudar junto — e é de propósito que ele quebra "
            "alto em vez de gerar um notebook vazio.")
    return m.group(1)


def _ipynb(codigo: str) -> dict:
    linhas = codigo.splitlines(keepends=True)
    return {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                   "outputs": [], "source": linhas}],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def _usuario_do_json() -> str | None:
    """Lê SÓ o campo `username` do kaggle.json legado, quando ele existir.

    ⚠️ O `key` do arquivo nunca é lido nem impresso. O único motivo de tocar neste
    arquivo é o nome de usuário, que compõe o slug do dataset — e o token novo e o
    OAuth não carregam esse campo, então esta é a única fonte automática dele.
    """
    p = Path.home() / ".kaggle/kaggle.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("username") or None
    except Exception:
        return None


def _usuario_da_sessao() -> str | None:
    """Pergunta à CLI quem está autenticado. `kaggle config view` não tem efeito.

    ⚠️ Não use `kaggle auth login` para descobrir isto. Quando já há sessão ele
    responde "You are already logged-in to Kaggle as [nome]", mas quando NÃO há ele
    abre o navegador e começa um fluxo de autorização — um sondador que autentica é
    efeito colateral inaceitável numa função de leitura.

    `config view` imprime `- username: nome` e sai.
    """
    try:
        r = subprocess.run([sys.executable, "-m", "kaggle", "config", "view"],
                           capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    m = re.search(r"^-\s*username:\s*(\S+)\s*$", r.stdout, re.M)
    nome = m.group(1) if m else None
    return None if nome in (None, "None") else nome


# Nomes que sao claramente o exemplo do README, nao uma conta. Colar o placeholder
# custou um upload de 160 MB em 2026-08-24: o Kaggle envia os arquivos ANTES de
# validar o dono, entao a rejeicao chega no fim.
PLACEHOLDERS = {"seu_usuario", "seuusuario", "usuario", "username", "exemplo",
                "your_username", "yourusername", "user", "me"}


def _tem_credencial() -> bool:
    """Alguma das três formas de autenticação da CLI 2.2.4 está presente?

    Checa presença, não validade: validar exigiria uma chamada à API, e falhar por
    rede seria indistinguível de falhar por credencial. A CLI dá a mensagem boa
    quando de fato tenta.

    O OAuth do `kaggle auth login` guarda o estado dentro de ~/.kaggle, e o nome do
    arquivo é detalhe interno da CLI — por isso a checagem é por "existe algo em
    ~/.kaggle além do kaggle.json", e não por um nome fixo que uma atualização
    mudaria sem avisar.
    """
    import os

    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    d = Path.home() / ".kaggle"
    if not d.is_dir():
        return False
    return any(f.name != "kaggle.json" for f in d.iterdir())


# ⚠️ Frases que a CLI imprime AO FALHAR mantendo codigo de saida 0.
#
# Medido em 2026-08-24: `kernels push` com uma fonte de dados invalida imprimiu
# "The following are not valid dataset sources" e saiu com 0. O script confiou no
# codigo de retorno e anunciou "✅ enviado" — o notebook foi criado SEM dados, e
# quem leu a mensagem achou que estava pronto. Um falso sucesso e pior que a falha.
#
# `datasets create` faz o mesmo com "Invalid Owner Id": sobe os 160 MB, rejeita o
# dono e sai com 0.
FALHAS_SILENCIOSAS = (
    "Invalid Owner Id",
    "Dataset creation error",
    "not valid dataset sources",
    "does not resolve to the specified id",
)


def _dataset_existe(id_dados: str) -> bool:
    """O dataset já está lá? Decide entre `datasets create` e `datasets version`.

    Busca pelo slug dentro dos datasets do próprio usuário. Em caso de dúvida
    devolve False, porque `create` num que já existe dá um erro claro, enquanto
    `version` num que não existe dá um erro obscuro.
    """
    dono, _, slug = id_dados.partition("/")
    try:
        r = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "list",
                            "--user", dono, "--search", slug],
                           capture_output=True, text=True, timeout=120)
    except Exception:
        return False
    return id_dados in r.stdout


def _rodar(cmd: list[str]) -> str:
    """Roda a CLI, ECOA a saída ao vivo e a devolve para conferência.

    Ecoa em vez de só capturar porque o upload leva dezenas de segundos e a barra de
    progresso é a única evidência de que algo acontece. Devolve porque o código de
    saída da CLI não é confiável — ver `FALHAS_SILENCIOSAS`.
    """
    proc = subprocess.Popen(cmd, cwd=RAIZ, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    partes = []
    assert proc.stdout is not None
    for linha in proc.stdout:
        sys.stdout.write(linha)
        sys.stdout.flush()
        partes.append(linha)
    proc.wait()
    saida = "".join(partes)
    if proc.returncode:
        raise SystemExit(
            f"a CLI do Kaggle saiu com {proc.returncode}. O upload é retomável: "
            "rode de novo. Se disser que o dataset já existe, use "
            f"`kaggle datasets version -p {PACOTE} -m 'atualização'`.")
    return saida


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--usuario", default=None,
                   help="seu usuário do Kaggle. Normalmente NÃO precisa: sai de "
                        "`kaggle config view` quando há sessão ativa")
    p.add_argument("--enviar", action="store_true",
                   help="chama a CLI do Kaggle; sem isto, só prepara e imprime")
    p.add_argument("--so-notebook", action="store_true",
                   help="não toca no dataset; útil quando ele já subiu e só o "
                        "notebook falhou")
    a = p.parse_args()

    # ⚠️ O console do Windows entrega cp1252 e este script imprime ✅ e Φ. Sem isto
    # ele levanta UnicodeEncodeError DEPOIS de já ter escrito os arquivos — o
    # trabalho fica feito e a saída diz que falhou, que é a pior das combinações.
    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if not (PACOTE / "MANIFESTO.json").exists():
        raise SystemExit(
            f"{PACOTE}/MANIFESTO.json não existe. Rode "
            "scripts/empacotar_kaggle.py primeiro.")

    # ⚠️ Guarda contra o placeholder colado literal. Em 2026-08-24 rodei com
    # `--usuario SEU_USUARIO` e o Kaggle subiu 160 MB antes de rejeitar o dono: ele
    # envia os arquivos primeiro e valida o slug do proprietário no fim. Barrar aqui
    # custa nada; descobrir lá custa o upload inteiro.
    if a.usuario and a.usuario.strip().lower() in PLACEHOLDERS:
        raise SystemExit(
            f"'{a.usuario}' é o placeholder do exemplo, não uma conta. O Kaggle só "
            "cria dataset sob o dono autenticado, e valida isso DEPOIS de receber os "
            "arquivos — colar isto custaria o upload inteiro para falhar no fim.\n\n"
            "Simplesmente omita --usuario: com sessão ativa ele sai de "
            "`kaggle config view`.")

    # Ordem: o que foi pedido explicitamente, senão quem está autenticado, senão o
    # kaggle.json legado. A sessão vem antes do arquivo porque é ela que manda no
    # que o Kaggle vai aceitar como dono.
    usuario = a.usuario or _usuario_da_sessao() or _usuario_do_json()
    if not usuario:
        raise SystemExit(
            "não sei seu usuário do Kaggle e não há sessão ativa. Rode "
            "`python -m kaggle auth login` primeiro, ou passe --usuario.")
    if not a.usuario:
        print(f"  usuário detectado da sessão: {usuario}")

    man = json.loads((PACOTE / "MANIFESTO.json").read_text(encoding="utf-8"))
    id_dados = f"{usuario}/{SLUG_DADOS}"
    id_nb = f"{usuario}/{SLUG_NB}"

    # ── metadados do dataset ────────────────────────────────────────────────
    (PACOTE / "dataset-metadata.json").write_text(json.dumps({
        "title": "PhiFM T1a — pares de citação arXiv",
        "id": id_dados,
        # `other`: os resumos seguem a licença de cada submissão do arXiv.
        "licenses": [{"name": "other"}],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── notebook + metadados dele ───────────────────────────────────────────
    SAIDA_NB.mkdir(parents=True, exist_ok=True)
    nb = SAIDA_NB / "t1a_phiemb.ipynb"
    nb.write_text(json.dumps(_ipynb(_celula()), indent=1), encoding="utf-8")
    (SAIDA_NB / "kernel-metadata.json").write_text(json.dumps({
        "id": id_nb,
        "title": TITULO_NB,
        "code_file": "t1a_phiemb.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        # ⚠️ Obrigatório: o `all-MiniLM-L6-v2` vem do HuggingFace e não há fallback.
        "enable_internet": True,
        "dataset_sources": [id_dados],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    tam = sum(f.stat().st_size for f in PACOTE.iterdir() if f.is_file()) / 1e6
    print("=" * 70)
    print(f"  preparado · git {man['git_sha']} · {man['linhas_treino']:,} pares · "
          f"{tam:.1f} MB")
    print(f"  dataset  : {id_dados}")
    print(f"  notebook : {id_nb}  (GPU ✅ · internet ✅ · privado ✅)")
    print("=" * 70)

    # ⚠️ `sys.executable -m kaggle`, nunca `["kaggle", ...]`. O executável fica em
    # `.venv/Scripts/kaggle.exe` e NÃO está no PATH a menos que a venv esteja
    # ativada — o script falharia com "não encontrado" só na hora do envio, depois
    # de já ter gerado tudo.
    #
    # `create` só na primeira vez; depois é `version`. Sem isto, rodar de novo (o
    # que a mensagem de erro do próprio script recomenda) morre em "dataset já
    # existe" — um roteiro que manda repetir e quebra na repetição.
    ja_existe = a.enviar and _dataset_existe(id_dados)
    if ja_existe:
        print(f"  dataset já existe — enviando VERSÃO nova em vez de criar")
        cmd_dados = [sys.executable, "-m", "kaggle", "datasets", "version",
                     "-p", str(PACOTE), "--dir-mode", "zip",
                     "-m", f"git {man['git_sha']}"]
    else:
        cmd_dados = [sys.executable, "-m", "kaggle", "datasets", "create",
                     "-p", str(PACOTE), "--dir-mode", "zip"]
    cmds = [
        cmd_dados,
        [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(SAIDA_NB)],
    ]
    if a.so_notebook:
        cmds = cmds[1:]
        print("  --so-notebook: o dataset não será tocado")
    if not a.enviar:
        print("\n  para enviar, depois de `python -m kaggle auth login`:\n")
        for c in cmds:
            print("    " + " ".join(c))
        print("\n  (ou rode este script de novo com --enviar)")
        return 0

    if not _tem_credencial():
        raise SystemExit("""sem credencial do Kaggle. A CLI 2.2.4 aceita três
caminhos, em ordem de menos atrito:

  1. `python -m kaggle auth login` — fluxo OAuth no navegador. Nada é copiado
     nem colado, e é o que a própria CLI recomenda.
  2. kaggle.com/settings/api → 'Generate New Token' (o de CIMA), e salvar o
     token em ~/.kaggle/access_token
  3. variável de ambiente KAGGLE_API_TOKEN

⚠️ 'Create Legacy API Key' (o botão de baixo) baixa um kaggle.json que esta
versão da CLI NÃO usa para autenticar — ele serve só para eu ler o nome de
usuário de lá.

Eu não faço nenhum dos três: exige entrar na sua conta.""")

    for c in cmds:
        print(f"\n$ {' '.join(c)}")
        saida = _rodar(c)
        ruins = [f for f in FALHAS_SILENCIOSAS if f in saida]
        if ruins:
            raise SystemExit(
                "a CLI saiu com 0 mas a saída acusa falha: "
                + "; ".join(f"'{f}'" for f in ruins)
                + ".\nNão vou chamar isto de enviado. Corrija e rode de novo.")
    print(f"\n✅ enviado. Dataset: kaggle.com/datasets/{id_dados}")
    print(f"   Notebook: kaggle.com/code/{id_nb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
