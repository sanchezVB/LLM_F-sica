"""O executor das medidas do G2 no Kaggle (a 8B, DOC-12 §3.7.5): prazo, retomada e estado.

As medidas em si são os scripts congelados (`medir_g2*.py`), sem mudança. Daqui sai só o
que o Kaggle exige em volta deles:

- **Prazo.** Uma versão do notebook morre em 12 h, e o Output de quem estoura pode se
  perder. Cada medida roda num subprocesso com prazo. No prazo, e também ao fim de TODA
  medida, o grupo de processos (Python e llama-server) recebe SIGTERM e, depois da carência,
  SIGKILL; só se segue quando o grupo sumiu, para nenhum servidor órfão segurar a GPU.
- **Linha cortada.** Morto no meio de uma escrita, um jsonl pode ficar com uma linha
  incompleta no fim. Ela é a amostra que não terminou; `aparar_jsonl` a tira (e só ela), e
  a retomada a gera de novo, com a mesma semente.
- **Estado.** O que uma sessão grava (`g2_saidas/`, os agregados da 8B, o livro de acessos)
  vai para o Output num zip, com o número da sessão e o SHA-256 de cada arquivo; a sessão
  seguinte o recebe como Input e o devolve ao repositório antes de retomar.
- **Livro de acessos.** O livro do pacote é uma foto do daqui, que continua crescendo. A
  sessão junta por REGISTRO: o livro do pacote mais as linhas da 8B que o estado traz. Toda
  linha que não é da 8B no estado tem de estar no pacote, e nenhuma linha some.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
import zipfile
from collections.abc import Callable
from pathlib import Path


def sha256_arquivo(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        while bloco := f.read(1 << 24):
            h.update(bloco)
    return h.hexdigest()


def aparar_jsonl(arquivo: Path) -> int:
    """Tira a linha incompleta do fim (bytes depois do último "\\n" que não são JSON) e
    devolve quantos bytes saíram. Uma linha inteira sem o "\\n" final ganha o "\\n" (sem ele,
    o próximo acréscimo a colaria na seguinte). Uma linha completa inválida no meio não é
    tocada: isso não é corte, e a leitura vai falhar alto, como deve."""
    arquivo = Path(arquivo)
    if not arquivo.exists():
        return 0
    dados = arquivo.read_bytes()
    fim = dados.rfind(b"\n") + 1
    resto = dados[fim:]
    if not resto.strip():
        return 0
    try:
        json.loads(resto.decode("utf-8"))
        with arquivo.open("ab") as f:
            f.write(b"\n")
        return 0
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    with arquivo.open("r+b") as f:
        f.truncate(fim)
    return len(resto)


def rodar_com_prazo(cmd: list[str], prazo: float, log: Path, cwd: Path | None = None,
                    env: dict | None = None, carencia_s: float = 30.0,
                    periodico: Callable[[], object] | None = None,
                    intervalo_s: float = 1200.0) -> str:
    """Roda `cmd` até terminar ou até o instante `prazo` (time.time()). Devolve "ok",
    "falhou" (código ≠ 0) ou "prazo". Enquanto roda, chama `periodico` a cada `intervalo_s`
    (o estado vai ao Output, para uma sessão que morra no meio não perder o que gravou). No
    fim, em qualquer caso, limpa o grupo de processos."""
    if time.time() >= prazo:
        return "prazo"
    posix = os.name == "posix"
    with Path(log).open("a", encoding="utf-8") as saida:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=saida, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=posix)
        estado = None
        proxima = time.time() + intervalo_s
        while proc.poll() is None:
            if time.time() >= prazo:
                estado = "prazo"
                break
            if periodico is not None and time.time() >= proxima:
                periodico()
                proxima = time.time() + intervalo_s
            time.sleep(5)
        limpar_grupo(proc, posix, carencia_s)
    if estado:
        return estado
    return "ok" if proc.returncode == 0 else "falhou"


def _grupo_vivo(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def limpar_grupo(proc: subprocess.Popen, posix: bool, carencia_s: float = 30.0) -> None:
    """SIGTERM ao grupo (ou ao processo, fora do POSIX), espera a carência e manda SIGKILL ao
    que sobrar. Volta só quando o líder e, no POSIX, o grupo inteiro sumiram."""
    if not posix:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=carencia_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        return
    pgid = proc.pid
    if _grupo_vivo(pgid):
        os.killpg(pgid, signal.SIGTERM)
        limite = time.time() + carencia_s
        while _grupo_vivo(pgid) and time.time() < limite:
            proc.poll()
            time.sleep(1)
        if _grupo_vivo(pgid):
            os.killpg(pgid, signal.SIGKILL)
            limite = time.time() + 60      # um zumbi sem quem o colha não segura a sessão
            while _grupo_vivo(pgid) and time.time() < limite:
                proc.poll()
                time.sleep(1)
    proc.wait()


# ── o estado entre sessões ──────────────────────────────────────────────────

# O que uma sessão cria ou muda: as gerações e os logs, os agregados da 8B, o pacote do
# EvalPlus (a execução é no outro notebook) e o livro de acessos. Os dados congelados
# (partição, exemplos, autoteste, a KL) NÃO entram: eles vêm sempre do pacote.
LIVRO = "data/processed/avaliacao/g2_acessos.jsonl"
PADROES_DO_ESTADO = ("data/processed/avaliacao/g2_saidas/*",
                     *(f"data/processed/avaliacao/g2_{m}_8B_*.json"
                       for m in ("dev", "teste", "mmlu_dev", "mmlu_teste", "ifeval_dev",
                                 "ifeval_teste", "evalplus_dev", "evalplus_teste")),
                     "data/processed/avaliacao/g2_vazao_8B.json",
                     "data/processed/avaliacao/g2_kl_8B_cuda.json",
                     "data/processed/kaggle_g2_evalplus/**/*",
                     LIVRO)


def arquivos_do_estado(repo: Path) -> list[Path]:
    return sorted({p for padrao in PADROES_DO_ESTADO for p in Path(repo).glob(padrao)
                   if p.is_file()})


def registros(dados: bytes) -> list[dict]:
    return [json.loads(x) for x in dados.decode("utf-8").split("\n") if x.strip()]


def _chave(r: dict) -> str:
    return json.dumps(r, sort_keys=True, ensure_ascii=False)


def juntar_livros(pacote: bytes, estado: bytes, tamanho: str = "8B") -> bytes:
    """O livro do pacote mais as linhas `tamanho` do estado que ele não tem, na ordem do
    estado. Recusa se uma linha que NÃO é desse tamanho estiver no estado e faltar no pacote
    (ela sumiria) ou se o estado repetir uma linha."""
    do_pacote = {_chave(r) for r in registros(pacote)}
    novas, vistas = [], set()
    for r in registros(estado):
        k = _chave(r)
        if k in vistas:
            raise SystemExit(f"o livro do estado repete um registro: {k[:120]}")
        vistas.add(k)
        if k in do_pacote:
            continue
        if r.get("tamanho") != tamanho:
            raise SystemExit(f"o livro do estado tem um registro que não é da {tamanho} e que "
                             f"o pacote não tem: {k[:120]}")
        novas.append(r)
    base = pacote if pacote.endswith(b"\n") or not pacote else pacote + b"\n"
    return base + b"".join(json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n"
                           for r in novas)


def linhas_do_tamanho(livro: bytes, tamanho: str = "8B") -> list[dict]:
    return [r for r in registros(livro) if r.get("tamanho") == tamanho]


def salvar_estado(repo: Path, destino: Path, sessao: int, extra: dict | None = None) -> dict:
    """Grava o estado do repositório em `destino` (no Output): um zip único (o Kaggle lida
    mal com muitos arquivos soltos), com `.bin` no fim (o Kaggle descompacta todo `.zip` que
    vira dataset) e o SHA-256 no nome, e o ESTADO.json, com o número da sessão, o nome e o
    SHA-256 do zip e o de cada arquivo (dos MESMOS bytes que foram para o zip). Os jsonl
    são aparados antes. A ordem é zip novo, depois o ESTADO.json, e só então sai o zip
    velho: uma morte em qualquer ponto deixa um par (ESTADO.json, zip) que confere."""
    repo, destino = Path(repo), Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    for p in repo.glob("data/processed/avaliacao/g2_saidas/*.jsonl"):
        aparar_jsonl(p)
    arquivos = {}
    temporario = destino / "estado.parcial"
    with zipfile.ZipFile(temporario, "w", zipfile.ZIP_DEFLATED) as z:
        for p in arquivos_do_estado(repo):
            rel = p.relative_to(repo).as_posix()
            dados = p.read_bytes()
            z.writestr(rel, dados)
            arquivos[rel] = {"bytes": len(dados), "sha256": hashlib.sha256(dados).hexdigest()}
    zip_sha = sha256_arquivo(temporario)
    nome_zip = f"estado_{zip_sha[:12]}.zip.bin"
    temporario.replace(destino / nome_zip)
    livro = (repo / LIVRO).read_bytes() if (repo / LIVRO).exists() else b""
    manifesto = {"estado": "g2_8b", "sessao": sessao, **(extra or {}),
                 "zip": nome_zip, "zip_sha256": zip_sha,
                 "livro_8B": linhas_do_tamanho(livro), "arquivos": arquivos}
    parcial = destino / "ESTADO.json.parcial"
    parcial.write_text(json.dumps(manifesto, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    parcial.replace(destino / "ESTADO.json")
    for velho in destino.glob("estado_*.zip.bin"):
        if velho.name != nome_zip:
            velho.unlink()
    return manifesto


def ler_estado(origem: Path) -> tuple[dict, dict[str, bytes]]:
    """O ESTADO.json e o conteúdo do zip de uma pasta de estado, conferidos (SHA-256 do zip
    e de cada arquivo, e exatamente os arquivos do manifesto)."""
    origem = Path(origem)
    manifesto = json.loads((origem / "ESTADO.json").read_text(encoding="utf-8"))
    if manifesto.get("estado") != "g2_8b":
        raise SystemExit(f"{origem} não é um estado do g2_8b")
    arq = origem / manifesto.get("zip", "estado.zip")
    if arq.is_dir():
        raise SystemExit(f"{arq.name} chegou como pasta: o Kaggle descompactou o zip do estado")
    if not arq.is_file() or sha256_arquivo(arq) != manifesto["zip_sha256"]:
        raise SystemExit(f"o zip do estado ({arq.name}) falta ou não confere com o ESTADO.json")
    with zipfile.ZipFile(arq) as z:
        nomes = set(z.namelist())
        if nomes != set(manifesto["arquivos"]):
            raise SystemExit("o zip do estado não traz exatamente os arquivos do ESTADO.json")
        conteudo = {rel: z.read(rel) for rel in nomes}
    ruins = [rel for rel, dados in conteudo.items()
             if hashlib.sha256(dados).hexdigest() != manifesto["arquivos"][rel]["sha256"]]
    if ruins:
        raise SystemExit(f"o estado não confere: {ruins[:5]}")
    return manifesto, conteudo


def restaurar_estado(origem: Path, repo: Path, sessao_esperada: int) -> dict:
    """Devolve ao repositório o estado da sessão `sessao_esperada`, conferido (`ler_estado`),
    e junta o livro (ver `juntar_livros`). Recusa outra sessão: um estado velho refaria itens
    e perderia acessos do livro."""
    repo = Path(repo)
    manifesto, conteudo = ler_estado(origem)
    if manifesto.get("sessao") != sessao_esperada:
        raise SystemExit(f"o estado anexado é da sessão {manifesto.get('sessao')}, e esta "
                         f"célula espera o da sessão {sessao_esperada}: atualize o dataset "
                         "do estado para a versão mais nova, ou corrija SESSAO")
    livro_pacote = (repo / LIVRO).read_bytes() if (repo / LIVRO).exists() else b""
    if LIVRO not in conteudo:
        raise SystemExit("o estado anterior não traz o livro de acessos")
    conteudo[LIVRO] = juntar_livros(livro_pacote, conteudo[LIVRO])
    for rel, dados in conteudo.items():
        alvo = repo / rel
        alvo.parent.mkdir(parents=True, exist_ok=True)
        alvo.write_bytes(dados)
    return manifesto


# ── o GGUF, tensor a tensor ─────────────────────────────────────────────────


def mapa_gguf(arquivo: Path, gguf_py: Path) -> dict:
    """{metadados: {chave: sha256 do campo}, tensores: {nome: sha256 dos bytes}} de um GGUF,
    lido pelo gguf-py do conversor da tag (`gguf_py`). Serve para comparar dois GGUF sem
    tê-los na mesma máquina: se só um metadado difere, os pesos são os mesmos."""
    import sys

    if str(gguf_py) not in sys.path:
        sys.path.insert(0, str(gguf_py))
    from gguf import GGUFReader

    leitor = GGUFReader(str(arquivo))
    metadados = {nome: hashlib.sha256(b"".join(parte.tobytes() for parte in campo.parts)).hexdigest()
                 for nome, campo in leitor.fields.items()}
    tensores = {t.name: hashlib.sha256(t.data.tobytes()).hexdigest() for t in leitor.tensors}
    return {"metadados": metadados, "tensores": tensores}


def comparar_mapas(a: dict, b: dict) -> dict:
    """O que difere entre dois mapas de GGUF."""
    def dif(x: dict, y: dict) -> list[str]:
        return sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))

    return {"metadados_diferentes": dif(a["metadados"], b["metadados"]),
            "tensores_diferentes": dif(a["tensores"], b["tensores"])}


# ── a etapa terminou de verdade? ────────────────────────────────────────────

# Só contadores dos agregados (o lacre permite): amostras esperadas, ERROR e sem previsão.
AGREGADOS = {"mmlu": "g2_mmlu_teste_{m}_{b}.json", "g2_1": "g2_teste_{m}_{b}_veredito.json",
             "ifeval": "g2_ifeval_teste_{m}_{b}.json", "evalplus": "g2_evalplus_teste_{m}_{b}.json"}
ESPERADO_IFEVAL = {"amostras": 541 * 16, "gulosa": 541, "crua": 541}
ESPERADO_EVALPLUS = {"amostras": {"humaneval": 1640, "mbpp": 3780},
                     "gulosa": {"humaneval": 164, "mbpp": 378}}


def conferir_etapa(rotulo: str, repo: Path, modelo: str = "8B", braco: str = "base") -> tuple[bool, str]:
    """(completa, porquê) de uma medida de teste, pelos contadores do agregado. O G2.1 tem
    de ter as MESMAS amostras por benchmark da medida oficial da 1,7B (os mesmos itens, sem
    os excluídos pelo autoteste) e nenhum ERROR; o MMLU, nenhum item sem previsão; o IFEval,
    as 541×16 + 541 + 541; o EvalPlus, 10 + 1 por problema."""
    av = Path(repo) / "data/processed/avaliacao"
    p = av / AGREGADOS[rotulo].format(m=modelo, b=braco)
    if not p.exists():
        return False, "sem o agregado"
    d = json.loads(p.read_text(encoding="utf-8"))
    if rotulo == "mmlu":
        if d.get("sem_previsao"):
            return False, f"{d['sem_previsao']} itens sem previsão"
    elif rotulo == "g2_1":
        ref = json.loads((av / "g2_teste_1.7B_base_veredito.json").read_text(encoding="utf-8"))
        for b, x in ref["por_benchmark"].items():
            y = d["por_benchmark"].get(b, {})
            if y.get("amostras") != x["amostras"]:
                return False, f"{b}: {y.get('amostras')} amostras, esperadas {x['amostras']}"
            if y.get("fracoes", {}).get("error", 0):
                return False, f"{b}: amostras com ERROR"
    elif rotulo == "ifeval":
        for rot, n in ESPERADO_IFEVAL.items():
            if d.get(rot, {}).get("n") != n:
                return False, f"{rot}: {d.get(rot, {}).get('n')} amostras, esperadas {n}"
    elif rotulo == "evalplus":
        for rot, por in ESPERADO_EVALPLUS.items():
            for conjunto, n in por.items():
                feitas = d.get("geracao", {}).get(rot, {}).get(conjunto, {}).get("amostras")
                if feitas != n:
                    return False, f"{rot}/{conjunto}: {feitas} amostras, esperadas {n}"
    return True, "completa"
