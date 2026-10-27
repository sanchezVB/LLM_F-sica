"""Pré-condições de execução G2, sem carregar enunciados ou iniciar o modelo.

A conferência lê metadados e hashes dos arquivos, nunca os itens de benchmark.
No congelamento, ``codigo_congelado.arquivos`` deve guardar o mapa completo de
``g2.codigo_do_corretor`` com os módulos de execução abaixo. O SHA de
``exemplos_few_shot`` é o do JSON de seleção; ``motor.sha256_executavel`` ancora
os bytes do binário. Este módulo não preenche nem congela esses campos.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from phifm.eval import g2

EXTRAS = tuple(f"src/phifm/eval/{n}.py" for n in
               ("g2_prompt", "g2_geracao_robusta", "g2_correcao", "g2_prerequisitos")) + (
                   "scripts/medir_g2_robusto.py",)
BENCHS = ("ugphysics", "olympiadbench", "phybench", "scibench")
_SHA = re.compile(r"[0-9a-f]{64}")


@dataclass
class Requisitos:
    bloqueios: list[str] = field(default_factory=list)
    exclusoes: dict[str, set[str]] = field(default_factory=dict)
    conferidos: list[str] = field(default_factory=list)

    @property
    def pronto(self) -> bool:
        return not self.bloqueios

    def como_dict(self) -> dict:
        return {"pronto": self.pronto, "bloqueios": self.bloqueios,
                "exclusoes": {b: sorted(ids) for b, ids in self.exclusoes.items()},
                "conferidos": self.conferidos}


class PrerequisitosAusentes(RuntimeError):
    def __init__(self, requisitos: Requisitos):
        self.requisitos = requisitos
        super().__init__("G2 não pode iniciar:\n- " + "\n- ".join(requisitos.bloqueios))


def sha256_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _json(path: Path, r: Requisitos) -> dict:
    try:
        dado = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(dado, dict):
            raise ValueError("objeto JSON esperado")
        return dado
    except (OSError, ValueError) as exc:
        r.bloqueios.append(f"{path}: ausente ou JSON inválido ({type(exc).__name__})")
        return {}


def _hash(path: Path, esperado: object, r: Requisitos, *, texto: bool = False) -> None:
    if not isinstance(esperado, str) or not _SHA.fullmatch(esperado):
        r.bloqueios.append(f"{path}: hash de referência ausente ou inválido")
        return
    try:
        atual = g2.sha256_arquivo(path) if texto else sha256_bytes(path)
    except OSError:
        r.bloqueios.append(f"{path}: arquivo ausente ou ilegível")
        return
    if atual != esperado:
        r.bloqueios.append(f"{path}: SHA-256 diverge da referência")
    else:
        r.conferidos.append(str(path))


def _dentro(raiz: Path, relativo: str) -> Path:
    caminho = raiz / relativo
    if not caminho.resolve().is_relative_to(raiz.resolve()):
        raise ValueError(f"caminho de artefato sai da raiz: {relativo}")
    return caminho


def _autoteste(aval: Path, raiz: Path, fase: str, r: Requisitos) -> None:
    dado = _json(aval / "g2_autoteste.json", r)
    try:
        atual = g2.codigo_do_corretor(raiz)
    except OSError:
        r.bloqueios.append("código do corretor ausente: autoteste não pode ser conferido")
        return
    if dado.get("codigo_sha256") != atual or any(v is None for v in atual.values()):
        r.bloqueios.append("autoteste desatualizado: código ou versões do corretor mudaram")
    conjuntos = dado.get("conjuntos", {})
    for bench in (BENCHS if fase == "teste" else ("ugphysics", "olympiadbench", "scibench")):
        chave = f"{bench}/{fase}"
        item = conjuntos.get(chave, {}) if isinstance(conjuntos, dict) else {}
        ids = item.get("ids_excluidos") if isinstance(item, dict) else None
        n = item.get("itens") if isinstance(item, dict) else None
        excluidos = item.get("excluidos") if isinstance(item, dict) else None
        if (not isinstance(ids, list) or not all(isinstance(i, str) and i for i in ids)
                or len(set(ids)) != len(ids) or type(n) is not int or n <= 0
                or type(excluidos) is not int or excluidos != len(ids) or excluidos >= n):
            r.bloqueios.append(f"autoteste {chave}: contagens ou exclusões ausentes/inválidas")
            continue
        if fase == "teste" and (excluidos / n > 0.05 or item.get("acima_do_limite")):
            r.bloqueios.append(f"autoteste {chave}: exclusões acima do limite de 5%")
        r.exclusoes[bench] = set(ids)


def _dados(conf: dict, raiz: Path, r: Requisitos) -> None:
    dados = conf.get("dados", {})
    manifesto = _json(_dentro(raiz, dados.get("manifesto", "data/processed/avaliacao/g2_downloads.json")), r)
    itens = manifesto.get("itens", {})
    for bench in BENCHS:
        entrada = itens.get(bench, {}) if isinstance(itens, dict) else {}
        arquivos = entrada.get("arquivos", {})
        if not arquivos:
            r.bloqueios.append(f"manifesto de downloads sem arquivos de {bench}")
            continue
        revisao = dados.get("revisoes", {}).get(entrada.get("repositorio"))
        if not revisao or entrada.get("revisao") != revisao:
            r.bloqueios.append(f"revisão de {bench} diverge do pré-registro")
        for nome, meta in arquivos.items():
            rel = f"{bench}/{nome if bench == 'ugphysics' else Path(nome).name}"
            esperado = dados.get("sha256", {}).get(rel)
            if not isinstance(meta, dict) or meta.get("sha256") != esperado:
                r.bloqueios.append(f"manifesto e pré-registro divergem em {rel}")
                continue
            _hash(_dentro(raiz / "data/raw/benchmarks" / bench, nome), esperado, r)
        # Um manifesto parcial não pode eliminar silenciosamente arquivos do protocolo.
        declarados = {f"{bench}/{n if bench == 'ugphysics' else Path(n).name}" for n in arquivos}
        requeridos = {n for n in dados.get("sha256", {}) if n.startswith(bench + "/")}
        if not requeridos or not requeridos <= declarados:
            r.bloqueios.append(f"manifesto de {bench} incompleto para o pré-registro")


def conferir(raiz: Path = g2.RAIZ, *, fase: str = "dev", modelo: str = "1.7B",
             braco: str = "base", gguf: Path | None = None, kl: Path | None = None,
             manifesto_cpt: Path | None = None, executavel: Path | None = None,
             hash_config: str | None = None) -> Requisitos:
    """Conferência somente leitura; ``hash_config`` é comparado antes de subir o motor."""
    if fase not in ("dev", "teste") or modelo not in ("1.7B", "8B"):
        raise ValueError("fase ou modelo G2 inválido")
    raiz = Path(raiz)
    r = Requisitos()
    prereg = raiz / "configs/eval/preregistration/g2.yaml"
    try:
        conf = yaml.safe_load(prereg.read_text(encoding="utf-8"))
        if not isinstance(conf, dict):
            raise ValueError("pré-registro não é um mapa")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        r.bloqueios.append(f"pré-registro ausente/inválido: {type(exc).__name__}")
        return r
    if not g2.REGRA_G2_ACEITA or conf.get("estado") != "ACEITA":
        r.bloqueios.append("regra G2 ainda não aceita")
    if fase == "teste":
        try:
            g2.exigir_aceite_e_congelamento(preregistro=prereg)
        except (g2.RegraNaoAceita, g2.ProtocoloNaoCongelado, g2.ProtocoloMudou) as exc:
            r.bloqueios.append(str(exc))
    aval = raiz / "data/processed/avaliacao"
    _autoteste(aval, raiz, fase, r)
    exemplos_path = aval / "g2_exemplos.json"
    exemplos = _json(exemplos_path, r).get("exemplos", {})
    for bench, quantidade in zip(BENCHS, (4, 4, 3, 9), strict=True):
        e = exemplos.get(bench, {}) if isinstance(exemplos, dict) else {}
        ids = e.get("ids", []) if isinstance(e, dict) else []
        if (not isinstance(ids, list) or len(ids) != quantidade
                or not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids)
                or not _SHA.fullmatch(str(e.get("sha256_do_texto", "")))):
            r.bloqueios.append(f"exemplos few-shot de {bench} ausentes/inválidos")
        elif fase == "dev":
            origem = "scibench" if bench == "scibench" else "ugphysics"
            if set(ids) & r.exclusoes.get(origem, set()):
                r.bloqueios.append(f"exemplos de {bench} contêm item excluído pelo autoteste")
    particao = conf.get("recortes", {}).get("ugphysics", {}).get("particao", {})
    _hash(_dentro(raiz, particao.get("arquivo", "data/processed/avaliacao/g2_particao.json")),
          particao.get("sha256"), r, texto=True)
    _dados(conf, raiz, r)
    motor = conf.get("motor", {})
    nome = f"Qwen3-{modelo}-Base"
    gguf = Path(gguf) if gguf else raiz / f"models/base/gguf/{nome}-Q8_0.gguf"
    kl = Path(kl) if kl else aval / f"g2_kl_{modelo}.json"
    quant = _json(kl, r)
    media = quant.get("kl_media_nats")
    limite = motor.get("kl_maxima_contra_f16_nats")
    if (type(media) not in (int, float) or not math.isfinite(media) or media < 0
            or type(limite) not in (int, float) or not math.isfinite(limite)
            or limite != 0.01 or media > limite or quant.get("passa") is not True
            or quant.get("tamanho") != modelo):
        r.bloqueios.append("conferência KL ausente, inválida ou acima de 0,01 nats")
    q8 = quant.get("gguf", {}).get("q8_0", {})
    sha = q8.get("sha256")
    _hash(gguf, sha, r)
    sha_base = motor.get("gguf_sha256", {}).get(nome)
    if braco == "base" and sha != sha_base:
        r.bloqueios.append("GGUF da base não corresponde ao pré-registro")
    if braco != "base" and sha == sha_base:
        r.bloqueios.append("GGUF do candidato é idêntico à base")
    executavel = Path(executavel) if executavel else raiz / "ferramentas/llama.cpp/llama-server.exe"
    if not executavel.is_file():
        r.bloqueios.append(f"motor ausente: {executavel}")
    if fase == "teste":
        _hash(exemplos_path, conf.get("exemplos_few_shot", {}).get("sha256"), r, texto=True)
        _hash(executavel, motor.get("sha256_executavel"), r)
        _hash(raiz / f"models/base/{nome}/tokenizer.json", motor.get("tokenizer_sha256"), r)
        try:
            atual = g2.codigo_do_corretor(raiz, extras=EXTRAS)
        except OSError:
            atual = {}
        congelado = conf.get("codigo_congelado", {}).get("arquivos")
        if not atual or congelado != atual or any(v is None for v in atual.values()):
            r.bloqueios.append("código/pacotes atuais divergem de codigo_congelado.arquivos")
        hc = motor.get("hash_da_configuracao")
        if isinstance(hc, dict):
            hc = hc.get(f"{modelo}/{braco}")
        if not isinstance(hc, str) or not _SHA.fullmatch(hc):
            r.bloqueios.append("hash da configuração do motor não congelado")
        elif hash_config is not None and hc != hash_config:
            r.bloqueios.append("configuração real do motor diverge do congelamento")
    if braco != "base":
        if manifesto_cpt is None:
            r.bloqueios.append("candidato CPT exige manifesto do corpus auditado")
        else:
            try:
                from phifm.corpus.auditoria_cpt import validar_manifesto
                validar_manifesto(Path(manifesto_cpt), exigir_referencias_g2=True,
                    referencias_g2=(conf.get("contaminacao", {}).get("referencias_g2")
                                    if fase == "teste" and g2.PROTOCOLO_G2_SHA256 else None))
            except (ImportError, OSError, ValueError) as exc:
                r.bloqueios.append(f"auditoria CPT não confirmada: {exc}")
    return r


def exigir(*args, **kwargs) -> Requisitos:
    requisitos = conferir(*args, **kwargs)
    if not requisitos.pronto:
        raise PrerequisitosAusentes(requisitos)
    return requisitos
