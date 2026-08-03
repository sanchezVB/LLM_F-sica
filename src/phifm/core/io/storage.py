"""Camadas de armazenamento — onde cada coisa mora, e por quê.

O DOC-17A §6.1 estabeleceu a arquitetura de custo zero: processar localmente,
alugar só a GPU. Com o Google Drive disponível (>4 TB), a camada fria deixa de
precisar de HD externo — mas a distinção entre **disco de trabalho** e
**arquivo frio** continua valendo, e ignorá-la seria caro.

    QUENTE   disco local        pipeline lê e reescreve dezenas de vezes
    FRIO     Google Drive       escrito uma vez, lido raramente
    REMOTO   bucket B2/R2       checkpoints de treino, escrita a cada 15 min

**Por que o Drive não serve como disco de trabalho.** O Google Drive for
Desktop é um sistema de arquivos por streaming: ler um arquivo baixa, escrever
sobe. A filtragem e a deduplicação (DOC-04) percorrem o corpus várias vezes —
sobre o Drive isso seria centenas de GB de tráfego por passagem, com latência
que inviabiliza o pipeline. Local para trabalhar, Drive para guardar.

**Por que checkpoints não vão para o Drive.** O DOC-08 §7.1 exige gravação a
cada ≤15 min durante o treino, com upload assíncrono que não bloqueie. O
cliente do Drive sincroniza por varredura, não por API transacional — não há
garantia de durabilidade no instante do checkpoint. Isso continua em B2/R2,
por US$ 1–3/mês.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

# O ponto de montagem do Drive contém o e-mail da conta, então é descoberto
# em tempo de execução em vez de fixado no código.
_CLOUD = Path.home() / "Library" / "CloudStorage"
_DRIVE_SUBDIRS = ("Meu Drive", "My Drive")
PROJECT_FOLDER = "LLMFísica"


def find_drive_root() -> Path | None:
    """Localiza a pasta do projeto no Google Drive montado, se houver.

    Devolve ``None`` sem levantar exceção: o Drive é opcional, e todo o
    pipeline precisa funcionar sem ele (é o que o DOC-17A §6.1 garante).
    """
    if env := os.environ.get("PHIFM_DRIVE_ROOT"):
        p = Path(env)
        return p if p.is_dir() else None
    if not _CLOUD.is_dir():
        return None
    for mount in sorted(_CLOUD.glob("GoogleDrive-*")):
        for sub in _DRIVE_SUBDIRS:
            candidate = mount / sub / PROJECT_FOLDER
            if candidate.is_dir():
                return candidate
    return None


@dataclass(frozen=True)
class Storage:
    """Resolve caminhos por camada."""

    local: Path
    drive: Path | None

    @classmethod
    def discover(cls, local: Path | None = None) -> Storage:
        return cls(local=local or PROJECT_ROOT / "data", drive=find_drive_root())

    # ── camada quente ─────────────────────────────────────────────────────
    @property
    def raw(self) -> Path:
        return self.local / "raw"

    @property
    def processed(self) -> Path:
        return self.local / "processed"

    # ── camada fria ───────────────────────────────────────────────────────
    @property
    def has_drive(self) -> bool:
        return self.drive is not None

    def cold(self, kind: str) -> Path | None:
        """Caminho no Drive por tipo de artefato, ou ``None`` sem Drive."""
        mapping = {
            "raw": "01-corpus-bruto",
            "processed": "02-corpus-processado",
            "checkpoints": "03-checkpoints",
            "reports": "04-relatorios",
            "manifests": "05-manifestos",
        }
        if self.drive is None or kind not in mapping:
            return None
        p = self.drive / mapping[kind]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def archive(self, src: Path, kind: str, name: str | None = None) -> Path | None:
        """Copia um artefato para a camada fria.

        Cópia, não movimentação: o original permanece local. Mover para o Drive
        transformaria a leitura seguinte em download, que é exatamente o
        antipadrão que este módulo existe para evitar.
        """
        dest_dir = self.cold(kind)
        if dest_dir is None:
            return None
        dest = dest_dir / (name or src.name)
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            tmp = dest.with_suffix(dest.suffix + ".partial")
            shutil.copy2(src, tmp)
            tmp.replace(dest)  # o arquivo só aparece completo para o cliente
        return dest

    def describe(self) -> str:
        d = str(self.drive) if self.drive else "não montado"
        return f"local (quente): {self.local}\ndrive (frio) : {d}"
