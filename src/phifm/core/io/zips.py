"""Descompactar um `.zip` sem herdar o separador de caminho de quem o criou.

## ⚠️ O defeito que este módulo existe para impedir

`Compress-Archive` (Windows PowerShell 5.1) grava os nomes das entradas no estilo do
Windows — `pasta\\arquivo` —, e o `ZipFile.extractall` do Python NÃO traduz isso: no
Linux ele cria **arquivos cujo nome contém a barra invertida**, em vez de uma pasta.
Sem erro nenhum.

Medido em 2026-09-22, no Colab, com o encoder do braço controle do caminho B: os 531
MB subiram para o Drive, a célula descompactou, imprimiu sucesso, e o
`model.safetensors` não estava em lugar nenhum — havia um arquivo chamado
`phienc-cpt-controle\\model.safetensors` solto na pasta de destino. O sintoma foi um
`FileNotFoundError` na conferência de blake3, depois de 40 s de descompactação; se a
conferência não existisse, o erro apareceria dentro do `from_pretrained`, ou pior,
não apareceria.

⚠️ E o `namelist()` do Python MOSTRA barra normal nesses zips quando lido no próprio
Windows, porque lá as duas barras são separadores — então o defeito é invisível na
máquina que empacotou e só aparece na que descompacta.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


def descompactar_normalizado(arquivo: Path, destino: Path) -> list[Path]:
    """Extrai `arquivo` em `destino`, tratando `\\` como separador. Devolve o que saiu.

    Além da barra, recusa entrada que escape do destino (`../`), que o `extractall`
    trata mas uma extração manual reintroduziria.
    """
    destino = Path(destino)
    raiz = destino.resolve()
    escritos: list[Path] = []
    with zipfile.ZipFile(arquivo) as z:
        for info in z.infolist():
            rel = info.filename.replace("\\", "/")
            if rel.endswith("/"):
                continue
            alvo = (destino / rel).resolve()
            if not alvo.is_relative_to(raiz):
                raise ValueError(
                    f"a entrada {info.filename!r} de {arquivo} aponta para fora de "
                    f"{destino}. Um zip pode carregar `../` no nome, e extrair isso "
                    "escreveria onde ninguém pediu.")
            alvo.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as origem, open(alvo, "wb") as saida:
                shutil.copyfileobj(origem, saida)
            escritos.append(alvo)
    return escritos
