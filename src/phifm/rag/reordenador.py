"""O reordenador do assistente: o ΦRank-PhysBERT sobre os primeiros da busca.

**Adotado em 2026-10-07 pela regra do DOC-13 §9.2**, escrita e aceita antes de o teste ser
tocado. No conjunto de teste (500 itens do primário), reordenar os 50 primeiros com ele:

- leva o artigo certo às 6 fontes em 0,818 das perguntas, contra 0,686 sem ele;
- sobe o acerto da RESPOSTA de 0,700 para 0,806 — +0,106, IC 95% [0,072; 0,142], inteiro
  acima do limiar de 0,03 (`avaliacao/assistente_reordenado.json`);
- custa ~11 s por pergunta (mediana de 49,8 s contra 31 s; limite de 60 s).

O `perguntar.py` e a página o ligam por padrão; `--sem-reordenar` desliga.

Ele roda na CPU por padrão: com o Qwen3-8B na GPU sobram ~1 GB dos 8 da RX 7600. Custo
medido: ~11 s para 50 pares de 384 tokens (6 threads).

O texto de cada documento é o de `textos_de_documentos` — `título. resumo`, o mesmo do
desenvolvimento. Outro formato daria uma ordem plausível e diferente da medida.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
PHIRANK = RAIZ / "models/phirank-physbert-melhor"
PROFUNDIDADE = 50


def carregar_ou_avisar(modelo: Path = PHIRANK) -> Reordenador | None:
    """O reordenador, ou None com um aviso se os pesos não estiverem em disco: o
    assistente continua funcionando como antes da adoção, só com a busca pior."""
    if not (modelo / "model.safetensors").exists():
        print(f"⚠️ o ΦRank não está em {modelo}: o assistente segue SEM reordenar "
              "(acerto medido de 0,700 em vez de 0,806).")
        return None
    return Reordenador(modelo)


class Reordenador:
    def __init__(self, modelo: Path = PHIRANK, dispositivo: str = "cpu",
                 profundidade: int = PROFUNDIDADE):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        cfg = json.loads((modelo / "phirank.json").read_text(encoding="utf-8"))["config"]
        self.max_tokens, self.profundidade = cfg["max_tokens"], profundidade
        self._torch = torch
        self.dev = torch.device(dispositivo)
        self.tok = AutoTokenizer.from_pretrained(modelo)
        self.mod = AutoModelForSequenceClassification.from_pretrained(modelo).to(self.dev).eval()

    def escores(self, consulta: str, textos: list[str]) -> np.ndarray:
        with self._torch.no_grad():
            b = self.tok([consulta] * len(textos), textos, padding=True, truncation=True,
                         max_length=self.max_tokens, return_tensors="pt")
            logits = self.mod(**{k: v.to(self.dev) for k, v in b.items()}).logits
        return logits.float().view(-1).cpu().numpy()

    def ordem(self, consulta: str, textos: list[str]) -> list[int]:
        """Os índices de `textos` do mais ao menos relevante. Empate fica na ordem da
        busca (`stable`)."""
        return list(np.argsort(-self.escores(consulta, textos), kind="stable"))
