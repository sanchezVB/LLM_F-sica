"""O reordenador do assistente: o ΦRank-PhysBERT sobre os primeiros da busca.

No desenvolvimento da busca do assistente (DOC-13 §9.2, `avaliacao/assistente_busca_dev.json`),
reordenar os 50 primeiros com ele levou o artigo certo às 6 fontes em 0,827 das perguntas
do primário, contra 0,733 sem ele. ⚠️ Isso é exploratório: a confirmação no conjunto de
teste ainda não rodou, e por isso o assistente só usa o reordenador quando pedido
(`--reordenar`).

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
