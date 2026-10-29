"""O IFEval do G2.3 (DOC-12 §3.7.4): o envelope congelado, a correção pelo código OFICIAL do
Google (`ferramentas/ifeval-oficial`, revisão em `g2_downloads.json`) e os agregados.

- **Consulta:** envelope + "# Query\\n" + prompt + "\\n\\n# Answer\\n" (g2.yaml); paradas
  "\\n# Query", <|endoftext|>, <|im_end|>; até 1.280 fichas.
- **Amostragem:** k=16 a T=0,7, top-p 0,8, top-k 20, mais 1 gulosa. Relatado também o
  IFEval CRU e guloso (o prompt sozinho, como o Open LLM Leaderboard roda bases).
- **O que decide:** o *prompt-level strict* médio das 16 amostras. Relatados: o strict da
  gulosa, o *loose* e o *instruction-level*.
- **kwargs:** os `None` são tirados antes do `build_description` (como no lm-eval): com
  `None`, várias instruções sorteiam o próprio limite, e a correção não seria determinística.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
DADOS = RAIZ / "data/raw/benchmarks/ifeval/ifeval_input_data.jsonl"
OFICIAL = RAIZ / "ferramentas/ifeval-oficial"
NLTK_DADOS = RAIZ / "ferramentas/nltk_data"
ENVELOPE = RAIZ / "configs/eval/preregistration/g2_ifeval_envelope.txt"
PARADAS = ["\n# Query", "<|endoftext|>", "<|im_end|>"]
FICHAS_MAX = 1280
K = 16


def amostragem(gulosa: bool = False) -> dict:
    """T=0,7, top-p 0,8, top-k 20 (ou guloso); o resto desligado, como na geração do G2."""
    from phifm.eval.g2_geracao import amostragem as base

    if gulosa:
        return base(temperatura=0.0, top_p=1.0)
    return base(temperatura=0.7, top_p=0.8) | {"top_k": 20}


def carregar(arquivo: Path = DADOS) -> list[dict]:
    return [json.loads(x) for x in Path(arquivo).read_text(encoding="utf-8").splitlines()
            if x.strip()]


def consulta(prompt: str, envelope: bool = True) -> str:
    if not envelope:
        return prompt
    return ENVELOPE.read_text(encoding="utf-8") + f"# Query\n{prompt}\n\n# Answer\n"


def _oficial():
    """O módulo oficial de correção, com o NLTK lendo os dados do HD."""
    if str(OFICIAL) not in sys.path:
        sys.path.insert(0, str(OFICIAL))
    import nltk

    if str(NLTK_DADOS) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DADOS))
    from instruction_following_eval import evaluation_lib

    return evaluation_lib


def corrigir(item: dict, resposta: str) -> dict:
    """{"strict": bool, "loose": bool, "strict_lista": [...], "loose_lista": [...]} de UMA
    resposta, pelo código oficial."""
    ev = _oficial()
    kwargs = [{k: v for k, v in kw.items() if v is not None} for kw in item["kwargs"]]
    inp = ev.InputExample(key=item["key"], instruction_id_list=item["instruction_id_list"],
                          prompt=item["prompt"], kwargs=kwargs)
    s = ev.test_instruction_following_strict(inp, {item["prompt"]: resposta})
    lo = ev.test_instruction_following_loose(inp, {item["prompt"]: resposta})
    return {"strict": s.follow_all_instructions, "loose": lo.follow_all_instructions,
            "strict_lista": s.follow_instruction_list, "loose_lista": lo.follow_instruction_list}


def agregar(notas: list[dict]) -> dict:
    """Prompt-level e instruction-level, strict e loose, sobre uma lista de correções."""
    if not notas:
        return {}
    n = len(notas)
    inst = [b for x in notas for b in x["strict_lista"]]
    inst_l = [b for x in notas for b in x["loose_lista"]]
    return {"n": n,
            "prompt_strict": round(sum(x["strict"] for x in notas) / n, 4),
            "prompt_loose": round(sum(x["loose"] for x in notas) / n, 4),
            "instrucao_strict": round(sum(inst) / len(inst), 4) if inst else None,
            "instrucao_loose": round(sum(inst_l) / len(inst_l), 4) if inst_l else None}
