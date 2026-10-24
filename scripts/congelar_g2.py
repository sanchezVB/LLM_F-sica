#!/usr/bin/env python3
"""O congelamento do protocolo do G2 (DOC-12 §3.7.3), uma vez, depois do dev:

    .venv\\Scripts\\python.exe scripts\\congelar_g2.py

1. Confere a auditoria cega cumprida e o autoteste da MESMA versão do corretor.
2. Grava `configs/eval/preregistration/g2_congelamento.json`: o SHA-256 do código e dos dados
   da medida (`g2.estado_congelavel`) e o hash da configuração da geração da 1,7B.
3. Preenche no `g2.yaml` os campos do congelamento (guarda, exemplos, envelope do IFEval,
   hash da configuração, o arquivo do congelamento e o hash dele).
4. Grava o SHA-256 do `g2.yaml` em `PROTOCOLO_G2_SHA256` (src/phifm/eval/g2.py).

Depois disso, o `medir_g2 teste` recusa se o g2.yaml, o código ou os dados mudarem.
Rodar de novo recusa: o protocolo já está congelado.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2, g2_geracao  # noqa: E402

AVAL = RAIZ / "data/processed/avaliacao"
G2_PY = RAIZ / "src/phifm/eval/g2.py"
GGUF_1_7B = RAIZ / "models/base/gguf/Qwen3-1.7B-Base-Q8_0.gguf"


def _trocar(texto: str, velho: str, novo: str) -> str:
    if texto.count(velho) != 1:
        raise SystemExit(f"no g2.yaml, esperava uma ocorrência de: {velho!r}")
    return texto.replace(velho, novo)


def main() -> int:
    if g2.PROTOCOLO_G2_SHA256:
        raise SystemExit("o protocolo já está congelado; mudança só por emenda registrada")
    aud = json.loads((AVAL / "g2_auditoria.json").read_text(encoding="utf-8"))
    if not aud.get("auditoria", {}).get("alvo_cumprido"):
        raise SystemExit("a auditoria cega não cumpriu o alvo: não congelo")
    codigo = g2.codigo_do_corretor(RAIZ)
    if json.loads((AVAL / "g2_autoteste.json").read_text(encoding="utf-8"))["codigo_sha256"] != codigo:
        raise SystemExit("o autoteste é de outra versão do corretor: refaça antes de congelar")
    if aud["codigo_sha256"] != codigo:
        raise SystemExit("a auditoria é de outra versão do corretor: refaça antes de congelar")

    kl = json.loads((AVAL / "g2_kl_1.7B.json").read_text(encoding="utf-8"))
    hc = g2_geracao.hash_da_configuracao(g2_geracao.ServidorG2(GGUF_1_7B),
                                         kl["gguf"]["q8_0"]["sha256"], "RX 7600 (Vulkan)",
                                         g2_geracao.amostragem())
    estado = g2.estado_congelavel(RAIZ)
    congelamento = {
        "o_que_e": "o congelamento do protocolo do G2 (DOC-12 §3.7.3): o código e os dados da "
                   "medida por SHA-256; o g2.yaml guarda o hash deste arquivo",
        "em": date.today().isoformat(), **estado,
        "hash_da_configuracao": {"Qwen3-1.7B-Base": hc},
        "auditoria": {"concordancia_pass": aud["auditoria"]["concordancia_pass"],
                      "kappa_pass": aud["auditoria"]["kappa_pass"]},
    }
    g2.CONGELAMENTO.write_text(json.dumps(congelamento, ensure_ascii=False, indent=1) + "\n",
                               encoding="utf-8", newline="\n")
    sha_cong = g2.sha256_arquivo(g2.CONGELAMENTO)

    yaml = g2.PREREGISTRO.read_text(encoding="utf-8")
    yaml = _trocar(yaml, "guarda_de_repeticao: {janela_fichas: 32, repeticoes: 4}",
                   "guarda_de_repeticao: {janela_fichas: 32, repeticoes: 4, folga_fichas: 2, "
                   "teto: 8, regra: intervalos iguais}")
    yaml = _trocar(yaml, "  sha256: null               # congelamento",
                   "  sha256: " + estado["dados"]["data/processed/avaliacao/g2_exemplos.json"])
    yaml = _trocar(yaml, "    envelope_sha256: null    # congelamento",
                   "    envelope_sha256: " + estado["dados"][
                       "configs/eval/preregistration/g2_ifeval_envelope.txt"])
    yaml = _trocar(yaml, "  hash_da_configuracao: null                                   # congelamento",
                   f"  hash_da_configuracao: {{Qwen3-1.7B-Base: {hc},\n"
                   "                         Qwen3-8B-Base: no livro de acessos, na medida do Kaggle}")
    yaml = yaml.rstrip("\n") + (
        "\n\n# Preenchido por scripts/congelar_g2.py. Depois disto, o medir_g2 teste recusa se o\n"
        "# g2.yaml, o código ou os dados da medida mudarem (g2.exigir_aceite_e_congelamento).\n"
        f"congelamento: {{arquivo: configs/eval/preregistration/g2_congelamento.json,\n"
        f"               sha256: {sha_cong},\n"
        f"               em: \"{date.today().isoformat()}\"}}\n")
    g2.PREREGISTRO.write_text(yaml, encoding="utf-8", newline="\n")
    sha_yaml = g2.sha256_arquivo(g2.PREREGISTRO)

    fonte = G2_PY.read_text(encoding="utf-8")
    velho = "PROTOCOLO_G2_SHA256: str | None = None"
    if fonte.count(velho) != 1:
        raise SystemExit("não achei a trava PROTOCOLO_G2_SHA256 em g2.py")
    G2_PY.write_text(fonte.replace(velho, f'PROTOCOLO_G2_SHA256: str | None = "{sha_yaml}"'),
                     encoding="utf-8", newline="")
    print(f"congelamento: {sha_cong}\ng2.yaml:      {sha_yaml}\nconfiguração: {hc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
