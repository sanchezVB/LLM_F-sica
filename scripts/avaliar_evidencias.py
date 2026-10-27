#!/usr/bin/env python3
"""Avalia o juiz EXPERIMENTAL de sustentação em casos rotulados, sem busca nem treino.

    python scripts/avaliar_evidencias.py --casos tests/fixtures/rag_evidencias_dev.json \
        --juiz-id Qwen3-8B-Q4_K_M --saida /caminho/no/HD/evidencias-dev.json

Requer llama-server JÁ em execução no computador local. Não baixa pesos nem inicia
servidores. O conjunto incluído é sintético e de desenvolvimento; passar nele não
substitui uma confirmação nova com revisão humana. O relatório conta falsos aceites,
rejeições indevidas e falhas do juiz separadamente. O avaliador não envia os rótulos ao
juiz e recusa sobrescrever resultados.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.rag.assistente import Fonte  # noqa: E402
from phifm.rag.evidencias import (  # noqa: E402
    SISTEMA_EVIDENCIAS,
    VEREDITOS,
    RelatorioEvidencias,
    VerificadorEvidencias,
)
from phifm.rag.llm import ModeloLocal  # noqa: E402


@dataclass(frozen=True)
class Caso:
    id: str
    pergunta: str
    resposta: str
    fontes: list[Fonte]
    esperado: str


def carregar_casos(caminho: Path) -> tuple[str, list[Caso]]:
    d = json.loads(caminho.read_text(encoding="utf-8"))
    if (not isinstance(d, dict) or d.get("tipo") not in
            {"desenvolvimento_sintetico", "desenvolvimento", "confirmacao_humana"}
            or not isinstance(d.get("casos"), list) or not d["casos"]):
        raise ValueError("conjunto deve declarar tipo e casos não vazios")
    casos = []
    ids = set()
    for c in d["casos"]:
        if (not isinstance(c, dict) or any(not isinstance(c.get(k), str) or not c[k].strip()
                for k in ("id", "pergunta", "resposta", "esperado"))
                or c["id"] in ids or c["esperado"] not in VEREDITOS
                or not isinstance(c.get("fontes"), list)):
            raise ValueError("caso inválido, sem rótulo ou com id duplicado")
        fontes = []
        for n, f in enumerate(c["fontes"], 1):
            if (not isinstance(f, dict) or not isinstance(f.get("resumo"), str)
                    or not isinstance(f.get("titulo", ""), str)
                    or not isinstance(f.get("passagens", []), list)
                    or any(not isinstance(p, str) for p in f.get("passagens", []))):
                raise ValueError("fonte inválida")
            fontes.append(Fonte(n, "", f.get("titulo", ""), None, f["resumo"], 0.0,
                                 tuple(f.get("passagens", []))))
        ids.add(c["id"])
        casos.append(Caso(c["id"], c["pergunta"], c["resposta"], fontes, c["esperado"]))
    return d["tipo"], casos


def veredito_da_resposta(r: RelatorioEvidencias) -> str:
    if any(j.veredito == "contradita" for j in r.julgamentos):
        return "contradita"
    if r.julgamentos and r.rejeitadas == 0:
        return "sustentada"
    return "insuficiente"


def avaliar(casos: list[Caso], verificador: VerificadorEvidencias) -> dict:
    if not casos or len({c.id for c in casos}) != len(casos):
        raise ValueError("avaliação precisa de casos não vazios e únicos")
    linhas = []
    for c in casos:
        t0 = time.perf_counter()
        r = verificador.verificar(c.pergunta, c.resposta, c.fontes, portugues=True)
        linhas.append({"id": c.id, "esperado": c.esperado,
                       "veredito": veredito_da_resposta(r),
                       "falhas_do_juiz": sum(j.erro == "falha_do_juiz" for j in r.julgamentos),
                       "fragmentos_sustentados": r.sustentadas,
                       "fragmentos_retidos": r.rejeitadas,
                       "segundos": round(time.perf_counter() - t0, 3)})
    positivas = [r for r in linhas if r["esperado"] == "sustentada"]
    negativas = [r for r in linhas if r["esperado"] != "sustentada"]
    aceitas = [r for r in linhas if r["veredito"] == "sustentada"]
    falso_aceite = sum(r["veredito"] == "sustentada" for r in negativas)
    falsa_rejeicao = sum(r["veredito"] != "sustentada" for r in positivas)
    return {"itens": len(linhas),
            "acerto_rotulo": sum(r["veredito"] == r["esperado"] for r in linhas) / len(linhas),
            "suporte_verdadeiro": len(positivas), "sem_suporte_verdadeiro": len(negativas),
            "aceites": len(aceitas), "falsos_aceites": falso_aceite,
            "rejeicoes_indevidas": falsa_rejeicao,
            "taxa_falso_aceite": falso_aceite / len(negativas) if negativas else None,
            "taxa_rejeicao_indevida": falsa_rejeicao / len(positivas) if positivas else None,
            "precisao_dos_aceites": (len(aceitas) - falso_aceite) / len(aceitas) if aceitas else None,
            "cobertura": len(aceitas) / len(linhas),
            "falhas_do_juiz": sum(r["falhas_do_juiz"] for r in linhas),
            "resultados": linhas}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--casos", required=True, type=Path)
    p.add_argument("--saida", required=True, type=Path)
    p.add_argument("--juiz-id", required=True,
                   help="identificação do modelo carregado no servidor, declarada pelo operador")
    p.add_argument("--porta", type=int, default=8080)
    a = p.parse_args()
    if a.saida.exists():
        p.error("a saída já existe; use outro nome para preservar a medição")
    tipo, casos = carregar_casos(a.casos)
    modelo = ModeloLocal(porta=a.porta)
    if not modelo.no_ar():
        p.error("llama-server local indisponível; inicie o modelo antes da avaliação")
    r = {"experimental": True, "tipo_do_conjunto": tipo,
         "juiz_declarado": a.juiz_id, "servidor_local": modelo.url,
         "casos_sha256": hashlib.sha256(a.casos.read_bytes()).hexdigest(),
         "prompt_sha256": hashlib.sha256(SISTEMA_EVIDENCIAS.encode()).hexdigest(),
         "codigo_sha256": {str(q.relative_to(RAIZ)): hashlib.sha256(q.read_bytes()).hexdigest()
                            for q in (Path(__file__), RAIZ / "src/phifm/rag/evidencias.py")},
         "nota": "Identificação do juiz declarada pelo operador; não atesta o peso servido. "
                 "Casos de desenvolvimento não estabelecem qualidade em perguntas reais.",
         **avaliar(casos, VerificadorEvidencias(modelo))}
    a.saida.parent.mkdir(parents=True, exist_ok=True)
    with a.saida.open("x", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"{r['itens']} itens · falsos aceites {r['falsos_aceites']} · "
          f"rejeições indevidas {r['rejeicoes_indevidas']} · "
          f"falhas do juiz {r['falhas_do_juiz']} · {a.saida}")
    return 1 if r["falhas_do_juiz"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
