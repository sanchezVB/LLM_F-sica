#!/usr/bin/env python3
"""A condição da §9.9 do DOC-13: o produto com a busca fundida roda na máquina do dono sem
paginar, e em quanto tempo?

    .venv-treino\\Scripts\\python.exe scripts\\medir_produto_busca_lexical.py [-n 10]

Monta o assistente como a página (`servir_busca.py --assistente`): índice denso e ΦRank
na CPU, o texto completo carregado (a opção da página) e o índice lexical aberto do disco
sob demanda, com o modelo na GPU. Responde `n` perguntas do desenvolvimento de cada
conjunto (corpo 1 e primário dev), alternando sem e com a busca fundida na MESMA sessão, e
anota o tempo de ponta a ponta e a memória. O teste não é tocado.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.core.console import utf8  # noqa: E402

utf8()

AGREGADO = RAIZ / "data/processed/avaliacao/assistente_busca_lexical_produto.json"


class _Contadores:
    """`PROCESS_MEMORY_COUNTERS` do Windows, montado na primeira chamada."""

    tipo = None

    @classmethod
    def ler(cls, pid: int | None = None) -> dict:
        import ctypes
        from ctypes import wintypes

        if cls.tipo is None:
            class C(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            cls.tipo = C
            cls.k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            cls.psapi = ctypes.WinDLL("psapi", use_last_error=True)
            cls.k32.GetCurrentProcess.restype = wintypes.HANDLE
            cls.k32.OpenProcess.restype = wintypes.HANDLE
            cls.k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            cls.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(C),
                                                       wintypes.DWORD]
        c = cls.tipo()
        c.cb = ctypes.sizeof(c)
        h = (cls.k32.GetCurrentProcess() if pid is None
             else cls.k32.OpenProcess(0x1000 | 0x0010, False, pid))   # consulta + leitura
        if not h or not cls.psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
            return {"ws_gb": None, "privado_gb": None, "faltas": None}
        if pid is not None:
            cls.k32.CloseHandle(h)
        return {"ws_gb": round(c.WorkingSetSize / 2**30, 2),
                "privado_gb": round(c.PagefileUsage / 2**30, 2), "faltas": int(c.PageFaultCount)}


def livre_gb() -> float:
    import ctypes
    from ctypes import wintypes

    class Estado(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    e = Estado()
    e.dwLength = ctypes.sizeof(e)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(e))
    return round(e.ullAvailPhys / 2**30, 2)


def main() -> int:
    import numpy as np

    import explorar_busca_lexical as e
    from phifm.rag import passagens as ps
    from phifm.rag.assistente import SISTEMA_RESPOSTA_V2, Assistente
    from phifm.rag.llm import ModeloLocal
    from phifm.rag.reordenador import carregar_ou_avisar
    from phifm.retrieval.indice import Busca
    from phifm.retrieval.lexical import Lexica

    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=10, help="perguntas de cada conjunto")
    a = p.parse_args()

    corpo = json.loads((e.DIR / "itens_corpo.json").read_text(encoding="utf-8"))[:a.n]
    dev = [i for i in json.loads((e.DIR / "itens_dev_busca.json").read_text(encoding="utf-8"))
           if i["estrato"] == "primario"][:a.n]
    perguntas = [("corpo_1", i["pergunta"]) for i in corpo] + [("primario_dev", i["pergunta"])
                                                              for i in dev]
    t0 = time.perf_counter()
    busca = Busca(e.INDICE, dispositivo="cpu")
    modelo = ModeloLocal()
    if not modelo.no_ar():
        modelo.iniciar(log=RAIZ / "data/processed/llama_server.log")
    passagens = ps.carregar_ou_avisar(modelo, busca.tok, RAIZ / "data/processed/textos_por_artigo")
    lexica = Lexica(e.LEXICO, busca.docs["arxiv_id"].to_list())
    comum = {"reordenador": carregar_ou_avisar(), "passagens": passagens,
             "sistema_com_passagens": SISTEMA_RESPOSTA_V2}
    sem = Assistente(busca, modelo, e.SPINE, **comum)
    com = Assistente(busca, modelo, e.SPINE, lexica=lexica, **comum)
    pid_llama = modelo._processo.pid if modelo._processo is not None else None

    def foto() -> dict:
        return {"livre_gb": livre_gb(), "python": _Contadores.ler(),
                "llama": _Contadores.ler(pid_llama) if pid_llama else None}

    inicio = foto()
    print(f"carregado em {time.perf_counter() - t0:.0f} s · {inicio}", flush=True)
    linhas = []

    def faltas(m1, m0, quem):
        a, b = (m1[quem] or {}).get("faltas"), (m0[quem] or {}).get("faltas")
        return None if a is None or b is None else a - b

    try:
        for n, (conjunto, pergunta) in enumerate(perguntas, start=1):
            ordem = (("sem", sem), ("com", com)) if n % 2 else (("com", com), ("sem", sem))
            for rotulo, ass in ordem:
                m0, t = foto(), time.perf_counter()
                consulta = ass.consulta_hipotetica(pergunta)
                t_hyde = time.perf_counter()
                ingles = ass.em_ingles(pergunta) if ass.lexica is not None else None
                t_trad = time.perf_counter()
                fontes = ass.recuperar(consulta, ingles)
                t_busca = time.perf_counter()
                ass.responder_com(pergunta, fontes, consulta)
                t_fim = time.perf_counter()
                m1 = foto()
                linhas.append({
                    "n": n, "conjunto": conjunto, "busca": rotulo,
                    "segundos": round(t_fim - t, 1), "hyde_s": round(t_hyde - t, 1),
                    "traducao_s": round(t_trad - t_hyde, 1), "busca_s": round(t_busca - t_trad, 1),
                    "resposta_s": round(t_fim - t_busca, 1), "livre_gb": m1["livre_gb"],
                    "python_ws_gb": m1["python"]["ws_gb"],
                    "faltas_python": faltas(m1, m0, "python"),
                    "faltas_llama": faltas(m1, m0, "llama")})
                print(f"[{n}/{len(perguntas)}] {linhas[-1]}", flush=True)
    finally:
        modelo.parar()
    resumo = {"o_que_e": "o produto como a página, na máquina do dono: índice denso e ΦRank "
                         "na CPU, texto completo carregado, índice lexical do disco sob "
                         "demanda, modelo na GPU; perguntas do desenvolvimento",
              "inicio": inicio, "perguntas": len(perguntas)}
    for rotulo in ("sem", "com"):
        xs = [x for x in linhas if x["busca"] == rotulo]
        resumo[rotulo] = {k: round(float(np.median([x[k] for x in xs])), 1)
                          for k in ("segundos", "hyde_s", "traducao_s", "busca_s", "resposta_s")}
        resumo[rotulo].update({
            "maximo_s": max(x["segundos"] for x in xs),
            "livre_min_gb": min(x["livre_gb"] for x in xs),
            "faltas_python_mediana": int(np.median([x["faltas_python"] or 0 for x in xs])),
            "faltas_llama_mediana": int(np.median([x["faltas_llama"] or 0 for x in xs]))})
    resumo["por_pergunta"] = linhas
    AGREGADO.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resumo, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
