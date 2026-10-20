"""A correção da medida do G2 num processo trabalhador que pode ser MORTO.

O teto de `g2_corretor.com_teto` é por thread: ela devolve INCONCLUSIVE no tempo, mas a
thread que estourou continua viva, segura o GIL em chamadas longas do SymPy e atrasa os
itens seguintes. Isso foi achado na rodada 2 do estresse adversarial, com uma torre de
potências. Aqui cada correção vai para um processo à parte. Se a resposta não volta em
`teto_s`, o processo é morto e outro sobe no lugar; o item fica INCONCLUSIVE, e a
contagem de mortes entra no relatório.

    with CorretorIsolado() as cor:
        v = cor.corrigir(item, texto)
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import sys
from pathlib import Path

from phifm.eval.g2_corretor import Veredito
from phifm.eval.g2_itens import Item

TETO_RIGIDO_S = 60.0     # acima dos 2 × 10 s do teto por thread, com folga para a carga


def _trabalhar(entrada, saida, raiz_src: str) -> None:
    sys.path.insert(0, raiz_src)
    from phifm.eval import g2_itens  # aquece o parser na importação

    saida.put("pronto")
    while (pedido := entrada.get()) is not None:
        n, item, texto = pedido
        try:
            saida.put((n, str(g2_itens.corrigir(item, texto))))
        except Exception:
            saida.put((n, str(Veredito.ERROR)))


class CorretorIsolado:
    def __init__(self, teto_s: float = TETO_RIGIDO_S):
        self.teto_s = teto_s
        self.mortes = 0
        self._n = 0
        self._ctx = mp.get_context("spawn")
        self._proc = None

    def _iniciar(self) -> None:
        self._entrada, self._saida = self._ctx.Queue(), self._ctx.Queue()
        raiz_src = str(Path(__file__).resolve().parents[2])
        self._proc = self._ctx.Process(target=_trabalhar, daemon=True,
                                       args=(self._entrada, self._saida, raiz_src))
        self._proc.start()
        if self._saida.get(timeout=300) != "pronto":
            raise RuntimeError("o trabalhador de correção não subiu")

    def corrigir(self, item: Item, texto: str) -> Veredito:
        if self._proc is None or not self._proc.is_alive():
            self._iniciar()
        self._n += 1
        self._entrada.put((self._n, item, texto))
        try:
            while True:
                n, v = self._saida.get(timeout=self.teto_s)
                if n == self._n:
                    return Veredito(v)
        except queue.Empty:
            self._proc.kill()
            self._proc.join(10)
            self._proc = None
            self.mortes += 1
            return Veredito.INCONCLUSIVE

    def fechar(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._entrada.put(None)
            self._proc.join(10)
            if self._proc.is_alive():
                self._proc.kill()
        self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.fechar()
