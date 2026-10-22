"""A correção do G2 num processo trabalhador que pode ser MORTO e que se REINICIA.

O teto de `g2_corretor.com_teto` é por thread: ela devolve INCONCLUSIVE no tempo, mas a
thread que estourou continua viva, segura o GIL em chamadas longas do SymPy e atrasa os
itens seguintes. Isso foi achado na rodada 2 do estresse adversarial, com uma torre de
potências, e de novo em 2026-10-22: no autoteste, as threads vazadas se acumularam até o
processo parar no item 400 do UG-dev. Aqui:

- cada pedido vai para um processo à parte; se a resposta não volta em `teto_s`, o
  processo é morto e outro sobe no lugar, e o pedido sai INCONCLUSIVE (ou, no autoteste,
  não ok);
- se depois de um pedido sobra thread vazada, o trabalhador avisa e sai; o pedido é refeito
  UMA vez num trabalhador novo, sem a carga das threads vazadas, e vale o segundo
  resultado. Nenhum pedido seguinte roda num processo contaminado.

As contagens de mortes e de reinícios entram no relatório.

    with CorretorIsolado() as cor:
        v = cor.corrigir(item, texto)
        r = cor.autoteste(item)
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import sys
from pathlib import Path

from phifm.eval.g2_corretor import Veredito
from phifm.eval.g2_itens import Item

TETO_RIGIDO_S = 60.0     # acima dos 2 × 10 s do teto por thread, com folga para a carga
TETO_AUTOTESTE_S = 300.0  # o autoteste de um item faz o próprio e até ~6 perturbações


def _trabalhar(entrada, saida, raiz_src: str) -> None:
    sys.path.insert(0, raiz_src)
    from phifm.eval import g2_corretor, g2_itens  # aquece o parser na importação

    saida.put("pronto")
    while (pedido := entrada.get()) is not None:
        n, tipo, item, texto = pedido
        try:
            if tipo == "autoteste":
                r = g2_itens.autoteste(item)
            else:
                r = str(g2_itens.corrigir(item, texto))
        except Exception:
            r = ({"id": item.id, "ok": False, "proprio": str(Veredito.ERROR), "perturbacoes": 0,
                  "perturbacoes_que_passaram": 0} if tipo == "autoteste" else str(Veredito.ERROR))
        vazou = g2_corretor.threads_vazadas() > 0
        saida.put((n, r, vazou))
        if vazou:
            return                # processo contaminado: sai, e o pai sobe outro


class CorretorIsolado:
    def __init__(self, teto_s: float = TETO_RIGIDO_S, teto_autoteste_s: float = TETO_AUTOTESTE_S):
        self.teto_s = teto_s
        self.teto_autoteste_s = teto_autoteste_s
        self.mortes = 0
        self.reinicios = 0
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

    def _encerrar(self, matar: bool) -> None:
        if self._proc is not None:
            if matar:
                self._proc.kill()
            self._proc.join(10)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(10)
        self._proc = None

    def _uma_vez(self, tipo: str, item: Item, texto: str | None, teto: float):
        """(resultado, vazou), ou (None, True) se o processo foi morto no teto."""
        if self._proc is None or not self._proc.is_alive():
            self._iniciar()
        self._n += 1
        self._entrada.put((self._n, tipo, item, texto))
        try:
            while True:
                n, r, vazou = self._saida.get(timeout=teto)
                if n == self._n:
                    break
        except queue.Empty:
            self._encerrar(matar=True)
            self.mortes += 1
            return None, True
        if vazou:
            self._encerrar(matar=False)       # o trabalhador já saiu sozinho
            self.reinicios += 1
        return r, vazou

    def _pedir(self, tipo: str, item: Item, texto: str | None, teto: float):
        r, contaminado = self._uma_vez(tipo, item, texto, teto)
        if contaminado:
            # refeito uma vez num processo novo: o estouro pode ter sido da carga
            r, _ = self._uma_vez(tipo, item, texto, teto)
        return r

    def corrigir(self, item: Item, texto: str) -> Veredito:
        r = self._pedir("corrigir", item, texto, self.teto_s)
        return Veredito.INCONCLUSIVE if r is None else Veredito(r)

    def autoteste(self, item: Item) -> dict:
        r = self._pedir("autoteste", item, None, self.teto_autoteste_s)
        if r is None:
            return {"id": item.id, "ok": False, "proprio": "teto", "perturbacoes": 0,
                    "perturbacoes_que_passaram": 0}
        return r

    def fechar(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._entrada.put(None)
        self._encerrar(matar=False)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.fechar()
