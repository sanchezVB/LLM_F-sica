#!/usr/bin/env python3
"""Sonda de estrutura tensorial — DOC-05 §11.2, terceira das três avaliações.

Roda na venv de TREINO (Python 3.12), em CPU.

    .venv-treino/Scripts/python.exe scripts/avaliar_tensorial.py \\
        --controle models/phienc-controle --tratado models/phienc-tratado

## O que ela pergunta

O DOC-00 nomeia colapso de notação como modo de falha F5, e o repositório já
pagou por ele no próprio parser: `\\rho_{xy}` e `\\rho_{yx}` viravam o mesmo
símbolo. A sonda pergunta o análogo para a representação aprendida — **o encoder
distingue expressões que diferem só na estrutura de índices?**

Nada é treinado. Para cada expressão base há duas edições: uma que muda a Física
(trocar ordem de índices, subir/baixar um índice, quebrar uma contração) e uma
que **não muda nada** (renomear um índice mudo). A pergunta é se a primeira move
a representação mais que a segunda.

## Baselines

`--baseline nome=caminho` acrescenta modelos de fora (PhysBERT, SciBERT) pela
mesma sonda. Vale a pena: se nenhum deles distinguir índices, o resultado do ΦEnc
passa a ter com o que ser comparado — e se todos distinguirem, a sonda é fácil
demais e precisa de pares mais duros.

⚠️ Baselines vêm do HuggingFace e exigem rede; os braços do ΦEnc são locais.
"""
import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.eval.phienc import carregar  # noqa: E402
from phifm.eval.tensorial import PARES, _leitura, comparar_bracos, sondar  # noqa: E402

log = logging.getLogger("avaliar_tensorial")

TOKENIZER = Path("data/processed/tokenizer/variante_A.json")


def _do_hub(caminho: str, dispositivo: str):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(caminho)
    mod = AutoModel.from_pretrained(caminho, attn_implementation="eager")
    return mod.to(dispositivo).eval(), tok


def main(argv: list[str] | None = None) -> int:
    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--controle", type=Path, help="checkpoint com p_equacao=0")
    p.add_argument("--tratado", type=Path, help="checkpoint com p_equacao>0")
    p.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    p.add_argument("--baseline", action="append", default=[], metavar="NOME=CAMINHO",
                   help="modelo do HuggingFace pela mesma sonda; repetível")
    p.add_argument("--sem-moldura", action="store_true",
                   help="mede a expressão nua, sem a frase em volta. Serve para "
                        "checar que a conclusão não depende da moldura — o encoder "
                        "viu equações DENTRO de texto, então a com moldura é o padrão")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--dispositivo", default="cpu")
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/avaliacao/phienc_tensorial.json"))
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    if not a.baseline and not (a.controle and a.tratado):
        log.error("passe --controle e --tratado, e/ou pelo menos um --baseline")
        return 2

    alvos = {}
    if a.controle and a.tratado:
        for nome, caminho in (("controle", a.controle), ("tratado", a.tratado)):
            mod, tok, _ = carregar(caminho, a.tokenizer, a.dispositivo)
            alvos[nome] = (mod, tok)
    for spec in a.baseline:
        if "=" not in spec:
            log.error("--baseline espera NOME=CAMINHO, recebi %r", spec)
            return 2
        nome, caminho = spec.split("=", 1)
        alvos[nome] = _do_hub(caminho, a.dispositivo)

    sondagens = {
        nome: sondar(mod, tok, nome, com_moldura=not a.sem_moldura,
                     max_tokens=a.max_tokens, dispositivo=a.dispositivo)
        for nome, (mod, tok) in alvos.items()}

    d = comparar_bracos(sondagens)
    d["moldura"] = not a.sem_moldura

    print(f"\nSONDA DE ESTRUTURA TENSORIAL — {len(PARES)} pares mínimos, "
          f"{'com' if not a.sem_moldura else 'sem'} moldura de prosa\n")
    # Razão = distância por caractere editado. A crua fica na quebra por par.
    print(f"  {'modelo':<16} {'r(signif.)':>11} {'r(controle)':>12} {'índice':>8} "
          f"{'a favor':>9}")
    for nome, s in sondagens.items():
        print(f"  {nome:<16} {s.dist_significativo:>11.5f} {s.dist_controle:>12.5f} "
              f"{s.indice:>+8.3f} {s.bases_a_favor:>4}/{len(PARES):<4}")
        print(f"  {'':<16} → {_leitura(s)}")
        cats = " · ".join(f"{c}: {v['a_favor']}/{v['pares']}"
                          for c, v in sorted(s.por_categoria.items()))
        if cats:
            print(f"  {'':<16}   por categoria — {cats}")
            print(f"  {'':<16}   (`ordem` é a mais difícil: a média mascarada é "
                  "quase invariante a permutação)")

    print("\n  por par (distância CRUA, e a edição que a normaliza):")
    for nome, s in sondagens.items():
        for linha in s.por_par:
            marca = "✓" if linha["a_favor"] else "·"
            print(f"    {marca} {nome:<12} {linha['nome']:<20} "
                  f"{linha['dist_significativo']:.5f} vs {linha['dist_controle']:.5f} "
                  f"(edição {linha['edicao_significativo']} vs "
                  f"{linha['edicao_controle']})")

    colaps = {n: s.colapsados for n, s in sondagens.items() if s.colapsados}
    if colaps:
        print("\n  ⚠️ PARES COLAPSADOS PELO TOKENIZER (excluídos da contagem):")
        for nome, lista in colaps.items():
            print(f"    {nome}: {lista}")
        print("    As duas cadeias viram os mesmos ids. O modelo não poderia "
              "distingui-las nem em princípio — o defeito é da tokenização.")

    if "ablacao" in d:
        ab = d["ablacao"]
        if ab.get("erro"):
            print(f"\n  ABLAÇÃO: {ab['erro']}")
        else:
            print(f"\n  ABLAÇÃO: tratado melhor em {ab['tratado_melhor']}, controle "
                  f"em {ab['controle_melhor']}, Δíndice {ab['delta_indice']:+.3f}")
            print(f"    {ab['leitura']}")

    print(f"\n  {d['ressalva']}\n")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str),
                     encoding="utf-8")
    log.info("gravado em %s", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
