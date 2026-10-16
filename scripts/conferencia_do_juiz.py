#!/usr/bin/env python3
"""O juiz independente da §9.6 do DOC-13: outro modelo, no Colab, corrige as mesmas
respostas — e diz se o desfecho da regra depende de quem corrige.

    P = .venv-treino\\Scripts\\python.exe scripts\\conferencia_do_juiz.py
    P --empacotar            # o que sobe para o Drive (só depois de o teste terminar)
    P --apurar <veredictos>  # a validade, os eixos pelo 2º juiz, a concordância e a leitura

O pacote (`data/processed/assistente/juiz2/pacote_juiz2.json`) tem perguntas, gabaritos e
respostas: fica fora do git (DOC-11 §8.1) e sobe para o Drive do dono por decisão dele. Vão
para o git o hash do pacote e o agregado, em `data/processed/avaliacao/`.

A regra está no DOC-13 §9.6 ("O juiz independente"), escrita antes de o teste rodar.
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

import medir_passagens as mp  # noqa: E402

DIR2 = mp.DIR / "juiz2"
PACOTE = DIR2 / "pacote_juiz2.json"
MANIFESTO = mp.AVALIACAO / "juiz_independente_pacote.json"
AGREGADO = mp.AVALIACAO / "assistente_juiz_independente.json"

# As respostas que o Claude julgou às cegas, com o hash dos veredictos no git ANTES de o
# juiz da regra rodar nelas: (amostra, veredictos, selo, campo do id, campos do selo).
SELADOS = {
    "9.1_rodada_2": ("i3_para_julgar_r2.json", "veredictos_i3_claude_r2.json",
                     "assistente_i3_selo_r2.json", "indice",
                     ("amostra_sha256", "veredictos_do_claude_sha256")),
    "9.5_corpo_1": ("corpo_i3c_para_julgar.json", "veredictos_i3c_claude.json",
                    "assistente_corpo_i3c_selo.json", "id",
                    ("sha256_amostra", "sha256_veredictos")),
    "9.6_corpo_2": ("passagens_i3_para_julgar.json", "veredictos_passagens_i3_claude.json",
                    "assistente_passagens_i3_selo.json", "id",
                    ("sha256_amostra", "sha256_veredictos")),
}


def selados() -> dict[str, list[tuple[str, str, str, str]]]:
    """`{conjunto: [(pergunta, gabarito, resposta, veredicto do Claude)]}`, conferido
    contra os selos versionados."""
    import hashlib

    def hashes(arq: Path) -> set[str]:
        # O selo da §9.1 é dos bytes do arquivo; os das §9.5 e §9.6, do texto lido.
        return {hashlib.sha256(arq.read_bytes()).hexdigest(),
                mp._sha(arq.read_text(encoding="utf-8"))}

    saida = {}
    for nome, (amostra, veredictos, selo, campo, (h_a, h_v)) in SELADOS.items():
        bruto_a = (mp.DIR / amostra).read_text(encoding="utf-8")
        bruto_v = (mp.DIR / veredictos).read_text(encoding="utf-8")
        s = json.loads((mp.AVALIACAO / selo).read_text(encoding="utf-8"))
        if s[h_a] not in hashes(mp.DIR / amostra) or s[h_v] not in hashes(mp.DIR / veredictos):
            raise SystemExit(f"{nome}: a amostra ou os veredictos não são os selados.")
        v = json.loads(bruto_v)
        if isinstance(v, list):
            v = {str(d[campo]): d["veredicto"] for d in v}
        saida[nome] = [(d["pergunta"], d["gabarito"], d["resposta"], v[str(d[campo])])
                       for d in json.loads(bruto_a)]
    return saida


def _conjuntos():
    """O teste e o corpo 2, com as respostas de cada braço: `{grupo: [(item, resposta)]}`."""
    from phifm.eval import assistente_bracos as b

    itens_t, a2, cache_t = mp._teste_primario()
    r4t = {(r.estrato, r.arxiv_id): r for r in b.ler_respostas(cache_t)}
    if len(r4t) < len(itens_t):
        raise SystemExit(f"o teste ainda não terminou: {len(r4t)} de {len(itens_t)} itens "
                         "(medir_passagens.py --primario).")
    itens_c, _, sha = mp.carregar()
    corpo = b.ler_respostas(mp.DIR / f"bracos_corpo2_{mp.assinatura(sha)[:12]}.jsonl")
    de = {(r.braco, r.arxiv_id): r for r in [*corpo, *b.ler_respostas(mp._cache_v2(sha))]}
    grupos = {"teste_A2": [(i, a2[(i["estrato"], i["arxiv_id"])]) for i in itens_t],
              "teste_A4": [(i, r4t[(i["estrato"], i["arxiv_id"])]) for i in itens_t]}
    for braco in (b.BRACO_REORDENADO, b.BRACO_PASSAGENS, b.BRACO_PASSAGENS_V2):
        grupos[f"corpo2_{braco}"] = [(i, de[(braco, i["arxiv_id"])]) for i in itens_c]
    return grupos


def empacotar() -> None:
    from phifm.eval import juiz_independente as ji

    grupos = {g: [(i["pergunta"], i["gabarito"], r.texto) for i, r in pares]
              for g, pares in _conjuntos().items()}
    for nome, textos in selados().items():
        grupos[f"selados_{nome}"] = [(p, g, r) for p, g, r, _ in textos]
    pacote, contagem = ji.empacotar(grupos)
    DIR2.mkdir(parents=True, exist_ok=True)
    bruto = json.dumps(pacote, ensure_ascii=False, indent=0)
    PACOTE.write_bytes(bruto.encode("utf-8"))
    manifesto = {
        "o_que_e": "o pacote do juiz independente (DOC-13 §9.6): pergunta, gabarito e "
                   "resposta sem as marcas [n], um texto distinto por linha, sem braço e "
                   "sem artigo. O conteúdo fica fora do git; aqui vai o hash.",
        "arquivo": PACOTE.name, "sha256": mp._sha(bruto), "bytes": len(bruto.encode("utf-8")),
        "textos": len(pacote), "grupos": contagem,
        "modelo": {"repo": ji.MODELO_REPO, "arquivo": ji.MODELO_ARQUIVO,
                   "revisao": ji.MODELO_REVISAO, "bytes": ji.MODELO_BYTES,
                   "sha256": ji.MODELO_SHA256, "llama_cpp": ji.LLAMA_CPP},
        "empacotado_em": time.strftime("%Y-%m-%dT%H:%M:%S")}
    MANIFESTO.write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifesto, ensure_ascii=False, indent=2))
    print(f"→ {PACOTE}  (é o arquivo que sobe para o Drive)")


def apurar(arquivo: Path) -> None:
    from phifm.eval import assistente_bracos as b
    from phifm.eval import juiz_independente as ji

    manifesto = json.loads(MANIFESTO.read_text(encoding="utf-8"))
    if mp._sha(PACOTE.read_text(encoding="utf-8")) != manifesto["sha256"]:
        raise SystemExit("o pacote local não é o do manifesto versionado.")
    pacote = json.loads(PACOTE.read_text(encoding="utf-8"))
    j2 = ji.ler(arquivo)
    falta = [e["chave"] for e in pacote if e["chave"] not in j2]
    if falta:
        raise SystemExit(f"faltam {len(falta)} de {len(pacote)} textos no arquivo do juiz "
                         "independente: a rodada do Colab não terminou.")
    j1 = b.ler_julgamentos(mp._cache_do_juiz())

    def v2(item, texto):
        return j2[ji.chave(item["pergunta"], item["gabarito"], texto)]["veredicto"]

    def v1(item, texto):
        return j1[b.chave_do_julgamento(item["pergunta"], item["gabarito"], texto)]["veredicto"]

    # 1. Ele vale? Contra os veredictos selados do Claude.
    sel = selados()
    do_claude = [v for textos in sel.values() for *_, v in textos]
    dele = [j2[ji.chave(p, g, r)]["veredicto"] for textos in sel.values() for p, g, r, _ in textos]
    validade = b.apurar_i3(do_claude, dele)
    n0 = 0
    por_conjunto = {}
    for nome, textos in sel.items():
        r = b.apurar_i3(do_claude[n0:n0 + len(textos)], dele[n0:n0 + len(textos)])
        por_conjunto[nome] = {k: r[k] for k in ("n", "kappa_certo", "concordancia_certo")}
        n0 += len(textos)

    # 2. Os eixos que dependem de juiz, refeitos com ele.
    grupos = _conjuntos()
    teste = {braco: {(i["estrato"], i["arxiv_id"]): v2(i, r.texto) for i, r in grupos[g]}
             for braco, g in ((b.BRACO_REORDENADO, "teste_A2"),
                              (b.BRACO_PASSAGENS_V2, "teste_A4"))}
    nao_piora = b.nao_piora(teste, [r for _, r in grupos["teste_A4"]], mp.ESTRATO_DO_TESTE,
                            b.BRACO_PASSAGENS_V2)
    corpo = {braco: {(i["estrato"], i["arxiv_id"]): v2(i, r.texto)
                     for i, r in grupos[f"corpo2_{braco}"]}
             for braco in (b.BRACO_REORDENADO, b.BRACO_PASSAGENS, b.BRACO_PASSAGENS_V2)}
    ganho = b.comparar_instrucao(
        corpo, [r for _, r in grupos[f"corpo2_{b.BRACO_PASSAGENS_V2}"]])

    # 3. O juiz da regra, nos mesmos eixos — do agregado versionado — e a concordância.
    regra = json.loads(mp.AGREGADO.read_text(encoding="utf-8"))
    if not isinstance(regra["teste_primario"], dict):
        raise SystemExit("o agregado da regra ainda não tem o teste (medir_passagens.py "
                         "--resumir).")
    da_regra = {"ganho_no_corpo": regra["emenda_instrucao_v2"]["resultado"]["ganho_no_corpo"]
                ["passa"], "nao_piora_o_primario": regra["teste_primario"]["passa"]}
    independente = {"ganho_no_corpo": ganho["ganho_no_corpo"]["passa"],
                    "nao_piora_o_primario": nao_piora["passa"]}
    pares = [(v1(i, r.texto), v2(i, r.texto)) for g in grupos.values() for i, r in g]
    so_teste = [(v1(i, r.texto), v2(i, r.texto))
                for g in ("teste_A2", "teste_A4") for i, r in grupos[g]]

    saida = {
        "regra": "DOC-13 §9.6, 'O juiz independente' — escrita antes de o teste rodar",
        "modelo": manifesto["modelo"], "pacote_sha256": manifesto["sha256"],
        "arquivo_do_juiz": {"nome": arquivo.name,
                            "sha256": mp._sha(arquivo.read_text(encoding="utf-8")),
                            "textos": len(j2),
                            "ilegiveis": sum(1 for d in j2.values()
                                             if d["veredicto"] == "ilegivel")},
        "limites": ["continua sendo um modelo corrigindo outro; a régua da validade são os "
                    "veredictos do Claude",
                    "as guardas do corpo (Bp e C) não foram rejulgadas",
                    "um modelo, uma rodada, temperatura 0"],
        "validade": {**validade, "por_conjunto": por_conjunto},
        "eixos_pelo_juiz_independente": {"nao_piora_o_primario": nao_piora,
                                         "ganho_no_corpo": ganho},
        "entre_os_juizes": {
            "teste_e_corpo_2": ji.entre_juizes([x for x, _ in pares], [y for _, y in pares]),
            "so_o_teste": ji.entre_juizes([x for x, _ in so_teste], [y for _, y in so_teste])},
        "conferencia": ji.conferir(bool(validade["passa"]), da_regra, independente),
        "decisao_da_regra": regra["decisao"]}
    AGREGADO.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in saida.items() if k not in ("limites", "modelo")},
                     ensure_ascii=False, indent=1))
    print(f"→ {AGREGADO.relative_to(RAIZ)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--empacotar", action="store_true")
    p.add_argument("--apurar", type=Path, metavar="VEREDICTOS")
    a = p.parse_args()
    if a.empacotar:
        empacotar()
    elif a.apurar:
        apurar(a.apurar)
    else:
        p.error("--empacotar ou --apurar <veredictos>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
