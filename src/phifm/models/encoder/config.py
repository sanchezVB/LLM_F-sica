"""Configurações do ΦEnc, do DOC-07 §2.2, com a contagem de parâmetros conferida.

## Por que este módulo não implementa as camadas

O DOC-07 §2.1 pede ModernBERT (Warner et al., 2024): RoPE, contexto 8.192, atenção
alternada local(128)/global, GeGLU, pre-norm sem bias, unpadding. O `transformers`
4.48 implementa tudo isso em `ModernBertForMaskedLM`.

Reimplementar 22 camadas disso aqui seria um **segundo** ModernBERT para manter em
sincronia, e a divergência entre os dois seria invisível — os dois rodariam, com
números diferentes, e nada apontaria qual está certo. É o mesmo argumento que impede
o notebook do Kaggle de reimplementar o laço de treino.

Então este módulo escreve a **configuração**, não as camadas. O que ele acrescenta é
a conferência: `parametros()` calcula o total analiticamente e um teste compara com o
modelo de verdade, para que uma mudança de estrutura no `transformers` apareça como
falha aqui em vez de como um modelo silenciosamente de outro tamanho.

## ⚠️ Os 16% do DOC-07 §2.2 estão errados

O documento afirma "Embedding como % do modelo: **16%**" para o ΦEnc-150M com
V=40.960. Medido, construindo o modelo:

| V | total | embedding | share |
|---|---|---|---|
| 32.768 | 136,1 M | 25,2 M | **18,5%** |
| **40.960** | **142,4 M** | **31,5 M** | **22,1%** |
| 65.536 | 161,4 M | 50,3 M | **31,2%** |

De onde vem o 16%: a tabela do §7.2 calcula `V × d` e divide pelos **150 M
nominais**, não pelo total real — e usa V=32.000 em vez de 32.768. Dividir por um
número redondo em vez do que o modelo tem dá 16,4%.

**A consequência prática é o contrário de acadêmica.** Eu disse ao usuário que
V=65.536 levaria a embedding "de 16% para ~24%". O real é **22,1% → 31,2%**: a troca
é mais dura que o documento diz, e o argumento da §7.3 para ficar em 40.960 fica
**mais** forte, não menos.

## O que a conferência pegou, e o que ela me impediu de errar

O `conferir_declarado` reprovou o meu primeiro ΦEnc-400M: com `ffn=1536` ele dava
292,7 M contra os 400 M declarados. A tabela do §2.2 não fixa o `ffn` do 400 M, e o
valor certo é o do ModernBERT-large, 2624. **Era o palpite que estava errado, não o
documento** — que é o desfecho que uma invariante existe para produzir.

E o `flops_de_treino` quase saiu errado no sentido oposto. Eu havia excluído a
embedding dos FLOPs com o argumento de que "consulta de tabela não é matmul" — certo
para a entrada, errado para o resto: com pesos amarrados, a mesma matriz volta como
**projeção de saída**, e aí é `d × V` por token. Ver a docstring da função.

## ⚠️ E o proxy de 50 M do DOC-05 §11.2 não funciona como está

O §11.2 manda treinar "um encoder de ~50 M parâmetros" por variante de tokenizer.
Com V=40.960, um modelo de 50 M é **53,6% embedding** — e com V=65.536 a embedding
sozinha passa de 50 M, antes de qualquer camada.

Nesse regime a comparação entre vocabulários mede sobretudo *quantos parâmetros de
embedding cada variante tem*, não *qual tokenizer produz o melhor modelo*. O
`PROXY_BAKEOFF` daqui fixa o corpo (camadas e `d_model`) e deixa o total variar com
V, o que torna a comparação sobre a qualidade do tokenizer a custo de corpo igual.
É uma escolha, e a alternativa — total igual, corpo variável — mede outra coisa.
Registrado para o §11.2 ser revisado antes de rodar, não depois.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Da tabela do DOC-07 §2.2 e do DOC-05 §7.3.
VOCAB_PHIENC = 40_960
CONTEXTO_PHIENC = 8_192

# Ids que o tokenizer treinado (`data/processed/tokenizer/variante_A.json`) usa.
# ⚠️ Fixos aqui porque o modelo os grava no `config.json` e um desencontro entre o
# `pad_token_id` do modelo e o do tokenizer faz o unpadding do ModernBERT descartar
# as posições erradas — sem erro, só com perda que não desce.
ESPECIAIS = {"pad_token_id": 0, "unk": 1, "cls_token_id": 2, "sep_token_id": 3,
             "mask": 4}


@dataclass(frozen=True)
class ConfigEnc:
    """Os campos que o DOC-07 §2.2 fixa, e nada além."""

    nome: str
    camadas: int
    d_model: int
    cabecas: int
    ffn: int
    vocab: int = VOCAB_PHIENC
    contexto: int = CONTEXTO_PHIENC
    # DOC-07 §2.1: janela local de 128, global a cada 3 camadas (ModernBERT).
    janela_local: int = 128
    global_a_cada: int = 3
    # O tamanho que o DOC-07 declara, para a conferência abaixo. `None` quando o
    # documento não declara nenhum (o proxy do bake-off).
    declarado_M: float | None = None

    def __post_init__(self) -> None:
        if self.d_model % self.cabecas:
            raise ValueError(
                f"{self.nome}: d_model={self.d_model} não divide por "
                f"cabecas={self.cabecas} — a atenção multi-cabeça exige divisão exata")
        if self.contexto % self.janela_local:
            raise ValueError(
                f"{self.nome}: contexto={self.contexto} não é múltiplo da janela "
                f"local {self.janela_local}; a atenção alternada do ModernBERT "
                "assume que é")

    def parametros(self) -> dict[str, int]:
        """Contagem analítica, por bloco. Torch-free de propósito.

        A estrutura é a do `ModernBertForMaskedLM` com `tie_word_embeddings=True` e
        `decoder_bias=True`, que são os padrões:

        - embeddings: `V × d` + uma norm
        - por camada: `Wqkv` (d × 3d), `Wo` (d × d), `Wi` (d × 2·ffn),
          `Wo` do MLP (ffn × d), duas norms — **menos** a `attn_norm` da camada 0,
          que é `Identity`
        - final norm, cabeça densa (d × d) + norm, e o bias do decoder (V)

        Sem bias em atenção e MLP: DOC-07 §2.1 pede pre-norm sem termos de bias.
        """
        d, ffn, v = self.d_model, self.ffn, self.vocab
        emb = v * d + d
        por_camada = (d * 3 * d) + (d * d) + (d * 2 * ffn) + (ffn * d) + 2 * d
        camadas = por_camada * self.camadas - d  # attn_norm da camada 0 é Identity
        cabeca = d + (d * d) + d + v
        return {"embedding": emb, "camadas": camadas, "cabeca": cabeca,
                "total": emb + camadas + cabeca}

    def fracao_de_embedding(self) -> float:
        p = self.parametros()
        return p["embedding"] / p["total"]

    def conferir_declarado(self, tolerancia: float = 0.08) -> None:
        """Levanta se o total real fugir do que o DOC-07 §2.2 declara.

        ⚠️ Existe porque "ΦEnc-150M" é um nome, e um nome não impõe nada. Uma
        mudança de `ffn` ou de vocabulário move o total sem que nada reclame, e o
        orçamento de computação do §2.4 (`C = 6 × N × D`) é linear em N — um modelo
        20% maior que o declarado custa 20% mais e ninguém percebe até a fatura.
        """
        if self.declarado_M is None:
            return
        real = self.parametros()["total"] / 1e6
        desvio = abs(real - self.declarado_M) / self.declarado_M
        if desvio > tolerancia:
            raise ValueError(
                f"{self.nome}: o DOC-07 §2.2 declara {self.declarado_M:.0f} M e a "
                f"configuração dá {real:.1f} M ({100 * desvio:.0f}% de desvio). Ou a "
                "configuração está errada, ou o documento precisa ser corrigido — as "
                "duas coisas mudam o orçamento do §2.4, que é linear em N.")

    def para_transformers(self) -> dict:
        """Argumentos de `ModernBertConfig`. Sem importar transformers aqui.

        Manter isto como dicionário deixa o módulo torch-free e testável na venv
        rápida; quem constrói o modelo é `construir()`, em `modelo.py`.
        """
        return {
            "vocab_size": self.vocab,
            "hidden_size": self.d_model,
            "num_hidden_layers": self.camadas,
            "num_attention_heads": self.cabecas,
            "intermediate_size": self.ffn,
            "max_position_embeddings": self.contexto,
            "local_attention": self.janela_local,
            "global_attn_every_n_layers": self.global_a_cada,
            "pad_token_id": ESPECIAIS["pad_token_id"],
            "cls_token_id": ESPECIAIS["cls_token_id"],
            "sep_token_id": ESPECIAIS["sep_token_id"],
            # Sem bias em nenhum lugar: DOC-07 §2.1.
            "norm_bias": False,
            "mlp_bias": False,
            "attention_bias": False,
        }

    def como_dict(self) -> dict:
        d = asdict(self)
        d["parametros"] = self.parametros()
        d["fracao_de_embedding"] = round(self.fracao_de_embedding(), 4)
        return d


# ── As configurações do DOC-07 §2.2 ─────────────────────────────────────────
#
# `ffn = 1152` com GeGLU: o `Wi` projeta para `2 × 1152` e a metade que sobra é o
# gate. É o valor do ModernBERT-base, e é o que faz o total fechar em 142 M.
PHIENC_150M = ConfigEnc(nome="ΦEnc-150M", camadas=22, d_model=768, cabecas=12,
                        ffn=1152, declarado_M=150)

# ⚠️ `ffn = 2624`, o do ModernBERT-large, e não um valor redondo. A tabela do
# DOC-07 §2.2 não declara o `ffn` do 400 M, e o meu primeiro palpite (1536) dava
# 292,7 M — 27% abaixo dos 400 M declarados. Foi o `conferir_declarado` que pegou:
# era o palpite que estava errado, não o documento.
PHIENC_400M = ConfigEnc(nome="ΦEnc-400M", camadas=28, d_model=1024, cabecas=16,
                        ffn=2624, declarado_M=400)

# ⚠️ Proxy do bake-off de tokenizer (DOC-05 §11.2), com o CORPO fixo. Ver a ressalva
# na docstring do módulo: a `fracao_de_embedding` deste proxy é alta de propósito, e
# é ela que revela que o §11.2 a 50 M compararia embeddings e não tokenizers.
PROXY_BAKEOFF = ConfigEnc(nome="proxy-bakeoff", camadas=12, d_model=512, cabecas=8,
                          ffn=768)

CONFIGS: dict[str, ConfigEnc] = {
    c.nome: c for c in (PHIENC_150M, PHIENC_400M, PROXY_BAKEOFF)}


def obter(nome: str) -> ConfigEnc:
    try:
        cfg = CONFIGS[nome]
    except KeyError:
        raise SystemExit(
            f"configuração {nome!r} não existe. Conhecidas: "
            f"{sorted(CONFIGS)}") from None
    cfg.conferir_declarado()
    return cfg


def flops_de_treino(cfg: ConfigEnc, tokens: int) -> float:
    """`C = 6 × N × D` do DOC-07 §2.4, com N = corpo **+ a projeção de saída**.

    ⚠️ A parte contraintuitiva, e eu errei nela primeiro. "Consulta de tabela não é
    matmul, então a embedding não entra nos FLOPs" está certo **para a entrada** e
    errado para o resto: numa MLM com `tie_word_embeddings`, a mesma matriz é usada
    de novo como **projeção de saída**, e aí é um matmul de `d × V` por token.

    Na escala do ΦEnc-150M isso não é detalhe: `768 × 40.960 = 31,5 M` de MACs por
    token, contra 110 M do corpo — 29% a mais. Excluir a projeção subestimaria o
    custo em 26%, e eu quase registrei isso como "o DOC-07 §2.4 superestima". O
    §2.4 usa `N = 1,5e8` e dá 2,7e19 para 30 B tokens; a conta correta dá 2,56e19,
    dentro de 5%. **O documento estava certo.**

    O que NÃO entra: a consulta de entrada, os vieses e as normas — desprezíveis.
    """
    p = cfg.parametros()
    # corpo + cabeça densa + a projeção de saída amarrada (V × d), sem o bias.
    n_efetivo = p["camadas"] + (p["cabeca"] - cfg.vocab) + cfg.vocab * cfg.d_model
    return 6.0 * n_efetivo * tokens
