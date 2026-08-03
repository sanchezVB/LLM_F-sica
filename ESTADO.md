# Estado do projeto — 2026-08-03

Ponto de retomada para migração de máquina. Instalação em [SETUP.md](SETUP.md).

## Onde estamos

**Corpus de projeto:** completo. 19 documentos + 1 ADR, cobrindo os 20 pipelines.

**Execução:** Sprint S1 em andamento, S2 parcial, barramento de verificação
iniciado.

| Sprint | Estado | Observação |
|---|---|---|
| **S1** · espinha de metadados | 🟡 em curso | arXiv ~50%; OpenAlex bloqueado por cota |
| **S2** · classificador de Física | 🟡 parcial | Subárea treinado; binária aguarda negativos |
| **S3** · fatias do HuggingFace | ⚪ não iniciado | ~280 GB, processar em lote |
| Barramento de verificação | 🟡 3 de 6 | `bus`, `symbolic`, `numeric` prontos |

## Coletas — como retomar

Ambas são **idempotentes e retomáveis**. Basta rodar de novo:

```bash
./scripts/run_harvest.sh arxiv
./scripts/run_harvest.sh openalex
```

O `_manifest.json` em cada pasta guarda o cursor durável. A retomada refaz
apenas o lote pendente (entrega "ao menos uma vez"), e a duplicação é
removida pela dedup exata.

| Fonte | Coletado | Situação |
|---|---|---|
| arXiv | ~600 mil registros, 0 falhas | Roda até o fim sozinho, ~2–3 h |
| OpenAlex | 150 mil obras, **10,1 M arestas de citação** | ⚠️ ver abaixo |

### ⚠️ Decisão pendente: OpenAlex

A API passou a ser cotada — 1.000 requisições/dia grátis, US$ 0,0001 cada.
Precisamos de 18.336. Três rotas:

| Rota | Custo | Tempo |
|---|---|---|
| Esperar a cota | US$ 0 | 18 dias, automático |
| Pagar a API | US$ 1,83 | ~5 h |
| **Snapshot S3** | **US$ 0** | **~2 h** a 444 Mbps |

O snapshot continua gratuito e sem cota. **É a rota recomendada** e ainda não
foi implementada. Detalhes em [DOC-02 §3.1](docs/01-data/DOC-02-aquisicao-corpus.md).

Não é bloqueio imediato: os 10,1 M de arestas já coletados bastam para
começar a treinar o ΦEmb.

## O que fazer a seguir, em ordem

1. **Terminar o S1** — deixar o arXiv completar; decidir a rota do OpenAlex
2. **Implementar o snapshot S3** do OpenAlex, se essa for a escolha
3. **Coletar negativos** (arXiv `cs`, `q-bio`, `econ`) para a binária do S2 —
   só depois que a coleta de Física terminar, por A5
4. **Completar o barramento**: falta `dimensional`, `limits`, `conservation`
5. **Sprint S3** — fatias do HuggingFace, em lote

## Achados desta sessão que alteraram documentos

| Achado | Impacto | Registrado em |
|---|---|---|
| Endpoint OAI do arXiv mudou | `export.arxiv.org` dá 301 | DOC-02 §3.1 |
| Set `physics` filtra no servidor | 1,2 M em vez de 2,7 M | DOC-02 §3.1 |
| Espinha ocupa ~700 MB, não 150 GB | Cabe em disco comum | DOC-02 §3.1 |
| `primary_location` exclui publicados | Perderia 1,44 M revisados por pares | DOC-02 |
| Chave de junção não está em `ids.arxiv` | 1,5% vs 98,5% de cobertura | `openalex.py` |
| IDs antigos truncados por regex | 41,5% do acervo | teste de regressão |
| **OpenAlex passou a cobrar** | 18 dias grátis ou US$ 1,83 | DOC-02 §3.1 |
| Licenças: 45,9% redistribuível em 2024 | Dimensiona o `PhysCorpus-Open` | ADR-0001 §4 |
| Precisão float64 no verificador | Reprovava toda resposta correta | `numeric.py` |

## Onde está cada coisa

| | Local | Drive | GitHub |
|---|---|---|---|
| Código, docs, testes | ✅ | — | ✅ |
| Manifestos | ✅ | ✅ | ✅ |
| Coletas brutas (285 MB) | ✅ | ✅ | ❌ por decisão |
| Espinha consolidada (150 MB) | ✅ | ✅ | ❌ |
| Classificador (59 MB) | ✅ | ✅ | ❌ |
| `.env` | ✅ | ❌ | ❌ recriar |

## Pendência conhecida

O histórico do git carrega um `model.pkl` de 56 MB commitado por engano antes
de eu corrigir o `.gitignore`. O `.git` está em ~32 MB. Limpar exige reescrever
o histórico e dar force-push — operação destrutiva, não executada sem
autorização explícita:

```bash
.venv/bin/pip install git-filter-repo
.venv/bin/git-filter-repo --path models --invert-paths --force
git remote add origin https://github.com/sanchezVB/LLM_F-sica.git
git push --force origin main
```
