# `src/phifm/` — mapa de módulos

**Ainda sem código.** A estrutura existe porque as fronteiras entre módulos são uma decisão de arquitetura, não um detalhe de implementação — ver [DOC-01 §4](../docs/00-foundations/DOC-01-system-architecture.md#4-organização-do-repositório).

## O que vai em cada lugar

| Módulo | Responsabilidade | Documento que o especifica |
|---|---|---|
| `core/` | Primitivas transversais. **Não importa de mais nada.** Schema (`PhysicsDocumentRecord`), linhagem, licenças, álgebra de unidades, canonicalizador LaTeX, I/O | DOC-01 §6 |
| `corpus/` | Aquisição, parsing, normalização, filtragem, deduplicação, descontaminação, mistura | DOC-02 a DOC-06 |
| `tokenizer/` | Treino, avaliação (fertilidade, compressão) e extensão de tokenizer | DOC-05 |
| `models/` | **Apenas definições de arquitetura.** `nn.Module` puros, sem qualquer consciência de treino distribuído | DOC-07 |
| `training/` | Laços de treino, paralelismo, callbacks. O sharding é aplicado aqui, nunca em `models/` | DOC-08, DOC-09 |
| `verify/` | **O barramento de verificação.** Equivalência simbólica, checagem dimensional, numérica, casos-limite, leis de conservação, sandbox | DOC-10 |
| `eval/` | PhysBench, harness, correção, estatística, avaliação humana, contaminação | DOC-11, DOC-12 |
| `retrieval/` | Chunking, indexação, busca híbrida, reranking, atribuição de citação | DOC-13 |
| `agents/` | Planejador, roteador de ferramentas, memória, registro de traços | DOC-14 |
| `tools/` | Integrações: SymPy, NumPy/SciPy, JAX, Wolfram, FEniCS, NASA CEA, GMAT, Orekit | DOC-14 |
| `serving/` | Adaptadores vLLM/SGLang, batching, guardrails | DOC-15 |
| `monitoring/` | Drift, telemetria de qualidade, custo, ganchos de incidente | DOC-16 |

## Fronteiras impostas (não são sugestão)

Violação de camadas é como repositórios de ML apodrecem. O DAG abaixo é verificado por `import-linter` no CI e **quebra o build** se violado:

```
core    ←  corpus, tokenizer, models, verify, eval, retrieval, tools
verify  ←  corpus.filter, training.rl, eval.grading, serving
models  ←  training, serving

Nenhum módulo importa de `training`, exceto `pipelines/` e `scripts/`.
Nenhum módulo importa de `notebooks/`.
```

### Por que `verify/` fica embaixo de tudo

`verify/` é importado por `corpus.filter` (filtragem de dados), `training.rl` (recompensa de RLVR), `eval.grading` (correção de benchmark) e `serving` (auto-checagem em inferência).

Isso é a imposição mecânica do princípio **P3** do DOC-01: *"tudo é verificável, ou é hipótese."* A maioria dos projetos de LLM tem três implementações separadas e silenciosamente divergentes de "esta resposta está certa" — uma no filtro de dados, uma na recompensa, uma no corretor. Aqui existe **uma só**, e é estruturalmente impossível que a recompensa de treino divirja do corretor de avaliação, porque são o mesmo caminho de código.

Consequência direta: **um bug em `verify/` é um bug global.** Por isso esse módulo carrega o requisito de cobertura de testes mais estrito do repositório, e por isso existe `tests/golden/` — casos de Física congelados que o barramento nunca pode quebrar.

### Por que `models/` não sabe que existe treino distribuído

`models/` define `nn.Module` puros. Todo sharding (FSDP2, tensor parallel, pipeline parallel, context parallel) é aplicado por `training/parallel/` no momento do wrap.

É a fronteira que sustenta o princípio **P5** (portabilidade entre perfis de computação): o mesmo modelo roda em 1 GPU e em 256 mudando apenas `configs/compute/profile_*.yaml`. Se paralelismo vazar para dentro de `models/`, essa propriedade morre e o projeto fica preso ao hardware em que nasceu.
