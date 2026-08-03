# Instalação e continuidade em outra máquina

O projeto tem **duas partes que vivem em lugares diferentes**, por decisão de
arquitetura (DOC-01 §4.1):

| Parte | Onde | Como obter |
|---|---|---|
| Código, documentos, testes | **GitHub** | `git clone` |
| Dados coletados | **Google Drive** (`LLMFísica/`) | sincronização do Drive |
| Ambiente Python | nenhum dos dois | recriado por `requirements.lock` |

---

## 1. Clonar e instalar

```bash
git clone https://github.com/sanchezVB/LLM_F-sica.git
cd LLM_F-sica
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
```

Verificar que funcionou:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/regression/ -q
```

Devem passar **28 testes**. Se passarem, o ambiente está correto.

## 2. Configurar identificação

Os coletores precisam de um e-mail real de contato — é exigência de cortesia
do arXiv (DOC-02 §8.2) e condição para o *polite pool* do OpenAlex.

```bash
cp .env.example .env
# editar .env e preencher PHIFM_CONTACT
```

O `.env` **não está no git** (contém dado pessoal). Precisa ser recriado em
cada máquina.

## 3. Trazer os dados

Duas rotas, e a escolha depende do que você vai fazer.

### Rota A — recoletar do zero (recomendada)

Não precisa copiar nada. Os manifestos versionados descrevem exatamente o que
coletar, e os coletores são idempotentes e retomáveis:

```bash
./scripts/run_harvest.sh arxiv
./scripts/run_harvest.sh openalex
```

É o critério **G1.5** do DOC-00 em funcionamento: *o corpus é reconstruível
ponta a ponta a partir de um único hash de manifesto*. Custa tempo (~5 h),
não custa dinheiro, e garante que o pipeline realmente reproduz.

### Rota B — copiar do Drive

Mais rápido, e necessário quando você quer continuar de onde parou:

```
Meu Drive/LLMFísica/
  01-corpus-bruto/        →  data/raw/
  02-corpus-processado/   →  data/processed/
  03-checkpoints/         →  models/
  05-manifestos/          →  data/raw/<fonte>/_manifest.json
```

O manifesto é o que importa: com ele no lugar, `run_harvest.sh` retoma do
último checkpoint durável em vez de recomeçar.

## 4. Retomar o trabalho

```bash
# estado das coletas
.venv/bin/python -c "
import json,pathlib
for d in pathlib.Path('data/raw').glob('*/_manifest.json'):
    m=json.load(open(d))
    print(f\"{d.parent.name:20} {m['actual_count']:>9,} registros · falhas {len(m['failures'])}\")"

# reconstruir a espinha consolidada
PYTHONPATH=src .venv/bin/python scripts/build_spine.py

# treinar o classificador de subárea
PYTHONPATH=src .venv/bin/python scripts/train_classifier.py
```

---

## O que NÃO copiar

| Item | Por quê |
|---|---|
| `.venv/` | 621 MB, específico de plataforma. Recriar com `requirements.lock` |
| Fatias brutas do HuggingFace | Datasets públicos versionados — rebaixar é mais rápido que sincronizar |
| PDFs de NTRS/OSTI/teses | Processados em fluxo, nunca armazenados (DOC-03 §8) |

---

## Onde ler primeiro

1. [`README.md`](README.md) — visão geral e a escada de degraus
2. [`docs/README.md`](docs/README.md) — índice dos 19 documentos, com ordem de leitura
3. [`docs/00-foundations/DOC-00-project-charter.md`](docs/00-foundations/DOC-00-project-charter.md) — por que o projeto existe
4. [`docs/01-data/DOC-02-aquisicao-corpus.md`](docs/01-data/DOC-02-aquisicao-corpus.md) §9 — o cronograma de sprints, e onde estamos

**Estado atual:** Sprint S1 em execução (coleta de metadados), S2 parcialmente
concluído (classificador de subárea treinado; a versão binária Física
vs. não-Física aguarda exemplos negativos).
