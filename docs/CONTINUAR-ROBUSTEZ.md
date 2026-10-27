# Continuidade: avaliação, treino e evidências

Implementação de 27/10/2026. Os resultados históricos permanecem como referência; estas
mudanças não demonstram um aumento de acerto do modelo.

## Entregas de código

- G2: identidade da execução inclui prompts, parâmetros e artefatos; logs de outra
  configuração são recusados. Retentativas contam uma vez e o agregado exige todos
  os itens/amostras. Resultados publicados não são recalculados com nova tentativa.
- Pré-condições G2: código, autoteste, exclusões, partição, dados, quantização e motor
  são conferidos antes da geração. O protocolo oficial já foi congelado no main. O executor novo é `medir_g2_robusto.py`; requer uma emenda e nova validação antes de uso no teste.
- Pré-treino: checkpoint completo do detector, métricas, contadores e RNG por processo;
  escrita atômica e cópia anterior. Rollback preserva a memória dos spikes. A continuidade
  de checkpoint antigo exige `train_phienc.py --permitir-legado`, sem equivalência prometida.
- CI: treino CPU com versões fixadas, modelos sintéticos e verificação de que nenhum
  alvo obrigatório ficou vazio ou saltado.
- Corpus CPT: auditoria de partições, identidade entre splits e coincidências literais
  com referências explícitas. Triagem por 13 palavras não detecta toda paráfrase; a
  validação G2 exige banco e contagens ancorados no protocolo, não autoafirmação do manifesto.
- Assistente: `--verificar-evidencias` é experimental. O juiz precisa apresentar trechos
  literais das fontes; respostas sem suporte são retidas. Testes com juiz falso validam
  a integração, não a precisão semântica. A opção continua desligada por padrão.
- Piloto causal: objetivo de próximo token, avaliação de desenvolvimento, identidade
  de dados/base e retomada. Não implementa SFT/RLVR nem prova ganho no G2.

## Máquina local e HD

Na janela PowerShell do checkout que está no HD, informe o caminho real (o exemplo
`D:\LLMFísica` vem da documentação, não foi confirmado por detecção física):

```powershell
. .\scripts\ambiente_hd.ps1 -RaizHD 'D:\LLMFísica'
```

Isso configura a sessão para caches, temporários, modelos e ambientes dentro do HD.
Recusa checkout ou destinos que escapem da raiz, inclusive por links. Não instala nada
por padrão. Para preparar a venv CPU separada, usando Python 3.12 já instalado, acrescente
`-Instalar`. A venv GPU existente não é substituída. PowerShell/Windows ainda precisam
de validação na máquina local; nesta nuvem foram testados os contratos Python de caminhos.

Depois, usando o interpretador apropriado da venv:

```powershell
.venv-treino-cpu\Scripts\python.exe scripts\conferir_g2.py --fase dev
.venv-treino-cpu\Scripts\python.exe scripts\auditar_corpus_cpt.py --help
.venv-treino-cpu\Scripts\python.exe scripts\treinar_phigen.py --help
```

O piloto recebe `--base` local, `--manifesto` da auditoria e `--saida` nova, todos sob
`PHIFM_HD_ROOT`. `--verificar` confere os pré-requisitos sem treinar. O orçamento é explícito
(`--passos`, `--contexto`, `--max-blocos`). A implementação faz ajuste de todos os parâmetros;
uma base de 1,7B exige memória para pesos, gradientes, Adam e ativações. A RX 7600 de 8 GB
documentada no projeto não atende este caminho CUDA. A GPU local continua adequada ao
fluxo de inferência Vulkan documentado; treino grande requer hardware/stack apropriados.

## O que falta executar com os artefatos reais

1. Disponibilizar corpus, referências e modelos locais; conferir os caminhos do HD.
2. Preservar a avaliação oficial 1,7B registrada no main. Para o executor robusto, atualizar autoteste depois do parser atual e validar no dev.
3. Registrar emenda e novo congelamento para o executor robusto antes de remedir bases. Não preencher hashes
   para contornar a ausência dos arquivos.
4. Auditar corpus completo e executar piloto causal no hardware adequado; medir retenção
   de capacidades gerais e ganho físico com os controles pré-registrados.
5. Avaliar o juiz de evidências com perguntas novas e revisão humana, medindo falso
   aceite, falsa rejeição e abstenção indevida antes de adotar a opção no produto.

Os arquivos externos não estão nesta máquina. Não foram executados treinos de 1,7B,
benchmarks lacrados ou medições reais de melhoria do assistente durante esta alteração.

## Estado incorporado do main

A integração inclui o commit `6058d0e`: avaliação oficial da base 1,7B já registrada,
auditoria cega do corretor e ADR-0005. `medir_g2.py`, `g2.py`, `g2_geracao.py` e o
pré-registro oficial foram preservados. `medir_g2_robusto.py` é o executor novo para
desenvolvimento e proposta de emenda; seus logs não retomam os logs históricos.
