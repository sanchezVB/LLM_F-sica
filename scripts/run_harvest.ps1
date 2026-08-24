# Lança um coletor do Sprint S1 destacado do terminal, no Windows.
#
#   .\scripts\run_harvest.ps1 arxiv       # espinha de metadados
#   .\scripts\run_harvest.ps1 openalex    # grafo de citações (API cotada)
#   .\scripts\run_harvest.ps1 snapshot    # grafo de citações (snapshot, US$ 0)
#
# Equivalente de `run_harvest.sh`, que é de macOS: `caffeinate`, `pgrep` e
# `.venv/bin/python` não existem aqui. As duas proteções do original valem
# igual, mas nenhuma das duas se resolve do mesmo jeito.
#
# ─── Suspensão ────────────────────────────────────────────────────────────
# Resolvida DENTRO do coletor, por `phifm.core.sistema.impedir_suspensao()`.
# Fica no processo, não no lançador, para valer em qualquer forma de invocação
# e ser liberada sozinha quando ele morre.
#
# ─── Desacoplamento: por que WMI e não Start-Process ─────────────────────
# Medido em 2026-08-06, e custou uma coleta: `Start-Process -WindowStyle Hidden`
# NÃO sobrevive. No Windows, processo criado por um shell entra no **job
# object** dele, e quando o job é encerrado — reciclagem do terminal, fim da
# sessão do agente — todos os descendentes morrem junto. A primeira coleta
# durou 8 min 40 s e morreu sem deixar traceback, o que torna o sintoma
# especialmente traiçoeiro: o log simplesmente para.
#
# `Win32_Process.Create` por WMI cria o processo a partir do serviço WMI, fora
# do nosso job object. É o análogo funcional do `nohup ... & disown`.
#
# Efeito colateral bem-vindo: o `cmd /c` com `>>` e `2>&1` junta os dois fluxos
# num arquivo só, como o `.sh` fazia — o `Start-Process` obrigava a separá-los
# e o log útil acabava no `.err`.
#
# Idempotente: detecta execução em andamento e não duplica. Após interrupção,
# retoma do cursor durável no `_manifest.json`.

# NOTA DE NOME: o script comecou como lancador de COLETA e passou a cobrir
# treino tambem ('phiemb'). O nome ficou estreito, mas a alternativa era um
# segundo par de scripts com trava e supervisor duplicados — e mecanismo
# duplicado e ramo sem teste. Preferi o nome torto ao codigo torto.
param([ValidateSet('arxiv', 'negativos', 'negativos-math', 'negativos-stat', 'openalex', 'snapshot',
                   'phiemb', 'phiemb-mini', 'phiemb-mini-1m5', 'phiemb-gc', 'phiemb-duros', 'phirank', 'g1',
                   'redpajama')][string]$Fonte = 'arxiv',
    # ⚠️ Roda o python NESTE processo, bloqueando, em vez de destacar por WMI.
    #
    # Existe para a Tarefa Agendada. Ela chama este lancador, e o lancador
    # destacava por WMI e RETORNAVA — a tarefa ia para "Ready" em segundos e o
    # treino voltava a ser orfao, que e exatamente o que morre quando a sessao
    # termina. Com isto a tarefa SEGURA o processo, que e o ponto dela.
    #
    # No modo normal (sem esta bandeira) nada muda: quem chama do terminal quer o
    # prompt de volta.
    [switch]$EmPrimeiroPlano)

$ErrorActionPreference = 'Stop'
$raiz = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $raiz

$python = Join-Path $raiz '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "venv nao encontrada em $python - ver SETUP.md" }

switch ($Fonte) {
    'arxiv'    { $script = 'scripts\harvest_arxiv.py'
                 $argumentos = '--out data/raw/arxiv_metadata --set physics' }
    'negativos' { $script = 'scripts\harvest_negativos.py'
                  $argumentos = '--out data/raw/arxiv_negativos' }
    'negativos-stat' { $script = 'scripts\harvest_negativos.py'
                  # NAO precisa fatiar, ao contrario do math: medido em
                  # 2026-08-14, `ListRecords set=stat` responde 200 em 5,1 s.
                  $argumentos = '--out data/raw/arxiv_negativos --sets stat' }
    'negativos-math' { $script = 'scripts\harvest_negativos.py'
                  # Set separado, e nao `-Fonte negativos`, por DUAS razoes.
                  #
                  # A trava por PID e por MODO (`.trava_$Fonte`). Se `math` usasse
                  # o modo `negativos`, ele nao poderia rodar enquanto o outro
                  # roda — e mais importante: uma trava orfa de `negativos`
                  # bloquearia `math` sem motivo.
                  #
                  # E o manifesto de cada set e independente, entao coletar so
                  # `math` nao toca no que ja esta concluido.
                  $argumentos = '--out data/raw/arxiv_negativos --sets math' }
    'redpajama' { $script = 'scripts/coletar_redpajama.py'
                  # 81 GB em fluxo, ~25 min a 61 MB/s medidos. O bruto nao aterra;
                  # so a fatia de Fisica (~10 GB de parquet) fica em disco.
                  $argumentos = '--out data/processed/redpajama_fisica' }
    'openalex' { $script = 'scripts\harvest_openalex.py'
                 $argumentos = '--out data/raw/openalex_works' }
    'snapshot' { $script = 'scripts\harvest_openalex_snapshot.py'
                 $argumentos = '--out data/raw/openalex_snapshot' }
    'phiemb'   { $script = 'scripts/train_embedding.py'
                 # A venv de treino e Python 3.12: o torch-directml nao suporta
                 # o 3.14 da venv principal.
                 $python = Join-Path $raiz '.venv-treino\Scripts\python.exe'
                 $argumentos = '--out models/phiemb --lote 8 --passos-aval 500 --max-pares 400000' }
    'phiemb-mini' { $script = 'scripts/train_embedding.py'
                    $python = Join-Path $raiz '.venv-treino\Scripts\python.exe'
                    # Lote 128 medido no DirectML: 16,2 pares/s e 127 negativos
                    # por ancora, contra 4,1 e 7 do SciBERT. Negativos no lote
                    # sao o limite de QUALIDADE do contrastivo, nao so de
                    # velocidade — 127 esta na faixa de 64-256 da literatura.
                    #
                    # Mesmos 400 mil pares do treino anterior, de proposito: a
                    # comparacao isola a mudanca de base e de lote.
                    #
                    # `n-candidatos 1000` porque a escolha do melhor checkpoint
                    # se baseia no MRR da avaliacao, e 256 candidatos davam
                    # ruido de +-0,031.
                    $argumentos = '--out models/phiemb-minilm ' +
                                  '--base sentence-transformers/all-MiniLM-L6-v2 ' +
                                  '--lote 128 --passos-aval 200 --max-pares 400000 ' +
                                  '--n-candidatos 1000' }
    'phiemb-mini-1m5' { $script = 'scripts/train_embedding.py'
                  $python = Join-Path $raiz '.venv-treino\Scripts\python.exe'
                  # MAIS DADOS, com o lote que ganhou por medicao.
                  #
                  # O experimento de 511 negativos perdeu: nDCG@10 0,449 contra
                  # 0,458 do lote 128, e 2,3x mais lento. Entao o lote fica em 128
                  # e a variavel passa a ser o volume: 1,5 M de pares contra os
                  # 400 mil, que sao 22% dos 6,7 M disponiveis contra 6%.
                  #
                  # A evidencia de que vale: a curva do treino de 400 mil AINDA
                  # SUBIA no fim — MRR 0,471 no passo 2990 e o melhor 0,4766 no
                  # 2800. Nao saturou.
                  #
                  # Saida em diretorio NOVO: o modelo de 400 mil e o campeao atual
                  # do G1.1 e nao pode ser sobrescrito por um experimento.
                  #
                  # `passos-aval 500` porque sao 11.719 passos; a cada 200 daria 58
                  # avaliacoes de ~15 s cada, que e tempo tirado do treino.
                  #
                  # `--sub-lote 64` NAO e mudanca de experimento: o lote LOGICO
                  # segue 128, entao os negativos e a comparabilidade com o
                  # campeao ficam iguais. O que muda e o forward FISICO.
                  #
                  # Motivo, medido duas vezes: o caminho direto estourou a VRAM
                  # pedindo 226.492.416 bytes = 128 x 12 cabecas x 192 x 192 x 4,
                  # os escores de atencao de um lote. Uma vez no passo 800 e outra
                  # minutos depois de retomar. Duas mortes na MESMA alocacao sao
                  # padrao, e repetir a tentativa identica seria esperar sorte.
                  #
                  # Com 64 a mesma alocacao cai para 113 MB. O DirectML nao devolve
                  # memoria num `del` (medido: 1 GB alocado continua marcado), logo
                  # fragmentacao crescente e o comportamento do alocador, nao
                  # anomalia — e pedir metade e o que se pode fazer sobre isso.
                  #
                  # Custo: o GradCache faz um forward a mais. O treino fica mais
                  # lento; um treino lento que termina vale mais que um rapido que
                  # morre no passo 800.
                  $argumentos = '--out models/phiemb-minilm-1m5 ' +
                                '--base sentence-transformers/all-MiniLM-L6-v2 ' +
                                '--lote 128 --sub-lote 64 ' +
                                '--passos-aval 500 --max-pares 1500000 ' +
                                '--n-candidatos 1000' }
    'phirank'  { $script = 'scripts/train_rerank.py'
                  $python = Join-Path $raiz '.venv-treino\Scripts\python.exe'
                  # PhiRank, o T1b: cross-encoder que reordena o top-100 do PhiEmb.
                  #
                  # Le os negativos LIMPOS (sem os co-citados com o positivo). Com os
                  # nao filtrados, 15,4% dos negativos sao documentos que a
                  # literatura cita junto com o positivo — treinar um reranker a
                  # rebaixa-los e ensinar a rebaixar o que e relevante, e a perda
                  # desce normalmente enquanto isso acontece.
                  #
                  # 12.500 grupos de 8 = 100 mil exemplos, ~1,7 h medidas nesta
                  # maquina com `--grupos 2` (16,2 exemplos/s).
                  #
                  # ⚠️ Os negativos vem de `minerar_do_recuperador.py`, NAO de
                  # `minerar_negativos.py`. Medido em 2026-08-24: os do segundo sao o
                  # top-K do denso menos a citacao verdadeira, o que rotula NEGATIVO
                  # tudo que o recuperador poe no topo. O reranker aprendeu
                  # `muito recuperado => nao e a resposta` e INVERTEU a fusao —
                  # escore medio subindo monotonicamente da posicao 0-4 (-4,32) para
                  # a 30-49 (-3,72), Spearman +0,179, positivo em 83% das consultas.
                  # Resultado: nDCG@10 da composicao caiu de 0,1393 (fusao sozinha)
                  # para 0,0179, pior que ordem aleatoria.
                  #
                  # Os novos vem do RRF top-50 DE VERDADE, a distribuicao exata da
                  # avaliacao: 14.289 grupos, 43,8 negativos cada, alvo em posicao
                  # media 11,2. Assim "estar no topo" deixa de prever o rotulo.
                  #
                  # ⚠️ E `--out models/phirank-rrf`, dir NOVO. Reaproveitar
                  # `models/phirank-minilm` faria `retomar()` carregar o
                  # `estado_rank.pt` da corrida quebrada e continuar do passo 1.600
                  # dela — o treino novo herdaria os pesos do invertido.
                  #
                  # ⚠️ Desvio de especificacao registrado: o DOC-07 §4 pede
                  # inicializacao do PhiEnc, que nao existe. Usado o MiniLM, a mesma
                  # base do PhiEmb campeao.
                  #
                  # ⚠️ `--grupos 2`, nao 4. Com 4 sao 32 exemplos a 384 tokens, e o
                  # tensor de atencao e 32 x 12 x 384 x 384 x 4 = 226.492.416 bytes
                  # — exatamente a alocacao que falta nos 8 GB. Passou numa corrida
                  # e estourou na seguinte, entao estava no limite. Com 2 grupos o
                  # tensor cai para 113 MB.
                  #
                  # NAO muda o experimento: o grupo segue 1 positivo + 7 negativos, e
                  # a perda continua sobre o grupo. So passam menos grupos por passo,
                  # o que e ruido de otimizacao, nao mudanca de tarefa. `max-grupos`
                  # e em LINHAS do conjunto, entao o volume de exemplos e o mesmo.
                  $argumentos = '--max-grupos 12500 --grupos 2 --n-negativos 7 ' +
                                '--passos-aval 1000 --grupos-aval 500 ' +
                                '--out models/phirank-rrf ' +
                                '--negativos data/processed/negativos_dificeis/' +
                                'pares_do_recuperador_limpos.parquet' }
    'phiemb-duros' { $script = 'scripts/train_embedding.py'
                  $python = Join-Path $raiz '.venv-treino\Scripts\python.exe'
                  # NEGATIVOS DIFICEIS minerados pelo proprio campeao (DOC-07 §4).
                  #
                  # A ultima alavanca barata do G1.2: as duas testadas mexiam na
                  # QUANTIDADE de negativos e de dados, e as duas deram empate. Esta
                  # mexe na DIFICULDADE deles.
                  #
                  # Mesmos 400 mil pares e mesmo lote 128 do campeao de proposito: a
                  # unica variavel e a dificuldade dos negativos.
                  #
                  # `--pedaco-negativos 32` porque tres codificacoes de 128 nao cabem
                  # nos 8 GB. Baixar o lote para 64 resolveria a memoria e mudaria os
                  # negativos DO LOTE de 127 para 63 — duas variaveis de uma vez, e o
                  # experimento perderia o sentido.
                  #
                  # ⚠️ Vai por AQUI e nao por um envelope .cmd caseiro, e a razao
                  # custou um estouro de VRAM: `cmd.exe` rele o arquivo de lote do
                  # disco a cada linha, por deslocamento de BYTES. Editei o envelope
                  # para forcar UTF-8 enquanto ele executava, os offsets deslocaram e
                  # o cmd RE-EXECUTOU a linha do python — duas sessoes de treino na
                  # mesma placa. Este lancador tem trava por PID e nao permite isso.
                  $argumentos = '--out models/phiemb-minilm-duros ' +
                                '--base sentence-transformers/all-MiniLM-L6-v2 ' +
                                '--lote 128 --passos-aval 200 --max-pares 400000 ' +
                                '--n-candidatos 1000 --pedaco-negativos 32 ' +
                                '--negativos data/processed/negativos_dificeis/pares_com_negativos.parquet' }
    'phiemb-gc' { $script = 'scripts/train_embedding.py'
                  $python = Join-Path $raiz '.venv-treino\Scripts\python.exe'
                  # GradCache: lote LOGICO 512 (511 negativos) com a memoria de
                  # 128. O experimento que faltava — o treino anterior mudou base
                  # E lote ao mesmo tempo, entao nao isolava nada. Aqui a base e a
                  # mesma e so os negativos mudam: 127 -> 511.
                  #
                  # sub-lote 64, nao 128. Com 128 o treino estourou a VRAM no
                  # passo ~200 pedindo 226.492.416 bytes — que e exatamente
                  # 128 x 12 cabecas x 192 x 192 x 4, os escores de atencao de um
                  # pedaco. Rodou os passos 100-200 antes de estourar, entao e
                  # fragmentacao acumulando, nao pico unico.
                  #
                  # O sub-lote NAO afeta os negativos: o lote logico segue 512, e
                  # 511 negativos continuam entrando na perda. Custa so tempo —
                  # 8 pedacos por passo em vez de 4. E o ponto todo do GradCache.
                  $argumentos = '--out models/phiemb-minilm-gc ' +
                                '--base sentence-transformers/all-MiniLM-L6-v2 ' +
                                '--lote 512 --sub-lote 64 --passos-aval 100 ' +
                                '--max-pares 400000 --n-candidatos 1000' }
    'g1'       { $script = 'scripts/avaliar_encoders.py'
                 $python = Join-Path $raiz '.venv-treino\Scripts\python.exe'
                 # ~2,5 h de CPU: o GTE-large de 335M leva ~45 min sozinho. Vai
                 # por aqui e nao por `Start-Process` porque o lancador ja tem
                 # trava por PID e escapa do job object do shell — sem isso a
                 # avaliacao morre junto com a sessao que a disparou.
                 #
                 # O cache guarda as posicoes por item, entao a proxima rodada
                 # so paga pelos modelos novos.
                 $argumentos = '--n 2000' }
}

# ─── Trava por PID, e por que a checagem antiga nao servia ───────────────
#
# A guarda anterior filtrava `Get-CimInstance Win32_Process` pela linha de
# comando. Quando essa consulta falha transitoriamente ela devolve VAZIO, e
# vazio e indistinguivel de "nao esta rodando" — entao o lancador lancava de
# novo. Medido em 2026-08-07: o supervisor acumulou TRES coletores de negativos
# simultaneos, triplicando a taxa vista pelo arXiv, que e violacao direta do
# principio A5.
#
# A trava guarda o PID do `cmd.exe` criado por WMI. Ele vive exatamente enquanto
# o python vive (`cmd /c` espera o filho), e `Get-Process -Id` responde de forma
# binaria e confiavel: existe ou nao existe. Sem depender de ler CommandLine.
$trava = Join-Path $raiz "data\raw\.trava_$Fonte"
New-Item -ItemType Directory -Force (Split-Path -Parent $trava) | Out-Null

if (Test-Path $trava) {
    $pidAntigo = (Get-Content $trava -Raw).Trim()
    if ($pidAntigo -match '^\d+$' -and (Get-Process -Id ([int]$pidAntigo) -ErrorAction SilentlyContinue)) {
        Write-Output "$Fonte ja em execucao (PID $pidAntigo, pela trava)"
        exit 0
    }
    Remove-Item $trava -Force   # trava orfa: o processo morreu
}

$log = Join-Path $raiz "data\raw\harvest_$Fonte.log"
New-Item -ItemType Directory -Force (Split-Path -Parent $log) | Out-Null

# PYTHONUTF8: sem isto o log sai com acento quebrado, porque o console herda
# cp1252 e o coletor registra mensagens acentuadas.
#
# O `& del` no fim remove a trava quando o python sai. Sem isso NADA a removia
# num termino normal — so o proximo lancamento, ao encontrar a trava orfa. O
# resultado e que `estado_processos.ps1` reportava ORFA depois de toda execucao
# concluida, e ruido permanente num diagnostico e como uma trava de verdade passa
# desapercebida. Medido em 2026-08-13: duas travas orfas de tarefas que terminaram
# bem horas antes.
#
# `&` e nao `&&`: a trava tem de sair mesmo se o python falhar. Se o processo for
# MORTO, o del nao roda e a trava fica orfa — que e o caso que o lancador ja trata.
# ⚠️ `-u`. Sem ele o Python bufferiza a saida em blocos de 8 KB quando ela vai
# para ARQUIVO, e um treino que loga uma linha a cada 50 passos leva ~20 min para
# encher o primeiro bloco. Ate la o log fica com ZERO bytes com o processo vivo,
# o que e indistinguivel de "morreu na partida" — e me custou dois diagnosticos
# errados: em 2026-08-18 no minerador e em 2026-08-21 no PhiRank.
if ($EmPrimeiroPlano) {
    # A trava tambem e escrita aqui: e o mesmo contrato do modo destacado, e sem
    # ela `estado_processos.ps1` e o supervisor nao enxergariam este treino.
    Set-Content -Path $trava -Value $PID -Encoding ascii
    $env:PYTHONPATH = 'src'
    $env:PYTHONUTF8 = '1'
    Write-Output "$Fonte em primeiro plano (PID $PID) - log: $log"
    try {
        & $python -u $script @($argumentos -split ' ') 2>&1 |
            ForEach-Object { $_ | Out-File -FilePath $log -Append -Encoding utf8 }
    } finally {
        Remove-Item $trava -Force -ErrorAction SilentlyContinue
    }
    exit 0
}

$linha = 'cmd.exe /c "set PYTHONPATH=src&& set PYTHONUTF8=1&& ' +
         "`"$python`" -u $script $argumentos >> `"$log`" 2>&1" +
         " & del /q `"$trava`"`""

$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine      = $linha
    CurrentDirectory = $raiz
}
if ($r.ReturnValue -ne 0) {
    Write-Output "FALHOU ao criar o processo (Win32_Process.Create devolveu $($r.ReturnValue))"
    exit 1
}

# Trava escrita IMEDIATAMENTE, antes de qualquer espera. Escrever depois do
# `Start-Sleep` abriria 8 s de janela para o supervisor lancar um segundo.
Set-Content -Path $trava -Value $r.ProcessId -Encoding ascii

Start-Sleep -Seconds 8
if (Get-Process -Id $r.ProcessId -ErrorAction SilentlyContinue) {
    Write-Output "$Fonte iniciado (PID $($r.ProcessId)) - log: $log"
    Write-Output "Acompanhar:  Get-Content -Wait -Tail 20 '$log'"
} else {
    Remove-Item $trava -Force -ErrorAction SilentlyContinue
    Write-Output "FALHOU: o processo nao esta vivo apos 8 s - ver $log"
    exit 1
}
