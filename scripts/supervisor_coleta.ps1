# Mantém uma coleta viva, relançando-a quando ela morre.
#
#   .\scripts\supervisor_coleta.ps1 snapshot
#   .\scripts\supervisor_coleta.ps1 negativos
#
# ─── Por que isto existe ──────────────────────────────────────────────────
#
# Em 2026-08-07 duas coletas morreram com 40 s de diferença — snapshot às
# 09:44:58 na partição 422/2446, negativos às 09:45:38 na página 13 — sem
# traceback, sem evento no log do Windows, com 5,7 GB de RAM livre e ambas
# lançadas por `Win32_Process.Create` (que escapa do job object; o coletor do
# arXiv sobreviveu 5 h por essa via na véspera).
#
# A causa não foi identificada. O que se sabe é que é externa e indiscriminada:
# atingiu dois processos independentes ao mesmo tempo.
#
# **Perseguir a causa é o caminho errado aqui.** O coletor já é idempotente e
# retomável por projeto (A4): uma morte custa, no pior caso, o lote pendente
# desde o último flush durável. O que faltava não era robustez do coletor, era
# alguém para reerguê-lo. Este supervisor é esse alguém.
#
# Ele NÃO mascara falha real: registra cada relançamento com horário e
# progresso, e desiste após `-MaxParado` mortes SEGUIDAS sem avanço durável —
# porque relançar em laço um processo que morre sempre no mesmo ponto é laço
# infinito, não resiliência.
#
# O `-MaxParado` era 1 e virou 4, e a razão é instrutiva: o progresso durável só
# existe no flush, então um processo morto antes do primeiro flush parece
# parado mesmo tendo lido partições. Com tolerância 1, este guarda abortou por
# engano às 09:55 — no exato modo de falha que existia para sobreviver. A
# contrapartida foi baixar `FLUSH_PARTICOES` de 10 para 3 no coletor, para que
# progresso durável apareça a cada ~30 s em vez de ~1,5 min.

param(
    [ValidateSet('arxiv', 'negativos', 'openalex', 'snapshot')][string]$Fonte = 'snapshot',
    [int]$IntervaloSegundos = 60,
    [int]$MaxReinicios = 40,
    [int]$MaxParado = 4
)

$ErrorActionPreference = 'Stop'
$raiz = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $raiz

switch ($Fonte) {
    'arxiv'     { $alvo = 'harvest_arxiv.py';            $manifesto = 'data\raw\arxiv_metadata\_manifest.json' }
    'negativos' { $alvo = 'harvest_negativos.py';        $manifesto = $null }
    'openalex'  { $alvo = 'harvest_openalex.py';         $manifesto = 'data\raw\openalex_works\_manifest.json' }
    'snapshot'  { $alvo = 'harvest_openalex_snapshot.py'; $manifesto = 'data\raw\openalex_snapshot\_manifest.json' }
}

$log = Join-Path $raiz "data\raw\supervisor_$Fonte.log"
$reinicios = 0
$paradas = 0
$progressoAnterior = -1

function Registrar($texto) {
    $linha = "{0}  {1}" -f (Get-Date -Format 'HH:mm:ss'), $texto
    Add-Content -Path $log -Value $linha -Encoding utf8
    Write-Output $linha
}

function ProgressoAtual {
    # Registros duráveis no manifesto. Serve para distinguir "morreu mas
    # avançou" de "morre sempre no mesmo ponto".
    if (-not $manifesto -or -not (Test-Path $manifesto)) { return -1 }
    try { (Get-Content $manifesto -Raw -Encoding utf8 | ConvertFrom-Json).actual_count }
    catch { return -1 }
}

Registrar "supervisor de '$Fonte' iniciado (intervalo ${IntervaloSegundos}s, ate $MaxReinicios reinicios)"

$trava = Join-Path $raiz "data\raw\.trava_$Fonte"
$mortesSeguidas = 0

while ($true) {
    # Vida pela TRAVA, nao por consulta de linha de comando. A consulta antiga
    # devolvia vazio quando falhava transitoriamente, e vazio parecia "morreu":
    # em 2026-08-07 isso acumulou TRES coletores de negativos simultaneos,
    # triplicando a taxa vista pelo arXiv (violacao do A5). `Get-Process -Id`
    # responde de forma binaria sobre um PID conhecido.
    $vivo = $false
    if (Test-Path $trava) {
        $pidAlvo = (Get-Content $trava -Raw).Trim()
        if ($pidAlvo -match '^\d+$') {
            $vivo = [bool](Get-Process -Id ([int]$pidAlvo) -ErrorAction SilentlyContinue)
        }
    }

    # Duas leituras seguidas de morte antes de agir. Uma so pode ser corrida com
    # o lancamento, e relancar em cima de um processo vivo e o defeito que esta
    # trava existe para impedir.
    if (-not $vivo) { $mortesSeguidas++ } else { $mortesSeguidas = 0 }

    if ($mortesSeguidas -ge 2) {
        $mortesSeguidas = 0
        $p = ProgressoAtual

        # Concluida? O manifesto marca `completed_at` ao fim.
        if ($manifesto -and (Test-Path $manifesto)) {
            $m = Get-Content $manifesto -Raw -Encoding utf8 | ConvertFrom-Json
            if ($m.completed_at) {
                Registrar "CONCLUIDA: $($m.actual_count) registros, $($m.failures.Count) falhas"
                break
            }
        }

        # Uma morte sem avanco NAO prova defeito. O progresso durável só aparece
        # no flush, então um processo morto antes do primeiro flush parece
        # parado mesmo tendo lido partições — foi assim que este guarda abortou
        # por engano em 2026-08-07 09:55, no exato modo de falha que existia
        # para sobreviver. Exige-se `$MaxParado` mortes seguidas sem avanço.
        if ($p -eq $progressoAnterior -and $p -ge 0) {
            $paradas++
            Registrar "morta sem avanco durável ($paradas de $MaxParado tolerados; progresso $p)"
            if ($paradas -ge $MaxParado) {
                Registrar "ABORTANDO: $MaxParado mortes seguidas sem avancar de $p. Nao e queda aleatoria."
                break
            }
        } else {
            $paradas = 0
        }
        if ($reinicios -ge $MaxReinicios) {
            Registrar "ABORTANDO: $MaxReinicios reinicios atingidos"
            break
        }

        $reinicios++
        $progressoAnterior = $p
        Registrar "morta (progresso $p) - reinicio #$reinicios"
        & (Join-Path $raiz 'scripts\run_harvest.ps1') $Fonte | ForEach-Object { Registrar "  $_" }
    }

    Start-Sleep -Seconds $IntervaloSegundos
}

Registrar "supervisor de '$Fonte' encerrado apos $reinicios reinicio(s)"
