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
# progresso, e desiste após `-MaxReinicios` seguidos sem avanço — porque
# relançar em laço um processo que morre na mesma partição é laço infinito, não
# resiliência.

param(
    [ValidateSet('arxiv', 'negativos', 'openalex', 'snapshot')][string]$Fonte = 'snapshot',
    [int]$IntervaloSegundos = 60,
    [int]$MaxReinicios = 40
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

while ($true) {
    $vivo = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -like "*$alvo*" }

    if (-not $vivo) {
        $p = ProgressoAtual

        # Concluida? O manifesto marca `completed_at` ao fim.
        if ($manifesto -and (Test-Path $manifesto)) {
            $m = Get-Content $manifesto -Raw -Encoding utf8 | ConvertFrom-Json
            if ($m.completed_at) {
                Registrar "CONCLUIDA: $($m.actual_count) registros, $($m.failures.Count) falhas"
                break
            }
        }

        if ($reinicios -ge 1 -and $p -eq $progressoAnterior -and $p -ge 0) {
            Registrar "ABORTANDO: morreu sem avancar (progresso parado em $p). Nao e queda aleatoria."
            break
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
