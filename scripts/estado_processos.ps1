# Lista e opcionalmente encerra processos do projeto.
#
#   .\scripts\estado_processos.ps1            # so lista
#   .\scripts\estado_processos.ps1 -Encerrar phiemb
#
# ─── Por que um SCRIPT e nao um comando inline ───────────────────────────
#
# Consultar processos por linha de comando com um comando inline e uma armadilha:
# a propria consulta contem o texto buscado, entao ela se encontra. Em
# 2026-08-07 isso me enganou SEIS vezes — a pior foi com `wmic /format:csv`, que
# ainda quebra linhas de comando longas em varias linhas do CSV, multiplicando o
# falso positivo.
#
# Um script em arquivo nao tem esse problema: sua linha de comando e so o caminho
# do arquivo. E `-Encerrar` filtra por `-File`, que so aparece em processo
# lancado de verdade, nunca numa consulta.

param([string]$Encerrar = '')

$ErrorActionPreference = 'Continue'

$alvos = @{
    'coleta-arxiv'  = 'harvest_arxiv.py'
    'coleta-neg'    = 'harvest_negativos.py'
    'coleta-snap'   = 'harvest_openalex_snapshot.py'
    'treino-phiemb' = 'train_embedding.py'
    'espinha'       = 'build_spine.py'
    'pares'         = 'build_pairs.py'
}

$todos = Get-CimInstance Win32_Process -Filter "Name='python.exe' or Name='powershell.exe' or Name='cmd.exe'" -ErrorAction SilentlyContinue |
         Where-Object { $_.CommandLine -and $_.CommandLine -notlike '*estado_processos*' }

Write-Output '=== processos do projeto ==='
$achados = @()
foreach ($k in $alvos.Keys | Sort-Object) {
    $p = $todos | Where-Object { $_.CommandLine -like "*$($alvos[$k])*" }
    $n = ($p | Measure-Object).Count
    if ($n) {
        Write-Output ("  {0,-15} {1} processo(s): {2}" -f $k, $n, (($p.ProcessId) -join ' '))
        $achados += [pscustomobject]@{ Nome = $k; Procs = $p }
    }
}
if (-not $achados) { Write-Output '  (nenhum)' }

Write-Output ''
Write-Output '=== supervisores ==='
$sup = $todos | Where-Object { $_.CommandLine -like '*-File*supervisor_coleta*' }
if ($sup) {
    foreach ($s in $sup) {
        $qual = if ($s.CommandLine -match 'supervisor_coleta\.ps1.\s*(\w+)') { $Matches[1] } else { '?' }
        Write-Output ("  {0,-15} PID {1}" -f $qual, $s.ProcessId)
    }
} else { Write-Output '  (nenhum)' }

Write-Output ''
Write-Output '=== travas ==='
Get-ChildItem 'data\raw\.trava_*' -ErrorAction SilentlyContinue | ForEach-Object {
    $p = (Get-Content $_.FullName -Raw).Trim()
    $vivo = if (Get-Process -Id ([int]$p) -ErrorAction SilentlyContinue) { 'PID vivo' } else { 'ORFA' }
    Write-Output ("  {0,-22} {1}  {2}" -f $_.Name, $p, $vivo)
}

if ($Encerrar) {
    Write-Output ''
    Write-Output "=== encerrando '$Encerrar' ==="
    $mortos = 0
    foreach ($s in ($sup | Where-Object { $_.CommandLine -match [regex]::Escape($Encerrar) })) {
        try { Stop-Process -Id $s.ProcessId -Force -ErrorAction Stop; Write-Output "  supervisor $($s.ProcessId) encerrado"; $mortos++ } catch {}
    }
    $chave = if ($Encerrar -eq 'phiemb') { 'train_embedding.py' } else { $alvos["coleta-$Encerrar"] }
    if ($chave) {
        foreach ($p in ($todos | Where-Object { $_.CommandLine -like "*$chave*" })) {
            try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; Write-Output "  processo $($p.ProcessId) encerrado"; $mortos++ } catch {}
        }
    }
    Write-Output "  total: $mortos"
}
