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

param([ValidateSet('arxiv', 'negativos', 'openalex', 'snapshot')][string]$Fonte = 'arxiv')

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
    'openalex' { $script = 'scripts\harvest_openalex.py'
                 $argumentos = '--out data/raw/openalex_works' }
    'snapshot' { $script = 'scripts\harvest_openalex_snapshot.py'
                 $argumentos = '--out data/raw/openalex_snapshot' }
}

# Ja rodando? Compara a linha de comando, porque o nome do processo e sempre
# `python.exe` e mataria a distincao entre os coletores.
$alvo = Split-Path -Leaf $script
$emCurso = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
           Where-Object { $_.CommandLine -like "*$alvo*" }
if ($emCurso) {
    Write-Output "$Fonte ja em execucao (PID $($emCurso.ProcessId -join ' '))"
    exit 0
}

$log = Join-Path $raiz "data\raw\harvest_$Fonte.log"
New-Item -ItemType Directory -Force (Split-Path -Parent $log) | Out-Null

# PYTHONUTF8: sem isto o log sai com acento quebrado, porque o console herda
# cp1252 e o coletor registra mensagens acentuadas.
$linha = 'cmd.exe /c "set PYTHONPATH=src&& set PYTHONUTF8=1&& ' +
         "`"$python`" $script $argumentos >> `"$log`" 2>&1`""

$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine      = $linha
    CurrentDirectory = $raiz
}
if ($r.ReturnValue -ne 0) {
    Write-Output "FALHOU ao criar o processo (Win32_Process.Create devolveu $($r.ReturnValue))"
    exit 1
}

Start-Sleep -Seconds 8
$vivo = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -like "*$alvo*" }
if ($vivo) {
    Write-Output "$Fonte iniciado (PID $($vivo.ProcessId -join ' ')) - log: $log"
    Write-Output "Acompanhar:  Get-Content -Wait -Tail 20 '$log'"
} else {
    Write-Output "FALHOU: o processo nao esta vivo apos 8 s - ver $log"
    exit 1
}
