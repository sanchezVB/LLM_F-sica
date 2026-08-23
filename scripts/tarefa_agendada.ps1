# Roda um treino longo DESACOPLADO de qualquer sessao, via Tarefa Agendada.
#
#   .\scripts\tarefa_agendada.ps1 -Fonte phirank
#   .\scripts\tarefa_agendada.ps1 -Fonte phirank -Remover
#   .\scripts\tarefa_agendada.ps1 -Listar
#
# ─── Por que isto existe ──────────────────────────────────────────────────
#
# Todo treino longo deste projeto morreu quando a sessao do agente terminou:
# o 1m5 duas vezes, a mineracao de negativos, o `phiemb-duros`, o PhiRank. Os
# lancamentos sao feitos por `Win32_Process.Create`, que cria o processo com o
# WmiPrvSE como pai justamente para escapar do job object do shell — e mesmo
# assim eles vao junto. O supervisor, que existe para reerguer, morre pelo mesmo
# motivo.
#
# O estado duravel salva o TRABALHO (o PhiRank voltou do passo 200 sem perder
# nada), mas nao salva o RELOGIO: um treino de 2,4 h vira dois dias se avanca em
# blocos de 15 min.
#
# Tarefa Agendada e o mecanismo do proprio Windows para "rode isto sem sessao".
# O processo fica sob o Task Scheduler, nao sob o shell que o criou.
#
# ⚠️ Isto e configuracao PERSISTENTE na maquina. Fica registrada com nome
# previsivel (`PhiFM-<fonte>`), e `-Remover` a apaga. `-Listar` mostra o que
# existe, para nao deixar tarefa esquecida rodando.

param(
    [Parameter(Mandatory = $false)][string]$Fonte,
    [switch]$Remover,
    [switch]$Listar
)

$ErrorActionPreference = 'Stop'
$raiz = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$prefixo = 'PhiFM-'

if ($Listar) {
    $tarefas = Get-ScheduledTask -TaskName "$prefixo*" -ErrorAction SilentlyContinue
    if (-not $tarefas) { Write-Output "nenhuma tarefa $prefixo* registrada"; exit 0 }
    foreach ($t in $tarefas) {
        $i = Get-ScheduledTaskInfo -TaskName $t.TaskName
        '{0,-24} {1,-12} ultima: {2}  resultado: {3}' -f `
            $t.TaskName, $t.State, $i.LastRunTime, $i.LastTaskResult
    }
    exit 0
}

if (-not $Fonte) { throw "informe -Fonte (ou use -Listar)" }
$nome = "$prefixo$Fonte"

if ($Remover) {
    Unregister-ScheduledTask -TaskName $nome -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "tarefa '$nome' removida"
    exit 0
}

# A tarefa chama o MESMO lancador que ja tem trava por PID. Nao ha um segundo
# caminho de lancamento — duplicar o lancamento foi o que criou dois treinos na
# mesma placa em 2026-08-18.
$lancador = Join-Path $raiz 'scripts\run_harvest.ps1'
# ⚠️ `-EmPrimeiroPlano`. Sem ele o lancador destaca por WMI e RETORNA: a tarefa vai
# para "Ready" em segundos e o treino volta a ser orfao — exatamente o que esta
# tarefa existe para evitar. Medido: a primeira versao registrou a tarefa, o estado
# ficou "Ready" e o processo continuou pendurado no WmiPrvSE.
#
# ⚠️ E o comentario fica AQUI, acima, e nao entre as linhas do comando. Comentario
# depois de uma continuacao com crase quebra o parser do PowerShell — foi assim que
# a primeira tentativa deste conserto virou erro de sintaxe.
$argumento = "-NoProfile -ExecutionPolicy Bypass -File `"$lancador`" $Fonte -EmPrimeiroPlano"
$acao = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $argumento `
    -WorkingDirectory $raiz

# `-RunLevel Limited`: nao pede privilegio de administrador. Um treino nao
# precisa, e pedir seria escalar permissao sem motivo.
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive `
    -RunLevel Limited

# Sem gatilho de tempo: a tarefa e disparada a mao com `Start-ScheduledTask`.
# Um gatilho recorrente relançaria o treino sozinho e criaria instancias novas
# sobre um treino ja concluido.
#
# `-ExecutionTimeLimit 0` desliga o limite de 72 h do padrao. `-MultipleInstances
# IgnoreNew` e a segunda barreira contra instancia duplicada, alem da trava por PID.
$opcoes = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable

Register-ScheduledTask -TaskName $nome -Action $acao -Principal $principal `
    -Settings $opcoes -Description "PhiFM: $Fonte (ver scripts/tarefa_agendada.ps1)" `
    -Force | Out-Null

Start-ScheduledTask -TaskName $nome
Start-Sleep -Seconds 3
$info = Get-ScheduledTaskInfo -TaskName $nome
Write-Output "tarefa '$nome' registrada e iniciada"
Write-Output "  estado: $((Get-ScheduledTask -TaskName $nome).State)"
Write-Output "  ultima execucao: $($info.LastRunTime)"
Write-Output ""
Write-Output "  remover:  .\scripts\tarefa_agendada.ps1 -Fonte $Fonte -Remover"
Write-Output "  listar :  .\scripts\tarefa_agendada.ps1 -Listar"
