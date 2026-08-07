# Espera o snapshot fechar, reconstroi os pares e dispara o treino do ΦEmb.
#
#   .\scripts\encadear_phiemb.ps1
#
# ─── Por que existe ───────────────────────────────────────────────────────
#
# O treino precisa dos pares COMPLETOS, e os pares dependem do snapshot, que
# fecha em algumas horas. Fazer a emenda a mao exige alguem acordado no momento
# certo; este script faz a mao ficar dispensavel — e sobrevive ao fim da sessao
# do agente, que e o ponto.
#
# Nao decide nada: a decisao de treinar local, com todos os pares, ja foi
# tomada. Ele so garante que a ordem seja respeitada.
#
# ─── A ordem importa e nao e negociavel ──────────────────────────────────
#
#   1. snapshot concluido      -> senao os pares saem incompletos
#   2. espinha reconstruida    -> para casar o grafo completo com titulo/resumo
#   3. pares reconstruidos     -> ~2,8 M em vez de 1,65 M
#   4. treino disparado        -> sob supervisor, porque leva dias
#
# Reconstruir a espinha antes dos pares nao e detalhe: `attach_citations` junta
# o grafo a espinha, e um grafo maior sem espinha atualizada joga fora as
# citacoes novas em silencio.

param([int]$IntervaloSegundos = 300)

$ErrorActionPreference = 'Stop'
$raiz = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $raiz

$log = Join-Path $raiz 'data\raw\encadear_phiemb.log'
$py  = Join-Path $raiz '.venv\Scripts\python.exe'          # 3.14: dados
$pyT = Join-Path $raiz '.venv-treino\Scripts\python.exe'   # 3.12: treino
$manifesto = Join-Path $raiz 'data\raw\openalex_snapshot\_manifest.json'

function Registrar($t) {
    $linha = "{0}  {1}" -f (Get-Date -Format 'HH:mm:ss'), $t
    Add-Content -Path $log -Value $linha -Encoding utf8
    Write-Output $linha
}

Registrar 'aguardando o snapshot concluir'

while ($true) {
    if (Test-Path $manifesto) {
        $m = Get-Content $manifesto -Raw -Encoding utf8 | ConvertFrom-Json
        if ($m.completed_at) {
            Registrar "snapshot concluido: $($m.actual_count) obras, $($m.failures.Count) falhas"
            break
        }
    }
    Start-Sleep -Seconds $IntervaloSegundos
}

$env:PYTHONPATH = 'src'
$env:PYTHONUTF8 = '1'

Registrar 'reconstruindo a espinha com o grafo completo'
& $py 'scripts\build_spine.py' --arxiv 'data/raw/arxiv_metadata' `
      --openalex 'data/raw/openalex_snapshot' --out 'data/processed/spine.parquet' `
      *>> $log
if ($LASTEXITCODE -ne 0) { Registrar 'FALHOU na espinha — treino nao disparado'; exit 1 }

Registrar 'reconstruindo os pares de citacao'
& $py 'scripts\build_pairs.py' *>> $log
if ($LASTEXITCODE -ne 0) { Registrar 'FALHOU nos pares — treino nao disparado'; exit 1 }

# Pares novos invalidam o estado antigo: retomar do passo N com outra ordem de
# lotes significaria pular pares que nunca foram vistos. Comeca limpo.
$estado = Join-Path $raiz 'models\phiemb\estado_treino.pt'
if (Test-Path $estado) {
    Registrar 'descartando estado anterior — os pares mudaram, a ordem dos lotes tambem'
    Remove-Item $estado -Force
}

Registrar 'disparando o treino sob supervisor'
Start-Process powershell -ArgumentList '-NoProfile','-File',
    (Join-Path $raiz 'scripts\supervisor_coleta.ps1'),'phiemb' -WindowStyle Hidden | Out-Null

Start-Sleep -Seconds 90
if (Test-Path (Join-Path $raiz 'data\raw\.trava_phiemb')) {
    Registrar 'treino em execucao — acompanhar data\raw\harvest_phiemb.log'
} else {
    Registrar 'ATENCAO: trava do treino ausente apos 90 s — verificar o log'
}
