# Configura somente a sessao corrente; -Instalar e explicitamente opcional.
# Exemplo (depois de confirmar que D: e o HD):
# . .\scripts\ambiente_hd.ps1 -RaizHD 'D:\LLMFisica'
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RaizHD,
    [string]$Python = 'python',
    [switch]$Instalar
)

$ErrorActionPreference = 'Stop'
$repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$helper = Join-Path $repo 'src\phifm\core\armazenamento.py'

# -B impede cache .pyc antes de aplicar o destino do HD; -I ignora PYTHONPATH herdado.
$version = & $Python -B -I -c 'import sys; print("%d.%d" % sys.version_info[:2])'
if ($LASTEXITCODE -ne 0 -or $version -ne '3.12') {
    throw 'Informe em -Python um interpretador Python 3.12 ja instalado.'
}
$planoJson = & $Python -B -I $helper --raiz-hd $RaizHD --repositorio $repo --criar
if ($LASTEXITCODE -ne 0) { throw 'Armazenamento recusado. Nenhuma instalacao foi iniciada.' }
$plano = $planoJson | ConvertFrom-Json
foreach ($nome in $plano.remover_variaveis) {
    [Environment]::SetEnvironmentVariable($nome, $null, 'Process')
}
foreach ($item in $plano.variaveis.PSObject.Properties) {
    [Environment]::SetEnvironmentVariable($item.Name, [string]$item.Value, 'Process')
}
Set-Location $plano.repositorio

if ($Instalar) {
    $cfg = Join-Path $plano.venv 'pyvenv.cfg'
    if (-not (Test-Path -LiteralPath $cfg)) {
        if ((Test-Path -LiteralPath $plano.venv) -and
            @(Get-ChildItem -LiteralPath $plano.venv -Force).Count -gt 0) {
            throw 'A pasta da venv existe e nao e uma venv reconhecida. Preserve-a antes de continuar.'
        }
        & $Python -B -I -m venv $plano.venv
        if ($LASTEXITCODE -ne 0) { throw 'Falha ao criar a venv no HD.' }
    }
    $venvPython = Join-Path $plano.venv 'Scripts\python.exe'
    & $venvPython -B -I -c 'import pathlib,sys; assert sys.version_info[:2] == (3,12); assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(sys.argv[1]).resolve(); assert sys.prefix != sys.base_prefix' $plano.venv
    if ($LASTEXITCODE -ne 0) { throw 'A venv existente nao corresponde ao destino e versao esperados.' }
    & $venvPython -m pip install --index-url 'https://download.pytorch.org/whl/cpu' --only-binary=:all: 'torch==2.4.1+cpu'
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar Torch CPU com verificacao TLS.' }
    & $venvPython -m pip install -r (Join-Path $repo 'requirements-treino-cpu.lock')
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar o lock de treino CPU.' }
    & $venvPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependencias inconsistentes na venv CPU.' }
}

Write-Host "Destinos de instalacao, downloads e caches preparados em: $($plano.raiz_hd)"
Write-Host 'A identificacao fisica como HD e fornecida por voce; o script verifica os caminhos.'
Write-Host "Venv CPU separada: $($plano.venv)"
Write-Host 'Use esta mesma janela para os comandos seguintes. Argumentos de saida devem ficar dentro dessa raiz.'
