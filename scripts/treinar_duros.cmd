@echo off
REM Ver o comentario em minerar.cmd sobre %~dp0.. e a pagina de codigo OEM.
pushd "%~dp0.."
".venv-treino\Scripts\python.exe" -u "scripts\train_embedding.py" %* > "data\raw\harvest_phiemb-duros.log" 2>&1
popd
