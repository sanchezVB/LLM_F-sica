@echo off
REM Envelope para lancar a mineracao destacada da sessao, sem inferno de aspas.
REM
REM ATENCAO: nao escreva o caminho do projeto aqui. O cmd.exe le este arquivo na
REM pagina de codigo OEM, e "LLMFisica" tem acento — o caminho literal virava
REM mojibake e o `cd` falhava em silencio, sem criar nem o log. `%~dp0..` resolve
REM a partir da localizacao deste arquivo e nao tem caractere nenhum fora do ASCII.
pushd "%~dp0.."
".venv-treino\Scripts\python.exe" -u "scripts\minerar_negativos.py" %* > "data\raw\minerar_negativos.log" 2>&1
popd
