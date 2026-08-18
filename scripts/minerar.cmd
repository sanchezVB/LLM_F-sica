@echo off
REM Envelope para lancar a mineracao destacada da sessao, sem inferno de aspas.
REM
REM ATENCAO: nao escreva o caminho do projeto aqui. O cmd.exe le este arquivo na
REM pagina de codigo OEM, e "LLMFisica" tem acento — o caminho literal virava
REM mojibake e o `cd` falhava em silencio, sem criar nem o log. `%~dp0..` resolve
REM a partir da localizacao deste arquivo e nao tem caractere nenhum fora do ASCII.
REM UTF-8 na saida. O cmd redireciona na pagina de codigo OEM (cp1252 aqui), e
REM caracteres como * (U+2605) e -> (U+2192) nao existem nela: o `logging` do
REM Python levanta no handler, engole a excecao e PERDE a linha. Foi assim que a
REM linha do melhor checkpoint desapareceu do log em 2026-08-18 — o trabalho foi
REM feito, o registro dele nao.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
pushd "%~dp0.."
".venv-treino\Scripts\python.exe" -u "scripts\minerar_negativos.py" %* > "data\raw\minerar_negativos.log" 2>&1
popd
