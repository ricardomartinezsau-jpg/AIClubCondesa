@echo off
title Visor de libretas - Cohorte 2
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo No encuentro el entorno local ^(.venv^). Crealo una sola vez con:
    echo     python -m venv .venv
    echo     .venv\Scripts\python -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo ==============================================
echo   Visor de libretas - Cohorte 2
echo   Se abre en el navegador: http://localhost:8501
echo   Para apagarlo, cierra esta ventana.
echo ==============================================
"%PY%" -m streamlit run app.py --server.port 8501
pause
