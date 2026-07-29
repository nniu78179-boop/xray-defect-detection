@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
set "ROOT=%~dp0.."
set "PYTHON=%ROOT%\.venv\python.exe"
set "LOG=%ROOT%startup_error.log"

if not exist "%PYTHON%" (
    echo [run_app] ERROR: python.exe not found: %PYTHON% >> "%LOG%" 2>&1
    echo Portable Python environment not found.
    echo Expected: %PYTHON%
    pause
    exit /b 1
)

set "KMP_DUPLICATE_LIB_OK=TRUE"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
set "STREAMLIT_CONSOLE_LOGGING=minimal"
cd /d "%~dp0"

echo [run_app] Starting Streamlit >> "%LOG%" 2>&1
echo. | "%PYTHON%" -m streamlit run app.py

set EXIT_CODE=!errorlevel!
echo [run_app] Streamlit exited, code: !EXIT_CODE! >> "%LOG%" 2>&1
pause
