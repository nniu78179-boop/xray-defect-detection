@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
set "ROOT=%~dp0.."
set "PYTHON=%ROOT%\.venv\python.exe"
set "LOG=%ROOT%startup_error.log"

if not exist "%PYTHON%" (
    echo [f2] ERROR: python.exe not found: %PYTHON% >> "%LOG%" 2>&1
    echo Portable Python environment not found.
    echo Expected: %PYTHON%
    pause
    exit /b 1
)

set "KMP_DUPLICATE_LIB_OK=TRUE"
cd /d "%~dp0"

echo [f2] Starting crop program >> "%LOG%" 2>&1
"%PYTHON%" adaptive_center_crop.py ^
  --input-dir ..\input ^
  --output-dir ..\output ^
  --target-ratio keep ^
  --padding 0.10 ^
  --keep-x 0.995 ^
  --keep-y 0.995 ^
  --bg-threshold-ratio 0.72

set EXIT_CODE=!errorlevel!
echo [f2] Crop program exited, code: !EXIT_CODE! >> "%LOG%" 2>&1
exit /b !EXIT_CODE!
