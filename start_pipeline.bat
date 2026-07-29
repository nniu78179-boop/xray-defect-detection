@echo off
chcp 65001 >nul
set "ROOT=%~dp0"
set "LOG=%ROOT%startup_error.log"

echo ============================== > "%LOG%"
echo Startup Log >> "%LOG%"
echo %date% %time% >> "%LOG%"
echo Path: %ROOT% >> "%LOG%"
echo ============================== >> "%LOG%"

set "PYTHON=%ROOT%.venv\python.exe"
set "ENV_TAR=%ROOT%yolo_env.tar.gz"

if not exist "%PYTHON%" (
    if exist "%ENV_TAR%" (
        echo .venv not found, extracting yolo_env.tar.gz ...
        echo This may take a few minutes. Please wait.
        md ".venv" 2>nul
        tar -xzf "%ENV_TAR%" -C "%ROOT%.venv" >> "%LOG%" 2>&1
        if errorlevel 1 (
            echo [FAIL] Failed to extract yolo_env.tar.gz >> "%LOG%"
            echo Extraction failed. Try extracting manually:
            echo   1. Delete the .venv folder
            echo   2. Use 7-Zip to extract yolo_env.tar.gz
            pause
            exit /b 1
        )
        echo [OK] yolo_env.tar.gz extracted >> "%LOG%"
        if exist "%ROOT%.venv\Scripts\conda-unpack.exe" (
            echo Running conda-unpack to fix paths...
            "%ROOT%.venv\Scripts\conda-unpack.exe" >> "%LOG%" 2>&1
            if errorlevel 1 (
                echo [WARN] conda-unpack reported issues >> "%LOG%"
            ) else (
                echo [OK] conda-unpack finished >> "%LOG%"
            )
        )
    )
)

if not exist "%PYTHON%" (
    echo [FAIL] python.exe not found >> "%LOG%"
    echo Expected: %PYTHON%
    echo.
    echo Make sure .venv is properly extracted.
    pause
    exit /b 1
)
echo [OK] python.exe found >> "%LOG%"

echo [2] Check key files >> "%LOG%"
if exist "%ROOT%f3-yolo\app.py" (echo [OK] f3-yolo\app.py >> "%LOG%") else (echo [FAIL] f3-yolo\app.py >> "%LOG%")
if exist "%ROOT%f2\adaptive_center_crop.py" (echo [OK] f2\adaptive_center_crop.py >> "%LOG%") else (echo [FAIL] f2\adaptive_center_crop.py >> "%LOG%")
if exist "%ROOT%input" (echo [OK] input >> "%LOG%") else (echo [FAIL] input >> "%LOG%")
if exist "%ROOT%output" (echo [OK] output >> "%LOG%") else (echo [FAIL] output >> "%LOG%")
if exist "%ROOT%save" (echo [OK] save >> "%LOG%") else (echo [FAIL] save >> "%LOG%")

echo [3] Test Python >> "%LOG%"
"%PYTHON%" --version >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [FAIL] Python cannot run >> "%LOG%"
    echo Python cannot run. .venv may be corrupted.
    pause
    exit /b 1
)
echo [OK] Python works >> "%LOG%"

echo [4] Test dependencies >> "%LOG%"
"%PYTHON%" -c "import streamlit; import ultralytics; import torch; import cv2; import pydicom; print('OK')" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [FAIL] Missing packages >> "%LOG%"
    echo Some Python packages are missing.
    pause
    exit /b 1
)
echo [OK] All packages OK >> "%LOG%"

echo [5] Start Streamlit >> "%LOG%"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
set "STREAMLIT_CONSOLE_LOGGING=minimal"
cd /d "%ROOT%f3-yolo"

echo ==============================
echo   X-Ray Defect Detection
echo ==============================
echo Starting...
echo Open http://localhost:8501
echo Press Ctrl+C to stop
echo ==============================

echo. | "%PYTHON%" -m streamlit run app.py >> "%LOG%" 2>&1

echo Streamlit exit code: %errorlevel% >> "%LOG%" 2>&1
echo.
echo Program exited. See startup_error.log for details.
pause
