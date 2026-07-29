@echo off
chcp 65001 >nul
set "ROOT=%~dp0"

echo =========== Environment Test ===========
echo.
echo [1] Script location: %ROOT%
echo.

set "PYTHON=%ROOT%.venv\python.exe"
set "ENV_TAR=%ROOT%yolo_env.tar.gz"

if not exist "%PYTHON%" (
    if exist "%ENV_TAR%" (
        echo [*] Auto-extracting yolo_env.tar.gz ...
        echo     This may take a few minutes. Please wait.
        md ".venv" 2>nul
        tar -xzf "%ENV_TAR%" -C "%ROOT%.venv" 2>&1
        if errorlevel 1 (
            echo     [FAIL] Extraction failed.
            pause
            exit /b 1
        )
        if exist "%ROOT%.venv\Scripts\conda-unpack.exe" (
            echo     Running conda-unpack to fix paths...
            "%ROOT%.venv\Scripts\conda-unpack.exe" >nul 2>&1
        )
        echo     [OK] Extracted successfully
        echo.
    )
)

echo [2] Check python.exe ...
if not exist "%PYTHON%" (
    echo    [FAIL] python.exe not found
    echo    Expected: %PYTHON%
    echo.
    echo    Current folder contents:
    dir /b "%ROOT%"
    pause
    exit /b 1
)
echo    [OK] python.exe found
echo.
echo [3] Python version:
"%PYTHON%" --version
if errorlevel 1 (
    echo    [FAIL] Python cannot start
    pause
    exit /b 1
)
echo.
echo [4] Test key packages:
echo    testing streamlit ...
"%PYTHON%" -c "import streamlit; print('       [OK] streamlit', streamlit.__version__)"
echo    testing ultralytics ...
"%PYTHON%" -c "import ultralytics; print('       [OK] ultralytics', ultralytics.__version__)"
echo    testing torch ...
"%PYTHON%" -c "import torch; print('       [OK] torch', torch.__version__)"
echo    testing cv2 ...
"%PYTHON%" -c "import cv2; print('       [OK] opencv-python', cv2.__version__)"
echo    testing pydicom ...
"%PYTHON%" -c "import pydicom; print('       [OK] pydicom', pydicom.__version__)"
echo    testing numpy ...
"%PYTHON%" -c "import numpy; print('       [OK] numpy', numpy.__version__)"
echo.
echo =========== Test Complete ===========
echo.
echo If all [OK], you can now run start_pipeline.bat
pause
