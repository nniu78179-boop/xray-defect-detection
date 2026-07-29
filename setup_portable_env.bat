@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
set "CONDA=D:\gzc\condaa\Scripts\conda.exe"
set "ENV_NAME=yolo"
set "OUTPUT=%ROOT%yolo_env.tar.gz"

if not exist "%CONDA%" (
    echo [FAIL] conda.exe not found at: %CONDA%
    echo Edit this script and set CONDA to your conda.exe path.
    pause
    exit /b 1
)

echo ============================================
echo  Portable Environment Packer (conda-pack)
echo ============================================
echo.
echo This script packs the '%ENV_NAME%' conda environment
echo into a portable archive: yolo_env.tar.gz
echo.
echo Conda: %CONDA%
echo Environment: %ENV_NAME%
echo Output: %OUTPUT%
echo.

"%CONDA%" install conda-pack -c conda-forge -y
if errorlevel 1 (
    echo [FAIL] Failed to install conda-pack
    pause
    exit /b 1
)

if exist "%OUTPUT%" (
    echo Previous archive found, deleting...
    del "%OUTPUT%"
)

echo Packaging environment '%ENV_NAME%'...
echo This may take several minutes...
"%CONDA%" pack -n "%ENV_NAME%" -o "%OUTPUT%" --force --ignore-missing-files
if errorlevel 1 (
    echo [FAIL] conda-pack failed
    pause
    exit /b 1
)

echo.
echo [OK] Packing complete!
echo.
echo Archive created: %OUTPUT%
echo.
echo ---- Deploy Instructions ----
echo 1. Copy the entire project folder to the target machine
echo 2. Double-click diagnose.bat or start_pipeline.bat
echo    The script will auto-extract yolo_env.tar.gz to .venv
echo.
pause
