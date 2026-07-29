@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0..
set KMP_DUPLICATE_LIB_OK=TRUE
set CUDA_VISIBLE_DEVICES=-1
echo Starting X-Ray Defect Detection desktop app...
"%~dp0..\.venv\python.exe" "%~dp0desktop_app.py"
pause
