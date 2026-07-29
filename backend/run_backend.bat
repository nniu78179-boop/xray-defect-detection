@echo off
set "ROOT=%~dp0.."
set "PYTHON=%ROOT%\.venv\python.exe"
cd /d "%ROOT%\backend"

echo ================================
echo   X-Ray Defect Detection API
echo ================================
echo.
echo [1] Starting API server (model loading ...)
echo.

start "API-Server" "%PYTHON%" -m uvicorn main:app --host 0.0.0.0 --port 8000

echo [2] Waiting for server to be ready ...
echo     Will open browser automatically soon.

:wait
ping -n 2 127.0.0.1 >nul
"%PYTHON%" -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" 2>nul
if errorlevel 1 goto wait

echo [3] Server ready, opening browser ...
start http://localhost:8000

echo.
echo ================================
echo   Browser opened successfully!
echo   API docs: http://localhost:8000/docs
echo   Close this window to stop.
echo ================================
pause
