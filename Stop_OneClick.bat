@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
echo [INFO] Workspace: %CD%

echo [INFO] Stopping API on port 8000 (if running)...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"

echo [INFO] Stopping LLM on port 8080 (if running)...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"

echo [INFO] Stopping leftover llama-server processes...
taskkill /IM llama-server.exe /F >nul 2>&1

echo [DONE] Stop sequence completed.
echo [DONE] If a window stays open, you can still close it manually.
exit /b 0
