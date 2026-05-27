@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
echo [INFO] Workspace: %CD%
goto :main

:resolve_npm
set "NPM_CMD="
where npm.cmd >nul 2>&1
if not errorlevel 1 set "NPM_CMD=npm.cmd"
if not defined NPM_CMD (
  where npm >nul 2>&1
  if not errorlevel 1 set "NPM_CMD=npm"
)
if not defined NPM_CMD (
  echo [ERROR] npm is not installed or not on PATH.
  echo [ERROR] Install Node.js 18+ first, then re-run this script.
  exit /b 1
)
exit /b 0

:resolve_python_bootstrap
set "PYTHON_BOOTSTRAP="
where py >nul 2>&1
if not errorlevel 1 (
  py -3.11 -c "import sys" >nul 2>&1
  if not errorlevel 1 set "PYTHON_BOOTSTRAP=py -3.11"
)
if not defined PYTHON_BOOTSTRAP (
  where py >nul 2>&1
  if not errorlevel 1 (
    py -3 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PYTHON_BOOTSTRAP=py -3"
  )
)
if not defined PYTHON_BOOTSTRAP (
  where python >nul 2>&1
  if not errorlevel 1 (
    python -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PYTHON_BOOTSTRAP=python"
  )
)
if not defined PYTHON_BOOTSTRAP (
  for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Python\pythoncore-3.11-64\python.exe"
    "%LOCALAPPDATA%\Python\pythoncore-3.12-64\python.exe"
    "%LOCALAPPDATA%\Python\pythoncore-3.13-64\python.exe"
    "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
  ) do (
    if not defined PYTHON_BOOTSTRAP if exist %%~P set "PYTHON_BOOTSTRAP=%%~fP"
  )
)
exit /b 0

:ensure_working_venv
if not exist ".venv\Scripts\python.exe" exit /b 0
".\.venv\Scripts\python.exe" -c "import sys" >nul 2>&1
if not errorlevel 1 exit /b 0
echo [WARN] Existing .venv is not usable on this machine. Recreating...
if exist ".venv.broken" rmdir /s /q ".venv.broken" >nul 2>&1
move ".venv" ".venv.broken" >nul 2>&1
if exist ".venv\Scripts\python.exe" (
  echo [ERROR] Failed to replace broken .venv automatically.
  exit /b 1
)
exit /b 0

:ensure_desktop_dependencies
if exist "desktop_app\node_modules\electron\dist\electron.exe" (
  exit /b 0
)
echo [INFO] Installing desktop app npm dependencies...
pushd "desktop_app" >nul
call %NPM_CMD% install
set "NPM_EXIT=%ERRORLEVEL%"
popd >nul
if not "%NPM_EXIT%"=="0" (
  echo [ERROR] npm install failed for desktop_app.
  exit /b 1
)
exit /b 0

:build_desktop_app
set "SKIP_BUILD=0"
powershell -NoProfile -Command ^
  "if(Test-Path 'desktop_app\dist\index.html'){$age=(Get-Date)-(Get-Item 'desktop_app\dist\index.html').LastWriteTime;if($age.TotalMinutes -lt 10){exit 0}}; exit 1" >nul 2>&1
if not errorlevel 1 (
  echo [INFO] Frontend build is fresh, skipping rebuild.
  set "SKIP_BUILD=1"
  exit /b 0
)
echo [INFO] Building desktop renderer...
pushd "desktop_app" >nul
call %NPM_CMD% run build
set "NPM_EXIT=%ERRORLEVEL%"
popd >nul
if not "%NPM_EXIT%"=="0" (
  echo [ERROR] npm run build failed for desktop_app.
  exit /b 1
)
exit /b 0

:launch_desktop_app
if not exist "desktop_app\node_modules\electron\dist\electron.exe" (
  echo [ERROR] Electron executable is missing after npm install.
  exit /b 1
)
echo [INFO] Opening desktop app window...
pushd "desktop_app" >nul
start "" "%CD%\node_modules\electron\dist\electron.exe" .
set "START_EXIT=%ERRORLEVEL%"
popd >nul
if not "%START_EXIT%"=="0" (
  echo [ERROR] Failed to launch desktop app.
  exit /b 1
)
exit /b 0

:resolve_llm_assets
set "LLM_START_ENABLED=0"
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "$server=$env:LLAMA_SERVER_EXE; if(-not $server -or -not (Test-Path $server)){ foreach($candidate in @((Join-Path (Get-Location) 'tools\llama.cpp\llama-server.exe'),(Join-Path (Get-Location) 'tools\llama-server.exe'),'C:\tools\llama.cpp\llama-server.exe',(Join-Path $env:USERPROFILE '.ai-navigator\micromamba\envs\cuda\Library\bin\llama-server.exe'))){ if(Test-Path $candidate){ $server=$candidate; break } }; if(-not $server){ $cmd = Get-Command 'llama-server.exe' -ErrorAction SilentlyContinue; if($cmd){ $server=$cmd.Source } } }; $model=$env:GGUF_MODEL_PATH; if(-not $model -or -not (Test-Path $model)){ $patterns=@((Join-Path (Get-Location) 'models\*.gguf'),(Join-Path (Get-Location) 'models\*\*.gguf'),(Join-Path (Get-Location) '*.gguf'),'C:\models\Qwen2.5-7B-Instruct-GGUF\*.gguf'); foreach($pattern in $patterns){ $matches=Get-ChildItem -Path $pattern -File -ErrorAction SilentlyContinue | Sort-Object Length -Descending; if($matches){ $preferred=$matches | Where-Object { $_.Name -notmatch '-\d{5}-of-\d{5}\.gguf$' } | Select-Object -First 1; if($preferred){ $model=$preferred.FullName } else { $model=$matches[0].FullName }; break } } }; if($server -and (Test-Path $server) -and $model -and (Test-Path $model)){ '1' } else { '0' }"` ) do set "LLM_START_ENABLED=%%I"
exit /b 0

:resolve_case_roots
set "AI_CONTEXTS_ROOT=%CD%\outputs\cases"
if exist "%CD%\outputs\runs_first10_reportllm_relaxed_v1" (
  set "AI_RUNS_ROOT=%CD%\outputs\runs_first10_reportllm_relaxed_v1"
) else (
  set "AI_RUNS_ROOT=%CD%\outputs\runs"
)
exit /b 0

:stop_all_on_port_8000
echo [INFO] Stopping ALL processes on port 8000...
powershell -NoProfile -Command ^
  "Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"
timeout /t 2 /nobreak >nul
exit /b 0

:main
set "START_MODE=%~1"
set "OPEN_API_DOCS=1"
set "OPEN_DESKTOP_APP=0"
if /I "%START_MODE%"=="desktop" (
  set "OPEN_API_DOCS=0"
  set "OPEN_DESKTOP_APP=1"
)

set "LLM_START_ENABLED=0"
call :resolve_llm_assets

set "PYTHON_BOOTSTRAP="
call :resolve_python_bootstrap
if not defined PYTHON_BOOTSTRAP (
  echo [ERROR] Python is not installed or not on PATH.
  echo [ERROR] Install Python 3.11+ first, then re-run this script.
  exit /b 1
)

call :ensure_working_venv
if errorlevel 1 exit /b 1

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating virtual environment .venv ...
  %PYTHON_BOOTSTRAP% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create .venv.
    exit /b 1
  )
)

echo [INFO] Checking required Python packages...
".\.venv\Scripts\python.exe" -c "import fastapi,uvicorn,pydantic,rapidfuzz,spellchecker,numpy,nibabel,SimpleITK,pydicom" >nul 2>&1
if errorlevel 1 (
  echo [INFO] Installing requirements, first run may take time...
  ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
  if errorlevel 1 exit /b 1
  ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 exit /b 1
  ".\.venv\Scripts\python.exe" -m pip install -r requirements-vision.txt
  if errorlevel 1 exit /b 1
)

if "%LLM_START_ENABLED%"=="1" (
  set "AI_USE_LLM=1"
  set "AI_REPORT_USE_LLM=1"
  set "AI_QA_USE_LLM_NONSTRUCTURED=1"
  set "AI_LLM_PROVIDER=embedded_llama"
  set "AI_LLAMACPP_URL=http://127.0.0.1:8080"
  echo [INFO] Starting embedded llama.cpp server...
  powershell -NoProfile -Command "$args = @('-NoExit','-ExecutionPolicy','Bypass','-File','%CD%\Start_Embedded_LlamaCpp.ps1'); if($env:LLAMA_SERVER_EXE){$args += @('-LlamaServerExe',$env:LLAMA_SERVER_EXE)}; if($env:GGUF_MODEL_PATH){$args += @('-ModelPath',$env:GGUF_MODEL_PATH)}; Start-Process -FilePath 'powershell.exe' -ArgumentList $args -WindowStyle Normal"

  if "%OPEN_DESKTOP_APP%"=="1" (
    echo [INFO] Desktop mode: not blocking on LLM health.
  ) else (
    echo [INFO] Waiting for LLM health at http://127.0.0.1:8080/health ...
    powershell -NoProfile -Command "$ok=$false; for($i=0;$i -lt 20;$i++){ try { $r=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health -TimeoutSec 2; if($r.StatusCode -eq 200){$ok=$true; break} } catch {}; Start-Sleep -Seconds 2 }; if($ok){ exit 0 } else { exit 1 }"
    if errorlevel 1 echo [WARN] LLM health check not ready yet. Continuing...
  )
) else (
  set "AI_USE_LLM=0"
  set "AI_REPORT_USE_LLM=0"
  set "AI_QA_USE_LLM_NONSTRUCTURED=0"
  echo [WARN] llama.cpp server or GGUF model not found. Running without LLM.
)

call :resolve_case_roots

echo [INFO] AI_CONTEXTS_ROOT = %AI_CONTEXTS_ROOT%
echo [INFO] AI_RUNS_ROOT     = %AI_RUNS_ROOT%

if not exist "%AI_CONTEXTS_ROOT%" (
  echo [WARN] Case directory not found: %AI_CONTEXTS_ROOT%
)
if not exist "%AI_RUNS_ROOT%" (
  echo [WARN] Runs directory not found: %AI_RUNS_ROOT%
)

call :stop_all_on_port_8000

echo [INFO] Starting API server on http://127.0.0.1:8000 ...
powershell -NoProfile -Command ^
  "$ws  = '%CD%';" ^
  "$ctx = '%AI_CONTEXTS_ROOT%';" ^
  "$rns = '%AI_RUNS_ROOT%';" ^
  "$py  = Join-Path $ws '.venv\Scripts\python.exe';" ^
  "$env:AI_CONTEXTS_ROOT = $ctx;" ^
  "$env:AI_RUNS_ROOT     = $rns;" ^
  "$env:AI_USE_LLM                    = '%AI_USE_LLM%';" ^
  "$env:AI_REPORT_USE_LLM             = '%AI_REPORT_USE_LLM%';" ^
  "$env:AI_QA_USE_LLM_NONSTRUCTURED   = '%AI_QA_USE_LLM_NONSTRUCTURED%';" ^
  "$env:AI_WORKSPACE_ROOT             = $ws;" ^
  "if ('%AI_LLM_PROVIDER%' -ne '') { $env:AI_LLM_PROVIDER  = '%AI_LLM_PROVIDER%' };" ^
  "if ('%AI_LLAMACPP_URL%' -ne '') { $env:AI_LLAMACPP_URL   = '%AI_LLAMACPP_URL%' };" ^
  "Start-Process -FilePath $py -ArgumentList @('-m','uvicorn','ai_assistant.api.main:app','--host','127.0.0.1','--port','8000') -WorkingDirectory $ws -WindowStyle Hidden"

echo [INFO] Waiting for API to be healthy...
powershell -NoProfile -Command ^
  "$ok=$false; for($i=0;$i -lt 40;$i++){ try { $r=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health -TimeoutSec 2; if($r.StatusCode -eq 200){$ok=$true; break} } catch {}; Start-Sleep -Seconds 1 }; if($ok){ Write-Host '[INFO] API is healthy.'; exit 0 } else { Write-Host '[WARN] API health check timed out.'; exit 1 }"
if errorlevel 1 (
  echo [WARN] API may not be ready. Continuing anyway...
)

echo [INFO] Waiting for case data to be indexed...
timeout /t 3 /nobreak >nul

if "%OPEN_DESKTOP_APP%"=="1" (
  call :resolve_npm
  if errorlevel 1 exit /b 1
  call :ensure_desktop_dependencies
  if errorlevel 1 exit /b 1
  call :build_desktop_app
  if errorlevel 1 exit /b 1
  call :launch_desktop_app
  if errorlevel 1 exit /b 1
  echo [DONE] Desktop app started. To stop background services, run Stop_OneClick.bat
) else (
  if "%OPEN_API_DOCS%"=="1" (
    echo [INFO] Opening API docs...
    start "" http://127.0.0.1:8000/docs
  )
  echo [DONE] One-click startup completed.
)

exit /b 0