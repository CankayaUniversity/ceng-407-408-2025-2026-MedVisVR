@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1

cd /d "%~dp0"
set "APP_ROOT=%CD%"

cls
echo.
echo  ============================================================
echo   CARVIS Clinical AI Workspace -- First-Time Setup
echo  ============================================================
echo   This script will:
echo     [1] Verify or auto-install Python 3.11+
echo     [2] Verify or auto-install Node.js 18+
echo     [3] Create Python virtual environment
echo     [4] Install all Python dependencies
echo     [5] GPU detection + CUDA torch (if NVIDIA GPU found)
echo     [6] Install npm packages and build the desktop app
echo     [7] Check / download LLM model (Qwen2.5-7B GGUF)
echo     [8] Run verification tests
echo     [9] Create desktop shortcut
echo  ============================================================
echo.
echo   Workspace: %APP_ROOT%
echo.
pause

echo.
echo  -- [1/9] Python 3.11+ -------------------------------------------
echo.

set "PYTHON_BOOTSTRAP="
call :resolve_python_bootstrap
if not defined PYTHON_BOOTSTRAP (
  echo  [INFO] Python 3.11+ not found. Attempting automatic installation...
  set "WINGET_AVAIL=0"
  where winget >nul 2>&1
  if not errorlevel 1 set "WINGET_AVAIL=1"
  if "!WINGET_AVAIL!"=="1" (
    echo  [INFO] Installing Python 3.11 via winget...
    winget install --id Python.Python.3.11 -e --silent --accept-package-agreements --accept-source-agreements
    call :refresh_path
    call :resolve_python_bootstrap
  )
  if not defined PYTHON_BOOTSTRAP (
    echo  [INFO] Downloading Python 3.11.9 installer...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile '%TEMP%\python_setup.exe'"
    if exist "%TEMP%\python_setup.exe" (
      echo  [INFO] Running Python installer silently...
      "%TEMP%\python_setup.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
      call :refresh_path
      call :resolve_python_bootstrap
    )
  )
  if not defined PYTHON_BOOTSTRAP (
    echo  [ERROR] Python 3.11+ could not be installed automatically.
    echo         Please install manually: https://www.python.org/downloads/
    echo         Enable "Add Python to PATH" during setup, then re-run this script.
    pause & exit /b 1
  )
  echo  [OK] Python installed successfully.
)
echo  [OK] Python: %PYTHON_BOOTSTRAP%

echo.
echo  -- [2/9] Node.js 18+ --------------------------------------------
echo.

set "NPM_CMD="
set "NODE_OK=0"
where npm.cmd >nul 2>&1
if not errorlevel 1 ( set "NPM_CMD=npm.cmd" & set "NODE_OK=1" )
if "!NODE_OK!"=="0" (
  where npm >nul 2>&1
  if not errorlevel 1 ( set "NPM_CMD=npm" & set "NODE_OK=1" )
)
if "!NODE_OK!"=="0" (
  echo  [INFO] Node.js not found. Attempting automatic installation...
  set "WINGET_AVAIL=0"
  where winget >nul 2>&1
  if not errorlevel 1 set "WINGET_AVAIL=1"
  if "!WINGET_AVAIL!"=="1" (
    echo  [INFO] Installing Node.js LTS via winget...
    winget install --id OpenJS.NodeJS.LTS -e --silent --accept-package-agreements --accept-source-agreements
    call :refresh_path
  )
  if "!WINGET_AVAIL!"=="0" (
    echo  [INFO] Downloading Node.js 20 LTS installer...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://nodejs.org/dist/v20.18.0/node-v20.18.0-x64.msi' -OutFile '%TEMP%\node_setup.msi'"
    if exist "%TEMP%\node_setup.msi" (
      echo  [INFO] Running Node.js installer silently...
      msiexec /i "%TEMP%\node_setup.msi" /quiet /norestart
      call :refresh_path
    )
  )
  set "NPM_CMD="
  set "NODE_OK=0"
  where npm.cmd >nul 2>&1
  if not errorlevel 1 ( set "NPM_CMD=npm.cmd" & set "NODE_OK=1" )
  if "!NODE_OK!"=="0" (
    where npm >nul 2>&1
    if not errorlevel 1 ( set "NPM_CMD=npm" & set "NODE_OK=1" )
  )
  if "!NODE_OK!"=="0" (
    for %%P in (
      "%ProgramFiles%\nodejs\npm.cmd"
      "%APPDATA%\npm\npm.cmd"
    ) do (
      if "!NODE_OK!"=="0" if exist "%%~P" ( set "NPM_CMD=%%~fP" & set "NODE_OK=1" )
    )
  )
  if "!NODE_OK!"=="0" (
    echo  [ERROR] Node.js could not be installed automatically.
    echo         Please install manually: https://nodejs.org/en/download
    echo         Restart this script after installation.
    pause & exit /b 1
  )
  echo  [OK] Node.js installed successfully.
)
for /f "tokens=*" %%V in ('node --version 2^>^&1') do set "NODE_VER=%%V"
echo  [OK] Node.js: !NODE_VER!

echo.
echo  -- [3/9] Python virtual environment ------------------------------
echo.

call :ensure_working_venv
if errorlevel 1 exit /b 1

if not exist ".venv\Scripts\python.exe" (
  echo  [INFO] Creating virtual environment...
  %PYTHON_BOOTSTRAP% -m venv .venv
  if errorlevel 1 (
    echo  [ERROR] Failed to create .venv
    pause & exit /b 1
  )
  echo  [OK] .venv created.
) else (
  echo  [OK] .venv already exists.
)

echo.
echo  -- [4/9] Python dependencies -------------------------------------
echo.

".venv\Scripts\python.exe" -c "import fastapi,uvicorn,pydantic,rapidfuzz,spellchecker,numpy,nibabel,SimpleITK,pydicom" >nul 2>&1
if not errorlevel 1 (
  echo  [OK] Python packages already installed -- skipping.
) else (
  echo  [INFO] Installing Python packages, may take 3-5 minutes...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
  if errorlevel 1 ( echo  [ERROR] pip upgrade failed. & pause & exit /b 1 )

  if not exist "requirements.txt" (
    echo  [ERROR] requirements.txt not found.
    pause & exit /b 1
  )
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 ( echo  [ERROR] requirements.txt failed. & pause & exit /b 1 )

  if exist "requirements-vision.txt" (
    ".venv\Scripts\python.exe" -m pip install -r requirements-vision.txt
    if errorlevel 1 ( echo  [ERROR] requirements-vision.txt failed. & pause & exit /b 1 )
  )
  echo  [OK] Python packages installed.
)

echo.
echo  -- [5/9] GPU detection + CUDA torch --------------------------------
echo.

set "HAS_GPU=0"
set "CUDA_INDEX="
set "GPU_NAME="

powershell -NoProfile -ExecutionPolicy Bypass -File "%APP_ROOT%\_gpu_detect.ps1"

for /f "usebackq tokens=1,* delims==" %%A in ("%TEMP%\carvis_gpu.txt") do (
  if "%%A"=="HAS_GPU"    set "HAS_GPU=%%B"
  if "%%A"=="GPU_NAME"   set "GPU_NAME=%%B"
  if "%%A"=="CUDA_INDEX" set "CUDA_INDEX=%%B"
)

if not "!HAS_GPU!"=="1" goto :no_gpu

echo  [OK] NVIDIA GPU: !GPU_NAME!

".venv\Scripts\python.exe" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 (
  echo  [OK] CUDA torch already active -- skipping.
  goto :gpu_done
)

if "!CUDA_INDEX!"=="" goto :gpu_no_version

echo  [INFO] CPU torch detected, installing CUDA torch (!CUDA_INDEX!) -- this may take a few minutes...
".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/!CUDA_INDEX! --force-reinstall --quiet
if errorlevel 1 (
  echo  [WARN] CUDA torch install failed. Falling back to CPU torch.
) else (
  echo  [OK] CUDA torch installed. GPU segmentation active.
)
goto :gpu_done

:gpu_no_version
echo  [WARN] Could not determine CUDA version. Keeping CPU torch.
goto :gpu_done

:no_gpu
echo  [INFO] No NVIDIA GPU found -- using CPU torch.

:gpu_done

echo.
echo  -- [6/9] Desktop app (npm + build) --------------------------------
echo.

if not exist "desktop_app" (
  echo  [ERROR] desktop_app folder not found.
  pause & exit /b 1
)

pushd "desktop_app" >nul

if exist "node_modules\electron\dist\electron.exe" (
  echo  [OK] npm packages already installed -- skipping npm install.
) else (
  echo  [INFO] Running npm install, first time may take a few minutes...
  call !NPM_CMD! install
  if errorlevel 1 ( popd >nul & echo  [ERROR] npm install failed. & pause & exit /b 1 )
  echo  [OK] npm packages installed.
)

if exist "dist\index.html" (
  echo  [OK] Frontend dist already exists -- skipping build.
) else (
  echo  [INFO] Building frontend...
  call !NPM_CMD! run build
  if errorlevel 1 ( popd >nul & echo  [ERROR] Frontend build failed. & pause & exit /b 1 )
  echo  [OK] Frontend built.
)

popd >nul

echo.
echo  -- [7/9] LLM assets (llama-server + GGUF model) -------------------
echo.

set "FOUND_LLAMA="
if not "%LLAMA_SERVER_EXE%"=="" if exist "%LLAMA_SERVER_EXE%" set "FOUND_LLAMA=%LLAMA_SERVER_EXE%"
if "!FOUND_LLAMA!"=="" (
  for %%P in (
    "%APP_ROOT%\tools\llama.cpp\llama-server.exe"
    "%APP_ROOT%\tools\llama-server.exe"
    "C:\tools\llama.cpp\llama-server.exe"
    "%USERPROFILE%\.ai-navigator\micromamba\envs\cuda\Library\bin\llama-server.exe"
  ) do (
    if "!FOUND_LLAMA!"=="" if exist "%%~P" set "FOUND_LLAMA=%%~fP"
  )
)
if "!FOUND_LLAMA!"=="" (
  for /f "delims=" %%P in ('where.exe llama-server.exe 2^>nul') do (
    if "!FOUND_LLAMA!"=="" set "FOUND_LLAMA=%%P"
  )
)

if "!FOUND_LLAMA!"=="" (
  echo  [WARN] llama-server.exe not found -- app will run without LLM.
  echo         To enable: place llama-server.exe in tools\llama.cpp\
) else (
  echo  [OK] llama-server: !FOUND_LLAMA!
)

set "FOUND_MODEL="
if not "%GGUF_MODEL_PATH%"=="" if exist "%GGUF_MODEL_PATH%" set "FOUND_MODEL=%GGUF_MODEL_PATH%"
if "!FOUND_MODEL!"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%APP_ROOT%\_gguf_find.ps1" -AppRoot "%APP_ROOT%"
  for /f "usebackq delims=" %%F in ("%TEMP%\carvis_gguf.txt") do (
    if not "%%F"=="" if "!FOUND_MODEL!"=="" set "FOUND_MODEL=%%F"
  )
)

if not "!FOUND_MODEL!"=="" goto :gguf_found
echo  [WARN] GGUF model not found.
echo.
if "!FOUND_LLAMA!"=="" goto :gguf_no_llama

echo  [INFO] llama-server found but GGUF model is missing.
echo         Qwen2.5-7B model is required for LLM reporting (~4.5 GB).
echo.
set /p DL_MODEL="  Download GGUF model now? (Y/N): "
if /I "!DL_MODEL!"=="Y" goto :gguf_download
echo  [INFO] Model download skipped.
echo         To download later: run Download_Qwen25_GGUF.ps1
goto :gguf_done

:gguf_no_llama
echo  [INFO] llama-server not found either -- skipping LLM download.
echo         To enable LLM: place llama-server.exe in tools\llama.cpp\
echo         and place a .gguf model in models\
goto :gguf_done

:gguf_download
echo.
echo  [INFO] Downloading GGUF model, this may take 10-30 minutes...
powershell -NoProfile -ExecutionPolicy Bypass -File "%APP_ROOT%\Download_Qwen25_GGUF.ps1"
if errorlevel 1 (
  echo  [WARN] Model download failed. Continuing without LLM.
) else (
  echo  [OK] GGUF model downloaded successfully.
)
goto :gguf_done

:gguf_found
echo  [OK] GGUF model: !FOUND_MODEL!

:gguf_done

echo.
echo  -- [7b/9] nnUNet segmentation models (HuggingFace) ---------------
echo.

set "NNUNET_OK=1"
for %%M in (nnunet_brain nnunet_liver nnunet_lung) do (
  if not exist "%APP_ROOT%\%%M" set "NNUNET_OK=0"
)

if "!NNUNET_OK!"=="1" (
  echo  [OK] nnUNet models already present -- skipping download.
  goto :nnunet_done
)

echo  [INFO] One or more nnUNet models missing.
echo         Downloading from HuggingFace ^(Yusufinhoo/carvis-nnunet-models^)...
echo         This may take 10-20 minutes depending on connection speed.
echo.

".venv\Scripts\python.exe" -c "from huggingface_hub import snapshot_download; import os; missing=[m for m in ['nnunet_brain','nnunet_liver','nnunet_lung'] if not os.path.isdir(m)]; snapshot_download(repo_id='Yusufinhoo/carvis-nnunet-models', local_dir='.', allow_patterns=[f'{m}/**' for m in missing]) if missing else None"
if errorlevel 1 (
  echo  [WARN] nnUNet model download failed. Segmentation will not work.
  echo         Retry later or manually place models in:
  echo           %APP_ROOT%\nnunet_brain
  echo           %APP_ROOT%\nnunet_liver
  echo           %APP_ROOT%\nnunet_lung
) else (
  echo  [OK] nnUNet models downloaded successfully.
)

:nnunet_done

echo.
echo  -- [8/9] Verification + tests ------------------------------------
echo.

if exist "tests\test_report_generator.py" (
  echo  [INFO] Running quick tests...
  ".venv\Scripts\python.exe" -m pytest tests\test_report_generator.py tests\test_qa_engine.py -q --tb=short 2>nul
  if errorlevel 1 (
    echo  [WARN] Some tests failed -- may be normal if case data is missing.
  ) else (
    echo  [OK] Tests passed.
  )
) else (
  echo  [INFO] No tests found -- skipping.
)

echo.
echo  -- [9/9] Desktop shortcut ----------------------------------------
echo.

call :create_shortcut

echo.
echo  ============================================================
echo   [DONE] Installation complete!
echo.
echo   HOW TO LAUNCH:
echo     * Desktop shortcut: "CARVIS"
echo     * Or run: Start_Desktop_App_OneClick.bat
echo.
echo   HOW TO STOP services:
echo     * Run: Stop_OneClick.bat
echo  ============================================================
echo.

set /p LAUNCH="Launch CARVIS now? (Y/N): "
if /I "!LAUNCH!"=="Y" (
  echo.
  echo  [INFO] Starting CARVIS...
  start "" "%APP_ROOT%\Start_Desktop_App_OneClick.bat"
)

pause
exit /b 0


:create_shortcut
set "VBS=%APP_ROOT%\Launch_CARVIS.vbs"

(
  echo Dim scriptDir, batPath
  echo scriptDir = Left^(WScript.ScriptFullName, InStrRev^(WScript.ScriptFullName, "\"^)^)
  echo batPath    = scriptDir ^& "Start_Desktop_App_OneClick.bat"
  echo Dim ws
  echo Set ws = CreateObject^("WScript.Shell"^)
  echo ws.Run Chr^(34^) ^& batPath ^& Chr^(34^), 0, False
) > "%VBS%"

for /f "usebackq delims=" %%D in (
  `powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"`
) do set "REAL_DESKTOP=%%D"

set "ICON_PATH=%APP_ROOT%\desktop_app\build\lumina.ico"
if not exist "!ICON_PATH!" set "ICON_PATH=%APP_ROOT%\desktop_app\node_modules\electron\dist\electron.exe"
if not exist "!ICON_PATH!" set "ICON_PATH="

set "LNK=!REAL_DESKTOP!\CARVIS.lnk"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ws  = New-Object -ComObject WScript.Shell;" ^
  "$lnk = $ws.CreateShortcut('!LNK!');" ^
  "$lnk.TargetPath       = '!VBS!';" ^
  "$lnk.WorkingDirectory = '!APP_ROOT!';" ^
  "$lnk.Description      = 'CARVIS - Clinical Anatomical Rendering Visualization Intelligent System';" ^
  "$lnk.WindowStyle      = 1;" ^
  "if ('!ICON_PATH!' -ne '' -and (Test-Path '!ICON_PATH!')) { $lnk.IconLocation = '!ICON_PATH!,0' };" ^
  "$lnk.Save()" >nul 2>&1

if exist "!LNK!" (
  echo  [OK] Desktop shortcut created: CARVIS
) else (
  echo  [WARN] Shortcut could not be created automatically.
  echo         Right-click this bat and choose "Run as administrator".
)
exit /b 0


:refresh_path
for /f "usebackq tokens=2,*" %%A in (`reg query "HKCU\Environment" /v PATH 2^>nul`) do set "USR_PATH=%%B"
for /f "usebackq tokens=2,*" %%A in (`reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH 2^>nul`) do set "SYS_PATH=%%B"
if defined SYS_PATH if defined USR_PATH set "PATH=!SYS_PATH!;!USR_PATH!"
if defined SYS_PATH if not defined USR_PATH set "PATH=!SYS_PATH!"
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
echo  [WARN] Existing .venv is not usable on this machine. Recreating...
if exist ".venv.broken" rmdir /s /q ".venv.broken" >nul 2>&1
move ".venv" ".venv.broken" >nul 2>&1
if exist ".venv\Scripts\python.exe" (
  echo  [ERROR] Failed to replace broken .venv automatically.
  exit /b 1
)
exit /b 0
