@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1

cd /d "%~dp0"
set "APP_ROOT=%CD%"
set "OUT_ZIP=%APP_ROOT%\CARVIS_Distribution.zip"
set "TMP_DIR=%APP_ROOT%\_dist_tmp"

cls
echo.
echo  ============================================================
echo   CARVIS Clinical AI -- Create Distribution Package
echo  ============================================================
echo   Excluded (large / auto-generated / machine-specific):
echo     - .venv                      (recreated by Install bat)
echo     - desktop_app\node_modules   (recreated by npm install)
echo     - desktop_app\.claude        (dev worktrees)
echo     - desktop_app\release        (Electron installer output)
echo     - __pycache__  *.pyc  tmp
echo     - Launch_LUMINA.vbs          (hardcoded legacy, replaced)
echo     - Launch_NeuroLens.vbs       (legacy)
echo     - _dist_tmp                  (this temp folder)
echo.
echo   Included (pre-built so target PC skips npm build):
echo     - desktop_app\dist\          (pre-built frontend UI)
echo  ============================================================
echo.
echo   Output: %OUT_ZIP%
echo.
pause

if exist "%OUT_ZIP%" del /q "%OUT_ZIP%"
if exist "%TMP_DIR%" rd /s /q "%TMP_DIR%"
mkdir "%TMP_DIR%"

echo.
echo  [1/3] Copying files (robocopy)...
echo.

robocopy "%APP_ROOT%" "%TMP_DIR%" /E /NFL /NDL /NJH /NJS ^
  /XD ".venv" ^
  /XD ".venv.broken" ^
  /XD ".git" ^
  /XD ".claude" ^
  /XD "__pycache__" ^
  /XD "tmp" ^
  /XD "_dist_tmp" ^
  /XD "desktop_app\node_modules" ^
  /XD "desktop_app\.claude" ^
  /XD "desktop_app\release" ^
  /XD "nnunet_inference_input" ^
  /XF "*.pyc" ^
  /XF "*.pyo" ^
  /XF "*.log" ^
  /XF "Launch_LUMINA.vbs" ^
  /XF "Launch_NeuroLens.vbs" ^
  /XF "CARVIS_Distribution.zip" ^
  /XF "LUMINA_Distribution.zip"

if !errorlevel! geq 8 (
  echo  [ERROR] robocopy failed with code !errorlevel!
  rd /s /q "%TMP_DIR%" 2>nul
  pause & exit /b 1
)

if not exist "%TMP_DIR%\desktop_app\dist\index.html" (
  echo  [WARN] desktop_app\dist not found in package!
  echo         Run "npm run build" inside desktop_app first, then re-pack.
)

echo.
echo  [2/3] Creating ZIP...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Compress-Archive -Path '%TMP_DIR%\*' -DestinationPath '%OUT_ZIP%' -CompressionLevel Optimal"

if not exist "%OUT_ZIP%" (
  echo  [ERROR] ZIP creation failed.
  rd /s /q "%TMP_DIR%" 2>nul
  pause & exit /b 1
)

echo.
echo  [3/3] Cleaning up temp folder...
rd /s /q "%TMP_DIR%"

for %%F in ("%OUT_ZIP%") do set "ZIP_SIZE=%%~zF"
set /a ZIP_MB=!ZIP_SIZE! / 1048576

echo.
echo  ============================================================
echo   [DONE] Package ready:
echo   %OUT_ZIP%
echo   Size: ~!ZIP_MB! MB
echo.
echo   On the target PC:
echo     1. Extract the ZIP to any folder (e.g. C:\CARVIS)
echo     2. Run Install_New_PC_OneClick.bat (first-time setup)
echo        - Python 3.11+ and Node.js 18+ must be installed
echo        - .venv + pip install runs automatically
echo        - npm install runs automatically (build skipped, dist included)
echo        - Desktop CARVIS shortcut is created
echo     3. On subsequent runs double-click CARVIS on the desktop
echo        or run Start_Desktop_App_OneClick.bat
echo.
echo   LLM (optional):
echo     - Place llama-server.exe under tools\llama.cpp\
echo     - Place a .gguf model file under models\
echo  ============================================================
echo.
pause
exit /b 0