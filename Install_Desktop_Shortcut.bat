@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo  CARVIS -- Desktop Shortcut Setup
echo  ========================================
echo.

set "WORKSPACE=%~dp0"
if "%WORKSPACE:~-1%"=="\" set "WORKSPACE=%WORKSPACE:~0,-1%"

set "VBS_LAUNCHER=%WORKSPACE%\Launch_CARVIS.vbs"

for /f "usebackq delims=" %%D in (
  `powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"`
) do set "DESKTOP=%%D"

(
  echo Dim scriptDir, batPath
  echo scriptDir = Left^(WScript.ScriptFullName, InStrRev^(WScript.ScriptFullName, "\"^)^)
  echo batPath    = scriptDir ^& "Start_Desktop_App_OneClick.bat"
  echo Dim ws
  echo Set ws = CreateObject^("WScript.Shell"^)
  echo ws.Run Chr^(34^) ^& batPath ^& Chr^(34^), 0, False
) > "%VBS_LAUNCHER%"

set "ICO_SRC=%WORKSPACE%\desktop_app\build\lumina.ico"
if not exist "!ICO_SRC!" set "ICO_SRC=%WORKSPACE%\desktop_app\node_modules\electron\dist\electron.exe"
if not exist "!ICO_SRC!" set "ICO_SRC=%SystemRoot%\System32\shell32.dll"

echo [INFO] VBS Launcher : %VBS_LAUNCHER%
echo [INFO] Desktop      : %DESKTOP%
echo [INFO] Icon source  : !ICO_SRC!
echo.

if exist "%DESKTOP%\LUMINA.lnk" (
  del "%DESKTOP%\LUMINA.lnk" >nul 2>&1
  echo [INFO] Old LUMINA shortcut removed.
)
if exist "%DESKTOP%\CARVIS.lnk" (
  del "%DESKTOP%\CARVIS.lnk" >nul 2>&1
  echo [INFO] Old CARVIS shortcut removed (will be recreated).
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ws   = New-Object -ComObject WScript.Shell;" ^
  "$lnk  = $ws.CreateShortcut('%DESKTOP%\CARVIS.lnk');" ^
  "$lnk.TargetPath       = '%VBS_LAUNCHER%';" ^
  "$lnk.WorkingDirectory = '%WORKSPACE%';" ^
  "$lnk.Description      = 'CARVIS — Clinical Anatomical Rendering Visualization Intelligent System';" ^
  "$lnk.IconLocation     = '!ICO_SRC!,0';" ^
  "$lnk.Save();"

if errorlevel 1 (
  echo [ERROR] Shortcut creation failed. PowerShell error.
  goto :fail
)

if not exist "%DESKTOP%\CARVIS.lnk" (
  echo [ERROR] Shortcut file not found at expected location.
  goto :fail
)

echo [OK] Desktop shortcut created:
echo      %DESKTOP%\CARVIS.lnk
echo.
echo  Double-click the "CARVIS" icon on your desktop to launch.
echo  (First launch may take 20-30 seconds — this is normal.)
echo.
pause
exit /b 0

:fail
echo.
echo [FAILED] Shortcut could not be created. Check the errors above.
pause
exit /b 1