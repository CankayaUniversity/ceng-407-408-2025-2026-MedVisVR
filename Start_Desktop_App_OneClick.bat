@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
set "APP_ROOT=%CD%"

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

set "LNK=%REAL_DESKTOP%\CARVIS.lnk"

if not exist "%LNK%" (
  echo [INFO] Creating desktop shortcut for the first time...

  set "ICON_PATH=%APP_ROOT%\desktop_app\build\lumina.ico"
  if not exist "!ICON_PATH!" set "ICON_PATH=%APP_ROOT%\desktop_app\node_modules\electron\dist\electron.exe"
  if not exist "!ICON_PATH!" set "ICON_PATH="

  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ws = New-Object -ComObject WScript.Shell;" ^
    "$lnk = $ws.CreateShortcut('%LNK%');" ^
    "$lnk.TargetPath       = '%VBS%';" ^
    "$lnk.WorkingDirectory = '%APP_ROOT%';" ^
    "$lnk.Description      = 'CARVIS - Clinical Anatomical Rendering Visualization Intelligent System';" ^
    "$lnk.WindowStyle      = 1;" ^
    "$icon = '!ICON_PATH!';" ^
    "if ($icon -ne '' -and (Test-Path $icon)) { $lnk.IconLocation = $icon + ',0' };" ^
    "$lnk.Save()" >nul 2>&1

  if exist "%LNK%" (
    echo [INFO] Desktop shortcut created: CARVIS.lnk
  )
)

call "%APP_ROOT%\Start_OneClick.bat" desktop
exit /b %ERRORLEVEL%