@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo   MedDemo - Log Kaydetme
echo ============================================================
echo.

if not exist "adb\adb.exe" (
    echo [HATA] adb bulunamadi: adb\adb.exe
    pause
    exit /b 1
)

echo Log dosyaya kaydediliyor...
adb\adb.exe logcat -d > "C:\Users\sezer\OneDrive\Desktop\meddemo_log.txt"

echo.
echo ============================================================
echo  TAMAMLANDI! Log kaydedildi:
echo    C:\Users\sezer\OneDrive\Desktop\meddemo_log.txt
echo.
echo  Simdi Claude'a "tamam" yaz.
echo ============================================================
echo.
pause
