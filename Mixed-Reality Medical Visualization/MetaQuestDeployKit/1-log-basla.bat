@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo   MedDemo - Log Kaydi Baslatma
echo ============================================================
echo.

if not exist "adb\adb.exe" (
    echo [HATA] adb bulunamadi: adb\adb.exe
    echo Bu dosya UnityProje klasorunde olmali (install.bat ile ayni yerde).
    pause
    exit /b 1
)

echo [1/3] Uygulama tamamen kapatiliyor...
adb\adb.exe shell am force-stop com.DefaultCompany.MedDemo
echo [2/3] Eski loglar temizleniyor...
adb\adb.exe logcat -c
echo [3/3] Uygulama yeniden aciliyor...
adb\adb.exe shell monkey -p com.DefaultCompany.MedDemo -c android.intent.category.LAUNCHER 1 >nul 2>&1

echo.
echo ============================================================
echo  HAZIR! Simdi sirasiyla sunlari yap:
echo.
echo    1. Quest 3 gozlugu tak
echo    2. Uygulamanin acilmasini bekle
echo    3. TITREME baslayip belirginlesene kadar bekle
echo       (yaklasik 1.5 - 2 dakika gozlugu takili tut)
echo    4. Sonra "2-log-kaydet.bat" dosyasina CIFT TIKLA
echo ============================================================
echo.
pause
