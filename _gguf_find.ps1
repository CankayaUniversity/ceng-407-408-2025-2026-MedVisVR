# GGUF model finder for Install_New_PC_OneClick.bat
# Usage: powershell -File _gguf_find.ps1 -AppRoot "C:\path\to\app"
# Writes the found model path (or empty string) to %TEMP%\carvis_gguf.txt
param(
    [string]$AppRoot = ""
)

$outFile = "$env:TEMP\carvis_gguf.txt"

$searchDirs = @()
if ($AppRoot -ne '') {
    $searchDirs += Join-Path $AppRoot 'models'
    $searchDirs += $AppRoot
}
$searchDirs += 'C:\models'

$found = $null
foreach ($dir in $searchDirs) {
    if (-not (Test-Path $dir)) { continue }
    $f = Get-ChildItem -Path $dir -Filter '*.gguf' -Recurse -File -ErrorAction SilentlyContinue |
         Where-Object { $_.Name -notmatch '-\d{5}-of-\d{5}\.gguf$' } |
         Sort-Object Length -Descending |
         Select-Object -First 1
    if ($f) { $found = $f.FullName; break }
}

if ($found) {
    $found | Set-Content $outFile -Encoding ASCII
} else {
    '' | Set-Content $outFile -Encoding ASCII
}
