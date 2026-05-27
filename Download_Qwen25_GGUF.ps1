param(
    [string]$RepoId = "Qwen/Qwen2.5-7B-Instruct-GGUF",
    [string]$Quant  = "q4_k_m",
    [string]$OutDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptRoot

if (-not $OutDir) {
    $OutDir = Join-Path $scriptRoot "models\Qwen2.5-7B-Instruct-GGUF"
}

$venvPy = Join-Path $scriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPy) {
    $pythonExe = $venvPy
}
else {
    try   { $pythonExe = (Get-Command "py.exe"     -ErrorAction Stop).Source }
    catch { $pythonExe = (Get-Command "python.exe" -ErrorAction Stop).Source }
}
Write-Host "[INFO] Python: $pythonExe"

function Find-ExistingModel {
    param([string]$dir, [string]$quant)

    if (-not (Test-Path $dir)) { return $null }

    $merged = Get-ChildItem -Path $dir -Filter "*${quant}*.gguf" -File -ErrorAction SilentlyContinue |
              Where-Object { $_.Name -notmatch '-\d{5}-of-\d{5}\.gguf$' } |
              Sort-Object Length -Descending |
              Select-Object -First 1

    if ($merged) { return $merged.FullName }

    $any = Get-ChildItem -Path $dir -Filter "*.gguf" -File -ErrorAction SilentlyContinue |
           Where-Object { $_.Name -notmatch '-\d{5}-of-\d{5}\.gguf$' } |
           Select-Object -First 1

    if ($any) { return $any.FullName }

    $split = Get-ChildItem -Path $dir -Filter "*${quant}*-00001-of-*.gguf" -File -ErrorAction SilentlyContinue |
             Select-Object -First 1

    if ($split) { return $split.FullName }
    return $null
}

$existingModel = Find-ExistingModel -dir $OutDir -quant $Quant

if ($existingModel) {
    $sizeMB = [math]::Round((Get-Item $existingModel).Length / 1MB, 1)
    Write-Host ""
    Write-Host "[OK] Model already exists, skipping download."
    Write-Host "     File  : $existingModel"
    Write-Host "     Size  : $sizeMB MB"
    Write-Host ""
    Write-Host "GGUF_MODEL_PATH=$existingModel"
    exit 0
}

Write-Host ""
Write-Host "[INFO] Model not found, downloading from HuggingFace..."
Write-Host "       Repo  : $RepoId"
Write-Host "       Quant : $Quant"
Write-Host "       Target: $OutDir"
Write-Host ""

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$downloadScript = @"
import sys
from huggingface_hub import hf_hub_download, list_repo_files

repo_id = r"$RepoId"
out_dir = r"$OutDir"
quant   = r"$Quant".lower()

all_files = list(list_repo_files(repo_id))
matched   = [f for f in all_files if quant in f.lower() and f.endswith('.gguf')]

if not matched:
    print(f"ERROR: no .gguf found for '{quant}'. Available files:", file=sys.stderr)
    for f in all_files:
        if f.endswith('.gguf'):
            print(f"  {f}", file=sys.stderr)
    sys.exit(1)

print(f"[INFO] {len(matched)} file(s) to download:")
for f in matched:
    print(f"  {f}")

downloaded = []
for filename in matched:
    print(f"\n[INFO] Downloading: {filename}")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=out_dir,
        resume_download=True,
    )
    downloaded.append(path)
    print(f"[OK] {filename}")

merged = [p for p in downloaded if not __import__('re').search(r'-\d{5}-of-\d{5}\.gguf$', p)]
if merged:
    print(f"\nGGUF_MODEL_PATH={merged[0]}")
elif downloaded:
    print(f"\nGGUF_MODEL_PATH={downloaded[0]}")

print("DOWNLOAD_OK")
"@

$tmpPy = Join-Path $env:TEMP "carvis_download_gguf.py"
Set-Content -Path $tmpPy -Value $downloadScript -Encoding UTF8

try {
    & $pythonExe $tmpPy
    if ($LASTEXITCODE -ne 0) {
        throw "Python download script failed (exit $LASTEXITCODE)."
    }
}
finally {
    Remove-Item $tmpPy -ErrorAction SilentlyContinue
}

$finalModel = Find-ExistingModel -dir $OutDir -quant $Quant
if (-not $finalModel) {
    Write-Error "Download completed but merged .gguf not found: $OutDir"
    exit 1
}

$sizeMB = [math]::Round((Get-Item $finalModel).Length / 1MB, 1)
Write-Host ""
Write-Host "[DONE] Model ready."
Write-Host "       File  : $finalModel"
Write-Host "       Size  : $sizeMB MB"
Write-Host ""
Write-Host "To activate the LLM, run Start_Desktop_App_OneClick.bat."
