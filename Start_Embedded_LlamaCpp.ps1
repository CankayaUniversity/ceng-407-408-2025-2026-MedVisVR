param(
    [string]$LlamaServerExe = "",
    [string]$ModelPath = "",
    [int]$Port = 8080,
    [int]$Ctx = 4096,
    [int]$Threads = 8,
    [int]$GpuLayers = 0
)

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptRoot

function Find-FirstExistingPath {
    param([string[]]$Candidates)
    foreach ($candidate in $Candidates) {
        if (-not $candidate) { continue }
        if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
    }
    return $null
}

function Find-BestGguf {
    param([string[]]$Patterns)
    foreach ($pattern in $Patterns) {
        $matches = Get-ChildItem -Path $pattern -File -ErrorAction SilentlyContinue | Sort-Object FullName
        if (-not $matches) { continue }
        $preferred = $matches | Where-Object { $_.Name -notmatch '-\d{5}-of-\d{5}\.gguf$' } | Sort-Object Length -Descending | Select-Object -First 1
        if ($preferred) { return $preferred.FullName }
        return ($matches | Sort-Object Length -Descending | Select-Object -First 1).FullName
    }
    return $null
}

if (-not $LlamaServerExe -and $env:LLAMA_SERVER_EXE) { $LlamaServerExe = $env:LLAMA_SERVER_EXE }
if (-not $ModelPath -and $env:GGUF_MODEL_PATH)       { $ModelPath = $env:GGUF_MODEL_PATH }

if (-not $LlamaServerExe) {
    $LlamaServerExe = Find-FirstExistingPath -Candidates @(
        (Join-Path $scriptRoot "tools\llama.cpp\llama-server.exe"),
        (Join-Path $scriptRoot "tools\llama-server.exe"),
        "C:\tools\llama.cpp\llama-server.exe",
        (Join-Path $env:USERPROFILE ".ai-navigator\micromamba\envs\cuda\Library\bin\llama-server.exe")
    )
    if (-not $LlamaServerExe) {
        $cmd = Get-Command "llama-server.exe" -ErrorAction SilentlyContinue
        if ($cmd) { $LlamaServerExe = $cmd.Source }
    }
}

if (-not $ModelPath) {
    $ModelPath = Find-BestGguf -Patterns @(
        (Join-Path $scriptRoot "models\*.gguf"),
        (Join-Path $scriptRoot "models\*\*.gguf"),
        (Join-Path $scriptRoot "*.gguf"),
        "C:\models\Qwen2.5-7B-Instruct-GGUF\*.gguf"
    )
}

if (-not $LlamaServerExe -or -not (Test-Path $LlamaServerExe)) {
    throw "llama-server not found. Copy it under tools\llama.cpp\ or provide the full path via LLAMA_SERVER_EXE / -LlamaServerExe."
}
if (-not (Test-Path $ModelPath)) {
    throw "GGUF model not found. Place a .gguf file under models\ or provide the full path via GGUF_MODEL_PATH / -ModelPath."
}

$env:AI_USE_LLM       = "1"
$env:AI_LLM_PROVIDER  = "embedded_llama"
$env:AI_LLAMACPP_URL  = "http://127.0.0.1:$Port"
$env:AI_MODEL_DEFAULT = "qwen2.5:7b"
$env:AI_MODEL_FALLBACK = "qwen2.5:3b"
$env:AI_MODEL_HIGH    = "qwen2.5:14b"
$env:AI_N_CTX         = "$Ctx"

$args = @(
    "-m", $ModelPath,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "-c", "$Ctx",
    "-t", "$Threads",
    "-ngl", "$GpuLayers",
    "--no-warmup"
)

Write-Host "Starting llama-server..."
Write-Host "  exe   : $LlamaServerExe"
Write-Host "  model : $ModelPath"
Write-Host "  url   : $env:AI_LLAMACPP_URL"
Write-Host "  note  : first model load can still take a little while"

$proc = Start-Process -FilePath $LlamaServerExe -ArgumentList $args -PassThru
Write-Host "llama-server PID: $($proc.Id)"
Write-Host "Smoke test: .\Smoke_Test_Embedded_LlamaCpp.ps1"