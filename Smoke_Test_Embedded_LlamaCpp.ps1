param(
    [string]$BaseUrl = "http://127.0.0.1:8080"
)

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptRoot

$env:AI_USE_LLM        = "1"
$env:AI_LLM_PROVIDER   = "embedded_llama"
$env:AI_LLAMACPP_URL   = $BaseUrl
$env:AI_MODEL_DEFAULT  = "qwen2.5:7b"
$env:AI_MODEL_FALLBACK = "qwen2.5:3b"
$env:AI_MODEL_HIGH     = "qwen2.5:14b"

Write-Host "Checking health endpoint: $BaseUrl/health"
try {
    $health = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get -TimeoutSec 10
    Write-Host "health ok:" ($health | ConvertTo-Json -Compress)
}
catch {
    Write-Warning "health endpoint did not respond, trying /v1/models"
}

Write-Host "Checking model endpoint: $BaseUrl/v1/models"
try {
    $models = Invoke-RestMethod -Uri "$BaseUrl/v1/models" -Method Get -TimeoutSec 10
    Write-Host "models ok:" ($models | ConvertTo-Json -Compress)
}
catch {
    Write-Warning "v1/models endpoint did not respond"
}

Write-Host "Running Python local_llm smoke test..."
$venvPy = Join-Path $scriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPy) {
    & $venvPy -c "from ai_assistant.core.local_llm import health_check,generate_text; ok=health_check(); print('health_check=', ok); txt, model=generate_text(system_prompt='You are concise.', user_prompt='Reply with only: LLAMA_OK', use_high_model=False); print('model_used=', model); print('text=', txt[:200])"
}
else {
    py -3 -c "from ai_assistant.core.local_llm import health_check,generate_text; ok=health_check(); print('health_check=', ok); txt, model=generate_text(system_prompt='You are concise.', user_prompt='Reply with only: LLAMA_OK', use_high_model=False); print('model_used=', model); print('text=', txt[:200])"
}