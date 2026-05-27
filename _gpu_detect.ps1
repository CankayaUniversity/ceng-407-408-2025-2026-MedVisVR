# GPU detection helper for Install_New_PC_OneClick.bat
# Writes KEY=VALUE lines to %TEMP%\carvis_gpu.txt

$outFile = "$env:TEMP\carvis_gpu.txt"

try {
    $raw = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
    if ($null -ne $raw -and "$raw".Trim() -ne '') {
        $line = (($raw | Select-Object -First 1) -as [string]).Trim()

        # Split on the last comma (driver version always comes after the last comma)
        $lastComma = $line.LastIndexOf(',')
        if ($lastComma -gt 0) {
            $gpuName = $line.Substring(0, $lastComma).Trim()
            $drvVer  = $line.Substring($lastComma + 1).Trim()
        } else {
            $gpuName = $line
            $drvVer  = ''
        }

        $major = 0
        if ($drvVer -match '^(\d+)') {
            $major = [int]$Matches[1]
        }

        $cuda = if     ($major -ge 565) { 'cu128' }
                elseif ($major -ge 545) { 'cu124' }
                elseif ($major -ge 530) { 'cu121' }
                elseif ($major -gt  0)  { 'cu118' }
                else                    { '' }

        @("HAS_GPU=1", "GPU_NAME=$gpuName", "CUDA_INDEX=$cuda") |
            Set-Content $outFile -Encoding ASCII
    } else {
        'HAS_GPU=0' | Set-Content $outFile -Encoding ASCII
    }
} catch {
    'HAS_GPU=0' | Set-Content $outFile -Encoding ASCII
}
