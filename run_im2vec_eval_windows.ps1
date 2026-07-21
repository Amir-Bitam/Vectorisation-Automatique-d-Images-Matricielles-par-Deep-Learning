param(
    [string]$Checkpoint = ".\logs\VectorVAEnLayers\version_110\epoch=667.ckpt",
    [string]$Config = ".\logs\VectorVAEnLayers\version_110\configs\emoji.yaml",
    [string]$OutputDirectory = ".\outputs\official_eval",
    [string]$EnvironmentName = "im2vec-win"
)

$ErrorActionPreference = "Stop"

function Get-Conda {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $fallback = "C:\Users\zak\anaconda3\Scripts\conda.exe"
    if (Test-Path $fallback) { return $fallback }
    throw "Conda was not found."
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

if (-not (Test-Path $Checkpoint)) { throw "Checkpoint not found: $Checkpoint" }
if (-not (Test-Path $Config)) { throw "Config not found: $Config" }
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

$logPath = Join-Path $OutputDirectory "run_im2vec_eval_windows.log"
$conda = Get-Conda
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

$command = "`"$conda`" run -n `"$EnvironmentName`" python eval_local.py --config `"$Config`" --checkpoint `"$Checkpoint`" --output-dir `"$OutputDirectory`" 2>&1"
& cmd /c $command | Tee-Object -FilePath $logPath
$exitCode = $LASTEXITCODE
$stopwatch.Stop()

Write-Host "Elapsed: $($stopwatch.Elapsed)"
Write-Host "Log: $((Resolve-Path $logPath).Path)"
Get-ChildItem -Path $OutputDirectory -Filter *.svg -Recurse | ForEach-Object {
    Write-Host "SVG: $($_.FullName)"
}

if ($exitCode -ne 0) {
    exit $exitCode
}
