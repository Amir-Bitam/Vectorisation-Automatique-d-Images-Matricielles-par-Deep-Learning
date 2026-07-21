param(
    [Parameter(Mandatory = $true)]
    [string]$InputImage,
    [string]$Checkpoint = ".\logs\VectorVAEnLayers\version_110\epoch=667.ckpt",
    [string]$Config = ".\logs\VectorVAEnLayers\version_110\configs\emoji.yaml",
    [string]$OutputDirectory = ".\outputs\custom_test",
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

if (-not (Test-Path $InputImage)) { throw "Input image not found: $InputImage" }
if (-not (Test-Path $Checkpoint)) { throw "Checkpoint not found: $Checkpoint" }
if (-not (Test-Path $Config)) { throw "Config not found: $Config" }
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

Write-Warning "The official checkpoint is trained on the emoji domain. This is encoder reconstruction, not universal raster-to-vector optimization."

$logPath = Join-Path $OutputDirectory "run_im2vec_image_windows.log"
$conda = Get-Conda
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

$command = "`"$conda`" run -n `"$EnvironmentName`" python reconstruct_image.py --input `"$InputImage`" --checkpoint `"$Checkpoint`" --config `"$Config`" --output-dir `"$OutputDirectory`" 2>&1"
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
