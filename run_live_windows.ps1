param(
    [Parameter(Mandatory = $true)]
    [string]$InputImage,
    [string]$Signature = "windows_smoke_test",
    [string]$Experiment = "experiment_5x1",
    [string]$Config = "config/base.yaml",
    [string]$LogDirectory = "log",
    [string]$EnvironmentName = "live-win"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$LiveRoot = Join-Path $RepoRoot "LIVE"
$ToolLogRoot = Join-Path $RepoRoot "windows_setup_logs"
New-Item -ItemType Directory -Force -Path $ToolLogRoot | Out-Null

function Get-CondaExe {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidate = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
    if (Test-Path -LiteralPath $candidate) { return $candidate }
    throw "Conda was not found. Run setup_live_windows.ps1 first or add conda.exe to PATH."
}

function Get-EnvPath {
    param([string]$CondaExe, [string]$Name)
    $json = & $CondaExe env list --json | ConvertFrom-Json
    $match = $json.envs | Where-Object { (Split-Path $_ -Leaf) -eq $Name } | Select-Object -First 1
    if (-not $match) { throw "Conda environment '$Name' was not found. Run setup_live_windows.ps1 first." }
    return $match
}

function Resolve-RunPath {
    param([string]$Path, [string]$Base)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return (Resolve-Path -LiteralPath $Path).Path
    }
    return (Resolve-Path -LiteralPath (Join-Path $Base $Path)).Path
}

function Quote-Arg {
    param([string]$Value)
    if ($Value -notmatch '[\s"]') { return $Value }
    return '"' + ($Value -replace '"', '\"') + '"'
}

$InputPath = Resolve-RunPath -Path $InputImage -Base (Get-Location).Path
$ConfigPath = Resolve-RunPath -Path $Config -Base $LiveRoot
if ([System.IO.Path]::IsPathRooted($LogDirectory)) {
    $LogPath = $LogDirectory
} else {
    $LogPath = Join-Path $LiveRoot $LogDirectory
}
New-Item -ItemType Directory -Force -Path $LogPath | Out-Null

$CondaExe = Get-CondaExe
$EnvPath = Get-EnvPath -CondaExe $CondaExe -Name $EnvironmentName
$PythonExe = Join-Path $EnvPath "python.exe"
$env:PATH = "$EnvPath;$EnvPath\Library\bin;$EnvPath\Scripts;$EnvPath\bin;$env:PATH"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunLog = Join-Path $ToolLogRoot "live_run_${Signature}_$timestamp.log"
$StdoutLog = Join-Path $ToolLogRoot "live_run_${Signature}_$timestamp.stdout.log"
$StderrLog = Join-Path $ToolLogRoot "live_run_${Signature}_$timestamp.stderr.log"

$args = @(
    "main.py",
    "--config", $ConfigPath,
    "--experiment", $Experiment,
    "--signature", $Signature,
    "--target", $InputPath,
    "--log_dir", $LogPath
)

Write-Host "LIVE command:"
Write-Host "$PythonExe $($args -join ' ')"

$start = Get-Date
$psiArgs = ($args | ForEach-Object { Quote-Arg $_ }) -join " "
$proc = Start-Process -FilePath $PythonExe `
    -ArgumentList $psiArgs `
    -WorkingDirectory $LiveRoot `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -NoNewWindow `
    -PassThru `
    -Wait
$end = Get-Date

"START=$($start.ToString('o'))" | Set-Content -LiteralPath $RunLog
"END=$($end.ToString('o'))" | Add-Content -LiteralPath $RunLog
"ELAPSED_SECONDS=$([Math]::Round(($end - $start).TotalSeconds, 3))" | Add-Content -LiteralPath $RunLog
"EXIT_CODE=$($proc.ExitCode)" | Add-Content -LiteralPath $RunLog
"--- STDOUT ---" | Add-Content -LiteralPath $RunLog
Get-Content -LiteralPath $StdoutLog -ErrorAction SilentlyContinue | Add-Content -LiteralPath $RunLog
"--- STDERR ---" | Add-Content -LiteralPath $RunLog
Get-Content -LiteralPath $StderrLog -ErrorAction SilentlyContinue | Add-Content -LiteralPath $RunLog

Remove-Item -LiteralPath $StdoutLog,$StderrLog -Force -ErrorAction SilentlyContinue

if ($proc.ExitCode -ne 0) {
    Write-Error "LIVE failed. Log: $RunLog"
    exit $proc.ExitCode
}

$resultDir = Get-ChildItem -LiteralPath $LogPath -Directory |
    Where-Object { $_.Name -like "*_$Signature" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $resultDir) {
    Write-Error "LIVE finished, but no result directory matching '*_$Signature' was found under $LogPath. Log: $RunLog"
    exit 2
}

$svg = Get-ChildItem -LiteralPath (Join-Path $resultDir.FullName "output-svg") -Filter "*.svg" -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $svg) {
    Write-Error "LIVE finished, but no SVG was found in $($resultDir.FullName). Log: $RunLog"
    exit 3
}

Write-Host "Elapsed seconds: $([Math]::Round(($end - $start).TotalSeconds, 3))"
Write-Host "Run log: $RunLog"
Write-Host "Result directory: $($resultDir.FullName)"
Write-Host "Generated SVG: $($svg.FullName)"
