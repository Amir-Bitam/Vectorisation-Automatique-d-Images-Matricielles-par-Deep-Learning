param(
    [string]$EnvironmentName = "live-win"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

function Get-CondaExe {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $candidate = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
    if (Test-Path -LiteralPath $candidate) { return $candidate }

    throw "Conda was not found. Install Anaconda/Miniconda or add conda.exe to PATH."
}

function Get-EnvPath {
    param([string]$CondaExe, [string]$Name)
    $json = & $CondaExe env list --json | ConvertFrom-Json
    $match = $json.envs | Where-Object { (Split-Path $_ -Leaf) -eq $Name } | Select-Object -First 1
    if (-not $match) { throw "Conda environment '$Name' was not found after creation/update." }
    return $match
}

function Test-CondaEnvExists {
    param([string]$CondaExe, [string]$Name)
    $json = & $CondaExe env list --json | ConvertFrom-Json
    return [bool]($json.envs | Where-Object { (Split-Path $_ -Leaf) -eq $Name })
}

function Get-VcVars64 {
    $known = "C:\Program\VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path -LiteralPath $known) { return $known }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $found = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Auxiliary\Build\vcvars64.bat" | Select-Object -First 1
        if ($found -and (Test-Path -LiteralPath $found)) { return $found }
    }

    throw "Visual Studio Build Tools with the MSVC x64 toolchain were not found. Install the 'Desktop development with C++' workload, including MSVC v143 x64/x86 build tools and a Windows SDK."
}

function Import-VsDevEnv {
    param([string]$VcVars64)
    $vars = cmd /c "call `"$VcVars64`" >nul && set"
    foreach ($line in $vars) {
        $idx = $line.IndexOf("=")
        if ($idx -gt 0) {
            [Environment]::SetEnvironmentVariable($line.Substring(0, $idx), $line.Substring($idx + 1), "Process")
        }
    }
}

function Invoke-Native {
    param([scriptblock]$Command, [string]$FailureMessage)
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Command
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    if ($code -ne 0) {
        throw "$FailureMessage Exit code: $code"
    }
}

function Remove-RepoDirectory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if (-not $resolved.StartsWith($RepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove unexpected path: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}

function Get-FreeSubstDrive {
    foreach ($letter in @("X", "Y", "Z", "W")) {
        if (-not (Get-PSDrive -Name $letter -ErrorAction SilentlyContinue)) {
            return $letter
        }
    }
    throw "No free drive letter was available for a temporary subst mapping."
}

$CondaExe = Get-CondaExe

Write-Step "Creating or updating Conda environment '$EnvironmentName'"
if (Test-CondaEnvExists -CondaExe $CondaExe -Name $EnvironmentName) {
    Invoke-Native { & $CondaExe env update -n $EnvironmentName -f (Join-Path $RepoRoot "environment-windows.yml") --prune } "Conda environment update failed."
} else {
    Invoke-Native { & $CondaExe env create -f (Join-Path $RepoRoot "environment-windows.yml") } "Conda environment creation failed."
}

$EnvPath = Get-EnvPath -CondaExe $CondaExe -Name $EnvironmentName
$PythonExe = Join-Path $EnvPath "python.exe"
$env:PATH = "$EnvPath;$EnvPath\Library\bin;$EnvPath\Scripts;$EnvPath\bin;$env:PATH"

Write-Step "Installing pip-only LIVE dependencies"
Invoke-Native { & $PythonExe -m pip install -r (Join-Path $RepoRoot "requirements-windows.txt") } "pip dependency installation failed."

Write-Step "Initializing repository submodules"
Invoke-Native { git submodule update --init --recursive } "git submodule initialization failed."

Write-Step "Preparing Visual Studio and CUDA build environment"
$VcVars64 = Get-VcVars64
Import-VsDevEnv -VcVars64 $VcVars64
$env:PATH = "$EnvPath;$EnvPath\Library\bin;$EnvPath\Scripts;$EnvPath\bin;$env:PATH"
$env:CUDA_PATH = "$EnvPath\Library"
$env:CUDA_HOME = "$EnvPath\Library"
$env:CMAKE_GENERATOR = "Ninja"
$env:DISTUTILS_USE_SDK = "1"
$env:MSSdk = "1"

Write-Step "Building bundled DiffVG inside '$EnvironmentName'"
Remove-RepoDirectory -Path (Join-Path $RepoRoot "DiffVG\build")
$badExtension = Join-Path $EnvPath "Lib\site-packages\diffvg"
if (Test-Path -LiteralPath $badExtension) {
    Remove-Item -LiteralPath $badExtension -Force
}

$drive = Get-FreeSubstDrive
cmd /c "subst $drive`: `"$RepoRoot`""
try {
    Push-Location "$drive`:\DiffVG"
    try {
        Invoke-Native { & $PythonExe setup.py install } "DiffVG build/install failed."
    }
    finally {
        Pop-Location
    }
}
finally {
    cmd /c "subst $drive`: /D" 2>$null | Out-Null
}

Write-Step "Verifying CUDA, imports, and DiffVG gradients"
Invoke-Native {
    & $PythonExe -c "import torch; print('Torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); raise SystemExit(0 if torch.cuda.is_available() else 1)"
} "CUDA verification failed."

Invoke-Native {
    & $PythonExe -c "import torch, cv2, numpy, skimage, svgwrite, svgpathtools, cssutils, numba, skfmm, easydict, pydiffvg; print('All major imports succeeded')"
} "Import verification failed."

$gradientTest = @'
import torch
import pydiffvg

pydiffvg.set_use_gpu(torch.cuda.is_available())
canvas_width, canvas_height = 32, 32
points = torch.tensor([[8.0, 8.0], [24.0, 8.0], [24.0, 24.0], [8.0, 24.0]], requires_grad=True)
polygon = pydiffvg.Polygon(points=points, is_closed=True)
color = torch.tensor([0.1, 0.6, 0.9, 1.0], requires_grad=True)
group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=color)
scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, [polygon], [group])
img = pydiffvg.RenderFunction.apply(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
loss = img[..., :3].sum()
loss.backward()
assert points.grad is not None and float(points.grad.abs().sum()) > 0
assert color.grad is not None and float(color.grad.abs().sum()) > 0
print("DiffVG use_gpu:", pydiffvg.get_use_gpu())
print("DiffVG device:", pydiffvg.get_device())
print("Gradient smoke test passed")
'@
Invoke-Native { $gradientTest | & $PythonExe - } "DiffVG gradient verification failed."

Write-Step "Setup completed successfully"
Write-Host "Use .\run_live_windows.ps1 -InputImage .\LIVE\figures\smile.png -Signature windows_smoke_test"
