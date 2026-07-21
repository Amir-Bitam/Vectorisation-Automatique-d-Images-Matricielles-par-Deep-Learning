param(
    [string]$EnvironmentName = "im2vec-win",
    [string]$CheckpointUrl = "http://geometry.cs.ucl.ac.uk/projects/2021/im2vec/paper_docs/epoch=667.ckpt",
    [string]$CheckpointPath = ".\logs\VectorVAEnLayers\version_110\epoch=667.ckpt",
    [switch]$ForceDiffVGBuild
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
    }
}

function Get-Conda {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $fallback = "C:\Users\zak\anaconda3\Scripts\conda.exe"
    if (Test-Path $fallback) { return $fallback }
    throw "Conda was not found. Install Anaconda/Miniconda or add conda to PATH."
}

function Get-EnvPrefix {
    param([string]$CondaExe, [string]$Name)
    $prefix = & $CondaExe run -n $Name python -c "import sys; print(sys.prefix)"
    if ($LASTEXITCODE -ne 0) { throw "Could not resolve Conda env prefix for $Name" }
    return ($prefix | Select-Object -Last 1).Trim()
}

function Test-EnvExists {
    param([string]$CondaExe, [string]$Name)
    $json = & $CondaExe env list --json | ConvertFrom-Json
    return [bool]($json.envs | Where-Object { Split-Path $_ -Leaf -eq $Name })
}

function Get-VcVars64 {
    $candidates = @(
        "C:\Program\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    throw "Visual Studio 2022 Build Tools vcvars64.bat was not found. Install Desktop development with C++ plus MSVC v143 and Windows SDK."
}

function Patch-DiffVGSource {
    param([string]$SourceDir)
    $patcher = @'
from pathlib import Path
import re

root = Path(r"__SOURCE_DIR__")
(root / "pyproject.toml").write_text('[build-system]\nrequires = ["setuptools", "wheel"]\nbuild-backend = "setuptools.build_meta"\n', encoding="utf-8")

setup = root / "setup.py"
text = setup.read_text(encoding="utf-8")
if "import sys" not in text:
    text = text.replace("import re\n", "import re\nimport sys\n")
if "DIFFVG_BUILD_TEMP" not in text:
    text = text.replace("if isinstance(ext, CMakeExtension):\n", "if isinstance(ext, CMakeExtension):\n            self.build_temp = os.environ.get('DIFFVG_BUILD_TEMP', self.build_temp)\n")
if "python{}{}.lib" not in text:
    text = text.replace(
        "python_library = get_config_var('LIBDIR')",
        "python_library = get_config_var('LIBDIR')\n            if platform.system() == \"Windows\":\n                python_library = os.path.join(sys.prefix, 'libs',\n                                              'python{}{}.lib'.format(sys.version_info.major,\n                                                                      sys.version_info.minor))",
    )
if "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" not in text:
    text = text.replace("'-DPYTHON_INCLUDE_PATH=' + include_path]", "'-DPYTHON_INCLUDE_PATH=' + include_path,\n                          '-DCMAKE_POLICY_VERSION_MINIMUM=3.5']")
if "CMAKE_GENERATOR" not in text:
    text = text.replace(
        "if sys.maxsize > 2**32:\n                    cmake_args += ['-A', 'x64']\n                build_args += ['--', '/m']",
        "cmake_generator = os.environ.get('CMAKE_GENERATOR', '')\n                if sys.maxsize > 2**32 and not cmake_generator.lower().startswith('ninja'):\n                    cmake_args += ['-A', 'x64']\n                build_args += ['--', '-j8' if cmake_generator.lower().startswith('ninja') else '/m']",
    )
setup.write_text(text, encoding="utf-8")

cmake = root / "CMakeLists.txt"
text = cmake.read_text(encoding="utf-8")
if "include_directories(thrust/dependencies/cub)" not in text:
    text = text.replace(
        "link_directories(${CUDA_LIBRARIES})",
        "link_directories(${CUDA_LIBRARIES})\n    include_directories(thrust)\n    include_directories(thrust/dependencies/cub)\n    include_directories($ENV{CUDA_PATH}/Library/include)\n    include_directories($ENV{CUDA_PATH}/Library/include/targets/x64)\n    include_directories($ENV{CUDA_PATH}/Library/include/targets/x64/cccl)",
    )
if "if(NOT MSVC)" not in text:
    text = text.replace(
        "add_compile_options(-Wall -g -O3 -fvisibility=hidden -Wno-unknown-pragmas)",
        "if(NOT MSVC)\n  add_compile_options(-Wall -g -O3 -fvisibility=hidden -Wno-unknown-pragmas)\nelse()\n  add_compile_options(/Wall /Zi)\n  add_link_options(/DEBUG)\nendif()",
    )
cmake.write_text(text, encoding="utf-8")
'@
    $patcher = $patcher.Replace("__SOURCE_DIR__", ($SourceDir -replace "\\", "\\"))
    $patcher | & (Join-Path (Get-EnvPrefix $script:CondaExe $script:EnvironmentName) "python.exe") -
    if ($LASTEXITCODE -ne 0) { throw "Failed to patch DiffVG source." }
}

$script:CondaExe = Get-Conda
$script:EnvironmentName = $EnvironmentName
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

if (-not (Test-EnvExists $CondaExe $EnvironmentName)) {
    Invoke-Checked $CondaExe @("create", "-n", $EnvironmentName, "-y", "python=3.11", "pip")
}

Invoke-Checked $CondaExe @("install", "-n", $EnvironmentName, "-y", "pytorch=2.5.1", "torchvision=0.20.1", "pytorch-cuda=12.4", "cuda-nvcc=12.4", "cuda-libraries-dev=12.4", "numpy=1.26.4", "cmake", "ninja", "-c", "pytorch", "-c", "nvidia", "-c", "conda-forge")
Invoke-Checked $CondaExe @("run", "-n", $EnvironmentName, "python", "-m", "pip", "install", "-r", "requirements-windows.txt")

$envPrefix = Get-EnvPrefix $CondaExe $EnvironmentName
$envPython = Join-Path $envPrefix "python.exe"

Invoke-Checked $envPython @("-c", "import torch; print('Torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); assert torch.cuda.is_available()")

$diffvgSource = Join-Path $repoRoot "external\diffvg"
if (-not (Test-Path $diffvgSource)) {
    New-Item -ItemType Directory -Path (Split-Path $diffvgSource -Parent) -Force | Out-Null
    Invoke-Checked "git" @("clone", "--recursive", "https://github.com/BachiLi/diffvg.git", $diffvgSource)
}
Invoke-Checked "git" @("-C", $diffvgSource, "submodule", "update", "--init", "--recursive")
Invoke-Checked "git" @("-C", (Join-Path $diffvgSource "pybind11"), "fetch", "--tags")
Invoke-Checked "git" @("-C", (Join-Path $diffvgSource "pybind11"), "checkout", "v2.13.6")
Patch-DiffVGSource $diffvgSource

$needsBuild = $ForceDiffVGBuild.IsPresent
if (-not $needsBuild) {
    & $envPython -c "import pydiffvg; print(pydiffvg.__file__)"
    $needsBuild = $LASTEXITCODE -ne 0
}
if ($needsBuild) {
    $vcvars = Get-VcVars64
    $buildTemp = Join-Path $env:TEMP "im2vec_diffvg_build_vs2022"
    if (Test-Path $buildTemp) { Remove-Item -LiteralPath $buildTemp -Recurse -Force }
    $cmd = "call `"$vcvars`" && set CONDA_PREFIX=$envPrefix&& set CUDA_HOME=$envPrefix&& set CUDA_PATH=$envPrefix&& set PATH=$envPrefix\bin;$envPrefix\Library\bin;$envPrefix\Scripts;!PATH!&& set DIFFVG_CUDA=1&& set CMAKE_GENERATOR=Ninja&& set DIFFVG_BUILD_TEMP=$buildTemp&& set CC=cl&& set CXX=cl&& `"$envPython`" -m pip install --no-build-isolation --verbose `"$diffvgSource`""
    & cmd /v:on /c $cmd
    if ($LASTEXITCODE -ne 0) { throw "DiffVG build failed." }
}

Invoke-Checked $envPython @("validate_pydiffvg_windows.py")

$resolvedCheckpoint = Resolve-Path -LiteralPath (Split-Path $CheckpointPath -Parent) -ErrorAction SilentlyContinue
if (-not $resolvedCheckpoint) {
    New-Item -ItemType Directory -Path (Split-Path $CheckpointPath -Parent) -Force | Out-Null
}
if (-not (Test-Path $CheckpointPath)) {
    $tmp = "$CheckpointPath.part"
    if (Test-Path $tmp) { Remove-Item -LiteralPath $tmp -Force }
    Invoke-WebRequest -Uri $CheckpointUrl -OutFile $tmp -UseBasicParsing
    $file = Get-Item $tmp
    if ($file.Length -lt 1000000) { throw "Downloaded checkpoint is implausibly small." }
    $bytes = [System.IO.File]::ReadAllBytes((Resolve-Path $tmp))
    $first = [System.Text.Encoding]::ASCII.GetString($bytes, 0, [Math]::Min(64, $bytes.Length))
    if ($first.TrimStart().StartsWith("<")) { throw "Downloaded checkpoint appears to be HTML." }
    Move-Item -LiteralPath $tmp -Destination $CheckpointPath
}

$hash = Get-FileHash $CheckpointPath -Algorithm SHA256
Write-Host "Checkpoint: $((Resolve-Path $CheckpointPath).Path)"
Write-Host "Checkpoint SHA256: $($hash.Hash)"
Invoke-Checked $envPython @("-c", "import torch, torchvision, yaml, numpy, PIL, pydiffvg; print('Core imports succeeded')")
