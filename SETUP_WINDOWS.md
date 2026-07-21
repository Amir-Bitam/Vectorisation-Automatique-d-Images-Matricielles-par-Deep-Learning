# LIVE Native Windows Setup

This repository has been validated for native Windows execution with an isolated Conda environment named `live-win`. LIVE performs per-image optimization with DiffVG; it does not load pretrained neural network weights for the example SVG generation workflow.

## Prerequisites

- Windows 10 or Windows 11, 64-bit.
- NVIDIA GPU and driver new enough for CUDA 12.4 runtime. Tested with NVIDIA driver 591.86 and an RTX 4070 SUPER.
- Anaconda or Miniconda.
- Git for Windows.
- Visual Studio 2022 Build Tools with:
  - `Desktop development with C++`
  - MSVC v143 x64/x86 build tools
  - Windows 10 or Windows 11 SDK
  - C++ CMake tools are acceptable, but this setup uses Conda CMake and Ninja.

The setup script looks for `vcvars64.bat` with `vswhere`, and also supports the tested path `C:\Program\VC\Auxiliary\Build\vcvars64.bat`.

## Tested Versions

- Repository commit: `679e1d16c5809367f2d2db3e403a8548c5419258`
- Python: 3.9.23
- PyTorch: 2.4.1
- PyTorch CUDA runtime: 12.4
- Conda CUDA compiler: nvcc 12.4.131
- CMake: 3.26.4
- Ninja: 1.13.2
- MSVC: 19.44.35227
- Windows SDK: 10.0.26100.0

Python 3.9 is used because it is modern enough for current Windows wheels while still compatible with this LIVE source and the bundled DiffVG/pybind11 code after the small Windows build patches in this repository.

## Setup

From the repository root:

```powershell
.\setup_live_windows.ps1
```

The script:

- creates or updates only the `live-win` Conda environment;
- installs CUDA-enabled PyTorch with CUDA 12.4;
- installs pip-only dependencies from `requirements-windows.txt`;
- runs `git submodule update --init --recursive`;
- builds the bundled `DiffVG` directory with MSVC, Ninja, and Conda CUDA;
- verifies CUDA availability, major imports, and a DiffVG backward pass.

The repository currently has no root `.gitmodules` file, so the submodule command is harmless. The bundled DiffVG support directories such as `pybind11` and `thrust` are already present in the tree.

## CUDA Verification

```powershell
$envPath = "$env:USERPROFILE\anaconda3\envs\live-win"
$env:PATH = "$envPath;$envPath\Library\bin;$envPath\Scripts;$envPath\bin;$env:PATH"
& "$envPath\python.exe" -c "import torch; print('Torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

CUDA must print `CUDA available: True`. If it does not, stop and fix the driver/environment before running LIVE.

## Run The Included Example

```powershell
.\run_live_windows.ps1 `
    -InputImage ".\LIVE\figures\smile.png" `
    -Signature "windows_smoke_test" `
    -Experiment "experiment_5x1" `
    -Config "config/base.yaml" `
    -LogDirectory "log"
```

The wrapper runs `LIVE\main.py` from the correct working directory and writes a timestamped command log under `windows_setup_logs`.

## Run A Custom Image

```powershell
.\run_live_windows.ps1 `
    -InputImage "C:\path\to\image.png" `
    -Signature "my_test_001" `
    -Experiment "experiment_5x1" `
    -Config "config/base.yaml" `
    -LogDirectory "log"
```

Use a unique signature for each run so result directories are easy to identify.

## Expected Outputs

For a signature such as `windows_smoke_test`, LIVE writes a directory like:

```text
LIVE\log\<timestamp>_windows_smoke_test\
```

Important files include:

- `config.yaml`: copied run configuration.
- `svg-init\*-init.svg`: initialized paths.
- `demo-png\*.png`: rendered raster previews after each path stage.
- `output-svg\*.svg`: generated vector outputs.
- `video-png\*.png`: optimization frames.
- `video-avi\*.avi`: optimization videos.

For `experiment_5x1`, the final SVG is normally:

```text
output-svg\1-1-1-1-1.svg
```

## Known Limitations

- The upstream DiffVG build uses legacy `setup.py install`; the script keeps that path because this repository is not packaged for modern PEP 517 installation on Windows.
- Visual Studio/MSBuild generator builds stalled in compiler probes under the long repository path. The setup uses Ninja and a temporary `subst` drive mapping to keep native Windows paths short.
- Build logs may contain MSVC warnings from upstream DiffVG C++ code. They did not prevent import, CUDA rendering, or gradient verification.
- LIVE's video frame saving can emit low-contrast image warnings for early optimization frames. These warnings are expected and are now visible in the logs.

## Remove Only The LIVE Environment

```powershell
conda env remove -n live-win
```

This removes only the isolated LIVE environment and does not affect other DiffVG or SuperSVG environments.
