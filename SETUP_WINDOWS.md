# Im2Vec Native Windows Setup

This repository is the official `preddy5/Im2Vec` CVPR 2021 codebase. The upstream GitHub repository is archived and read-only as of November 24, 2022: https://github.com/preddy5/Im2Vec

## Scope And Limitations

Im2Vec is a learned generative model, not a universal optimization-based vectorizer like LIVE or Potrace. The official checkpoint validated here is trained on the emoji domain.

Supported experiments in this repaired Windows workflow:

- Reconstruct images from the included emoji dataset/domain.
- Pass a user raster through the pretrained encoder and vector decoder.
- Generate random latent samples from the checkpoint.
- Generate latent-space interpolation PNG grids.
- Start a tiny training smoke test.

The custom-image wrapper does not fine-tune, optimize, or postprocess the SVG. Poor out-of-domain quality is expected and is not an installation failure.

## Tested Environment

- Windows 10/11 64-bit, PowerShell.
- GPU: NVIDIA GeForce RTX 4070 SUPER.
- NVIDIA driver observed: 591.86.
- Conda environment: `im2vec-win`.
- Python: 3.11.15.
- PyTorch: 2.5.1 with CUDA runtime 12.4.
- torchvision: 0.20.1.
- PyTorch Lightning: 1.9.5.
- NumPy: 1.26.4.
- DiffVG source: official `BachiLi/diffvg`, commit `85802a71fbcc72d79cb75716eb4da4392fd09532`.
- `pydiffvg`: native Windows build installed inside `im2vec-win`.

## Prerequisites

Required:

- Conda or Anaconda.
- Git.
- NVIDIA driver supporting the RTX 4070 SUPER.
- Visual Studio 2022 Build Tools with:
  - Desktop development with C++ workload.
  - MSVC v143 x64/x86 build tools.
  - Windows 10 or 11 SDK.
- CMake and Ninja. The setup installs Conda copies into `im2vec-win`.

Do not use Visual Studio 2026 for the DiffVG CUDA 12.4 build. `nvcc` rejects the newer MSVC host compiler. On this machine the working VS 2022 environment script is:

```powershell
C:\Program\VC\Auxiliary\Build\vcvars64.bat
```

## Install Or Repair

From the repository root:

```powershell
.\setup_im2vec_windows.ps1
```

The script:

- Creates `im2vec-win` only if it does not already exist.
- Installs CUDA-enabled PyTorch, torchvision, CUDA 12.4 toolkit components, CMake and Ninja.
- Installs pip runtime dependencies from `requirements-windows.txt`.
- Clones or reuses official DiffVG source under `external\diffvg`.
- Applies the Windows/Python 3.11 DiffVG build patches.
- Builds `pydiffvg` inside `im2vec-win`.
- Verifies CUDA and `pydiffvg` rendering/backpropagation.
- Downloads/verifies the official `epoch=667.ckpt`.

## Dependency Decisions

- `kaolin==0.0` is not installed. The repository does not import `kaolin`.
- PyPI `diffvg` is not used blindly. The Python module needed by Im2Vec is `pydiffvg`, built from official DiffVG source.
- PyTorch Lightning 0.9.0 is not used because it is not practical with the CUDA-enabled modern Windows/PyTorch stack needed for RTX 4070 SUPER. The code was patched minimally for Lightning 1.9.5.
- `setuptools<81` is pinned because Lightning 1.9 imports `pkg_resources`.
- SVG rerasterization is performed with `pydiffvg`, not CairoSVG.

## Checkpoint

Official URL:

```powershell
http://geometry.cs.ucl.ac.uk/projects/2021/im2vec/paper_docs/epoch=667.ckpt
```

Expected local path:

```powershell
.\logs\VectorVAEnLayers\version_110\epoch=667.ckpt
```

Validated SHA-256:

```text
9162b31c54ec4abcde34d16f6b880276b1be797995a4a7c3f77697ee9d952feb
```

The checkpoint is a PyTorch Lightning checkpoint with `epoch=668`, `global_step=11462`, and a strict `state_dict` match. Missing keys: none. Unexpected keys: none.

## Official Evaluation

Use the log-directory snapshot config for the official checkpoint:

```powershell
.\run_im2vec_eval_windows.ps1 `
    -Checkpoint ".\logs\VectorVAEnLayers\version_110\epoch=667.ckpt" `
    -Config ".\logs\VectorVAEnLayers\version_110\configs\emoji.yaml" `
    -OutputDirectory ".\outputs\official_eval"
```

Why this config: the upstream README runs evaluation from `logs\VectorVAEnLayers\version_110`, where `configs\emoji.yaml` is a snapshot. That snapshot uses hard compositing. The repository-root config currently differs by using soft compositing, so it is not the safest default for reproducing the official logged checkpoint behavior.

Expected outputs include:

- `official_input_grid.png`
- `official_reconstruction_grid.png`
- `official_interpolate_img.png`
- `official_interpolate2D_image.png`
- `official_visualize_sampling_image.png`
- `official_naive_interpolate_image.png`
- `official_reconstruction_000.svg`
- `official_random_sample.svg`
- rerasterized PNGs and side-by-side comparisons
- `official_eval_summary.json`

## Custom Raster Reconstruction

The wrapper is supported because the model has an encoder. It is still an emoji-domain checkpoint.

```powershell
.\run_im2vec_image_windows.ps1 `
    -InputImage "C:\path\to\icon.png" `
    -OutputDirectory ".\outputs\icon_test"
```

Equivalent direct command:

```powershell
conda run -n im2vec-win python reconstruct_image.py `
    --input "C:\path\to\icon.png" `
    --checkpoint ".\logs\VectorVAEnLayers\version_110\epoch=667.ckpt" `
    --config ".\logs\VectorVAEnLayers\version_110\configs\emoji.yaml" `
    --output-dir ".\outputs\icon_test"
```

Preprocessing matches training data loading:

- `PIL.Image.open(...).convert("RGB")`
- Resize to 128.
- Center crop to 128.
- `ToTensor()`.
- No normalization beyond `[0, 1]`.

## Training Smoke Test

This is not full training. It verifies that training can start, computes a finite loss, backpropagates finite gradients, changes at least one parameter, saves a checkpoint, and reloads it strictly.

```powershell
conda run -n im2vec-win python train_smoke_windows.py `
    --config ".\configs\emoji_windows_smoke.yaml" `
    --output-dir ".\outputs\training_smoke"
```

## Validation Commands

CUDA PyTorch:

```powershell
conda run -n im2vec-win python -c "import torch; print('Torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('Capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'NONE')"
```

Core imports:

```powershell
conda run -n im2vec-win python -c "import torch, torchvision, yaml, numpy, PIL, pydiffvg; print('Core imports succeeded')"
```

DiffVG renderer/backpropagation:

```powershell
conda run -n im2vec-win python validate_pydiffvg_windows.py
```

Syntax:

```powershell
conda run -n im2vec-win python -m compileall .
```

Environment report:

```powershell
conda run -n im2vec-win python write_im2vec_environment_report.py `
    --checkpoint ".\logs\VectorVAEnLayers\version_110\epoch=667.ckpt" `
    --official-summary ".\outputs\official_eval\official_eval_summary.json" `
    --output ".\outputs\im2vec_environment_report.json"
```

## Known Warnings

- Lightning 1.9 emits a `pkg_resources` deprecation warning. `setuptools<81` is pinned so the import still works.
- PyTorch 2.5 warns that the default `torch.load(weights_only=False)` behavior will change in the future. This workflow loads the official Im2Vec checkpoint from the upstream project.
- The root config and log snapshot config differ. Use the log snapshot for official checkpoint evaluation.
- Generated SVGs are exported from decoded vector layers. The model's hard/soft raster compositing is not guaranteed to be identical to every SVG viewer's compositing.

## Delete Only This Environment

```powershell
conda env remove -n im2vec-win
```

This does not touch other DiffVG, LIVE or SuperSVG environments.
