# Raster-to-SVG Vectorization with DINOv3 and DiffVG

This repository contains the project's vectorization model and a web app that
uses it directly. The active backend no longer runs SuperSVG: it imports the
real pipeline from `implementation/`, loads the model once at startup, then
reuses the same instance for every request.

## Project structure

```text
implementation/
  inference.py                    Full inference pipeline
  dataset.py                      RGB reading, SLIC, region preparation
  encoder.py                      DINOv3 encoder and path head
  model.py                        Model, DiffVG rendering and SVG export
  checkpoints/                    Trained checkpoints
  diffvg/                         Local DiffVG submodule, with Windows patches

vectorization-app/
  backend/
    main.py                       FastAPI API
    model_service.py              Persistent adapter to implementation/
    config.py                     .env configuration and pathlib paths
    SuperSVG/                     Legacy engine, kept but unused
  frontend/                       Existing React/Vite/Tailwind interface
```

## Pipeline used

The reference entry point is `implementation/inference.py`. The backend
directly reuses its `build_model`, `prepare_regions`, and
`predict_global_strokes` functions, along with the rendering/export functions
from `implementation/model.py`.

For each image:

1. Pillow converts the full image to `float32` RGB in `[0, 1]`.
2. SLIC segments the image; each region is cropped and resized to
   `224 x 224`, with the background set to `-1`.
3. The DINOv3 `vit_small_patch16_dinov3` model predicts 128 paths per region.
4. The 12 points of each closed path (four cubic Béziers) are remapped into
   the full image's coordinates.
5. DiffVG generates the SVG at the original dimensions and a PNG preview on a
   black background.

The default checkpoint used is:

```text
implementation/checkpoints/raster_to_svg_128paths/epoch_0019.pt
```

It corresponds to the `ours_final` model evaluated in the repo. Do not replace
it with `raster_to_svg_128paths/latest.pt`: the latter is epoch 14.

## Windows prerequisites

- Windows 10 or 11, 64-bit;
- Python 3.11 recommended;
- Node.js 18+ (Node.js 20 or 24 recommended) and npm;
- Git with submodules initialized;
- CMake and Visual Studio Build Tools with the C++ component to compile
  DiffVG;
- the checkpoint above, or another compatible checkpoint set in `.env`.

Initialize DiffVG if the submodule is empty:

```powershell
git submodule update --init --recursive
```

### GPU and CUDA (RTX 4070 SUPER)

The GPU is optional: `MODEL_DEVICE=auto` uses CUDA when
`torch.cuda.is_available()` is `True`, otherwise it falls back to CPU.

The configuration actually validated on the project's machine is: Python
3.11.15, PyTorch 2.5.1, torchvision 0.20.1, PyTorch CUDA runtime 12.4, and an
RTX 4070 SUPER. This combination is a verified reference, not a requirement
for every machine. To compile a fresh GPU DiffVG extension, a CUDA Toolkit
compatible with the PyTorch build and `nvcc` are also required.

Quick check:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Backend installation

From PowerShell:

```powershell
cd vectorization-app/backend
py -3.11 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
```

Install the desired PyTorch variant first. For the validated CUDA 12.4
configuration:

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

For a CPU backend:

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1
```

Then install the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
```

DiffVG/pydiffvg is a local native extension with no portable Windows wheel.
After PyTorch, install it from the submodule. For CUDA:

```powershell
Push-Location ..\..\implementation\diffvg
$env:DIFFVG_CUDA="1"
python setup.py install
Pop-Location
```

For a CPU build, use `$env:DIFFVG_CUDA="0"`. Then verify:

```powershell
python -c "import torch, pydiffvg, diffvg; print(torch.cuda.is_available()); print(pydiffvg.__file__); print(diffvg.__file__)"
```

## Backend configuration

Create the local `.env` file:

```powershell
Copy-Item .env.example .env
```

Provided values:

```env
MODEL_CHECKPOINT=../../implementation/checkpoints/raster_to_svg_128paths/epoch_0019.pt
MODEL_DEVICE=auto
IMPLEMENTATION_DIR=../../implementation
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
MAX_UPLOAD_MB=25
```

Relative paths are resolved with `pathlib` from `vectorization-app/backend`.
`MODEL_DEVICE` only accepts `auto`, `cpu`, or `cuda`.

Start with code reload:

```powershell
python -m uvicorn main:app --reload
```

`python main.py` reads `BACKEND_HOST` and `BACKEND_PORT` directly. With the
Uvicorn CLI, add `--host` and `--port` if values other than the defaults are
needed.

## API

- `GET /`: server status;
- `GET /health`: model status, device, and checkpoint;
- `POST /vectorize`: multipart form containing `file` (PNG or JPEG) and
  `num_regions` (optional integer between 2 and 256, default: 64);
- `GET /download/{job_id}/{filename}`: SVG download;
- `GET /preview/{job_id}/{filename}`: PNG preview produced by the pipeline.

Vectorization response:

```json
{
  "job_id": "...",
  "svg_filename": "vectorized.svg",
  "download_url": "/download/.../vectorized.svg",
  "preview_filename": "preview.png",
  "preview_url": "/preview/.../preview.png",
  "device": "cuda",
  "num_regions": 64,
  "region_count": 49,
  "path_count": 6272
}
```

The model and checkpoint are loaded in the FastAPI lifespan, once per
process. Inferences use `torch.inference_mode()` and are serialized by a
lock, since DiffVG keeps its device in a global state.

## Frontend installation

```powershell
cd vectorization-app/frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Vite configuration:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Without this variable, the frontend falls back to the same default URL. The
design, drag-and-drop, comparison viewer, retry, and SVG download are
preserved. The old SuperSVG parameters are no longer sent.

Production build:

```powershell
npm run build
```

## Common errors

### Checkpoint not found

Check `http://127.0.0.1:8000/health`. The `checkpoint` field shows the
resolved path and `error` specifies the missing file. Fix
`MODEL_CHECKPOINT`; from the backend, `implementation/` is located at
`../../implementation`.

### CUDA unavailable

Temporarily use `MODEL_DEVICE=cpu`, or check the NVIDIA driver and the
installed PyTorch variant. `MODEL_DEVICE=cuda` fails explicitly if
`torch.cuda.is_available()` is false.

### PyTorch and CUDA incompatible

Compare `torch.version.cuda`, the driver version shown by `nvidia-smi`, and
the version used to compile DiffVG. Reinstall PyTorch, torchvision, and the
DiffVG extension together in the same Python environment.

### pydiffvg or DiffVG missing

A `No module named pydiffvg` or `No module named diffvg` error means the
native submodule is not installed in the active `venv`. Redo the
`implementation/diffvg` step. An extension compiled for a different Python
version cannot be reused.

### CORS

The backend allows `localhost` and `127.0.0.1` on Vite ports 5173 and 5174.
Use one of these origins, or explicitly add another origin in
`backend/main.py`.

### Backend not running

If the interface shows `Backend is not running`, start Uvicorn, check
`/health`, then confirm that `VITE_API_BASE_URL` matches the backend's
address.

## How to start the app

Backend:

```powershell
cd vectorization-app/backend
.\venv\Scripts\activate
python -m uvicorn main:app --reload
```

Frontend:

```powershell
cd vectorization-app/frontend
npm install
npm run dev
```

Browser:

```text
http://localhost:5173
```
