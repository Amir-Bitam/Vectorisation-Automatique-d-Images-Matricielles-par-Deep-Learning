# DINOv3 Raster-to-SVG Path Vectorization

This repository contains a PyTorch pipeline that vectorizes raster images by:

1. extracting image tokens with a DINOv3 ViT encoder,
2. decoding a bounded sequence of SVG path queries,
3. predicting cubic Bezier stroke paths, colors, widths, alpha, visibility, and stop logits,
4. training with a pure PyTorch differentiable soft-stroke renderer,
5. writing a variable number of SVG `<path>` elements at inference time.

The sequence has a configurable upper bound (`--max-paths`), while the actual SVG output count is selected by the learned visibility and stop predictions.

## Setup

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r implementation\requirements.txt
```

If `python` is not installed on Windows, install Python 3.10+ first, then reopen PowerShell.

## Train

Train from a folder of ordinary images:

```powershell
.\.venv\Scripts\python implementation\main.py `
  --dataset-root path\to\images `
  --device auto `
  --dino-source timm `
  --image-size 128 `
  --batch-size 2 `
  --epochs 10 `
  --max-paths 64 `
  --num-segments 3 `
  --checkpoint-out implementation\checkpoints\dinov3_svg_paths.pt
```

For a quick smoke test:

```powershell
.\.venv\Scripts\python implementation\main.py `
  --dataset-root path\that\does\not\exist `
  --device cpu `
  --dino-source timm `
  --image-size 64 `
  --batch-size 1 `
  --epochs 1 `
  --max-batches 1 `
  --max-paths 8 `
  --num-segments 1 `
  --hidden-dim 128 `
  --decoder-heads 4 `
  --decoder-layers 2 `
  --checkpoint-out implementation\checkpoints\smoke_svg_paths.pt
```

To use pretrained DINOv3 weights, add `--dino-pretrained`. If your environment cannot access the public checkpoint, pass a local file or folder with `--dino-weights path\to\checkpoint`.

## Inference

```powershell
.\.venv\Scripts\python implementation\infer.py `
  path\to\input.png `
  implementation\outputs\input.svg `
  --checkpoint implementation\checkpoints\dinov3_svg_paths.pt `
  --device auto `
  --dino-source timm `
  --min-alpha 0.02 `
  --visible-threshold 0.35 `
  --stop-threshold 0.65
```

Lower `--visible-threshold` or `--min-alpha` to emit more paths. Raise them for a smaller SVG.
