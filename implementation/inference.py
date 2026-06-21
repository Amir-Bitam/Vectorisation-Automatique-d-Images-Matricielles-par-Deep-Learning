"""Run full-image raster-to-SVG inference with SLIC region remapping."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import pydiffvg

from dataset import crop_resize_region, load_rgb_image, segment_image
from model import (
    RasterToSVGModel,
    configure_pydiffvg,
    render_rgba_strokes,
    rgba_to_rgb_black,
    save_strokes_as_svg,
)


def checkpoint_args(checkpoint: dict) -> dict:
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else {}


def build_model(args: argparse.Namespace, device: torch.device) -> RasterToSVGModel:
    checkpoint = None
    ckpt_args: dict = {}
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ckpt_args = checkpoint_args(checkpoint)

    model = RasterToSVGModel(
        num_paths=args.num_paths or int(ckpt_args.get("num_paths", 32)),
        canvas_size=args.image_size or int(ckpt_args.get("image_size", 224)),
        dino_model_name=args.dino_model or str(ckpt_args.get("dino_model", "vit_small_patch16_dinov3")),
        pretrained=args.pretrained and checkpoint is None,
    ).to(device)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def prepare_regions(
    image: np.ndarray,
    n_segments: int,
    image_size: int,
    compactness: float,
    sigma: float,
) -> list[tuple[torch.Tensor, tuple[int, int, int, int]]]:
    labels = segment_image(image, n_segments=n_segments, compactness=compactness, sigma=sigma)
    regions: list[tuple[torch.Tensor, tuple[int, int, int, int]]] = []

    for label in np.unique(labels):
        mask = labels == label
        if not mask.any():
            continue
        sample = crop_resize_region(image, mask, output_size=image_size)
        regions.append((sample.model_input, sample.bbox))

    return regions


def local_to_global_strokes(strokes: torch.Tensor, bbox: tuple[int, int, int, int]) -> torch.Tensor:
    x1, y1, x2, y2 = bbox
    width = float(max(1, x2 - x1))
    height = float(max(1, y2 - y1))

    global_strokes = strokes.detach().float().clone()
    points = global_strokes[:, :24].reshape(-1, 12, 2)
    points[:, :, 0] = float(x1) + width * points[:, :, 0]
    points[:, :, 1] = float(y1) + height * points[:, :, 1]
    global_strokes[:, :24] = points.reshape(-1, 24)
    return global_strokes


@torch.no_grad()
def predict_global_strokes(
    model: RasterToSVGModel,
    regions: list[tuple[torch.Tensor, tuple[int, int, int, int]]],
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> torch.Tensor:
    all_strokes: list[torch.Tensor] = []
    amp_enabled = amp and device.type == "cuda"

    for start in tqdm(range(0, len(regions), batch_size), desc="regions", dynamic_ncols=True):
        chunk = regions[start : start + batch_size]
        inputs = torch.stack([item[0] for item in chunk], dim=0).to(device)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            predicted = model.encoder(inputs).float().cpu()

        for local_strokes, (_, bbox) in zip(predicted, chunk):
            all_strokes.append(local_to_global_strokes(local_strokes, bbox))

    if not all_strokes:
        raise RuntimeError("No SLIC regions were produced for the input image.")
    return torch.cat(all_strokes, dim=0)


def run_inference(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_pydiffvg(device)

    image = load_rgb_image(args.input)
    height, width = image.shape[:2]
    model = build_model(args, device)

    regions = prepare_regions(
        image,
        n_segments=args.num_regions,
        image_size=model.canvas_size,
        compactness=args.slic_compactness,
        sigma=args.slic_sigma,
    )
    print(f"regions={len(regions)} strokes_per_region={model.encoder.stroke_tokens.shape[1]}")

    global_strokes = predict_global_strokes(
        model=model,
        regions=regions,
        device=device,
        batch_size=args.region_batch_size,
        amp=args.amp,
    ).to(device)

    rgba = render_rgba_strokes(
        global_strokes,
        width=width,
        height=height,
        normalized_coordinates=False,
        samples=args.samples,
    )
    rgb = rgba_to_rgb_black(rgba)[0].permute(1, 2, 0).detach().cpu()

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    pydiffvg.imwrite(rgb, str(args.output_png), gamma=1.0)
    save_strokes_as_svg(
        args.output_svg,
        global_strokes.detach().cpu(),
        width=width,
        height=height,
        normalized_coordinates=False,
    )
    print(f"wrote PNG: {args.output_png.resolve()}")
    print(f"wrote SVG: {args.output_svg.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Single RGB image to vectorize.")
    parser.add_argument("--checkpoint", type=Path, help="Training checkpoint to load.")
    parser.add_argument("--output-svg", type=Path, default=Path("implementation") / "outputs" / "prediction.svg")
    parser.add_argument("--output-png", type=Path, default=Path("implementation") / "outputs" / "prediction.png")
    parser.add_argument("--device", default="auto")

    parser.add_argument("--dino-model", default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-paths", type=int)
    parser.add_argument("--image-size", type=int)

    parser.add_argument("--num-regions", type=int, default=64)
    parser.add_argument("--region-batch-size", type=int, default=16)
    parser.add_argument("--slic-compactness", type=float, default=10.0)
    parser.add_argument("--slic-sigma", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
