"""Evaluate raster-to-SVG checkpoints on a fixed held-out image manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lpips
import numpy as np
import pydiffvg
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from dataset import crop_resize_region, load_rgb_image, segment_image
from inference import checkpoint_args, local_to_global_strokes
from model import (
    RasterToSVGModel,
    configure_pydiffvg,
    render_rgba_strokes,
    rgba_to_rgb_black,
    save_strokes_as_svg,
)


DEFAULT_MODELS = (
    "paths32_e09=implementation/checkpoints/raster_to_svg_serious_v1/epoch_0009.pt",
    "paths128_e19=implementation/checkpoints/raster_to_svg_128paths/epoch_0019.pt",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_model_spec(value: str) -> ModelSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Model specifications must use NAME=CHECKPOINT.")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Model specifications must use NAME=CHECKPOINT.")
    return ModelSpec(name=name.strip(), checkpoint=Path(raw_path.strip()))


def read_manifest(manifest: Path, dataset_root: Path, limit: int | None) -> list[Path]:
    entries = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        if not path.is_absolute():
            path = dataset_root / path
        if not path.is_file():
            raise FileNotFoundError(f"Manifest image does not exist: {path}")
        entries.append(path)
    if limit is not None:
        entries = entries[:limit]
    if not entries:
        raise RuntimeError(f"No test images were listed in {manifest}.")
    return entries


def resize_longest_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = float(max_side) / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    uint8 = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    resized = Image.fromarray(uint8, mode="RGB").resize(
        (new_width, new_height), Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def resize_square(image: np.ndarray, size: int) -> np.ndarray:
    uint8 = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    resized = Image.fromarray(uint8, mode="RGB").resize(
        (size, size), Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    uint8 = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(uint8, mode="RGB").save(path)


def prepare_regions(
    image: np.ndarray,
    n_segments: int,
    image_size: int,
    compactness: float,
    sigma: float,
) -> tuple[np.ndarray, list[tuple[torch.Tensor, tuple[int, int, int, int]]]]:
    labels = segment_image(
        image,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
    )
    regions: list[tuple[torch.Tensor, tuple[int, int, int, int]]] = []
    for label in np.unique(labels):
        mask = labels == label
        if not mask.any():
            continue
        sample = crop_resize_region(image, mask, output_size=image_size)
        regions.append((sample.model_input, sample.bbox))
    if not regions:
        raise RuntimeError("SLIC produced no non-empty region.")
    return labels, regions


@torch.no_grad()
def predict_strokes(
    model: RasterToSVGModel,
    regions: list[tuple[torch.Tensor, tuple[int, int, int, int]]],
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> torch.Tensor:
    all_strokes: list[torch.Tensor] = []
    amp_enabled = amp and device.type == "cuda"
    for start in range(0, len(regions), batch_size):
        chunk = regions[start : start + batch_size]
        inputs = torch.stack([item[0] for item in chunk], dim=0).to(device)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            local_batch = model.encoder(inputs).float().cpu()
        for local_strokes, (_, bbox) in zip(local_batch, chunk):
            all_strokes.append(local_to_global_strokes(local_strokes, bbox))
    strokes = torch.cat(all_strokes, dim=0)
    if not torch.isfinite(strokes).all():
        raise RuntimeError("The model predicted non-finite SVG parameters.")
    return strokes


def load_model(spec: ModelSpec, device: torch.device) -> tuple[RasterToSVGModel, dict[str, Any]]:
    checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint_args(checkpoint)
    model = RasterToSVGModel(
        num_paths=int(saved_args.get("num_paths", 32)),
        canvas_size=int(saved_args.get("image_size", 224)),
        dino_model_name=str(saved_args.get("dino_model", "vit_small_patch16_dinov3")),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    metadata = {
        "name": spec.name,
        "checkpoint": str(spec.checkpoint.resolve()),
        "epoch_zero_based": int(checkpoint.get("epoch", -1)),
        "num_paths": int(model.encoder.stroke_tokens.shape[1]),
        "canvas_size": int(model.canvas_size),
        "dino_model": str(saved_args.get("dino_model", "vit_small_patch16_dinov3")),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "dino_parameters": int(sum(parameter.numel() for parameter in model.encoder.dino.parameters())),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "training_args": {key: str(value) if isinstance(value, Path) else value for key, value in saved_args.items()},
    }
    return model, metadata


def warm_up(model: RasterToSVGModel, device: torch.device, amp: bool) -> None:
    with torch.no_grad():
        dummy = torch.zeros(1, 3, model.canvas_size, model.canvas_size, device=device)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            strokes = model.encoder(dummy).float()
        render_rgba_strokes(strokes[:, :1], width=32, height=32, samples=1)
    synchronize(device)


def image_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    perceptual_model: torch.nn.Module,
    metric_device: torch.device,
) -> dict[str, float]:
    target = np.clip(target.astype(np.float32), 0.0, 1.0)
    prediction = np.clip(prediction.astype(np.float32), 0.0, 1.0)
    mse = float(np.mean((target - prediction) ** 2, dtype=np.float64))
    psnr = float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
    ssim = float(structural_similarity(target, prediction, data_range=1.0, channel_axis=-1))

    target_tensor = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).to(metric_device)
    prediction_tensor = torch.from_numpy(prediction).permute(2, 0, 1).unsqueeze(0).to(metric_device)
    with torch.no_grad():
        lpips_value = perceptual_model(2.0 * target_tensor - 1.0, 2.0 * prediction_tensor - 1.0)
    return {
        "mse": mse,
        "psnr_db": psnr,
        "ssim": ssim,
        "lpips": float(lpips_value.item()),
    }


def validate_svg(path: Path, expected_paths: int, width: int, height: int) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    path_count = sum(1 for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "path")
    if path_count != expected_paths:
        raise RuntimeError(f"{path} contains {path_count} paths; expected {expected_paths}.")

    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(str(path))
    if int(canvas_width) != width or int(canvas_height) != height:
        raise RuntimeError(
            f"Reloaded SVG dimensions are {canvas_width}x{canvas_height}; expected {width}x{height}."
        )
    if len(shapes) != expected_paths or len(shape_groups) != expected_paths:
        raise RuntimeError("DiffVG did not reload every exported path and shape group.")
    return {
        "svg_xml_valid": True,
        "svg_diffvg_reload_valid": True,
        "svg_path_count": path_count,
    }


def summarize(rows: list[dict[str, Any]], model_metadata: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "mse",
        "psnr_db",
        "ssim",
        "lpips",
        "preprocess_seconds",
        "network_seconds",
        "render_seconds",
        "svg_export_seconds",
        "inference_seconds",
        "metric_seconds",
        "peak_extra_gpu_mb",
        "svg_bytes",
        "path_count",
        "region_count",
    )
    summaries = []
    for metadata in model_metadata:
        selected = [row for row in rows if row["model"] == metadata["name"]]
        item: dict[str, Any] = {
            "model": metadata["name"],
            "checkpoint": metadata["checkpoint"],
            "epoch_zero_based": metadata["epoch_zero_based"],
            "num_paths_per_region": metadata["num_paths"],
            "images": len(selected),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summaries.append(item)
    return {"models": model_metadata, "summary": summaries, "rows": rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    metric_device = torch.device(args.metric_device)
    configure_pydiffvg(device)

    specs = args.model or [parse_model_spec(value) for value in DEFAULT_MODELS]
    image_paths = read_manifest(args.manifest, args.dataset_root, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device} metric_device={metric_device} images={len(image_paths)}")
    perceptual_model = lpips.LPIPS(net="alex", verbose=False).eval().to(metric_device)
    rows: list[dict[str, Any]] = []
    all_metadata: list[dict[str, Any]] = []

    for spec in specs:
        if not spec.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {spec.checkpoint}")
        print(f"\nloading {spec.name}: {spec.checkpoint}")
        model, metadata = load_model(spec, device)
        all_metadata.append(metadata)
        warm_up(model, device, args.amp)

        model_dir = args.output_dir / spec.name
        model_dir.mkdir(parents=True, exist_ok=True)

        for image_index, image_path in enumerate(image_paths, start=1):
            print(f"[{spec.name} {image_index:02d}/{len(image_paths):02d}] {image_path.name}")
            original = load_rgb_image(image_path)
            image = (
                resize_square(original, args.square_size)
                if args.square_size is not None
                else resize_longest_side(original, args.max_side)
            )
            height, width = image.shape[:2]
            image_id = image_path.stem
            input_path = args.output_dir / "inputs" / f"{image_id}.png"
            if not input_path.exists():
                save_rgb(input_path, image)

            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                baseline_memory = torch.cuda.memory_allocated(device)
            else:
                baseline_memory = 0

            synchronize(device)
            start = time.perf_counter()
            labels, regions = prepare_regions(
                image,
                n_segments=args.num_regions,
                image_size=model.canvas_size,
                compactness=args.slic_compactness,
                sigma=args.slic_sigma,
            )
            preprocess_seconds = time.perf_counter() - start

            synchronize(device)
            start = time.perf_counter()
            strokes = predict_strokes(
                model,
                regions,
                device=device,
                batch_size=args.region_batch_size,
                amp=args.amp,
            ).to(device)
            synchronize(device)
            network_seconds = time.perf_counter() - start

            synchronize(device)
            start = time.perf_counter()
            rendered_rgba = render_rgba_strokes(
                strokes,
                width=width,
                height=height,
                normalized_coordinates=False,
                samples=args.samples,
                seed=args.seed,
            )
            rendered = rgba_to_rgb_black(rendered_rgba)[0].permute(1, 2, 0).detach().cpu().numpy()
            synchronize(device)
            render_seconds = time.perf_counter() - start

            if not np.isfinite(rendered).all():
                raise RuntimeError("DiffVG produced non-finite pixels.")
            prediction_path = model_dir / f"{image_id}.png"
            svg_path = model_dir / f"{image_id}.svg"
            save_rgb(prediction_path, rendered)

            synchronize(device)
            start = time.perf_counter()
            save_strokes_as_svg(
                svg_path,
                strokes.detach().cpu(),
                width=width,
                height=height,
                normalized_coordinates=False,
            )
            svg_export_seconds = time.perf_counter() - start

            validation = validate_svg(svg_path, int(strokes.shape[0]), width, height)
            if image_index == 1:
                np.save(args.output_dir / "first_image_slic_labels.npy", labels)

            peak_extra_gpu_mb = 0.0
            if device.type == "cuda":
                peak_extra_gpu_mb = max(
                    0.0,
                    float(torch.cuda.max_memory_allocated(device) - baseline_memory) / (1024.0**2),
                )

            metric_start = time.perf_counter()
            metrics = image_metrics(image, rendered, perceptual_model, metric_device)
            metric_seconds = time.perf_counter() - metric_start
            inference_seconds = preprocess_seconds + network_seconds + render_seconds + svg_export_seconds

            row: dict[str, Any] = {
                "model": spec.name,
                "image": image_path.name,
                "source_index": image_index,
                "width": width,
                "height": height,
                "region_count": len(regions),
                "paths_per_region": metadata["num_paths"],
                "path_count": int(strokes.shape[0]),
                **metrics,
                "preprocess_seconds": preprocess_seconds,
                "network_seconds": network_seconds,
                "render_seconds": render_seconds,
                "svg_export_seconds": svg_export_seconds,
                "inference_seconds": inference_seconds,
                "metric_seconds": metric_seconds,
                "peak_extra_gpu_mb": peak_extra_gpu_mb,
                "svg_bytes": svg_path.stat().st_size,
                "input_png_bytes": input_path.stat().st_size,
                **validation,
            }
            rows.append(row)
            print(
                f"  MSE={row['mse']:.5f} PSNR={row['psnr_db']:.2f} "
                f"SSIM={row['ssim']:.4f} LPIPS={row['lpips']:.4f} "
                f"time={row['inference_seconds']:.3f}s paths={row['path_count']}"
            )
            write_csv(args.output_dir / "metrics_partial.csv", rows)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = summarize(rows, all_metadata)
    write_csv(args.output_dir / "metrics.csv", rows)
    write_csv(args.output_dir / "summary.csv", report["summary"])
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
    print(f"\nwrote evaluation report to {args.output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("implementation") / "evaluation_manifest.txt",
    )
    parser.add_argument("--model", type=parse_model_spec, action="append")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("implementation") / "evaluation_results" / "final_comparison",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metric-device", default="cpu")
    parser.add_argument("--max-side", type=int, default=256)
    parser.add_argument(
        "--square-size",
        type=int,
        help="Resize every evaluation image to an exact square for cross-method benchmarks.",
    )
    parser.add_argument("--num-regions", type=int, default=16)
    parser.add_argument("--region-batch-size", type=int, default=16)
    parser.add_argument("--slic-compactness", type=float, default=10.0)
    parser.add_argument("--slic-sigma", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
