"""Re-rasterize baseline SVGs and compute one common set of image metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import lpips
import numpy as np
import pydiffvg
import torch
from PIL import Image
from skimage.metrics import structural_similarity


@dataclass(frozen=True)
class MethodSpec:
    slug: str
    display_name: str
    svg_dir: Path
    timing_report: Path
    background: float


def image_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    perceptual_model: torch.nn.Module,
) -> dict[str, float]:
    target = np.clip(target.astype(np.float32), 0.0, 1.0)
    prediction = np.clip(prediction.astype(np.float32), 0.0, 1.0)
    mse = float(np.mean((target - prediction) ** 2, dtype=np.float64))
    psnr = float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
    ssim = float(structural_similarity(target, prediction, data_range=1.0, channel_axis=-1))
    target_tensor = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0)
    prediction_tensor = torch.from_numpy(prediction).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        value = perceptual_model(2.0 * target_tensor - 1.0, 2.0 * prediction_tensor - 1.0)
    return {"mse": mse, "psnr_db": psnr, "ssim": ssim, "lpips": float(value.item())}


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    uint8 = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(uint8, mode="RGB").save(path)


def resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    uint8 = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    resized = Image.fromarray(uint8, mode="RGB").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def load_timing(path: Path) -> dict[str, float]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        key = "inference_seconds"
    else:
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = report["rows"]
        key = "elapsed_seconds"
    timings = {Path(row["image"]).stem: float(row[key]) for row in rows}
    if not timings:
        raise RuntimeError(f"No timings found in {path}.")
    return timings


def count_xml_paths(svg_path: Path) -> int:
    text = svg_path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    if "nan" in lowered or " inf" in lowered or "-inf" in lowered:
        raise RuntimeError(f"Non-finite value found in {svg_path}.")
    root = ET.fromstring(text)
    return sum(1 for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "path")


def render_svg(
    svg_path: Path,
    target_size: tuple[int, int],
    background: float,
    samples: int,
) -> tuple[np.ndarray, dict[str, int]]:
    path_count = count_xml_paths(svg_path)
    width, height, shapes, shape_groups = pydiffvg.svg_to_scene(str(svg_path))
    cubic_segments = 0
    for shape in shapes:
        control_counts = getattr(shape, "num_control_points", None)
        if control_counts is not None:
            cubic_segments += int((control_counts == 2).sum().item())
    scene_args = pydiffvg.RenderFunction.serialize_scene(width, height, shapes, shape_groups)
    rgba = pydiffvg.RenderFunction.apply(
        width,
        height,
        samples,
        samples,
        1234,
        None,
        *scene_args,
    )
    rgba_np = rgba.detach().cpu().numpy().astype(np.float32)
    alpha = rgba_np[..., 3:4]
    rgb = rgba_np[..., :3] * alpha + background * (1.0 - alpha)
    if rgb.shape[1] != target_size[0] or rgb.shape[0] != target_size[1]:
        rgb = resize_rgb(rgb, target_size)
    if not np.isfinite(rgb).all():
        raise RuntimeError(f"Non-finite rendered pixels found in {svg_path}.")
    return rgb, {
        "path_count": path_count,
        "cubic_segment_count": cubic_segments,
        "native_width": int(width),
        "native_height": int(height),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], methods: list[MethodSpec]) -> list[dict]:
    metrics = (
        "mse",
        "psnr_db",
        "ssim",
        "lpips",
        "inference_seconds",
        "path_count",
        "cubic_segment_count",
        "svg_bytes",
    )
    result = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method.slug]
        item: dict[str, object] = {
            "method": method.slug,
            "display_name": method.display_name,
            "images": len(selected),
            "background": "black" if method.background == 0.0 else "white",
        }
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        result.append(item)
    return result


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[1]
    default_root = workspace / "implementation" / "evaluation_results" / "method_comparison"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--samples", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_dir = root / "ours_native" / "inputs"
    methods = [
        MethodSpec(
            "ours",
            "Notre methode",
            root / "ours_native" / "ours_final",
            root / "ours_native" / "metrics.csv",
            0.0,
        ),
        MethodSpec(
            "supersvg",
            "SuperSVG",
            root / "supersvg_native",
            root / "supersvg_native" / "benchmark.json",
            0.0,
        ),
        MethodSpec(
            "live",
            "LIVE",
            root / "live_native" / "svg",
            root / "live_native" / "benchmark.json",
            1.0,
        ),
        MethodSpec(
            "im2vec",
            "Im2Vec",
            root / "im2vec_native" / "svg",
            root / "im2vec_native" / "benchmark.json",
            1.0,
        ),
    ]
    inputs = sorted(input_dir.glob("*.png"))
    if not inputs:
        raise RuntimeError(f"No common inputs found in {input_dir}.")
    for method in methods:
        for required in (method.svg_dir, method.timing_report):
            if not required.exists():
                raise FileNotFoundError(required)

    pydiffvg.set_use_gpu(False)
    pydiffvg.set_device(torch.device("cpu"))
    perceptual_model = lpips.LPIPS(net="alex", verbose=False).eval()
    timings = {method.slug: load_timing(method.timing_report) for method in methods}
    rows: list[dict] = []

    for input_path in inputs:
        target = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.float32) / 255.0
        target_size = (target.shape[1], target.shape[0])
        for method in methods:
            svg_path = method.svg_dir / f"{input_path.stem}.svg"
            if not svg_path.is_file():
                raise FileNotFoundError(svg_path)
            if input_path.stem not in timings[method.slug]:
                raise RuntimeError(f"Missing timing for {method.slug}/{input_path.name}.")
            started = time.perf_counter()
            prediction, complexity = render_svg(
                svg_path,
                target_size=target_size,
                background=method.background,
                samples=args.samples,
            )
            render_seconds = time.perf_counter() - started
            render_path = root / "common_renders" / method.slug / input_path.name
            save_rgb(render_path, prediction)
            metrics = image_metrics(target, prediction, perceptual_model)
            row = {
                "method": method.slug,
                "display_name": method.display_name,
                "image": input_path.name,
                **metrics,
                "inference_seconds": timings[method.slug][input_path.stem],
                "common_render_seconds": render_seconds,
                **complexity,
                "svg_bytes": svg_path.stat().st_size,
                "background": "black" if method.background == 0.0 else "white",
                "svg": str(svg_path.resolve()),
                "render": str(render_path.resolve()),
            }
            rows.append(row)
            print(
                f"{method.slug:8s} {input_path.name}: MSE={row['mse']:.5f} "
                f"PSNR={row['psnr_db']:.2f} SSIM={row['ssim']:.4f} "
                f"LPIPS={row['lpips']:.4f}"
            )
            write_csv(root / "metrics_partial.csv", rows)

    summary = summarize(rows, methods)
    write_csv(root / "metrics.csv", rows)
    write_csv(root / "summary.csv", summary)
    report = {
        "protocol": {
            "images": len(inputs),
            "target_size": list(Image.open(inputs[0]).size),
            "common_renderer": "pydiffvg CPU, 2x2 samples",
            "metric_device": "CPU",
            "timing_excludes_model_load_and_common_metrics": True,
        },
        "summary": summary,
        "rows": rows,
    }
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
