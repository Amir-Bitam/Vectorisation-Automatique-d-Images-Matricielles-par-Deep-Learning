"""Run the available pretrained Im2Vec checkpoint on common benchmark images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch


def count_paths(svg_path: Path) -> int:
    root = ET.parse(svg_path).getroot()
    return sum(1 for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "path")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--im2vec-root",
        type=Path,
        default=workspace / "test & result" / "Im2Vec",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=workspace
        / "test & result"
        / "Im2Vec"
        / "logs"
        / "VectorVAEnLayers"
        / "version_110"
        / "configs"
        / "emoji.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=workspace
        / "test & result"
        / "Im2Vec"
        / "logs"
        / "VectorVAEnLayers"
        / "version_110"
        / "epoch=667.ckpt",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.im2vec_root = args.im2vec_root.resolve()
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    for required in (args.input_dir, args.im2vec_root, args.config, args.checkpoint):
        if not required.exists():
            raise FileNotFoundError(required)

    sys.path.insert(0, str(args.im2vec_root))
    from im2vec_windows_utils import (  # pylint: disable=import-error,import-outside-toplevel
        build_experiment,
        encode_to_latent,
        layer_points_from_latent,
        load_config,
        prepare_config,
        preprocess_image,
        reset_curve_count,
        save_latent_svg,
        set_seed,
    )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The validated Im2Vec environment requires CUDA.")
    config = prepare_config(load_config(args.config), args.im2vec_root)
    seed = int(config["logging_params"]["manual_seed"])
    set_seed(seed)
    experiment, load_info = build_experiment(config, args.checkpoint, device)
    input_size = int(config["exp_params"]["img_size"])
    curve_count = int(config["model_params"]["paths"])

    inputs = sorted(args.input_dir.glob("*.png"))
    if not inputs:
        raise RuntimeError(f"No PNG images found in {args.input_dir}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg_dir = args.output_dir / "svg"
    svg_dir.mkdir(exist_ok=True)

    warmup_tensor, _ = preprocess_image(inputs[0], input_size)
    reset_curve_count(experiment.model, curve_count)
    with torch.no_grad():
        warmup_z = encode_to_latent(experiment.model, warmup_tensor.to(device))
        layer_points_from_latent(experiment.model, warmup_z)
    torch.cuda.synchronize(device)
    set_seed(seed)

    rows: list[dict] = []
    for index, input_path in enumerate(inputs, start=1):
        print(f"[Im2Vec {index}/{len(inputs)}] {input_path.name}", flush=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        input_tensor, _ = preprocess_image(input_path, input_size)
        reset_curve_count(experiment.model, curve_count)
        svg_path = svg_dir / f"{input_path.stem}.svg"
        with torch.no_grad():
            z = encode_to_latent(experiment.model, input_tensor.to(device))
            save_latent_svg(experiment.model, z, svg_path, sample_index=0)
        torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - started
        path_count = count_paths(svg_path)
        row = {
            "image": input_path.name,
            "input": str(input_path.resolve()),
            "svg": str(svg_path.resolve()),
            "elapsed_seconds": elapsed_seconds,
            "path_count": path_count,
            "cubic_segments_per_path": curve_count,
            "svg_bytes": svg_path.stat().st_size,
        }
        rows.append(row)
        report = {
            "method": "Im2Vec",
            "checkpoint": str(args.checkpoint),
            "checkpoint_domain": "emoji",
            "config": str(args.config),
            "model_input_size": input_size,
            "curves_per_path": curve_count,
            "model_load_included": False,
            "warmup": True,
            "load_info": load_info,
            "rows": rows,
        }
        (args.output_dir / "benchmark.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        write_csv(args.output_dir / "benchmark.csv", rows)
        print(f"  time={elapsed_seconds:.3f}s paths={path_count}", flush=True)


if __name__ == "__main__":
    main()
