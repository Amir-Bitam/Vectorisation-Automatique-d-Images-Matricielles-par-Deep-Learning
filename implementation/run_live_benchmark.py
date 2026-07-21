"""Run LIVE on a directory of common benchmark images."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path


FINAL_SCHEDULE_NAME = "1-2-4-8-16-1.svg"


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


def write_report(path: Path, args: argparse.Namespace, rows: list[dict]) -> None:
    report = {
        "method": "LIVE",
        "configuration": "experiment_exp2_32",
        "paths": 32,
        "stages": [1, 2, 4, 8, 16, 1],
        "iterations_per_stage": 500,
        "total_optimization_steps": 3000,
        "model_load_included": False,
        "python": str(args.python.resolve()),
        "live_root": str(args.live_root.resolve()),
        "config": str(args.config.resolve()),
        "rows": rows,
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(path.with_suffix(".csv"), rows)


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--live-root",
        type=Path,
        default=workspace / "test & result" / "LIVE-Layerwise-Image-Vectorization" / "LIVE",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(r"C:\Users\zak\anaconda3\envs\live-win\python.exe"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=workspace / "implementation" / "baselines" / "live_benchmark.yaml",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.live_root = args.live_root.resolve()
    args.python = args.python.resolve()
    args.config = args.config.resolve()

    for required in (args.input_dir, args.live_root, args.python, args.config):
        if not required.exists():
            raise FileNotFoundError(required)

    inputs = sorted(args.input_dir.glob("*.png"))
    if args.limit is not None:
        inputs = inputs[: args.limit]
    if not inputs:
        raise RuntimeError(f"No PNG images found in {args.input_dir}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg_dir = args.output_dir / "svg"
    log_dir = args.output_dir / "logs"
    svg_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    report_path = args.output_dir / "benchmark.json"
    rows: list[dict] = []
    if args.skip_existing and report_path.is_file():
        rows = json.loads(report_path.read_text(encoding="utf-8")).get("rows", [])
    for row in rows:
        canonical_svg = svg_dir / f"{Path(row['image']).stem}.svg"
        if canonical_svg.is_file():
            row["svg"] = str(canonical_svg.resolve())
    completed_stems = {Path(row["image"]).stem for row in rows}

    for index, input_path in enumerate(inputs, start=1):
        stem = input_path.stem
        destination_svg = svg_dir / f"{stem}.svg"
        if args.skip_existing and stem in completed_stems and destination_svg.is_file():
            print(f"[LIVE {index}/{len(inputs)}] {input_path.name} already complete", flush=True)
            continue
        print(f"[LIVE {index}/{len(inputs)}] {input_path.name}", flush=True)
        before = {path.resolve() for path in args.output_dir.iterdir() if path.is_dir()}
        command = [
            str(args.python),
            str(args.live_root / "main.py"),
            "--config",
            str(args.config),
            "--experiment",
            "experiment_exp2_32",
            "--seed",
            str(args.seed),
            "--target",
            str(input_path.resolve()),
            "--log_dir",
            str(args.output_dir),
            "--signature",
            "benchmark",
            stem,
        ]
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["TQDM_DISABLE"] = "1"
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=args.live_root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        elapsed_seconds = time.perf_counter() - started
        (log_dir / f"{stem}.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (log_dir / f"{stem}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-30:])
            raise RuntimeError(f"LIVE failed for {input_path.name}:\n{tail}")

        candidates = [
            path
            for path in args.output_dir.iterdir()
            if path.is_dir()
            and path.resolve() not in before
            and path.name.endswith(f"_benchmark_{stem}")
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Could not identify LIVE output directory for {stem}: {candidates}")
        experiment_dir = candidates[0]
        source_svg = experiment_dir / "output-svg" / FINAL_SCHEDULE_NAME
        if not source_svg.is_file():
            raise FileNotFoundError(source_svg)
        shutil.copy2(source_svg, destination_svg)
        path_count = count_paths(destination_svg)
        if path_count != 32:
            raise RuntimeError(f"LIVE exported {path_count} paths for {stem}; expected 32.")

        row = {
            "image": input_path.name,
            "input": str(input_path.resolve()),
            "svg": str(destination_svg.resolve()),
            "elapsed_seconds": elapsed_seconds,
            "path_count": path_count,
            "svg_bytes": destination_svg.stat().st_size,
            "experiment_dir": str(experiment_dir.resolve()),
        }
        rows.append(row)
        write_report(report_path, args, rows)
        print(f"  time={elapsed_seconds:.3f}s paths={path_count}", flush=True)

    write_report(report_path, args, rows)


if __name__ == "__main__":
    main()
