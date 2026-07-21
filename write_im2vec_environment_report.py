import argparse
import json
import subprocess
from pathlib import Path

import pydiffvg

from im2vec_windows_utils import sha256_file, write_environment_report


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def official_eval_succeeded(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = data.get("reconstruction", {}).get("svg_validation", {})
    return (
        bool(data.get("load_info", {}).get("strict_load"))
        and validation.get("root") == "svg"
        and int(validation.get("path_count", 0)) > 0
        and not bool(validation.get("constant", True))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Im2Vec Windows environment metadata report.")
    parser.add_argument("--checkpoint", default="logs/VectorVAEnLayers/version_110/epoch=667.ckpt")
    parser.add_argument("--official-summary", default="outputs/official_eval/official_eval_summary.json")
    parser.add_argument("--output", default="outputs/im2vec_environment_report.json")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    summary = Path(args.official_summary).resolve()
    output = Path(args.output).resolve()
    report = write_environment_report(
        report_path=output,
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        repository_commit=git_commit(),
        pydiffvg_path=pydiffvg.__file__,
        successful_official_evaluation=official_eval_succeeded(summary),
        custom_image_supported=True,
        compatibility_patches=[
            "Built official DiffVG source natively on Windows with MSVC/Ninja/CUDA 12.4.",
            "Updated DiffVG build metadata for setuptools and Python 3.11.",
            "Updated DiffVG pybind11 submodule to v2.13.6 for Python 3.11 compatibility.",
            "Added Windows-safe eval_local.py path/checkpoint/output arguments.",
            "Added explicit SVG export and validation for decoded Im2Vec vector layers.",
            "Set DataLoader workers to 0 by default on Windows.",
            "Added Kornia PyrDown compatibility fallback.",
            "Replaced legacy TestTubeLogger/Trainer usage in root launchers with Lightning 1.9-compatible APIs.",
        ],
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
