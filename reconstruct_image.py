import argparse
import json
import time
from pathlib import Path

import torch

from im2vec_windows_utils import (
    REPO_ROOT,
    build_experiment,
    load_config,
    prepare_config,
    preprocess_image,
    reconstruct_tensor,
    set_seed,
)


def resolve_path(path: str, base: Path = REPO_ROOT) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (base / candidate).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct a raster image through the pretrained Im2Vec encoder.")
    parser.add_argument("--input", required=True, help="Input raster image.")
    parser.add_argument("--checkpoint", default="logs/VectorVAEnLayers/version_110/epoch=667.ckpt")
    parser.add_argument("--config", default="logs/VectorVAEnLayers/version_110/configs/emoji.yaml")
    parser.add_argument("--output-dir", default="outputs/custom_test")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    checkpoint_path = resolve_path(args.checkpoint)
    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    config = prepare_config(load_config(config_path))
    set_seed(int(config["logging_params"]["manual_seed"]))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the validated Windows Im2Vec workflow.")

    start = time.perf_counter()
    experiment, load_info = build_experiment(config, checkpoint_path, device)
    input_tensor, preprocess_info = preprocess_image(input_path, int(config["exp_params"]["img_size"]))
    stem = input_path.stem.replace(" ", "_")
    result = reconstruct_tensor(experiment, input_tensor, output_dir, stem, config, save_svg=True)
    elapsed_seconds = time.perf_counter() - start

    report = {
        "input": str(input_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "config": str(config_path.resolve()),
        "checkpoint_domain": "emoji",
        "custom_image_supported": True,
        "support_scope": "The encoder accepts RGB rasters resized/cropped to the training size; quality is only demonstrated for the emoji-domain checkpoint.",
        "preprocessing": preprocess_info,
        "load_info": load_info,
        "elapsed_seconds": elapsed_seconds,
        **result,
    }
    report_path = output_dir / f"{stem}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("WARNING: This checkpoint was trained on the emoji domain; this script does not make Im2Vec a universal vectorizer.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
