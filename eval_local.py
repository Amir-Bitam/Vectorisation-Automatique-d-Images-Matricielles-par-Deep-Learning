import argparse
import json
import time
from pathlib import Path

import torch

from im2vec_windows_utils import (
    REPO_ROOT,
    build_experiment,
    encode_to_latent,
    load_config,
    load_dataset_batch,
    prepare_config,
    reconstruct_from_latent,
    reconstruct_tensor,
    reset_curve_count,
    save_latent_svg,
    save_tensor_image,
    set_seed,
    validate_svg,
)


def latest_checkpoint(search_dir: Path) -> Path:
    checkpoints = sorted(search_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt files found in {search_dir}")
    return checkpoints[-1]


def resolve_path(path: str, base: Path = REPO_ROOT) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (base / candidate).resolve()


def save_grid_outputs(experiment, images, config, output_dir: Path) -> dict:
    device = next(experiment.parameters()).device
    batch_size = images.shape[0]
    reset_curve_count(experiment.model, int(config["model_params"]["paths"]))

    with torch.no_grad():
        images_device = images.to(device)
        z = encode_to_latent(experiment.model, images_device)
        reconstructions = reconstruct_from_latent(experiment.model, z)

    input_grid = output_dir / "official_input_grid.png"
    reconstruction_grid = output_dir / "official_reconstruction_grid.png"
    save_tensor_image(images, input_grid, nrow=min(batch_size, 8))
    save_tensor_image(reconstructions[:, :3], reconstruction_grid, nrow=min(batch_size, 8))
    return {
        "input_grid": str(input_grid.resolve()),
        "reconstruction_grid": str(reconstruction_grid.resolve()),
    }


def save_interpolation_outputs(experiment, images, config, output_dir: Path, run_other: bool) -> dict:
    device = next(experiment.parameters()).device
    images_device = images.to(device)
    outputs = {}
    reset_curve_count(experiment.model, int(config["model_params"]["paths"]))

    with torch.no_grad():
        interpolation = torch.cat(experiment.model.interpolate(images_device, verbose=False), dim=0)
    interpolation_path = output_dir / "official_interpolate_img.png"
    save_tensor_image(interpolation[:, :3], interpolation_path, nrow=10)
    outputs["interpolation"] = str(interpolation_path.resolve())

    if not run_other:
        return outputs

    with torch.no_grad():
        interpolation_2d = torch.cat(experiment.model.interpolate2D(images_device, verbose=False), dim=0)
    path = output_dir / "official_interpolate2D_image.png"
    save_tensor_image(interpolation_2d[:, :3], path, nrow=10)
    outputs["interpolation_2d"] = str(path.resolve())

    with torch.no_grad():
        visualize_sampling = torch.cat(experiment.model.visualize_sampling(images_device, verbose=False), dim=0)
    path = output_dir / "official_visualize_sampling_image.png"
    save_tensor_image(visualize_sampling[:, :3], path, nrow=int(config["exp_params"]["val_batch_size"]))
    outputs["visualize_sampling"] = str(path.resolve())
    reset_curve_count(experiment.model, int(config["model_params"]["paths"]))

    if config["model_params"].get("composite_fn") == "hard":
        with torch.no_grad():
            naive = torch.cat(experiment.model.naive_vector_interpolate(images_device, verbose=False), dim=0)
        path = output_dir / "official_naive_interpolate_image.png"
        save_tensor_image(naive[:, :3], path, nrow=10)
        outputs["naive_interpolation"] = str(path.resolve())
        reset_curve_count(experiment.model, int(config["model_params"]["paths"]))
    else:
        outputs["naive_interpolation_skipped"] = "requires hard compositing in the legacy implementation"

    return outputs


def save_random_sample(experiment, config, output_dir: Path) -> dict:
    device = next(experiment.parameters()).device
    reset_curve_count(experiment.model, int(config["model_params"]["paths"]))
    with torch.no_grad():
        z = torch.randn(1, int(config["model_params"]["latent_dim"]), device=device)
        sample = reconstruct_from_latent(experiment.model, z)
    sample_png = output_dir / "official_random_sample.png"
    sample_svg = output_dir / "official_random_sample.svg"
    sample_reraster = output_dir / "official_random_sample_rerasterized.png"
    save_tensor_image(sample[:, :3], sample_png)
    save_latent_svg(experiment.model, z, sample_svg)
    validation = validate_svg(sample_svg, sample_reraster)
    return {
        "sample_png": str(sample_png.resolve()),
        "sample_svg": str(sample_svg.resolve()),
        "sample_svg_validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Windows-safe Im2Vec evaluation")
    parser.add_argument("-c", "--config", default="logs/VectorVAEnLayers/version_110/configs/emoji.yaml")
    parser.add_argument("--checkpoint", default="logs/VectorVAEnLayers/version_110/epoch=667.ckpt")
    parser.add_argument("--output-dir", default="outputs/official_eval")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-other-interpolations", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        checkpoint_path = latest_checkpoint(Path.cwd())

    config = prepare_config(load_config(config_path))
    set_seed(int(config["logging_params"]["manual_seed"]))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is not available.")

    torch.empty(1, device=device)
    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    experiment, load_info = build_experiment(config, checkpoint_path, device)
    images, _, dataset_paths = load_dataset_batch(config)

    grid_outputs = save_grid_outputs(experiment, images, config, output_dir)
    reconstruction_result = reconstruct_tensor(
        experiment,
        images[:1],
        output_dir,
        "official_reconstruction_000",
        config,
        save_svg=True,
    )
    interpolation_outputs = save_interpolation_outputs(
        experiment,
        images,
        config,
        output_dir,
        run_other=bool(config["logging_params"].get("other_interpolations", False)) and not args.skip_other_interpolations,
    )
    sample_outputs = save_random_sample(experiment, config, output_dir)

    elapsed_seconds = time.perf_counter() - wall_start
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    summary = {
        "config": str(config_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "dataset_example_paths": dataset_paths,
        "load_info": load_info,
        "grid_outputs": grid_outputs,
        "reconstruction": reconstruction_result,
        "interpolations": interpolation_outputs,
        "random_sample": sample_outputs,
        "elapsed_seconds": elapsed_seconds,
        "peak_gpu_memory_mb": peak_memory_mb,
        "device": str(device),
        "pydiffvg_use_gpu": True,
    }
    summary_path = output_dir / "official_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
