import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from im2vec_windows_utils import (
    REPO_ROOT,
    build_experiment,
    load_config,
    prepare_config,
    preprocessing_transform,
    set_seed,
)


def resolve_path(path: str, base: Path = REPO_ROOT) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (base / candidate).resolve()


def first_changed_parameter(before, experiment) -> str:
    for name, parameter in experiment.named_parameters():
        if parameter.requires_grad and name in before:
            if not torch.allclose(before[name], parameter.detach().cpu()):
                return name
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one native-Windows Im2Vec training smoke step.")
    parser.add_argument("--config", default="configs/emoji_windows_smoke.yaml")
    parser.add_argument("--output-dir", default="outputs/training_smoke")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = prepare_config(load_config(config_path))
    set_seed(int(config["logging_params"]["manual_seed"]))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the training smoke test.")

    start = time.perf_counter()
    experiment, _ = build_experiment(config, checkpoint_path=None, device=device)
    for parameter in experiment.parameters():
        parameter.requires_grad = True
    experiment.train()
    dataset = datasets.ImageFolder(
        config["exp_params"]["data_path"],
        transform=preprocessing_transform(int(config["exp_params"]["img_size"])),
    )
    subset = Subset(dataset, list(range(min(2, len(dataset)))))
    loader = DataLoader(
        subset,
        batch_size=int(config["exp_params"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    images, labels = next(iter(loader))
    images = images.to(device)

    optimizer = torch.optim.Adam(
        [parameter for parameter in experiment.parameters() if parameter.requires_grad],
        lr=float(config["exp_params"]["LR"]),
        weight_decay=float(config["exp_params"]["weight_decay"]),
    )
    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in experiment.named_parameters()
        if parameter.requires_grad
    }

    optimizer.zero_grad(set_to_none=True)
    outputs = experiment.model(images, labels=labels)
    loss_info = experiment.model.loss_function(
        *outputs,
        M_N=float(config["exp_params"]["batch_size"]) / max(1, len(dataset)),
        optimizer_idx=0,
        batch_idx=0,
    )
    loss = loss_info["loss"]
    if not torch.isfinite(loss):
        raise RuntimeError(f"Training smoke loss is not finite: {loss.item()}")
    loss.backward()

    finite_gradients = True
    gradient_norm = 0.0
    for parameter in experiment.parameters():
        if parameter.grad is not None:
            finite_gradients = finite_gradients and bool(torch.isfinite(parameter.grad).all())
            gradient_norm += float(parameter.grad.detach().norm().item())
    if not finite_gradients:
        raise RuntimeError("Training smoke test produced non-finite gradients.")

    optimizer.step()
    changed_parameter = first_changed_parameter(before, experiment)
    if not changed_parameter:
        raise RuntimeError("No trainable parameter changed after the optimizer step.")

    checkpoint_path = output_dir / "emoji_windows_smoke.ckpt"
    torch.save(
        {
            "state_dict": experiment.state_dict(),
            "config": config,
            "loss": float(loss.item()),
        },
        checkpoint_path,
    )

    reloaded, _ = build_experiment(config, checkpoint_path=None, device=device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    reload_result = reloaded.load_state_dict(checkpoint["state_dict"], strict=True)

    report = {
        "config": str(config_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "dataset_size": len(dataset),
        "batch_size": int(config["exp_params"]["batch_size"]),
        "loss": float(loss.item()),
        "finite_gradients": finite_gradients,
        "gradient_norm_sum": gradient_norm,
        "changed_parameter": changed_parameter,
        "checkpoint": str(checkpoint_path.resolve()),
        "reload_missing_keys": list(reload_result.missing_keys),
        "reload_unexpected_keys": list(reload_result.unexpected_keys),
        "elapsed_seconds": time.perf_counter() - start,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
    }
    report_path = output_dir / "training_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
