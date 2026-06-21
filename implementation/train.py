"""Train the first-iteration DINOv3 + DiffVG raster-to-SVG model."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImageNetSuperpixelDataset
from model import RasterToSVGModel, configure_pydiffvg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lambda(global_step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and global_step < warmup_steps:
        return float(global_step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    progress = float(global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def mask_lambda_for_epoch(epoch: int, epochs: int, start: float = 0.05, end: float = 0.01) -> float:
    if epochs <= 1:
        return end
    t = epoch / float(epochs - 1)
    return start + (end - start) * t


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model: RasterToSVGModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "args": vars(args),
    }
    epoch_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(payload, epoch_path)
    torch.save(payload, latest_path)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    configure_pydiffvg(device)

    dataset = ImageNetSuperpixelDataset(
        root=args.imagenet_root,
        output_size=args.image_size,
        n_segments=args.slic_regions,
        compactness=args.slic_compactness,
        sigma=args.slic_sigma,
        max_images=args.max_images,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    if len(loader) == 0:
        raise RuntimeError("The DataLoader is empty. Add more images or lower --batch-size.")

    steps_per_epoch = len(loader)
    if args.max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(0, args.warmup_epochs * steps_per_epoch)

    model = RasterToSVGModel(
        num_paths=args.num_paths,
        canvas_size=args.image_size,
        dino_model_name=args.dino_model,
        pretrained=args.pretrained,
    ).to(device)

    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lambda(step, total_steps=total_steps, warmup_steps=warmup_steps),
    )

    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(args.checkpoint_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"device={device}")
    print(f"images={len(dataset)} batch_size={args.batch_size} steps_per_epoch={steps_per_epoch}")
    print(f"optimizer=AdamW(model.encoder.parameters) lr={args.lr} weight_decay={args.weight_decay}")

    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        lambda_mask = mask_lambda_for_epoch(epoch, args.epochs)
        running_loss = 0.0
        running_reconstruction = 0.0
        running_mask = 0.0

        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        for step, batch in enumerate(progress):
            if args.max_steps_per_epoch is not None and step >= args.max_steps_per_epoch:
                break

            image = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(image, target, mask, lambda_mask=lambda_mask)
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_value = float(loss.detach().cpu())
            reconstruction_value = float(outputs["reconstruction_loss"].detach().cpu())
            mask_value = float(outputs["mask_loss"].detach().cpu())
            running_loss += loss_value
            running_reconstruction += reconstruction_value
            running_mask += mask_value
            global_step += 1

            if step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                progress.set_postfix(
                    loss=f"{loss_value:.5f}",
                    rec=f"{reconstruction_value:.5f}",
                    mask=f"{mask_value:.5f}",
                    lr=f"{lr:.2e}",
                    lmask=f"{lambda_mask:.3f}",
                )

        completed_steps = min(step + 1, steps_per_epoch)
        print(
            f"epoch {epoch + 1}: "
            f"loss={running_loss / completed_steps:.6f} "
            f"reconstruction={running_reconstruction / completed_steps:.6f} "
            f"mask={running_mask / completed_steps:.6f} "
            f"lambda_mask={lambda_mask:.4f}"
        )
        save_checkpoint(args.checkpoint_dir, epoch, model, optimizer, scheduler, scaler, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenet-root", type=Path, required=True, help="Path to ImageNet train images.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("implementation") / "checkpoints" / "raster_to_svg")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--dino-model", default="vit_small_patch16_dinov3")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-paths", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)

    parser.add_argument("--slic-regions", type=int, default=16)
    parser.add_argument("--slic-compactness", type=float, default=10.0)
    parser.add_argument("--slic-sigma", type=float, default=1.0)
    parser.add_argument("--max-images", type=int)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
