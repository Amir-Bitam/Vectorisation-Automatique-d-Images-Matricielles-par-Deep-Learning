"""CUDA + DiffVG trainability smoke test.

This script verifies that PyTorch, CUDA, pydiffvg, and the native diffvg
extension can all cooperate in one differentiable training loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

import diffvg
import pydiffvg


def configure_diffvg(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError("This smoke test is meant to run on CUDA. Use --device cuda:0.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch in this environment.")

    pydiffvg.set_print_timing(False)
    pydiffvg.set_device(device)
    pydiffvg.set_use_gpu(True)


def render_circle(
    canvas_size: int,
    center: torch.Tensor,
    radius: torch.Tensor,
    fill_color: torch.Tensor,
    device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    circle = pydiffvg.Circle(radius=radius, center=center)
    shapes = [circle]
    shape_groups = [
        pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0], dtype=torch.long, device=device),
            fill_color=fill_color,
        )
    ]

    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_size,
        canvas_size,
        shapes,
        shape_groups,
    )
    return pydiffvg.RenderFunction.apply(
        canvas_size,
        canvas_size,
        2,
        2,
        seed,
        None,
        *scene_args,
    )


class TrainableCircle(nn.Module):
    def __init__(self, canvas_size: int, device: torch.device) -> None:
        super().__init__()
        self.canvas_size = canvas_size
        self.device = device

        self.raw_center = nn.Parameter(torch.tensor([-0.25, 0.25], device=device))
        self.raw_radius = nn.Parameter(torch.tensor(-0.20, device=device))
        self.raw_rgb = nn.Parameter(torch.tensor([1.2, -1.1, -0.7], device=device))

    def bounded_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        margin = 6.0
        center = margin + (self.canvas_size - 2 * margin) * torch.sigmoid(self.raw_center)
        radius = 5.0 + 16.0 * torch.sigmoid(self.raw_radius)
        color = torch.cat([torch.sigmoid(self.raw_rgb), torch.ones(1, device=self.device)])
        return center, radius, color

    def forward(self) -> torch.Tensor:
        center, radius, color = self.bounded_params()
        return render_circle(self.canvas_size, center, radius, color, self.device)


def assert_gradients_are_healthy(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if param.grad is None:
            raise RuntimeError(f"Missing gradient for parameter: {name}")
        if not torch.isfinite(param.grad).all():
            raise RuntimeError(f"Non-finite gradient for parameter: {name}")


def save_image(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pydiffvg.imwrite(image.detach().cpu(), str(path), gamma=1.0)


def composite_over_white(image: torch.Tensor) -> torch.Tensor:
    rgb = image[..., :3]
    alpha = image[..., 3:4]
    return rgb * alpha + (1.0 - alpha)


def run_trainability_test(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    configure_diffvg(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    target_center = torch.tensor([40.0, 24.0], device=device)
    target_radius = torch.tensor(11.0, device=device)
    target_color = torch.tensor([0.08, 0.72, 0.95, 1.0], device=device)

    with torch.no_grad():
        target = render_circle(
            args.canvas_size,
            target_center,
            target_radius,
            target_color,
            device,
            seed=args.seed,
        ).detach()
        target_rgb = composite_over_white(target).detach()

    model = TrainableCircle(args.canvas_size, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    initial_loss = None
    final_loss = None

    print(f"python torch cuda device: {torch.cuda.get_device_name(device)}")
    print(f"torch: {torch.__version__}")
    print(f"pydiffvg: {pydiffvg.__file__}")
    print(f"diffvg native extension: {diffvg.__file__}")
    print("training tiny DiffVG-backed model...")

    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model()
        prediction_rgb = composite_over_white(prediction)
        loss = torch.mean((prediction_rgb - target_rgb) ** 2)

        if step == 0:
            initial_loss = float(loss.detach().cpu())

        loss.backward()
        assert_gradients_are_healthy(model)
        optimizer.step()

        final_loss = float(loss.detach().cpu())
        if step % args.log_every == 0 or step == args.steps:
            center, radius, color = model.bounded_params()
            print(
                f"step {step:03d} loss={final_loss:.6f} "
                f"center={center.detach().cpu().tolist()} "
                f"radius={float(radius.detach().cpu()):.3f} "
                f"color={color.detach().cpu().tolist()}"
            )

    if initial_loss is None or final_loss is None:
        raise RuntimeError("Training loop did not run.")
    if final_loss >= initial_loss * args.required_improvement:
        raise RuntimeError(
            "Loss did not improve enough: "
            f"initial={initial_loss:.6f}, final={final_loss:.6f}, "
            f"required final < {initial_loss * args.required_improvement:.6f}"
        )

    with torch.no_grad():
        final_prediction = model()
    save_image(args.output_dir / "target.png", target_rgb)
    save_image(args.output_dir / "final_prediction.png", composite_over_white(final_prediction))

    print(
        "PASS: CUDA PyTorch + pydiffvg/native diffvg can render, backpropagate, "
        f"and train. Loss {initial_loss:.6f} -> {final_loss:.6f}."
    )
    print(f"Wrote visual outputs to: {args.output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use, for example cuda:0")
    parser.add_argument("--canvas-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--required-improvement",
        type=float,
        default=0.05,
        help="Final loss must be below this fraction of the initial loss.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("implementation") / "diffvg_smoke_outputs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_trainability_test(parse_args())

