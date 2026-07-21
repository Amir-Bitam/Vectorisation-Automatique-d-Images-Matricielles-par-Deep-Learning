from pathlib import Path

import pydiffvg
import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to validate a CPU-only DiffVG path.")

    pydiffvg.set_use_gpu(True)
    device = pydiffvg.get_device()

    out_dir = Path("outputs") / "diffvg_renderer_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    canvas_width = 64
    canvas_height = 64
    points = torch.tensor(
        [
            [10.0, 54.0],
            [12.0, 18.0],
            [30.0, 8.0],
            [48.0, 18.0],
            [56.0, 44.0],
            [40.0, 58.0],
            [10.0, 54.0],
        ],
        device=device,
        requires_grad=True,
    )
    path = pydiffvg.Path(
        num_control_points=torch.tensor([2, 2], dtype=torch.int32),
        points=points,
        stroke_width=torch.tensor(1.0, device=device),
        is_closed=True,
    )
    fill_color = torch.tensor([0.10, 0.35, 0.85, 1.0], device=device, requires_grad=True)
    shape_group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0], dtype=torch.int32),
        fill_color=fill_color,
    )

    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width,
        canvas_height,
        [path],
        [shape_group],
    )
    img = pydiffvg.RenderFunction.apply(
        canvas_width,
        canvas_height,
        2,
        2,
        0,
        None,
        *scene_args,
    )
    loss = img[..., :3].mean()
    loss.backward()

    if points.grad is None or not torch.isfinite(points.grad).all():
        raise RuntimeError("DiffVG did not produce finite point gradients.")
    if fill_color.grad is None or not torch.isfinite(fill_color.grad).all():
        raise RuntimeError("DiffVG did not produce finite color gradients.")

    output_path = out_dir / "bezier_cuda_render.png"
    pydiffvg.imwrite(img.detach().cpu(), str(output_path), gamma=1.0)

    print(f"pydiffvg: {pydiffvg.__file__}")
    print(f"CUDA rendering enabled: {pydiffvg.get_use_gpu()}")
    print(f"DiffVG device: {device}")
    print(f"Loss: {loss.item():.8f}")
    print(f"Point gradient norm: {points.grad.norm().item():.8f}")
    print(f"Color gradient norm: {fill_color.grad.norm().item():.8f}")
    print(f"Saved raster: {output_path.resolve()}")


if __name__ == "__main__":
    main()
