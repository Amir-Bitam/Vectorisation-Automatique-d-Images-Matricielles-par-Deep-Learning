"""Full raster-to-SVG model with pydiffvg rendering and training losses."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

import pydiffvg

from encoder import DinoV3StrokeEncoder


def configure_pydiffvg(device: torch.device) -> None:
    pydiffvg.set_print_timing(False)
    pydiffvg.set_device(device)
    pydiffvg.set_use_gpu(device.type == "cuda")


def rgba_to_rgb_black(rgba: torch.Tensor) -> torch.Tensor:
    rgb = rgba[:, :3]
    alpha = rgba[:, 3:4]
    return rgb * alpha


def dilate_mask(mask: torch.Tensor, radius: int = 2) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel_size = radius * 2 + 1
    return F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=radius)


def build_diffvg_scene(
    strokes: torch.Tensor,
    width: int,
    height: int | None = None,
    normalized_coordinates: bool = True,
    fill_rgb_override: torch.Tensor | None = None,
) -> tuple[list[pydiffvg.Path], list[pydiffvg.ShapeGroup]]:
    """Build closed cubic path shapes for one image.

    ``strokes`` is ``[num_paths, 27]``. The first 24 values are 12 point
    coordinates. With ``num_control_points=[2,2,2,2]``, DiffVG interprets these
    as four cubic Bezier segments in a closed path.
    """

    if height is None:
        height = width

    strokes = strokes.float()
    device = strokes.device
    dtype = strokes.dtype
    scale = torch.tensor([float(width), float(height)], device=device, dtype=dtype)
    num_control_points = torch.full((4,), 2, dtype=torch.int32)
    stroke_width = torch.tensor(0.0, device=device, dtype=dtype)
    alpha = torch.ones(1, device=device, dtype=dtype)

    shapes: list[pydiffvg.Path] = []
    groups: list[pydiffvg.ShapeGroup] = []

    for stroke_idx in range(strokes.shape[0]):
        points = strokes[stroke_idx, :24].reshape(12, 2)
        if normalized_coordinates:
            points = points * scale
        points = points.contiguous()

        if fill_rgb_override is None:
            rgb = strokes[stroke_idx, 24:27]
        else:
            rgb = fill_rgb_override.to(device=device, dtype=dtype)
        fill_color = torch.cat([rgb, alpha], dim=0).contiguous()

        shapes.append(
            pydiffvg.Path(
                num_control_points=num_control_points,
                points=points,
                stroke_width=stroke_width,
                is_closed=True,
            )
        )
        groups.append(
            pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([stroke_idx], dtype=torch.long),
                fill_color=fill_color,
            )
        )

    return shapes, groups


def render_rgba_strokes(
    strokes: torch.Tensor,
    width: int,
    height: int | None = None,
    normalized_coordinates: bool = True,
    fill_rgb_override: torch.Tensor | None = None,
    samples: int = 2,
    seed: int = 0,
) -> torch.Tensor:
    """Render a batch of strokes to ``[B, 4, H, W]`` using an explicit batch loop."""

    if height is None:
        height = width
    if strokes.ndim == 2:
        strokes = strokes.unsqueeze(0)

    device = strokes.device
    configure_pydiffvg(device)

    rendered = []
    with torch.amp.autocast(device_type=device.type, enabled=False):
        strokes = strokes.float()
        for batch_idx in range(strokes.shape[0]):
            shapes, groups = build_diffvg_scene(
                strokes[batch_idx],
                width=width,
                height=height,
                normalized_coordinates=normalized_coordinates,
                fill_rgb_override=fill_rgb_override,
            )
            scene_args = pydiffvg.RenderFunction.serialize_scene(width, height, shapes, groups)
            image = pydiffvg.RenderFunction.apply(
                width,
                height,
                samples,
                samples,
                seed + batch_idx,
                None,
                *scene_args,
            )
            rendered.append(image.permute(2, 0, 1))
    return torch.stack(rendered, dim=0)


def save_strokes_as_svg(
    path: str | Path,
    strokes: torch.Tensor,
    width: int,
    height: int,
    normalized_coordinates: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shapes, groups = build_diffvg_scene(
        strokes.detach().float().cpu(),
        width=width,
        height=height,
        normalized_coordinates=normalized_coordinates,
    )
    pydiffvg.save_svg(str(path), width, height, shapes, groups)


class RasterToSVGModel(nn.Module):
    def __init__(
        self,
        num_paths: int = 32,
        canvas_size: int = 224,
        dino_model_name: str = "vit_small_patch16_dinov3",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.canvas_size = canvas_size
        self.encoder = DinoV3StrokeEncoder(
            model_name=dino_model_name,
            pretrained=pretrained,
            num_paths=num_paths,
            image_size=canvas_size,
        )

    def render(self, strokes: torch.Tensor, fill_rgb_override: torch.Tensor | None = None) -> torch.Tensor:
        rgba = render_rgba_strokes(
            strokes,
            width=self.canvas_size,
            height=self.canvas_size,
            normalized_coordinates=True,
            fill_rgb_override=fill_rgb_override,
        )
        return rgba_to_rgb_black(rgba)

    def forward(
        self,
        image: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        lambda_mask: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        strokes = self.encoder(image)

        rendered_rgba = render_rgba_strokes(
            strokes,
            width=self.canvas_size,
            height=self.canvas_size,
            normalized_coordinates=True,
        )
        rendered_rgb = rgba_to_rgb_black(rendered_rgba)

        mask = mask.float()
        target = target.float()
        mask_area_ratio = mask.flatten(1).mean(dim=1).clamp_min(1.0 / (mask.shape[-1] * mask.shape[-2]))
        reconstruction_per_image = (((rendered_rgb - target) ** 2) * mask).mean(dim=(1, 2, 3)) / mask_area_ratio
        reconstruction_loss = reconstruction_per_image.mean()

        white_rgba = render_rgba_strokes(
            strokes,
            width=self.canvas_size,
            height=self.canvas_size,
            normalized_coordinates=True,
            fill_rgb_override=torch.ones(3, device=strokes.device),
        )
        white_alpha = white_rgba[:, 3:4]
        dilated = dilate_mask(mask, radius=2)
        outside = (1.0 - dilated).clamp(0.0, 1.0)
        outside_denom = outside.flatten(1).sum(dim=1).clamp_min(1.0)
        mask_penalty_per_image = (white_alpha * outside).flatten(1).sum(dim=1) / outside_denom
        mask_loss = mask_penalty_per_image.mean() * float(lambda_mask)

        total_loss = reconstruction_loss + mask_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss.detach(),
            "mask_loss": mask_loss.detach(),
            "rendered": rendered_rgb.detach(),
            "strokes": strokes,
        }
