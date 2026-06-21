"""SLIC superpixel sampling dataset for raster-to-SVG training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class RegionSample:
    model_input: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor
    bbox: tuple[int, int, int, int]


def find_image_files(root: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(root)
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    paths = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No image files found under: {root}")
    return paths


def load_rgb_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(_to_uint8(image), mode="RGB")
    pil = pil.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    pil = pil.resize((size, size), Image.Resampling.NEAREST)
    return (np.asarray(pil) > 127).astype(np.float32)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Cannot compute a bounding box for an empty mask.")
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def segment_image(
    image: np.ndarray,
    n_segments: int = 16,
    compactness: float = 10.0,
    sigma: float = 1.0,
) -> np.ndarray:
    height, width = image.shape[:2]
    safe_segments = max(2, min(n_segments, height * width))
    return slic(
        image,
        n_segments=safe_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        channel_axis=-1,
        enforce_connectivity=True,
    )


def crop_resize_region(image: np.ndarray, mask: np.ndarray, output_size: int = 224) -> RegionSample:
    x1, y1, x2, y2 = mask_bbox(mask)
    crop = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    resized_image = resize_rgb(crop, output_size)
    resized_mask = resize_mask(crop_mask, output_size)
    if resized_mask.sum() == 0:
        resized_mask = np.ones((output_size, output_size), dtype=np.float32)

    image_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).contiguous()
    mask_tensor = torch.from_numpy(resized_mask).unsqueeze(0).contiguous()

    # Requested formulation: region values stay in RGB range and the background is -1.
    model_input = image_tensor * mask_tensor - (1.0 - mask_tensor)
    return RegionSample(
        model_input=model_input.float(),
        target=image_tensor.float(),
        mask=mask_tensor.float(),
        bbox=(x1, y1, x2, y2),
    )


def random_superpixel_region(
    image: np.ndarray,
    n_segments: int = 16,
    output_size: int = 224,
    compactness: float = 10.0,
    sigma: float = 1.0,
) -> RegionSample:
    labels = segment_image(image, n_segments=n_segments, compactness=compactness, sigma=sigma)
    label_values = np.unique(labels)
    random.shuffle(label_values)

    for label in label_values:
        mask = labels == label
        if mask.any():
            return crop_resize_region(image, mask, output_size=output_size)

    raise ValueError("SLIC produced no non-empty superpixel regions.")


class ImageNetSuperpixelDataset(Dataset):
    """Samples one random SLIC superpixel crop from each ImageNet image."""

    def __init__(
        self,
        root: str | Path,
        output_size: int = 224,
        n_segments: int = 16,
        compactness: float = 10.0,
        sigma: float = 1.0,
        recursive: bool = True,
        max_images: int | None = None,
        retry_count: int = 5,
    ) -> None:
        self.paths = find_image_files(root, recursive=recursive)
        if max_images is not None:
            self.paths = self.paths[:max_images]
        self.output_size = output_size
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.retry_count = retry_count

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        last_error: Exception | None = None
        for attempt in range(self.retry_count):
            path = self.paths[(index + attempt) % len(self.paths)]
            try:
                image = load_rgb_image(path)
                sample = random_superpixel_region(
                    image,
                    n_segments=self.n_segments,
                    output_size=self.output_size,
                    compactness=self.compactness,
                    sigma=self.sigma,
                )
                return {
                    "input": sample.model_input,
                    "target": sample.target,
                    "mask": sample.mask,
                    "path": str(path),
                }
            except Exception as exc:  # Corrupt images should not stop long training runs.
                last_error = exc

        raise RuntimeError(f"Failed to sample a superpixel near index {index}: {last_error}")
