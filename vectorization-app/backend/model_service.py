"""Persistent adapter around the real pipeline in ../../implementation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import logging
from pathlib import Path
import sys
import threading
import traceback
from types import ModuleType
from typing import Any

from config import Settings


LOGGER = logging.getLogger(__name__)

# These values mirror implementation/inference.py. Only the SLIC region count is
# configurable per request; the remaining inference settings stay fixed.
DEFAULT_NUM_REGIONS = 64
MIN_NUM_REGIONS = 2
MAX_NUM_REGIONS = 256
REGION_BATCH_SIZE = 16
SLIC_COMPACTNESS = 10.0
SLIC_SIGMA = 1.0
RENDER_SAMPLES = 2
USE_AMP = True


class ModelNotReadyError(RuntimeError):
    """Raised when inference is requested after startup loading failed."""


@dataclass(frozen=True)
class InferenceResult:
    svg_path: Path
    preview_path: Path
    region_count: int
    path_count: int


class RasterVectorizationService:
    """Load the checkpoint once, then reuse the model for every request."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Any | None = None
        self.device: Any | None = None
        self.pipeline: ModuleType | None = None
        self.model_loaded = False
        self.load_error: str | None = None
        self.load_traceback: str | None = None
        self._inference_lock = threading.Lock()

    @property
    def device_name(self) -> str:
        return str(self.device) if self.device is not None else self.settings.model_device

    def health(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "ok" if self.model_loaded else "error",
            "model_loaded": self.model_loaded,
            "device": self.device_name,
            "checkpoint": str(self.settings.model_checkpoint),
            "backend": "fastapi",
        }
        if self.load_error:
            payload["error"] = self.load_error
        return payload

    def load(self) -> None:
        if self.model_loaded:
            return

        try:
            self._load_once()
        except Exception as exc:
            self.model = None
            self.model_loaded = False
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.load_traceback = traceback.format_exc()
            raise

    def _load_once(self) -> None:
        implementation_dir = self.settings.implementation_dir
        checkpoint = self.settings.model_checkpoint

        if not implementation_dir.is_dir():
            raise FileNotFoundError(
                f"IMPLEMENTATION_DIR does not exist or is not a directory: {implementation_dir}"
            )
        for required_name in ("inference.py", "model.py", "encoder.py", "dataset.py"):
            required_path = implementation_dir / required_name
            if not required_path.is_file():
                raise FileNotFoundError(f"Required implementation file is missing: {required_path}")
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "MODEL_CHECKPOINT was not found: "
                f"{checkpoint}. Set MODEL_CHECKPOINT in vectorization-app/backend/.env."
            )

        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyTorch is missing. Install the CUDA or CPU build before starting FastAPI."
            ) from exc

        requested_device = self.settings.model_device
        if requested_device not in {"auto", "cpu", "cuda"}:
            raise ValueError('MODEL_DEVICE must be one of: "auto", "cpu", or "cuda".')
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "MODEL_DEVICE=cuda, but torch.cuda.is_available() is false. "
                "Check the NVIDIA driver and the installed PyTorch CUDA build."
            )

        implementation_text = str(implementation_dir)
        if implementation_text in sys.path:
            sys.path.remove(implementation_text)
        sys.path.insert(0, implementation_text)
        importlib.invalidate_caches()

        existing_module = sys.modules.get("inference")
        if existing_module is not None:
            module_file = Path(getattr(existing_module, "__file__", "")).resolve()
            if module_file.parent != implementation_dir:
                raise ImportError(
                    "A different top-level 'inference' module is already loaded from "
                    f"{module_file}; expected {implementation_dir / 'inference.py'}."
                )

        try:
            pipeline = importlib.import_module("inference")
        except ModuleNotFoundError as exc:
            missing = exc.name or "unknown"
            raise ModuleNotFoundError(
                f"A model dependency is missing: {missing}. "
                "Install vectorization-app/backend/requirements.txt and local DiffVG."
            ) from exc

        pipeline_path = Path(pipeline.__file__).resolve()
        if pipeline_path.parent != implementation_dir:
            raise ImportError(
                f"Loaded the wrong inference module: {pipeline_path}; "
                f"expected {implementation_dir / 'inference.py'}."
            )

        device = torch.device(requested_device)
        pipeline.configure_pydiffvg(device)
        build_args = argparse.Namespace(
            checkpoint=checkpoint,
            num_paths=None,
            image_size=None,
            dino_model=None,
            pretrained=True,
        )
        model = pipeline.build_model(build_args, device)
        model.eval()

        self.pipeline = pipeline
        self.device = device
        self.model = model
        self.model_loaded = True
        self.load_error = None
        self.load_traceback = None
        LOGGER.info("Model loaded once from %s on %s", checkpoint, device)

    def vectorize(
        self,
        input_path: Path,
        output_dir: Path,
        *,
        num_regions: int = DEFAULT_NUM_REGIONS,
    ) -> InferenceResult:
        if not self.model_loaded or self.model is None or self.pipeline is None or self.device is None:
            raise ModelNotReadyError(self.load_error or "The model is not loaded.")
        if not MIN_NUM_REGIONS <= num_regions <= MAX_NUM_REGIONS:
            raise ValueError(
                f"num_regions must be between {MIN_NUM_REGIONS} and {MAX_NUM_REGIONS}."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        svg_path = output_dir / "vectorized.svg"
        preview_path = output_dir / "preview.png"
        torch = self.pipeline.torch

        # DiffVG stores its device globally. Serializing this block also prevents
        # concurrent requests from exhausting the GPU with duplicate render scenes.
        with self._inference_lock, torch.inference_mode():
            image = self.pipeline.load_rgb_image(input_path)
            height, width = image.shape[:2]
            regions = self.pipeline.prepare_regions(
                image,
                n_segments=num_regions,
                image_size=self.model.canvas_size,
                compactness=SLIC_COMPACTNESS,
                sigma=SLIC_SIGMA,
            )
            global_strokes = self.pipeline.predict_global_strokes(
                model=self.model,
                regions=regions,
                device=self.device,
                batch_size=REGION_BATCH_SIZE,
                amp=USE_AMP,
            ).to(self.device)
            rgba = self.pipeline.render_rgba_strokes(
                global_strokes,
                width=width,
                height=height,
                normalized_coordinates=False,
                samples=RENDER_SAMPLES,
            )
            rgb = self.pipeline.rgba_to_rgb_black(rgba)[0].permute(1, 2, 0).detach().cpu()
            self.pipeline.pydiffvg.imwrite(rgb, str(preview_path), gamma=1.0)
            self.pipeline.save_strokes_as_svg(
                svg_path,
                global_strokes.detach().cpu(),
                width=width,
                height=height,
                normalized_coordinates=False,
            )
            path_count = int(global_strokes.shape[0])

        if not svg_path.is_file() or svg_path.stat().st_size == 0:
            raise RuntimeError(f"Inference did not generate a non-empty SVG: {svg_path}")
        svg_header = svg_path.read_text(encoding="utf-8", errors="ignore")[:4096].lower()
        if "<svg" not in svg_header:
            raise RuntimeError(f"The generated file is not a valid SVG document: {svg_path}")
        if not preview_path.is_file() or preview_path.stat().st_size == 0:
            raise RuntimeError(f"Inference did not generate a non-empty PNG preview: {preview_path}")

        return InferenceResult(
            svg_path=svg_path,
            preview_path=preview_path,
            region_count=len(regions),
            path_count=path_count,
        )
