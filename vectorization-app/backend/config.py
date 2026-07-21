"""Backend configuration resolved relative to this directory."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # The defaults still let /health explain model issues.
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = BASE_DIR.parents[1]

if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")


def _resolve_path(raw_value: str | None, default: Path) -> Path:
    path = Path(raw_value).expanduser() if raw_value else default
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def _read_positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return value


@dataclass(frozen=True)
class Settings:
    implementation_dir: Path
    model_checkpoint: Path
    model_device: str
    backend_host: str
    backend_port: int
    max_upload_bytes: int

    @classmethod
    def from_environment(cls) -> "Settings":
        implementation_dir = _resolve_path(
            os.getenv("IMPLEMENTATION_DIR"),
            REPOSITORY_DIR / "implementation",
        )
        checkpoint_default = (
            implementation_dir
            / "checkpoints"
            / "raster_to_svg_128paths"
            / "epoch_0019.pt"
        )

        return cls(
            implementation_dir=implementation_dir,
            model_checkpoint=_resolve_path(os.getenv("MODEL_CHECKPOINT"), checkpoint_default),
            model_device=os.getenv("MODEL_DEVICE", "auto").strip().lower(),
            backend_host=os.getenv("BACKEND_HOST", "127.0.0.1").strip() or "127.0.0.1",
            backend_port=_read_positive_int("BACKEND_PORT", 8000),
            max_upload_bytes=_read_positive_int("MAX_UPLOAD_MB", 25) * 1024 * 1024,
        )
