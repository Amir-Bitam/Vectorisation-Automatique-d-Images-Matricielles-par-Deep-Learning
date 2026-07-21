"""FastAPI backend for the repository's DINOv3 + SLIC + DiffVG model."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import re
import shutil
import traceback
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import BASE_DIR, Settings
from model_service import (
    DEFAULT_NUM_REGIONS,
    MAX_NUM_REGIONS,
    MIN_NUM_REGIONS,
    ModelNotReadyError,
    RasterVectorizationService,
)


LOGGER = logging.getLogger(__name__)
SETTINGS = Settings.from_environment()
MODEL_SERVICE = RasterVectorizationService(SETTINGS)
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
CHUNK_SIZE = 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        MODEL_SERVICE.load()
    except Exception:
        # Keep FastAPI alive so /health exposes the precise startup problem.
        LOGGER.exception("The raster-to-SVG model could not be loaded at startup")
    yield


app = FastAPI(title="PFE Vectorization Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_filename(filename: Optional[str]) -> str:
    name = Path(filename or "input.png").name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._")
    return name or "input.png"


def _save_upload(file: UploadFile, destination: Path) -> int:
    total_bytes = 0
    with destination.open("xb") as output:
        while chunk := file.file.read(CHUNK_SIZE):
            total_bytes += len(chunk)
            if total_bytes > SETTINGS.max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"The uploaded image exceeds the {SETTINGS.max_upload_bytes // (1024 * 1024)} MB limit.",
                )
            output.write(chunk)
    return total_bytes


def _validate_image_file(input_path: Path, size: int) -> None:
    if size == 0:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    try:
        from PIL import Image, UnidentifiedImageError
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="Pillow is missing from the backend environment; image validation cannot run.",
        ) from exc

    try:
        with Image.open(input_path) as image:
            detected_format = (image.format or "").upper()
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"The uploaded file is not a decodable PNG or JPEG image: {exc}",
        ) from exc

    if detected_format not in {"PNG", "JPEG"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{detected_format or 'unknown'}'. Only PNG and JPEG are accepted.",
        )


def _validate_download_parts(job_id: str, filename: str, extension: str) -> Path:
    if not re.fullmatch(r"[A-Fa-f0-9]{32}", job_id):
        raise HTTPException(status_code=400, detail="Invalid job_id.")
    safe_name = _safe_filename(filename)
    if safe_name != filename or Path(filename).suffix.lower() != extension:
        raise HTTPException(status_code=400, detail=f"Invalid {extension} filename.")
    path = OUTPUTS_DIR / job_id / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Generated file not found.")
    return path


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Vectorization backend is running"}


@app.get("/health")
def health():
    return MODEL_SERVICE.health()


@app.post("/vectorize")
def vectorize(
    file: UploadFile | None = File(default=None),
    num_regions: int = Form(
        default=DEFAULT_NUM_REGIONS,
        ge=MIN_NUM_REGIONS,
        le=MAX_NUM_REGIONS,
    ),
):
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    suffix = Path(file.filename).suffix.lower()
    content_type = (file.content_type or "").lower()
    if suffix not in ALLOWED_EXTENSIONS or content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid file type. Upload a PNG or JPEG image "
                f"(received extension '{suffix or 'none'}', content type '{content_type or 'none'}')."
            ),
        )
    if not MODEL_SERVICE.model_loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "The vectorization model is not loaded.",
                "model_error": MODEL_SERVICE.load_error,
                "checkpoint": str(SETTINGS.model_checkpoint),
            },
        )

    job_id = uuid4().hex
    upload_job_dir = UPLOADS_DIR / job_id
    output_job_dir = OUTPUTS_DIR / job_id
    upload_job_dir.mkdir(parents=True, exist_ok=False)
    output_job_dir.mkdir(parents=True, exist_ok=False)
    input_path = upload_job_dir / _safe_filename(file.filename)

    try:
        size = _save_upload(file, input_path)
        _validate_image_file(input_path, size)
        result = MODEL_SERVICE.vectorize(
            input_path,
            output_job_dir,
            num_regions=num_regions,
        )
    except HTTPException:
        shutil.rmtree(upload_job_dir, ignore_errors=True)
        shutil.rmtree(output_job_dir, ignore_errors=True)
        raise
    except ModelNotReadyError as exc:
        shutil.rmtree(upload_job_dir, ignore_errors=True)
        shutil.rmtree(output_job_dir, ignore_errors=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        debug_traceback = traceback.format_exc()
        LOGGER.exception("Inference failed for job %s", job_id)
        is_cuda_error = "cuda" in f"{type(exc).__name__}: {exc}".lower()
        raise HTTPException(
            status_code=500,
            detail={
                "message": "CUDA inference failed." if is_cuda_error else "Model inference failed.",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "job_id": job_id,
                "traceback": debug_traceback,
            },
        ) from exc
    finally:
        file.file.close()
        # The generated files must remain downloadable, but the source upload is
        # temporary and is removed as soon as inference finishes or fails.
        shutil.rmtree(upload_job_dir, ignore_errors=True)

    return {
        "job_id": job_id,
        "svg_filename": result.svg_path.name,
        "download_url": f"/download/{job_id}/{result.svg_path.name}",
        "preview_filename": result.preview_path.name,
        "preview_url": f"/preview/{job_id}/{result.preview_path.name}",
        "device": MODEL_SERVICE.device_name,
        "num_regions": num_regions,
        "region_count": result.region_count,
        "path_count": result.path_count,
    }


@app.get("/download/{job_id}/{filename}")
def download_svg(job_id: str, filename: str):
    svg_path = _validate_download_parts(job_id, filename, ".svg")
    return FileResponse(svg_path, media_type="image/svg+xml", filename=filename)


@app.get("/preview/{job_id}/{filename}")
def preview_png(job_id: str, filename: str):
    png_path = _validate_download_parts(job_id, filename, ".png")
    return FileResponse(png_path, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=SETTINGS.backend_host, port=SETTINGS.backend_port)
