import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


BASE_DIR = Path(__file__).resolve().parent
SUPERSVG_DIR = BASE_DIR / "SuperSVG"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PFE Vectorization Backend")

# CORS configuration: allow the local Vite frontend to call the FastAPI backend.
app.add_middleware(
    CORSMiddleware,
    # Vite normally uses 5173, but may fall back to 5174 if 5173 is already busy.
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
    # Keep uploaded filenames safe before writing them inside the uploads folder.
    name = Path(filename or "input.png").name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._")
    return name or "input.png"


def _run_supersvg(
    upload_dir: Path,
    output_dir: Path,
    path_num: int,
    optimize_iter: int,
) -> subprocess.CompletedProcess[str]:
    # Build the SuperSVG inference command. path_num and optimize_iter come from the frontend form.
    command = [
        sys.executable,
        "inference.py",
        "--input_path",
        str(upload_dir),
        "--output_dir",
        str(output_dir),
        "--device",
        "cpu",
        "--path_num",
        str(path_num),
        "--optimize_iter",
        str(optimize_iter),
    ]

    # Run inference.py and capture stdout/stderr so failed jobs are easier to debug.
    return subprocess.run(
        command,
        cwd=str(SUPERSVG_DIR),
        capture_output=True,
        text=True,
    )


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Vectorization backend is running"}


@app.post("/vectorize")
def vectorize(
    file: UploadFile = File(...),
    path_num: int = Form(256),
    optimize_iter: int = Form(0),
):
    # Uploaded file and parameter validation happens before creating job folders.
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")
    if path_num <= 0:
        raise HTTPException(status_code=400, detail="path_num must be greater than 0.")
    if optimize_iter < 0:
        raise HTTPException(status_code=400, detail="optimize_iter must be greater than or equal to 0.")

    job_id = uuid4().hex
    upload_job_dir = UPLOADS_DIR / job_id
    output_job_dir = OUTPUTS_DIR / job_id

    # Each request gets isolated input/output folders to avoid mixing generated files.
    upload_job_dir.mkdir(parents=True, exist_ok=True)
    output_job_dir.mkdir(parents=True, exist_ok=True)

    # Save the uploaded raster image before passing the folder to SuperSVG.
    input_path = upload_job_dir / _safe_filename(file.filename)
    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = _run_supersvg(upload_job_dir, output_job_dir, path_num, optimize_iter)
    if result.returncode != 0:
        # Keep stdout/stderr in the response because SuperSVG errors are usually diagnosed from logs.
        raise HTTPException(
            status_code=500,
            detail={
                "message": "SuperSVG inference failed.",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )

    # SuperSVG writes SVG files into the job output folder. Use the newest SVG as the result.
    svg_files = sorted(output_job_dir.glob("*.svg"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not svg_files:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "SuperSVG finished, but no SVG was generated.",
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )

    svg_file = svg_files[0]

    # Return the SVG download URL and parameter values so the frontend can display the result.
    return {
        "job_id": job_id,
        "svg_filename": svg_file.name,
        "download_url": f"/download/{job_id}/{svg_file.name}",
        "path_num": path_num,
        "optimize_iter": optimize_iter,
    }


@app.get("/download/{job_id}/{filename}")
def download_svg(job_id: str, filename: str):
    # Validate path parts before serving a generated SVG from disk.
    if not re.fullmatch(r"[A-Fa-f0-9]{32}", job_id):
        raise HTTPException(status_code=400, detail="Invalid job_id.")

    safe_name = _safe_filename(filename)
    if safe_name != filename or Path(filename).suffix.lower() != ".svg":
        raise HTTPException(status_code=400, detail="Invalid SVG filename.")

    svg_path = OUTPUTS_DIR / job_id / filename
    if not svg_path.is_file():
        raise HTTPException(status_code=404, detail="SVG file not found.")

    return FileResponse(svg_path, media_type="image/svg+xml", filename=filename)
