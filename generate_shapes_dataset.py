import os
import csv
import random
import math
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional

# --- Try CairoSVG for rasterization ---
HAS_CAIROSVG = False
try:
    import cairosvg  # pip install cairosvg
    HAS_CAIROSVG = True
except Exception:
    HAS_CAIROSVG = False


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    out_dir: str = "dataset_shapes"
    n_samples: int = 5000
    width: int = 256
    height: int = 256
    seed: int = 42

    # distribution across shape types (must sum to 1.0)
    p_circle: float = 0.20
    p_rect: float = 0.20
    p_triangle: float = 0.20
    p_polygon: float = 0.20
    p_bezier: float = 0.20

    # complexity: number of shapes per image (start with 1 for easiest)
    min_shapes_per_image: int = 1
    max_shapes_per_image: int = 1

    # colors
    allow_color: bool = True       # if False => monochrome
    allow_stroke: bool = True      # strokes around shapes
    background: str = "white"      # "white" or "transparent"


cfg = Config()


# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs():
    os.makedirs(os.path.join(cfg.out_dir, "svg"), exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, "png"), exist_ok=True)


def rand_color() -> str:
    if not cfg.allow_color:
        return "black"
    # choose from a small palette (stable learning) + random sometimes
    palette = ["black", "#1f2937", "#111827", "#0f172a", "#334155", "#4b5563",
               "#ef4444", "#22c55e", "#3b82f6", "#f59e0b", "#a855f7"]
    if random.random() < 0.75:
        return random.choice(palette)
    # random hex
    return "#{:02x}{:02x}{:02x}".format(random.randint(0,255), random.randint(0,255), random.randint(0,255))


def rand_stroke() -> Tuple[str, float]:
    if not cfg.allow_stroke or random.random() < 0.4:
        return ("none", 0.0)
    col = "black" if not cfg.allow_color else random.choice(["black", "#111827", "#334155"])
    width = random.uniform(1.0, 4.0)
    return (col, width)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def svg_header() -> str:
    bg_rect = ""
    if cfg.background == "white":
        bg_rect = f'<rect x="0" y="0" width="{cfg.width}" height="{cfg.height}" fill="white"/>'
    # if transparent => nothing
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{cfg.width}" height="{cfg.height}" viewBox="0 0 {cfg.width} {cfg.height}">\n'
        f'{bg_rect}\n'
    )


def svg_footer() -> str:
    return "</svg>\n"


# -----------------------------
# Shape generators
# -----------------------------
def gen_circle() -> str:
    r = random.uniform(cfg.width * 0.05, cfg.width * 0.25)
    cx = random.uniform(r, cfg.width - r)
    cy = random.uniform(r, cfg.height - r)
    fill = rand_color()
    stroke, sw = rand_stroke()
    return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}"/>'


def gen_rect() -> str:
    w = random.uniform(cfg.width * 0.10, cfg.width * 0.50)
    h = random.uniform(cfg.height * 0.10, cfg.height * 0.50)
    x = random.uniform(0, cfg.width - w)
    y = random.uniform(0, cfg.height - h)
    rx = random.uniform(0, min(w, h) * 0.2) if random.random() < 0.5 else 0
    angle = random.uniform(-45, 45)

    # rotate around center
    cx = x + w/2
    cy = y + h/2

    fill = rand_color()
    stroke, sw = rand_stroke()
    return (f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}" '
            f'transform="rotate({angle:.2f} {cx:.2f} {cy:.2f})"/>')

def gen_triangle() -> str:
    # random triangle within image bounds
    pts = []
    for _ in range(3):
        px = random.uniform(0, cfg.width)
        py = random.uniform(0, cfg.height)
        pts.append((px, py))
    fill = rand_color()
    stroke, sw = rand_stroke()
    points_str = " ".join([f"{x:.2f},{y:.2f}" for x, y in pts])
    return f'<polygon points="{points_str}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}"/>'

def gen_polygon() -> str:
    # random convex-ish polygon: pick center then points around
    n = random.randint(4, 8)
    cx = random.uniform(cfg.width*0.3, cfg.width*0.7)
    cy = random.uniform(cfg.height*0.3, cfg.height*0.7)
    base_r = random.uniform(cfg.width*0.10, cfg.width*0.30)

    angles = sorted([random.uniform(0, 2*math.pi) for _ in range(n)])
    pts = []
    for a in angles:
        rr = base_r * random.uniform(0.6, 1.2)
        x = clamp(cx + rr * math.cos(a), 0, cfg.width)
        y = clamp(cy + rr * math.sin(a), 0, cfg.height)
        pts.append((x, y))

    fill = rand_color()
    stroke, sw = rand_stroke()
    points_str = " ".join([f"{x:.2f},{y:.2f}" for x, y in pts])
    return f'<polygon points="{points_str}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}"/>'

def gen_bezier() -> str:
    # simple cubic bezier path "M x y C x1 y1 x2 y2 x3 y3"
    x0 = random.uniform(0, cfg.width)
    y0 = random.uniform(0, cfg.height)
    x1 = random.uniform(0, cfg.width)
    y1 = random.uniform(0, cfg.height)
    x2 = random.uniform(0, cfg.width)
    y2 = random.uniform(0, cfg.height)
    x3 = random.uniform(0, cfg.width)
    y3 = random.uniform(0, cfg.height)

    stroke_col = rand_color() if cfg.allow_color else "black"
    sw = random.uniform(2.0, 6.0)
    fill = "none"

    # sometimes close a shape by adding a line back (harder)
    if random.random() < 0.25:
        d = f"M {x0:.2f} {y0:.2f} C {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {x3:.2f} {y3:.2f} Z"
        fill = rand_color()
        return f'<path d="{d}" fill="{fill}" stroke="{stroke_col}" stroke-width="{sw:.2f}"/>'

    d = f"M {x0:.2f} {y0:.2f} C {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {x3:.2f} {y3:.2f}"
    return f'<path d="{d}" fill="{fill}" stroke="{stroke_col}" stroke-width="{sw:.2f}" stroke-linecap="round"/>'

def sample_shape_type() -> str:
    r = random.random()
    thresholds = [
        ("circle", cfg.p_circle),
        ("rect", cfg.p_rect),
        ("triangle", cfg.p_triangle),
        ("polygon", cfg.p_polygon),
        ("bezier", cfg.p_bezier),
    ]
    acc = 0.0
    for name, p in thresholds:
        acc += p
        if r <= acc:
            return name
    return "circle"


def gen_shape(name: str) -> str:
    if name == "circle":
        return gen_circle()
    if name == "rect":
        return gen_rect()
    if name == "triangle":
        return gen_triangle()
    if name == "polygon":
        return gen_polygon()
    if name == "bezier":
        return gen_bezier()
    return gen_circle()




def svg_to_png(svg_path: str, png_path: str):
    inkscape_path = r"C:\Program Files\Inkscape\bin\inkscape.exe"
    subprocess.run([
        inkscape_path,
        svg_path,
        "--export-type=png",
        f"--export-filename={png_path}",
        f"--export-width={cfg.width}",
        f"--export-height={cfg.height}"
    ], check=True)


def main():
    random.seed(cfg.seed)
    ensure_dirs()

    meta_path = os.path.join(cfg.out_dir, "metadata.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "svg_file", "png_file", "shape_types"])

        for i in range(1, cfg.n_samples + 1):
            sample_id = f"{i:06d}"
            svg_file = f"{sample_id}.svg"
            png_file = f"{sample_id}.png"

            n_shapes = random.randint(cfg.min_shapes_per_image, cfg.max_shapes_per_image)
            shape_types: List[str] = []
            shapes_svg: List[str] = []

            for _ in range(n_shapes):
                st = sample_shape_type()
                shape_types.append(st)
                shapes_svg.append(gen_shape(st))

            svg_content = svg_header() + "\n".join(shapes_svg) + "\n" + svg_footer()

            svg_path = os.path.join(cfg.out_dir, "svg", svg_file)
            png_path = os.path.join(cfg.out_dir, "png", png_file)

            with open(svg_path, "w", encoding="utf-8") as sf:
                sf.write(svg_content)

            # rasterize
            svg_to_png(svg_path, png_path)

            writer.writerow([sample_id, f"svg/{svg_file}", f"png/{png_file}", "|".join(shape_types)])

            if i % 500 == 0:
                print(f"Generated {i}/{cfg.n_samples}")

    print("\nDone ✅")
    print(f"Dataset saved in: {os.path.abspath(cfg.out_dir)}")
    print(f"Metadata: {os.path.abspath(meta_path)}")


if __name__ == "__main__":
    main()