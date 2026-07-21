"""Create Chapter 3 figures from the reproducible evaluation outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries


MODEL_LABELS = {
    "paths32_e09": "32 chemins",
    "paths128_e19": "128 chemins",
}
MODEL_COLORS = {
    "paths32_e09": "#167D8D",
    "paths128_e19": "#D66A4A",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.resolve()}")


def make_slic_figure(results_dir: Path, output_dir: Path) -> None:
    first_row = read_csv(results_dir / "metrics.csv")[0]
    image_id = Path(first_row["image"]).stem
    image = read_image(results_dir / "inputs" / f"{image_id}.png")
    labels = np.load(results_dir / "first_image_slic_labels.npy")

    mean_color = np.zeros_like(image)
    for label in np.unique(labels):
        region = labels == label
        mean_color[region] = image[region].mean(axis=0)
    boundaries = mark_boundaries(image, labels, color=(0.95, 0.2, 0.12), mode="thick")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    panels = (
        (image, "Image redimensionnee"),
        (boundaries, f"Contours SLIC ({len(np.unique(labels))} regions)"),
        (mean_color, "Couleur moyenne par region"),
    )
    for axis, (panel, title) in zip(axes, panels):
        axis.imshow(panel)
        axis.set_title(title, pad=7)
        axis.axis("off")
    fig.subplots_adjust(wspace=0.04)
    save_figure(fig, output_dir / "ch3_segmentation_slic.png")


def make_qualitative_figure(results_dir: Path, output_dir: Path) -> None:
    selected = (
        ("0078807977709", "Scene complexe"),
        ("0078936124638016", "Objet isole"),
        ("007905920170535", "Texture fine"),
    )
    rows = read_csv(results_dir / "metrics.csv")
    metrics = {(row["model"], Path(row["image"]).stem): row for row in rows}

    fig, axes = plt.subplots(len(selected), 3, figsize=(9.2, 8.6))
    column_titles = ("Image matricielle", "32 chemins par region", "128 chemins par region")
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=11, pad=10)

    for row_index, (image_id, row_label) in enumerate(selected):
        images = (
            read_image(results_dir / "inputs" / f"{image_id}.png"),
            read_image(results_dir / "paths32_e09" / f"{image_id}.png"),
            read_image(results_dir / "paths128_e19" / f"{image_id}.png"),
        )
        for column, image in enumerate(images):
            axis = axes[row_index, column]
            axis.imshow(image)
            axis.axis("off")
            if column == 0:
                axis.set_ylabel(row_label, rotation=90, labelpad=10, fontsize=10)
            else:
                model = "paths32_e09" if column == 1 else "paths128_e19"
                item = metrics[(model, image_id)]
                axis.text(
                    0.5,
                    -0.045,
                    f"PSNR {float(item['psnr_db']):.2f} dB   SSIM {float(item['ssim']):.3f}",
                    ha="center",
                    va="top",
                    transform=axis.transAxes,
                    fontsize=8,
                )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.03, hspace=0.18, wspace=0.04)
    save_figure(fig, output_dir / "ch3_comparaison_qualitative.png")


def annotate_bars(axis: plt.Axes, bars, decimals: int) -> None:
    for bar in bars:
        height = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def make_quantitative_figure(results_dir: Path, output_dir: Path) -> None:
    summaries = read_csv(results_dir / "summary.csv")
    summaries.sort(key=lambda row: int(row["num_paths_per_region"]))
    models = [row["model"] for row in summaries]
    labels = [MODEL_LABELS[model] for model in models]
    colors = [MODEL_COLORS[model] for model in models]

    panels = (
        ("mse", "MSE", 4),
        ("psnr_db", "PSNR (dB)", 2),
        ("ssim", "SSIM", 3),
        ("lpips", "LPIPS", 3),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.3))
    for axis, (key, title, decimals) in zip(axes.flat, panels):
        means = [float(row[f"{key}_mean"]) for row in summaries]
        deviations = [float(row[f"{key}_std"]) for row in summaries]
        bars = axis.bar(labels, means, yerr=deviations, color=colors, capsize=4, width=0.62)
        axis.set_title(title)
        axis.grid(axis="y")
        axis.set_axisbelow(True)
        axis.set_ylim(bottom=0)
        annotate_bars(axis, bars, decimals)
    fig.subplots_adjust(hspace=0.34, wspace=0.25)
    save_figure(fig, output_dir / "ch3_comparaison_quantitative.png")


def make_progression_figure(progression_dir: Path, output_dir: Path) -> None:
    summaries = read_csv(progression_dir / "summary.csv")
    summaries.sort(key=lambda row: int(row["epoch_zero_based"]))
    epochs = [int(row["epoch_zero_based"]) + 1 for row in summaries]
    panels = (
        ("mse_mean", "MSE"),
        ("psnr_db_mean", "PSNR (dB)"),
        ("ssim_mean", "SSIM"),
        ("lpips_mean", "LPIPS"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.2))
    for axis, (key, title) in zip(axes.flat, panels):
        values = [float(row[key]) for row in summaries]
        axis.plot(epochs, values, color="#167D8D", marker="o", linewidth=2.0, markersize=5)
        axis.set_title(title)
        axis.set_xlabel("Epoque")
        axis.set_xticks(epochs)
        axis.grid(True)
        axis.set_axisbelow(True)
        axis.annotate(
            f"{values[-1]:.3f}" if key != "psnr_db_mean" else f"{values[-1]:.2f}",
            (epochs[-1], values[-1]),
            xytext=(-5, 8),
            textcoords="offset points",
            ha="right",
            fontsize=8,
        )
    fig.subplots_adjust(hspace=0.36, wspace=0.25)
    save_figure(fig, output_dir / "ch3_progression_entrainement.png")


def make_cost_figure(results_dir: Path, output_dir: Path) -> None:
    summaries = read_csv(results_dir / "summary.csv")
    summaries.sort(key=lambda row: int(row["num_paths_per_region"]))
    models = [row["model"] for row in summaries]
    labels = [MODEL_LABELS[model] for model in models]
    colors = [MODEL_COLORS[model] for model in models]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    time_parts = (
        ("preprocess_seconds_mean", "Preparation", "#A6A6A6"),
        ("network_seconds_mean", "Reseau", "#167D8D"),
        ("render_seconds_mean", "Rendu", "#D66A4A"),
        ("svg_export_seconds_mean", "Export SVG", "#D6A84A"),
    )
    bottoms = np.zeros(len(summaries), dtype=np.float64)
    for key, label, color in time_parts:
        values = np.asarray([float(row[key]) for row in summaries])
        axes[0].bar(labels, values, bottom=bottoms, label=label, color=color, width=0.62)
        bottoms += values
    axes[0].set_title("Temps moyen par image")
    axes[0].set_ylabel("Secondes")
    axes[0].grid(axis="y")
    axes[0].legend(fontsize=7, frameon=False)

    path_values = [float(row["path_count_mean"]) for row in summaries]
    path_bars = axes[1].bar(labels, path_values, color=colors, width=0.62)
    axes[1].set_title("Nombre moyen de chemins")
    axes[1].grid(axis="y")
    annotate_bars(axes[1], path_bars, 0)

    size_values = [float(row["svg_bytes_mean"]) / 1024.0 for row in summaries]
    size_bars = axes[2].bar(labels, size_values, color=colors, width=0.62)
    axes[2].set_title("Taille moyenne du SVG")
    axes[2].set_ylabel("Kio")
    axes[2].grid(axis="y")
    annotate_bars(axes[2], size_bars, 0)

    for axis in axes:
        axis.set_axisbelow(True)
        axis.set_ylim(bottom=0)
    fig.subplots_adjust(wspace=0.35)
    save_figure(fig, output_dir / "ch3_cout_vectorisation.png")


def make_figures(args: argparse.Namespace) -> None:
    configure_style()
    make_slic_figure(args.results_dir, args.output_dir)
    make_qualitative_figure(args.results_dir, args.output_dir)
    make_quantitative_figure(args.results_dir, args.output_dir)
    make_progression_figure(args.progression_dir, args.output_dir)
    make_cost_figure(args.results_dir, args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("implementation") / "evaluation_results" / "final_comparison",
    )
    parser.add_argument(
        "--progression-dir",
        type=Path,
        default=Path("implementation") / "evaluation_results" / "progression_32paths",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("memoire") / "figures",
    )
    return parser.parse_args()


if __name__ == "__main__":
    make_figures(parse_args())
