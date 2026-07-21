"""Create Chapter 3 figures from the cross-method benchmark report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


METHODS = ["ours", "supersvg", "live", "im2vec"]
TRADEOFF_METHODS = ["ours", "supersvg", "live"]
LABELS = {
    "ours": "Notre approche",
    "supersvg": "SuperSVG",
    "live": "LIVE",
    "im2vec": "Im2Vec",
}
COLORS = {
    "ours": "#167D8D",
    "supersvg": "#345995",
    "live": "#E07A5F",
    "im2vec": "#6B7280",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_quantitative(summary: dict[str, dict[str, str]], output: Path) -> None:
    definitions = [
        ("mse", "MSE (plus faible)", 5),
        ("psnr_db", "PSNR en dB (plus élevé)", 2),
        ("ssim", "SSIM (plus élevée)", 3),
        ("lpips", "LPIPS (plus faible)", 3),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    positions = np.arange(len(METHODS))
    for axis, (metric, title, decimals) in zip(axes.flat, definitions):
        means = [float(summary[method][f"{metric}_mean"]) for method in METHODS]
        errors = [float(summary[method][f"{metric}_std"]) for method in METHODS]
        bars = axis.bar(
            positions,
            means,
            yerr=errors,
            capsize=3,
            color=[COLORS[method] for method in METHODS],
            edgecolor="white",
            linewidth=0.8,
        )
        axis.set_title(title, fontsize=11)
        axis.set_xticks(positions, [LABELS[method] for method in METHODS], rotation=12)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, means):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.{decimals}f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.tight_layout(pad=1.4)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_qualitative(root: Path, output: Path) -> None:
    inputs = sorted((root / "ours_native" / "inputs").glob("*.png"))
    columns = ["input", *METHODS]
    titles = ["Image cible", *[LABELS[method] for method in METHODS]]
    figure, axes = plt.subplots(len(inputs), len(columns), figsize=(13.2, 10.7))
    for row_index, input_path in enumerate(inputs):
        image_paths = [
            input_path,
            *[root / "common_renders" / method / input_path.name for method in METHODS],
        ]
        for column_index, image_path in enumerate(image_paths):
            axes[row_index, column_index].imshow(Image.open(image_path).convert("RGB"))
            axes[row_index, column_index].axis("off")
            if row_index == 0:
                axes[row_index, column_index].set_title(titles[column_index], fontsize=11)
        axes[row_index, 0].text(
            -0.08,
            0.5,
            f"Image {row_index + 1}",
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=10,
        )
    figure.subplots_adjust(left=0.055, right=0.995, top=0.94, bottom=0.015, wspace=0.035, hspace=0.08)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_tradeoff(summary: dict[str, dict[str, str]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for method in TRADEOFF_METHODS:
        row = summary[method]
        time_value = float(row["inference_seconds_mean"])
        ssim = float(row["ssim_mean"])
        segments = float(row["cubic_segment_count_mean"])
        lpips = float(row["lpips_mean"])
        size_kib = float(row["svg_bytes_mean"]) / 1024.0
        marker_size = 90.0 + 28.0 * np.log10(max(size_kib, 1.0))
        axes[0].scatter(time_value, ssim, s=marker_size, color=COLORS[method], edgecolor="white")
        time_offset = (-5, 5) if method == "live" else (5, 5)
        time_align = "right" if method == "live" else "left"
        axes[0].annotate(
            LABELS[method],
            (time_value, ssim),
            xytext=time_offset,
            textcoords="offset points",
            ha=time_align,
        )
        axes[1].scatter(segments, lpips, s=marker_size, color=COLORS[method], edgecolor="white")
        segment_offset = (-5, 5) if method == "ours" else (5, 5)
        segment_align = "right" if method == "ours" else "left"
        axes[1].annotate(
            LABELS[method],
            (segments, lpips),
            xytext=segment_offset,
            textcoords="offset points",
            ha=segment_align,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Temps de vectorisation moyen (s, échelle logarithmique)")
    axes[0].set_ylabel("SSIM moyenne")
    axes[0].set_title("Fidélité structurelle et temps")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Segments cubiques moyens (échelle logarithmique)")
    axes[1].set_ylabel("LPIPS moyenne")
    axes[1].set_title("Similarité perceptuelle et complexité")
    for axis in axes:
        axis.margins(x=0.12, y=0.12)
        axis.grid(alpha=0.25)
        axis.set_axisbelow(True)
    figure.tight_layout(pad=1.3)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=workspace / "implementation" / "evaluation_results" / "method_comparison",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=workspace / "memoire" / "figures",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    figure_dir = args.figure_dir.resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = read_csv(root / "summary.csv")
    summary = {row["method"]: row for row in summary_rows}
    save_quantitative(summary, figure_dir / "ch3_comparaison_methodes_quantitative.png")
    save_qualitative(root, figure_dir / "ch3_comparaison_methodes_qualitative.png")
    save_tradeoff(summary, figure_dir / "ch3_compromis_methodes.png")


if __name__ == "__main__":
    main()
