#!/usr/bin/env python3
"""Paper-ready result figures for LCP-CNN / PulmoScan final run (processed_v3, 48³)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Exact epoch logs from final Kubeflow run
EPOCHS = np.arange(1, 29)
TRAIN_LOSS = np.array([
    0.9966, 0.8465, 0.7215, 0.8678, 0.5215, 0.7867, 0.6672, 0.8732,
    0.6543, 0.5791, 0.8663, 0.3594, 0.2699, 0.8478, 0.6099, 0.3319,
    0.2817, 0.4644, 0.4239, 0.5211, 0.4364, 0.3563, 0.3279, 0.3082,
    0.4129, 0.2605, 0.1276, 0.0061,
])
VAL_AUC = np.array([
    0.5743, 0.7428, 0.4710, 0.8351, 0.7754, 0.8533, 0.8514, 0.8460,
    0.9203, 0.9040, 0.9167, 0.9312, 0.9203, 0.8641, 0.8967, 0.8895,
    0.9185, 0.9167, 0.9239, 0.9493, 0.9475, 0.9257, 0.9330, 0.9384,
    0.9312, 0.9275, 0.9348, 0.9330,
])
VAL_ACC = np.array([
    0.657, 0.771, 0.714, 0.800, 0.886, 0.800, 0.771, 0.743,
    0.771, 0.829, 0.857, 0.886, 0.886, 0.857, 0.943, 0.829,
    0.914, 0.914, 0.914, 0.943, 0.886, 0.886, 0.886, 0.914,
    0.886, 0.886, 0.914, 0.886,
])

BEST_VAL_EPOCH = int(np.argmax(VAL_AUC)) + 1
BEST_VAL_AUC = float(VAL_AUC.max())

TEST = {
    "AUC": 0.9139,
    "AUPRC": 0.9583,
    "Accuracy": 0.7586,
    "Sensitivity": 0.7000,
    "Specificity": 0.8889,
    "Precision": 0.9333,
}
CM = np.array([[8, 1], [6, 14]])  # [[TN, FP], [FN, TP]] rows=true ben/mal, cols=pred


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, name: str):
    for ext in ("png", "pdf"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", edgecolor="none")
        print("wrote", path)
    plt.close(fig)


def fig_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(EPOCHS, TRAIN_LOSS, color="#1f4e79", marker="o", markersize=3.5, label="Train loss")
    ax.axvline(BEST_VAL_EPOCH, color="#b22222", ls="--", lw=1.2, label=f"Best val AUC (epoch {BEST_VAL_EPOCH})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_title("(a) Training loss")
    ax.set_xlim(1, 28)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25, linestyle=":")

    ax = axes[1]
    ax.plot(EPOCHS, VAL_AUC, color="#0b6e4f", marker="o", markersize=3.5, label="Val AUC")
    ax.plot(EPOCHS, VAL_ACC, color="#c47d0e", marker="s", markersize=3.2, label="Val accuracy")
    ax.scatter([BEST_VAL_EPOCH], [BEST_VAL_AUC], s=60, c="#b22222", zorder=5, label=f"Best AUC = {BEST_VAL_AUC:.3f}")
    ax.axhline(0.85, color="#666", ls=":", lw=1.0, label="Target AUC = 0.85")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("(b) Validation AUC and accuracy")
    ax.set_xlim(1, 28)
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25, linestyle=":")

    fig.suptitle("Training dynamics on processed_v3 (48³ hybrid CNN–Swin)", fontsize=13, y=1.02)
    save(fig, "fig_training_curves")


def fig_confusion_matrix():
    fig, ax = plt.subplots(figsize=(4.6, 4.0), constrained_layout=True)
    im = ax.imshow(CM, cmap="Blues", vmin=0, vmax=CM.max())
    ax.set_xticks([0, 1], ["Pred benign", "Pred malignant"])
    ax.set_yticks([0, 1], ["True benign", "True malignant"])
    ax.set_title("Test-set confusion matrix (n = 29)")
    labels = [["TN = 8", "FP = 1"], ["FN = 6", "TP = 14"]]
    thresh = CM.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, labels[i][j],
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if CM[i, j] > thresh else "#123",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count")
    ax.text(
        0.5, -0.18,
        "Accuracy = 0.759   |   Sensitivity = 0.70   |   Specificity = 0.89",
        transform=ax.transAxes, ha="center", fontsize=9,
    )
    save(fig, "fig_confusion_matrix")


def fig_test_metrics():
    names = list(TEST.keys())
    vals = [TEST[k] for k in names]
    colors = ["#0b6e4f", "#1f4e79", "#4a4a4a", "#8b4513", "#2f6f8f", "#6b3fa0"]

    fig, ax = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
    bars = ax.bar(names, vals, color=colors, width=0.72, edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Held-out test metrics (best.pt)")
    ax.axhline(0.85, color="#b22222", ls="--", lw=1.1, label="AUC target = 0.85")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    save(fig, "fig_test_metrics")


def fig_dataset_split():
    splits = ["Train", "Validation", "Test"]
    mal = np.array([45, 12, 20])
    ben = np.array([87, 23, 9])
    x = np.arange(len(splits))
    w = 0.55

    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    ax.bar(x, ben, w, label="Benign", color="#5b8fa8")
    ax.bar(x, mal, w, bottom=ben, label="Malignant", color="#c45c26")
    for i, (b, m) in enumerate(zip(ben, mal)):
        ax.text(i, b + m + 1.5, f"n={b+m}", ha="center", fontsize=9)
    ax.set_xticks(x, splits)
    ax.set_ylabel("Number of nodules")
    ax.set_title("Labeled nodule counts by split (processed_v3)")
    ax.legend()
    ax.set_ylim(0, 160)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    save(fig, "fig_dataset_split")


def fig_results_summary_card():
    """Single summary panel suitable for poster / results slide."""
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Final held-out test summary — processed_v3, VOI 48³, hybrid CNN–Swin", fontsize=12, pad=10)

    cards = [
        (0.4, 1.0, 2.1, 2.0, "Test AUC", "0.914", "#0b6e4f"),
        (2.8, 1.0, 2.1, 2.0, "Test AUPRC", "0.958", "#1f4e79"),
        (5.2, 1.0, 2.1, 2.0, "Accuracy", "0.759", "#4a4a4a"),
        (7.6, 1.0, 2.1, 2.0, "Best Val AUC", "0.949", "#8b4513"),
    ]
    for x, y, w, h, title, val, color in cards:
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.4, edgecolor=color, facecolor="#fafafa",
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center", fontsize=10, color="#333")
        ax.text(x + w / 2, y + h * 0.38, val, ha="center", va="center", fontsize=20, fontweight="bold", color=color)

    ax.text(
        5.0, 0.35,
        "Early stopping at epoch 28  ·  Test n = 29 nodules  ·  Research use only",
        ha="center", fontsize=9, style="italic", color="#444",
    )
    save(fig, "fig_results_summary")


def fig_val_vs_test():
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    metrics = ["AUC", "Accuracy"]
    val_vals = [BEST_VAL_AUC, float(VAL_ACC[BEST_VAL_EPOCH - 1])]
    test_vals = [TEST["AUC"], TEST["Accuracy"]]
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w / 2, val_vals, w, label="Validation (best ckpt epoch)", color="#0b6e4f")
    ax.bar(x + w / 2, test_vals, w, label="Held-out test", color="#1f4e79")
    for i, (v, t) in enumerate(zip(val_vals, test_vals)):
        ax.text(i - w / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
        ax.text(i + w / 2, t + 0.015, f"{t:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Validation (model selection) vs held-out test")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    save(fig, "fig_val_vs_test")


def main():
    apply_style()
    fig_training_curves()
    fig_confusion_matrix()
    fig_test_metrics()
    fig_dataset_split()
    fig_results_summary_card()
    fig_val_vs_test()
    print("All figures written to", OUT)


if __name__ == "__main__":
    main()
