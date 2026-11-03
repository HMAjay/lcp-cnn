"""Journal figures. Line art goes to EPS with embedded Arial; Fig 3 is combination art at 600 dpi TIFF."""
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

HERE = Path(__file__).parent
OUT = HERE / "figures_out"
OUT.mkdir(exist_ok=True)
MM = 1 / 25.4
FULL = 174 * MM
INK = "#000000"
BLUE, VERM = "#0072B2", "#D55E00"
GRID = "#D9D9D9"
BOX = "#F2F2F2"
HILITE = "#DCE9F3"

FONT_FILES = [
    *(f"/System/Library/Fonts/Supplemental/{f}" for f in ("Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf")),
    *(f"C:/Windows/Fonts/{f}" for f in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf")),
]
for f in FONT_FILES:
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "ps.fonttype": 42,
        "pdf.fonttype": 42,
        "ps.useafm": False,
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save_eps(fig, name):
    fig.savefig(OUT / f"{name}.eps", format="eps")
    fig.savefig(OUT / f"{name}_preview.pdf", format="pdf")
    plt.close(fig)


def box(ax, x, y, w, h, text, fill=BOX, lw=0.6, bold=False, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=INK, linewidth=lw, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, p, q, lw=0.8, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, linewidth=lw, color=INK,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def fig1():
    fig, ax = plt.subplots(figsize=(FULL, 96 * MM))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 174)
    ax.set_ylim(0, 96)
    ax.axis("off")
    w, h = 40, 16
    xs = [0, 44.67, 89.33, 134]
    y1, y2, y3, y4 = 80, 56, 28, 0
    box(ax, xs[0], y1, w, h, "TCIA DICOM series\nand reader XML files")
    box(ax, xs[1], y1, w, h, "Series selection\n(annotated, one\nper patient)")
    box(ax, xs[2], y1, w, h, "Load series in HU,\nkeep original\nDICOM geometry")
    box(ax, xs[3], y1, w, h, "Resample to 1 mm\nisotropic voxels,\nsave with 1 mm affine")
    for i in range(3):
        arrow(ax, (xs[i] + w, y1 + h / 2), (xs[i + 1], y1 + h / 2))
    box(ax, xs[0], y2, w, h, "Geometry verification:\nphantoms, all annotations,\nreal CT series", ls="--")
    box(ax, xs[1], y2, w, h, "Reader contours:\nmean pixel index,\nnearest slice")
    box(ax, xs[2], y2, w, h, "Pixel index to patient\nmm using original\ngeometry (Eq. 1)", fill=HILITE, lw=1.2)
    box(ax, xs[3], y2, w, h, "Cluster reader marks\nwithin 6 mm into\nphysical nodules")
    arrow(ax, (xs[1] + w / 2, y1), (xs[1] + w / 2, y2 + h))
    arrow(ax, (xs[1] + w, y2 + h / 2), (xs[2], y2 + h / 2))
    arrow(ax, (xs[2] + w, y2 + h / 2), (xs[3], y2 + h / 2))
    arrow(ax, (xs[2] + w / 2, y1), (xs[2] + w / 2, y2 + h))
    ax.plot([xs[0] + w / 2, xs[0] + w / 2, xs[2] + w / 2, xs[2] + w / 2], [y2, 50.5, 50.5, 52.5],
            color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
    arrow(ax, (xs[2] + w / 2, 52.5), (xs[2] + w / 2, y2))
    box(ax, xs[3], y3, w, h, "Label rule: mean rating\n\u2265 4.0 malignant,\n\u2264 2.0 benign")
    box(ax, xs[2], y3, w, h, "Patient-level partition,\n48 mm VOI crop,\nHU window")
    box(ax, xs[1], y3, w, h, "Stage 2: hybrid 3D\nCNN\u2013Swin classifier")
    box(ax, xs[0], y3, w, h, "P(malignant) and\nheld-out evaluation")
    arrow(ax, (xs[3] + w / 2, y2), (xs[3] + w / 2, y3 + h))
    arrow(ax, (xs[3], y3 + h / 2), (xs[2] + w, y3 + h / 2))
    arrow(ax, (xs[2], y3 + h / 2), (xs[1] + w, y3 + h / 2))
    arrow(ax, (xs[1], y3 + h / 2), (xs[0] + w, y3 + h / 2))
    box(ax, xs[2], y4, w, h, "Stage 1: 3D U-Net\ncandidate centers (built,\nnot evaluated on real CT)", ls="--")
    box(ax, xs[0], y4, w, h, "Research review page\n(not evaluated)", ls="--")
    arrow(ax, (xs[2] + w / 2, y4 + h), (xs[2] + w / 2, y3), ls=(0, (3, 2)))
    arrow(ax, (xs[0] + w / 2, y3), (xs[0] + w / 2, y4 + h), ls=(0, (3, 2)))
    save_eps(fig, "Fig1")


def fig2():
    fig, ax = plt.subplots(figsize=(FULL, 56 * MM))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 174)
    ax.set_ylim(0, 56)
    ax.axis("off")
    top = [
        ("VOI\n1 \u00d7 48\u00b3", 24),
        ("Stem: 7\u00b3 conv,\nmax pool\n32 \u00d7 12\u00b3", 36),
        ("Residual stage 1\n2 blocks\n32 \u00d7 12\u00b3", 34),
        ("Residual stage 2\n2 blocks\n64 \u00d7 6\u00b3", 34),
        ("Residual stage 3\n2 blocks\n128 \u00d7 3\u00b3", 34),
    ]
    y, h, gap = 34, 22, 3
    x, prev = 0, None
    for text, w in top:
        box(ax, x, y, w, h, text)
        if prev is not None:
            arrow(ax, (prev, y + h / 2), (x, y + h / 2))
        prev = x + w
        x += w + gap
    ax.plot([prev - 17, prev - 17, 13, 13], [y, 28, 28, 25], color=INK, linewidth=0.8)
    arrow(ax, (13, 26), (13, 22))
    bottom = [
        ("Projection\n1\u00b3 conv\n256 \u00d7 3\u00b3", 26),
        ("Swin stage 1\n2 blocks, 4 heads\nwindow 2\u00b3\n256 \u00d7 3\u00b3", 32),
        ("Patch\nmerging\n512 \u00d7 2\u00b3", 22),
        ("Swin stage 2\n2 blocks, 8 heads\nwindow 2\u00b3\n512 \u00d7 2\u00b3", 32),
        ("Class-token\nattention\npool, 512", 24),
        ("MLP head\n128 to 1,\nsigmoid", 21),
    ]
    x, prev = 0, None
    gap2 = (174 - sum(w for _, w in bottom)) / 5
    for text, w in bottom:
        box(ax, x, 0, w, 22, text, fill=HILITE if text.startswith(("Swin", "Patch", "Class")) else BOX)
        if prev is not None:
            arrow(ax, (prev, 11), (x, 11))
        prev = x + w
        x += w + gap2
    save_eps(fig, "Fig2")


def load_rows():
    rows = list(csv.DictReader((HERE / "phantom/results/phantom_results.csv").open()))
    for r in rows:
        for k in ("pixel_spacing_mm", "error_mm", "nodule_fraction_in_voi", "voi_padding_fraction", "true_i_px", "true_j_px"):
            r[k] = float(r[k])
    return rows


def fig3():
    ep = [
        (1, 0.9966, 0.5743, 0.657), (2, 0.8465, 0.7428, 0.771), (3, 0.7215, 0.4710, 0.714), (4, 0.8678, 0.8351, 0.800),
        (5, 0.5215, 0.7754, 0.886), (6, 0.7867, 0.8533, 0.800), (7, 0.6672, 0.8514, 0.771), (8, 0.8732, 0.8460, 0.743),
        (9, 0.6543, 0.9203, 0.771), (10, 0.5791, 0.9040, 0.829), (11, 0.8663, 0.9167, 0.857), (12, 0.3594, 0.9312, 0.886),
        (13, 0.2699, 0.9203, 0.886), (14, 0.8478, 0.8641, 0.857), (15, 0.6099, 0.8967, 0.943), (16, 0.3319, 0.8895, 0.829),
        (17, 0.2817, 0.9185, 0.914), (18, 0.4644, 0.9167, 0.914), (19, 0.4239, 0.9239, 0.914), (20, 0.5211, 0.9493, 0.943),
        (21, 0.4364, 0.9475, 0.886), (22, 0.3563, 0.9257, 0.886), (23, 0.3279, 0.9330, 0.886), (24, 0.3082, 0.9384, 0.914),
        (25, 0.4129, 0.9312, 0.886), (26, 0.2605, 0.9275, 0.886), (27, 0.1276, 0.9348, 0.914), (28, 0.0061, 0.9330, 0.886),
    ]
    e, loss, auc, acc = map(list, zip(*ep))
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 68 * MM), gridspec_kw={"wspace": 0.28})
    a.plot(e, loss, color=INK, marker="o", markersize=2.8, linestyle="-")
    a.set_ylabel("Training loss")
    a.set_ylim(0, 1.05)
    b.plot(e, auc, color=BLUE, marker="o", markersize=2.8, linestyle="-", label="AUC")
    b.plot(e, acc, color=VERM, marker="s", markersize=2.8, linestyle="--", label="Accuracy at 0.50")
    b.set_ylabel("Validation metric")
    b.set_ylim(0.4, 1.0)
    for ax, tag in ((a, "a"), (b, "b")):
        ax.axvline(20, color=INK, linewidth=0.6, linestyle=":")
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, 29)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 28])
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.text(-0.14, 1.03, tag, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    a.text(20.4, 1.0, "Selected\ncheckpoint", fontsize=8, va="top")
    b.annotate("AUC 0.949,\nepoch 20", xy=(20, 0.9493), xytext=(21.4, 0.56), fontsize=8,
               arrowprops=dict(arrowstyle="-", linewidth=0.6, color=INK, shrinkA=1, shrinkB=2))
    b.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.24, 0.02))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.16)
    save_eps(fig, "Fig3")


def fig4():
    s = json.loads((HERE / "lidc_audit/cohort153_summary.json").read_text())
    test = s["test_nodules"]
    aucs = np.load(HERE / "lidc_audit/random_partition_aucs.npy")
    thr = s["size_model_test"]["threshold_diameter_mm"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 72 * MM), gridspec_kw={"wspace": 0.3, "width_ratios": [0.85, 1.15]})
    rng = np.random.default_rng(3)
    for lab, x0, color, marker, name in ((0, 0, BLUE, "o", "Benign"), (1, 1, VERM, "s", "Malignant")):
        d = [t[3] for t in test if t[2] == lab]
        a.scatter(x0 + rng.uniform(-0.13, 0.13, len(d)), d, s=16, facecolor="none" if lab == 0 else color, edgecolor=color, marker=marker, linewidth=0.8, zorder=3)
    a.axhline(thr, color=INK, linewidth=0.7, linestyle="--")
    a.text(-0.45, thr + 1.0, f"Training-set cut-off {thr:.1f} mm", fontsize=8, va="bottom")
    a.set_xticks([0, 1])
    a.set_xticklabels(["Benign (n = 9)", "Malignant (n = 20)"])
    a.set_xlim(-0.5, 1.5)
    a.set_ylim(0, 50)
    a.set_ylabel("Mean largest diameter (mm)")
    b.hist(aucs, bins=np.arange(0.72, 1.0101, 0.0125), facecolor="#C9C9C9", edgecolor=INK, linewidth=0.3)
    top = b.get_ylim()[1]
    b.axvline(0.914, color=BLUE, linewidth=1.2, linestyle="-", label="Hybrid, project test set (0.914)")
    b.axvline(0.972, color=VERM, linewidth=1.2, linestyle="--", label="Diameter, same nodules (0.972)")
    b.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=2.4)
    b.set_ylim(0, top * 1.12)
    b.set_xlim(0.72, 1.01)
    b.set_xlabel("Diameter-only AUC")
    b.set_ylabel("Random test partitions (count)")
    for ax, tag in ((a, "a"), (b, "b")):
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.text(-0.15, 1.03, tag, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.15)
    save_eps(fig, "Fig4")


def fig5():
    d = np.load(HERE / "lidc_audit/real_example.npz")
    lo, hi = -1000.0, 400.0
    view = lambda a: np.clip((a - lo) / (hi - lo), 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 66 * MM), gridspec_kw={"width_ratios": [1.0, 0.72, 0.72], "wspace": 0.12})
    ax = axes[0]
    ax.imshow(view(d["axial"]), cmap="gray", vmin=0, vmax=1, origin="upper", interpolation="nearest")
    cz, bz = d["centre_corrected_zyx"], d["centre_defective_zyx"]
    ax.add_patch(Rectangle((cz[2] - 24, cz[1] - 24), 48, 48, fill=False, edgecolor="white", linewidth=1.6))
    ax.add_patch(Rectangle((cz[2] - 24, cz[1] - 24), 48, 48, fill=False, edgecolor=BLUE, linewidth=1.0))
    ax.add_patch(Rectangle((bz[2] - 24, bz[1] - 24), 48, 48, fill=False, edgecolor="white", linewidth=1.6))
    ax.add_patch(Rectangle((bz[2] - 24, bz[1] - 24), 48, 48, fill=False, edgecolor=VERM, linewidth=1.0, linestyle=(0, (3, 2))))
    ny, nx = d["axial"].shape
    ax.set_xlim(0, max(nx, bz[2] + 30))
    ax.set_ylim(max(ny, bz[1] + 30), 0)
    ax.plot([12, 62], [ny - 14, ny - 14], color="white", linewidth=2)
    ax.text(37, ny - 20, "50 mm", color="white", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.0, 1.02, "a", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    for ax, key, tag, lab in ((axes[1], "voi_corrected", "b", "Corrected"), (axes[2], "voi_defective", "c", "Defective")):
        ax.imshow(view(d[key][24]), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.6)
        ax.text(0.0, 1.02, tag, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
        ax.text(1.0, 1.02, lab, transform=ax.transAxes, fontsize=8, va="bottom", ha="right")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.03)
    png = OUT / "_fig6.png"
    fig.savefig(png, dpi=600)
    fig.savefig(OUT / "Fig6_preview.pdf")
    plt.close(fig)
    Image.open(png).convert("RGB").save(OUT / "Fig6.tif", compression="tiff_lzw", dpi=(600, 600))
    png.unlink()


def fig6():
    rows = load_rows()
    spacings = sorted({r["pixel_spacing_mm"] for r in rows})
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 72 * MM), gridspec_kw={"wspace": 0.3})
    series = (("corrected", "Corrected mapping", BLUE, "o", "-"), ("defective", "Defective mapping", VERM, "s", "--"))
    for key, label, color, marker, ls in series:
        med, lo, hi = [], [], []
        for s in spacings:
            e = sorted(r["error_mm"] for r in rows if r["mapping"] == key and r["pixel_spacing_mm"] == s)
            med.append(statistics.median(e))
            lo.append(e[0])
            hi.append(e[-1])
        x = [s + (-0.004 if key == "corrected" else 0.004) for s in spacings]
        a.errorbar(x, med, yerr=[[m - l for m, l in zip(med, lo)], [h - m for m, h in zip(med, hi)]], color=color,
                   marker=marker, markersize=3.5, linestyle=ls, elinewidth=0.7, capsize=1.6, capthick=0.7, label=label)
    a.axhline(24, color=INK, linewidth=0.6, linestyle=":")
    a.text(0.495, 29, "VOI half-width (24 mm)", fontsize=8, va="bottom")
    a.set_yscale("log")
    a.set_ylim(0.01, 500)
    a.set_yticks([0.01, 0.1, 1, 10, 100])
    a.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    a.minorticks_off()
    a.set_ylabel("3D centroid error (mm)")
    a.set_xlabel("In-plane pixel spacing (mm)")
    a.set_xlim(0.49, 1.01)
    a.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    a.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.585))
    rows2 = list(csv.DictReader((HERE / "lidc_audit/nodules.csv").open()))
    d = [float(r["displacement_mm"]) for r in rows2 if r["malignancy_mean"] not in ("", "None")]
    b.hist(d, bins=np.arange(0, 310, 10), facecolor="#C9C9C9", edgecolor=INK, linewidth=0.3)
    top = b.get_ylim()[1]
    b.axvline(24, color=INK, linewidth=0.8, linestyle=":")
    b.set_ylim(0, top * 1.22)
    b.text(29, top * 1.2, "VOI half-width (24 mm)", fontsize=8, va="top")
    b.set_xlabel("Displacement by the defective mapping (mm)")
    b.set_ylabel("Rated nodules, all LIDC-IDRI (count)")
    b.set_xlim(0, 300)
    for ax, tag in ((a, "a"), (b, "b")):
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.text(-0.14, 1.03, tag, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.16)
    save_eps(fig, "Fig5")


def fig7():
    tol = json.loads((HERE / "tolerance/tolerance_summary.json").read_text())
    series = (("test", "Test", BLUE, "o", "-", -0.3), ("validation", "Validation", VERM, "s", "--", 0.0), ("all", "Combined", INK, "^", ":", 0.3))
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 72 * MM), gridspec_kw={"wspace": 0.28})
    for key, label, color, marker, ls, dx in series:
        r = tol[key]
        x = np.array(r["shift_mm"]) + dx
        y, lo, hi = np.array(r["auc"]), np.array(r["auc_ci_low"]), np.array(r["auc_ci_high"])
        a.errorbar(x, y, yerr=[y - lo, hi - y], color=color, marker=marker, markersize=3.5, linestyle=ls, linewidth=1.0,
                   elinewidth=0.7, capsize=1.6, capthick=0.7, label=f"{label} ({r['nodules']} nodules)")
        if key != "validation":
            d, dlo, dhi = np.array(r["auc_change_from_0mm"]), np.array(r["change_ci_low"]), np.array(r["change_ci_high"])
            b.errorbar(x[1:], d[1:], yerr=[d[1:] - dlo[1:], dhi[1:] - d[1:]], color=color, marker=marker, markersize=3.5, linestyle=ls,
                       linewidth=1.0, elinewidth=0.7, capsize=1.6, capthick=0.7, label=label)
    a.axhline(0.5, color=INK, linewidth=0.6, linestyle=":")
    a.text(6, 0.51, "Chance", fontsize=8, ha="center", va="bottom")
    a.set_ylim(0.3, 1.02)
    a.set_ylabel("AUC")
    a.legend(frameon=False, loc="lower left")
    b.axhline(0, color=INK, linewidth=0.6)
    b.axhline(-0.05, color=INK, linewidth=0.6, linestyle="--")
    b.text(20.6, -0.035, "Drop of 0.05", fontsize=8, ha="right", va="bottom")
    b.set_ylim(-0.6, 0.1)
    b.set_ylabel("Change in AUC from 0 mm")
    b.legend(frameon=False, loc="lower left")
    for ax, tag in ((a, "a"), (b, "b")):
        ax.set_xticks(tol["test"]["shift_mm"])
        ax.set_xlim(-1, 21)
        ax.set_xlabel("VOI center offset from nodule center (mm)")
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.text(-0.14, 1.03, tag, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.16)
    save_eps(fig, "Fig7")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7"]
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6, "7": fig7}[w]()
    print("done", which)
