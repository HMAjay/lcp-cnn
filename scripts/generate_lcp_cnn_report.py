#!/usr/bin/env python3
"""Generate LCP-CNN / PulmoScan major-project report as DOCX."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, Cm, RGBColor

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "artifacts" / "report_assets"
OUT = ROOT / "artifacts" / "report" / "LCP_CNN_Project_Report.docx"
ASSET.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)

# Placeholder student names (edit later — VoteChain PDF was not readable from this environment)
STUDENTS = [
    ("1", "H M Ajay", "1XX20CS001"),
    ("2", "Student Name 2", "1XX20CS002"),
    ("3", "Student Name 3", "1XX20CS003"),
    ("4", "Student Name 4", "1XX20CS004"),
]
GUIDE = "Prof. Guide Name"
COLLEGE = "Your College / Department of Computer Science and Engineering"
TITLE = "Hybrid 3D CNN–Swin Transformer for Pulmonary Nodule Malignancy Risk Prediction on LIDC-IDRI"
SHORT = "PulmoScan / LCP-CNN"


def _set_run_font(run, size=12, bold=False, italic=False, name="Times New Roman", color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def add_para(doc, text, *, size=12, bold=False, italic=False, align="left", space_after=8, first_line=True):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if first_line and align == "justify":
        p.paragraph_format.first_line_indent = Cm(1.0)
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "justify":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    elif align == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    _set_run_font(run, size=size, bold=bold, italic=italic)
    return p


def add_heading_custom(doc, text, level=1):
    # Manual headings for consistent Times New Roman
    sizes = {0: 18, 1: 16, 2: 14, 3: 12}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(8)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    _set_run_font(run, size=sizes.get(level, 12), bold=True)
    return p


def add_page_break(doc):
    doc.add_page_break()


def set_narrow_margins(doc):
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        _set_run_font(run, size=11, bold=True)
    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            cell = table.rows[r_i + 1].cells[c_i]
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            _set_run_font(run, size=11)
    if col_widths:
        for row in table.rows:
            for idx, w in enumerate(col_widths):
                row.cells[idx].width = Inches(w)
    doc.add_paragraph()
    return table


def draw_box(ax, xy, w, h, text, fc="#E8F1FA", ec="#1F4E79"):
    box = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.5, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2, xy[1] + h / 2, text,
        ha="center", va="center", fontsize=9, fontweight="bold", wrap=True,
        color="#123",
    )


def arrow(ax, start, end):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="->", color="#333", lw=1.5),
    )


def fig_system_overview(path: Path):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.set_title("Figure 1. End-to-End System Overview (Stage-2 Malignancy Scoring)", fontsize=11, pad=8)
    boxes = [
        (0.2, 1.1, 1.7, 1.0, "LIDC-IDRI\nDICOM + XML"),
        (2.2, 1.1, 1.8, 1.0, "Preprocess\n1 mm NIfTI\n+ centroids"),
        (4.3, 1.1, 1.7, 1.0, "VOI Crop\n48×48×48"),
        (6.3, 1.1, 1.8, 1.0, "Hybrid\n3D CNN + Swin"),
        (8.4, 1.1, 1.4, 1.0, "P(malignant)\n+ metrics"),
    ]
    for b in boxes:
        draw_box(ax, (b[0], b[1]), b[2], b[3], b[4])
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + boxes[i][2]
        x1 = boxes[i + 1][0]
        y = boxes[i][1] + boxes[i][3] / 2
        arrow(ax, (x0 + 0.02, y), (x1 - 0.02, y))
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_preprocess_flow(path: Path):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Figure 2. Preprocessing Pipeline (CPU)", fontsize=11, pad=8)
    steps = [
        (3.5, 5.0, "TCIA download: CT series + LIDC XML annotations"),
        (3.5, 4.0, "Select annotated chest CT (skip angio / unannotated)"),
        (3.5, 3.0, "DICOM → HU → resample 1×1×1 mm → NIfTI volume"),
        (3.5, 2.0, "Map ROI pixels with original DICOM geometry → patient-mm centroids"),
        (3.5, 1.0, "Cluster reader marks → physical nodules + patient-level splits"),
    ]
    for i, (x, y, t) in enumerate(steps):
        draw_box(ax, (1.2, y - 0.35), 7.6, 0.7, t, fc="#F3F8F1", ec="#2E7D32")
        if i < len(steps) - 1:
            arrow(ax, (5, y - 0.35), (5, steps[i + 1][1] + 0.35))
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_model_arch(path: Path):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    ax.set_title("Figure 3. Hybrid 3D CNN–Swin Architecture", fontsize=11, pad=8)
    items = [
        (0.3, 1.1, 1.6, 1.2, "VOI\n1×48×48×48"),
        (2.2, 1.1, 1.8, 1.2, "3D CNN\nBackbone\n(ResNet-style)"),
        (4.3, 1.1, 2.0, 1.2, "Swin3D\nEncoder\n+ [CLS] token"),
        (6.6, 1.1, 1.5, 1.2, "MLP Head"),
        (8.4, 1.1, 1.3, 1.2, "Sigmoid\nP(malig.)"),
    ]
    for b in items:
        draw_box(ax, (b[0], b[1]), b[2], b[3], b[4], fc="#FFF4E8", ec="#A65D00")
    for i in range(len(items) - 1):
        y = items[i][1] + items[i][3] / 2
        arrow(ax, (items[i][0] + items[i][2], y), (items[i + 1][0], y))
    ax.text(5, 0.35, "Optional XAI: CLS→token attention overlay on feature grid", ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_train_eval_flow(path: Path):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Figure 4. Training & Evaluation Flow (Kubeflow GPU)", fontsize=11, pad=8)
    draw_box(ax, (0.5, 3.5), 2.2, 1.0, "processed_v3\npatient splits", fc="#EAF7EA")
    draw_box(ax, (3.5, 3.5), 2.4, 1.0, "Train loop\nAMP + AdamW\nearly stopping", fc="#EAF7EA")
    draw_box(ax, (6.8, 3.5), 2.4, 1.0, "best.pt\n(best val AUC)", fc="#EAF7EA")
    arrow(ax, (2.7, 4.0), (3.5, 4.0))
    arrow(ax, (5.9, 4.0), (6.8, 4.0))
    draw_box(ax, (3.5, 1.2), 2.4, 1.0, "Evaluate on\nheld-out TEST", fc="#FDECEC", ec="#8B0000")
    draw_box(ax, (6.8, 1.2), 2.4, 1.0, "test_metrics.json\nAUC / sens / spec", fc="#FDECEC", ec="#8B0000")
    arrow(ax, (8.0, 3.5), (8.0, 2.2))
    arrow(ax, (5.9, 1.7), (6.8, 1.7))
    ax.text(1.6, 2.2, "No patient leakage\nacross train/val/test", ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_stage_roadmap(path: Path):
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.set_title("Figure 5. Stage-1 vs Stage-2 Scope", fontsize=11, pad=8)
    draw_box(ax, (0.4, 1.0), 4.2, 1.4, "Stage-1 (Future)\nDetect nodules from raw CT\n(3D U-Net / detector)", fc="#F5F5F5", ec="#666")
    draw_box(ax, (5.2, 1.0), 4.4, 1.4, "Stage-2 (This Project) ✓\nScore malignancy of known VOIs\nHybrid 3D CNN–Swin", fc="#E8F5E9", ec="#2E7D32")
    arrow(ax, (4.6, 1.7), (5.2, 1.7))
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_doc(fig_paths: dict[str, Path]) -> Document:
    doc = Document()
    set_narrow_margins(doc)

    # ---------- COVER ----------
    for _ in range(2):
        doc.add_paragraph()
    add_para(doc, "A PROJECT REPORT", size=14, bold=True, align="center", first_line=False)
    add_para(doc, "ON", size=12, align="center", first_line=False)
    add_para(doc, TITLE, size=16, bold=True, align="center", first_line=False, space_after=14)
    add_para(doc, f"({SHORT})", size=12, italic=True, align="center", first_line=False)
    add_para(doc, "Submitted in partial fulfillment of the requirements for the award of the degree of", size=11, align="center", first_line=False)
    add_para(doc, "BACHELOR OF ENGINEERING", size=13, bold=True, align="center", first_line=False)
    add_para(doc, "IN", size=11, align="center", first_line=False)
    add_para(doc, "COMPUTER SCIENCE AND ENGINEERING", size=13, bold=True, align="center", first_line=False, space_after=16)
    add_para(doc, "Submitted by", size=12, align="center", first_line=False)
    for _, name, usn in STUDENTS:
        add_para(doc, f"{name}  ({usn})", size=12, bold=True, align="center", first_line=False, space_after=2)
    add_para(doc, "", size=12, align="center", first_line=False)
    add_para(doc, "Under the Guidance of", size=12, align="center", first_line=False)
    add_para(doc, GUIDE, size=12, bold=True, align="center", first_line=False, space_after=18)
    add_para(doc, COLLEGE, size=12, bold=True, align="center", first_line=False)
    add_para(doc, "Academic Year 2025–26", size=12, align="center", first_line=False)

    # ---------- CERTIFICATE ----------
    add_page_break(doc)
    add_heading_custom(doc, "CERTIFICATE", level=1)
    names = ", ".join(n for _, n, _ in STUDENTS)
    usns = ", ".join(u for _, _, u in STUDENTS)
    add_para(
        doc,
        f"This is to certify that the project work entitled “{TITLE}” is a bonafide work carried out by "
        f"{names} bearing USN {usns} in partial fulfillment for the award of Bachelor of Engineering in "
        f"Computer Science and Engineering of the Affiliating University during the academic year 2025–26. "
        f"It is certified that all corrections / suggestions indicated for Internal Assessment have been "
        f"incorporated in the report. The project report has been approved as it satisfies the academic "
        f"requirements in respect of project work prescribed for the said degree.",
        align="justify",
    )
    add_para(doc, "", first_line=False)
    add_para(doc, f"Guide: {GUIDE}", first_line=False)
    add_para(doc, "Head of the Department: _______________________", first_line=False)
    add_para(doc, "Principal: _______________________", first_line=False)
    add_para(doc, "External Examiner: _______________________", first_line=False)

    # ---------- DECLARATION ----------
    add_page_break(doc)
    add_heading_custom(doc, "DECLARATION", level=1)
    add_para(
        doc,
        f"We, {names}, students of {COLLEGE}, hereby declare that the project work entitled "
        f"“{TITLE}” submitted to the Affiliating University in partial fulfillment of the requirements "
        f"for the award of Bachelor of Engineering in Computer Science and Engineering is a record of "
        f"original work carried out by us under the guidance of {GUIDE}. We further declare that this "
        f"project work or any part thereof has not been submitted to any other University / Institution "
        f"for the award of any degree or diploma.",
        align="justify",
    )
    add_para(doc, "", first_line=False)
    for _, name, usn in STUDENTS:
        add_para(doc, f"{name} ({usn})  ____________________", first_line=False, space_after=4)

    # ---------- ACKNOWLEDGEMENT ----------
    add_page_break(doc)
    add_heading_custom(doc, "ACKNOWLEDGEMENT", level=1)
    add_para(
        doc,
        f"We express our sincere gratitude to our guide {GUIDE} for continuous support, valuable suggestions, "
        f"and encouragement throughout this project. We thank the Head of the Department and the Principal "
        f"for providing facilities to carry out this work. We also acknowledge The Cancer Imaging Archive (TCIA) "
        f"and the LIDC-IDRI consortium for providing publicly available imaging data for research use. "
        f"Finally, we thank our families and friends for their support during the course of this project.",
        align="justify",
    )

    # ---------- ABSTRACT ----------
    add_page_break(doc)
    add_heading_custom(doc, "ABSTRACT", level=1)
    add_para(
        doc,
        "Lung cancer remains a leading cause of cancer mortality worldwide. Low-dose CT screening increases "
        "early detection but produces many indeterminate pulmonary nodules that require risk stratification. "
        "Hand-crafted clinical models (e.g., Brock) ignore much of the volumetric imaging signal, while purely "
        "2D CNN approaches discard 3D context. This project presents PulmoScan / LCP-CNN: a Stage-2 malignancy "
        "risk predictor that classifies nodule volumes of interest (VOIs) using a hybrid 3D convolutional neural "
        "network and Swin Transformer encoder.",
        align="justify",
    )
    add_para(
        doc,
        "We build a reproducible preprocessing pipeline for LIDC-IDRI: DICOM CT series are converted to Hounsfield "
        "Units, resampled to 1×1×1 mm isotropic NIfTI, and linked to LIDC XML annotations. Reader marks are mapped "
        "to patient-space centroids using original DICOM geometry (critical for correct VOI placement), clustered "
        "into physical nodules, and labeled with the LIDC malignancy rule (≥4 malignant, ≤2 benign, 3 excluded). "
        "Patient-level train/validation/test splits prevent leakage.",
        align="justify",
    )
    add_para(
        doc,
        "On a first serious batch of 153 annotated series (~199 labeled nodules after filtering), a 48³ VOI hybrid "
        "model trained on a Kubeflow A100 GPU achieved validation AUC ≈ 0.949 (early-stopped) and held-out test "
        "AUC ≈ 0.914, with AUPRC ≈ 0.958. The work focuses on Stage-2 classification; Stage-1 detection from raw "
        "unannotated CT is identified as future work. Research use only — not a medical device or clinical diagnosis.",
        align="justify",
    )
    add_para(doc, "Keywords: LIDC-IDRI, pulmonary nodule, malignancy, 3D CNN, Swin Transformer, deep learning, CT", italic=True, first_line=False)

    # ---------- TOC (manual) ----------
    add_page_break(doc)
    add_heading_custom(doc, "TABLE OF CONTENTS", level=1)
    toc = [
        "Certificate",
        "Declaration",
        "Acknowledgement",
        "Abstract",
        "List of Figures",
        "List of Tables",
        "Chapter 1  Introduction",
        "Chapter 2  Literature Survey",
        "Chapter 3  System Requirements & Analysis",
        "Chapter 4  System Design",
        "Chapter 5  Implementation",
        "Chapter 6  Results & Discussion",
        "Chapter 7  Conclusion & Future Work",
        "References",
        "Appendix A  Commands & Reproducibility",
    ]
    for item in toc:
        add_para(doc, item, first_line=False, space_after=4)

    add_heading_custom(doc, "LIST OF FIGURES", level=1)
    for f in [
        "Figure 1. End-to-End System Overview (Stage-2 Malignancy Scoring)",
        "Figure 2. Preprocessing Pipeline (CPU)",
        "Figure 3. Hybrid 3D CNN–Swin Architecture",
        "Figure 4. Training & Evaluation Flow (Kubeflow GPU)",
        "Figure 5. Stage-1 vs Stage-2 Scope",
    ]:
        add_para(doc, f, first_line=False, space_after=4)

    add_heading_custom(doc, "LIST OF TABLES", level=1)
    for t in [
        "Table 1. Hardware & Software Environment",
        "Table 2. Dataset Split Summary (processed_v3)",
        "Table 3. Labeling Rule (LIDC Malignancy)",
        "Table 4. Training Configuration",
        "Table 5. Validation Training Trajectory (selected epochs)",
        "Table 6. Held-out Test Metrics",
    ]:
        add_para(doc, t, first_line=False, space_after=4)

    # ---------- CH 1 ----------
    add_page_break(doc)
    add_heading_custom(doc, "CHAPTER 1 — INTRODUCTION", level=1)
    add_heading_custom(doc, "1.1 Overview", level=2)
    add_para(
        doc,
        "Computed tomography (CT) is central to lung cancer screening and incidental nodule management. "
        "Many detected nodules are benign, yet uncertainty drives repeat imaging, PET, or biopsy—each with "
        "cost and risk. An image-native malignancy probability that uses the full 3D nodule context can support "
        "radiologists as a research decision-support signal (not a replacement for clinical judgment).",
        align="justify",
    )
    add_heading_custom(doc, "1.2 Problem Statement", level=2)
    add_para(
        doc,
        "Given a pulmonary nodule volume of interest extracted from chest CT, predict the probability that the "
        "nodule is malignant according to LIDC radiologist malignancy characteristics, under a strict patient-level "
        "evaluation protocol with no train/test patient leakage.",
        align="justify",
    )
    add_heading_custom(doc, "1.3 Objectives", level=2)
    for o in [
        "Design a hybrid 3D CNN + Swin Transformer classifier for nodule VOIs.",
        "Implement a reproducible LIDC-IDRI preprocessing pipeline (DICOM/XML → processed tree).",
        "Train and evaluate on patient-level splits; report AUC, sensitivity, specificity, and related metrics.",
        "Document limitations (Stage-2 scope, label subjectivity) and a roadmap for Stage-1 detection.",
    ]:
        add_para(doc, f"• {o}", first_line=False, space_after=4)
    add_heading_custom(doc, "1.4 Scope", level=2)
    add_para(
        doc,
        "In scope: Stage-2 malignancy scoring on LIDC-annotated nodules; preprocessing; training on GPU; "
        "metrics and checkpoint export. Out of scope for this phase: automatic nodule detection on arbitrary "
        "raw CTs (Stage-1), multi-center clinical deployment, and regulatory clearance.",
        align="justify",
    )
    add_heading_custom(doc, "1.5 Organization of the Report", level=2)
    add_para(
        doc,
        "Chapter 2 reviews related work. Chapter 3 states requirements. Chapter 4 presents design and diagrams. "
        "Chapter 5 describes implementation. Chapter 6 reports experiments and results. Chapter 7 concludes.",
        align="justify",
    )

    # ---------- CH 2 ----------
    add_page_break(doc)
    add_heading_custom(doc, "CHAPTER 2 — LITERATURE SURVEY", level=1)
    add_heading_custom(doc, "2.1 Clinical Risk Models", level=2)
    add_para(
        doc,
        "Models such as Brock (PanCan) estimate malignancy from clinical and simple imaging covariates. They are "
        "interpretable but underuse voxel-level appearance (texture, spiculation patterns, volumetric morphology).",
        align="justify",
    )
    add_heading_custom(doc, "2.2 Deep Learning on LIDC / LUNA", level=2)
    add_para(
        doc,
        "LIDC-IDRI provides multi-reader nodule annotations including malignancy ratings (1–5). LUNA16 focuses on "
        "detection benchmarks derived from LIDC. Prior work uses 2D/2.5D/3D CNNs for classification and U-Nets for "
        "segmentation. Transformers and hybrid CNN–attention models improve long-range context on volumetric patches.",
        align="justify",
    )
    add_heading_custom(doc, "2.3 Motivation for a Hybrid CNN–Swin Design", level=2)
    add_para(
        doc,
        "3D CNNs capture local morphology efficiently; Swin-style shifted-window attention models broader spatial "
        "relationships within the VOI with manageable compute. A CNN backbone producing a feature volume fed to a "
        "Swin encoder with a [CLS] token yields both a probability and an attention map for explainability.",
        align="justify",
    )

    # ---------- CH 3 ----------
    add_page_break(doc)
    add_heading_custom(doc, "CHAPTER 3 — SYSTEM REQUIREMENTS & ANALYSIS", level=1)
    add_heading_custom(doc, "3.1 Functional Requirements", level=2)
    for o in [
        "Ingest LIDC DICOM CT and XML annotations.",
        "Produce isotropic NIfTI volumes and physical-nodule metadata.",
        "Train hybrid classifier with AMP on GPU.",
        "Evaluate patient-level test metrics (AUC, etc.).",
        "Export best checkpoint for inference/demo.",
    ]:
        add_para(doc, f"• {o}", first_line=False, space_after=4)
    add_heading_custom(doc, "3.2 Non-Functional Requirements", level=2)
    add_para(
        doc,
        "Reproducibility (configs + manifests), memory-safe data loading (lazy mmap, num_workers=0 on constrained hosts), "
        "and clear research-only disclaimers.",
        align="justify",
    )
    add_heading_custom(doc, "3.3 Environment", level=2)
    add_table(
        doc,
        ["Component", "Details"],
        [
            ["Preprocess host", "Windows / WSL2, CPU, pydicom + SimpleITK"],
            ["Training host", "Kubeflow GPU notebook (NVIDIA A100 MIG ~21 GB)"],
            ["Python", "Conda env pulmo, Python 3.10"],
            ["Deep learning", "PyTorch 2.1.2+cu118 (driver-matched)"],
            ["Data source", "TCIA LIDC-IDRI (~303 series first batch → 153 selected)"],
            ["Code", "lcp-cnn repository, branch cursor/lidc-fresh-preprocess-1296"],
        ],
    )
    add_para(doc, "Table 1. Hardware & Software Environment", italic=True, align="center", first_line=False)

    # ---------- CH 4 ----------
    add_page_break(doc)
    add_heading_custom(doc, "CHAPTER 4 — SYSTEM DESIGN", level=1)
    add_heading_custom(doc, "4.1 Overall Architecture", level=2)
    add_para(
        doc,
        "The system separates CPU preprocessing from GPU training. Preprocessing writes a stable processed/ tree. "
        "Training lazily crops VOIs around centroids and optimizes binary malignancy probability.",
        align="justify",
    )
    doc.add_picture(str(fig_paths["overview"]), width=Inches(6.2))
    last = doc.paragraphs[-1]
    last.alignment = WD_ALIGN_PARAGRAPH.CENTER

    add_heading_custom(doc, "4.2 Preprocessing Design", level=2)
    add_para(
        doc,
        "TCIA downloads often include multiple series per patient (e.g., chest vs angio). We keep CT series with "
        "matching LIDC XML, prefer non-angio series, and one series per patient. A critical design choice is mapping "
        "LIDC edgeMap xCoord/yCoord using the original DICOM SimpleITK geometry before 1 mm resampling; treating those "
        "indices as 1 mm voxels misplaces VOIs and collapses AUC (observed in an earlier processed_v2 experiment).",
        align="justify",
    )
    doc.add_picture(str(fig_paths["preprocess"]), width=Inches(5.9))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    add_heading_custom(doc, "4.3 Model Design", level=2)
    add_para(
        doc,
        "Input VOI shape is 1×D×H×W (default 48³ for the primary experiment). A ResNet-style 3D CNN backbone emits a "
        "feature map; a Swin3D encoder with [CLS] pooling feeds an MLP head and sigmoid. Attention from [CLS] to spatial "
        "tokens supports qualitative explanation overlays.",
        align="justify",
    )
    doc.add_picture(str(fig_paths["model"]), width=Inches(6.2))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    add_heading_custom(doc, "4.4 Training & Evaluation Design", level=2)
    doc.add_picture(str(fig_paths["train"]), width=Inches(5.9))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    add_heading_custom(doc, "4.5 Stage Scope", level=2)
    doc.add_picture(str(fig_paths["stages"]), width=Inches(5.9))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    add_heading_custom(doc, "4.6 Data Schema (processed/)", level=2)
    add_para(
        doc,
        "volumes/<SeriesUID>.nii.gz; metadata/lidc/; metadata/physical_nodules/; manifests/dataset_manifest.json; "
        "manifests/patient_splits.json. Training reads manifests and crops using patient-mm centroids transformed "
        "through the NIfTI affine.",
        align="justify",
    )

    # ---------- CH 5 ----------
    add_page_break(doc)
    add_heading_custom(doc, "CHAPTER 5 — IMPLEMENTATION", level=1)
    add_heading_custom(doc, "5.1 Repository Layout", level=2)
    add_para(
        doc,
        "Package pulmoscan under src/: data (io, labels, dataset, preprocess_lidc), classification (backbone, swin3d, "
        "model, losses, train), evaluation, inference, xai. Configs live under configs/model|data|train. Scripts include "
        "run_local.py and report generation helpers.",
        align="justify",
    )
    add_heading_custom(doc, "5.2 Labeling Rule", level=2)
    add_table(
        doc,
        ["Condition (mean LIDC malignancy)", "Label"],
        [
            ["≥ 4.0", "Malignant (1)"],
            ["≤ 2.0", "Benign (0)"],
            ["Otherwise (e.g., ~3)", "Excluded (uncertain)"],
        ],
    )
    add_para(doc, "Table 3. Labeling Rule (LIDC Malignancy)", italic=True, align="center", first_line=False)

    add_heading_custom(doc, "5.3 Training Implementation Details", level=2)
    add_table(
        doc,
        ["Hyperparameter", "Primary run (v3 / 48³)"],
        [
            ["VOI size", "48 × 48 × 48"],
            ["Batch size", "2"],
            ["Optimizer", "AdamW (lr 1e-4, weight decay 1e-4)"],
            ["Loss", "Weighted BCE (pos_weight from train counts)"],
            ["AMP", "Enabled (torch.cuda.amp on torch 2.1)"],
            ["Epochs / early stop", "Max 30 / patience 8 on val AUC"],
            ["Params", "~12.16 M"],
        ],
    )
    add_para(doc, "Table 4. Training Configuration", italic=True, align="center", first_line=False)

    add_heading_custom(doc, "5.4 Engineering Challenges & Fixes", level=2)
    for o in [
        "WSL venv missing ensurepip — install python3-venv.",
        "Kubeflow base Python 3.8 incompatible — conda env pulmo with Python 3.10.",
        "Default pip torch (CUDA 13) incompatible with driver 11.4 — install torch 2.1.2+cu118.",
        "torch.amp.GradScaler API mismatch on 2.1 — use torch.cuda.amp fallbacks.",
        "Wrong VOI centroids in v2 — remap LIDC pixels via original DICOM geometry; rebuild processed_v3.",
        "Jupyter download root is /workspace — copy artifacts there for FileLink/browser download.",
    ]:
        add_para(doc, f"• {o}", first_line=False, space_after=4)

    # ---------- CH 6 ----------
    add_page_break(doc)
    add_heading_custom(doc, "CHAPTER 6 — RESULTS & DISCUSSION", level=1)
    add_heading_custom(doc, "6.1 Dataset Summary", level=2)
    add_table(
        doc,
        ["Split", "Nodules (n)", "Malignant", "Benign"],
        [
            ["Train", "132", "45", "87"],
            ["Validation", "35", "12", "23"],
            ["Test", "29", "20", "9"],
            ["Total labeled", "196", "—", "—"],
        ],
    )
    add_para(doc, "Table 2. Dataset Split Summary (processed_v3 verify/train summary)", italic=True, align="center", first_line=False)
    add_para(
        doc,
        "Series selected for preprocessing: 153 (from ~303 TCIA completed items / multi-series patients). "
        "Uncertain malignancy scores were excluded by design.",
        align="justify",
    )

    add_heading_custom(doc, "6.2 Training Trajectory", level=2)
    add_table(
        doc,
        ["Epoch", "Train loss", "Val AUC", "Val Acc"],
        [
            ["1", "0.997", "0.574", "0.657"],
            ["2", "0.847", "0.743", "0.771"],
            ["4", "0.868", "0.835", "0.800"],
            ["6", "0.787", "0.853", "0.800"],
            ["9", "0.654", "0.920", "0.771"],
            ["12", "0.359", "0.931", "0.886"],
            ["20", "0.521", "0.949", "0.943"],
            ["28 (early stop)", "0.006", "0.933", "0.886"],
        ],
    )
    add_para(doc, "Table 5. Validation Training Trajectory (selected epochs)", italic=True, align="center", first_line=False)
    add_para(
        doc,
        "Best validation AUC ≈ 0.949 at the checkpoint retained as best.pt. Early stopping halted at epoch 28. "
        "Validation n=35 is small, so epoch-to-epoch AUC noise is expected; held-out test is the primary reported figure.",
        align="justify",
    )

    add_heading_custom(doc, "6.3 Held-out Test Results", level=2)
    add_table(
        doc,
        ["Metric", "Value"],
        [
            ["AUC", "0.9139"],
            ["AUPRC", "0.9583"],
            ["Accuracy", "0.7586"],
            ["Sensitivity", "0.7000"],
            ["Specificity", "0.8889"],
            ["Precision", "0.9333"],
            ["Brier score", "0.2296"],
            ["Confusion (TP/TN/FP/FN)", "14 / 8 / 1 / 6"],
        ],
    )
    add_para(doc, "Table 6. Held-out Test Metrics (best.pt on test split)", italic=True, align="center", first_line=False)

    add_heading_custom(doc, "6.4 Discussion", level=2)
    add_para(
        doc,
        "Test AUC ≈ 0.914 meets the project’s stretch target (≥0.85) on this batch. Accuracy is lower than AUC because "
        "the operating threshold optimizing a particular criterion can differ from 0.5, and the test set is malignancy-heavy "
        "(20 vs 9). Compared with an earlier 7-patient exploratory run (~0.76 AUC, unstable) and a failed processed_v2 run "
        "(val AUC collapsing near chance due to misplaced VOIs), processed_v3 + 48³ hybrid training is the strongest result "
        "obtained in this work.",
        align="justify",
    )
    add_para(
        doc,
        "Limitations: LIDC malignancy is a radiologist characteristic, not histopathology; sample size remains modest; "
        "single-split evaluation without external cohort; Stage-1 detection not implemented. These limit clinical claims.",
        align="justify",
    )

    # ---------- CH 7 ----------
    add_page_break(doc)
    add_heading_custom(doc, "CHAPTER 7 — CONCLUSION & FUTURE WORK", level=1)
    add_heading_custom(doc, "7.1 Conclusion", level=2)
    add_para(
        doc,
        "This project delivered an end-to-end Stage-2 pipeline for pulmonary nodule malignancy risk prediction on "
        "LIDC-IDRI using a hybrid 3D CNN–Swin Transformer. Correct geometric linking of XML annotations to CT volumes "
        "proved essential. On processed_v3, the model achieved test AUC ≈ 0.914. The software artifacts (best.pt, metrics, "
        "configs, processed tree) support demonstration and further research. Research use only.",
        align="justify",
    )
    add_heading_custom(doc, "7.2 Future Work", level=2)
    for o in [
        "Scale dataset toward hundreds of patients; nested cross-validation / bootstrap CIs.",
        "Implement Stage-1 3D detection/segmentation for raw-CT demos without XML.",
        "Calibrate probabilities; compare against Brock/Lung-RADS baselines.",
        "Richer XAI and clinician-facing UI; prospective reader studies (beyond this academic scope).",
    ]:
        add_para(doc, f"• {o}", first_line=False, space_after=4)

    # ---------- REFERENCES ----------
    add_page_break(doc)
    add_heading_custom(doc, "REFERENCES", level=1)
    refs = [
        "Armato III, S.G., et al. The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans. Medical Physics, 2011.",
        "The Cancer Imaging Archive (TCIA): LIDC-IDRI collection. https://www.cancerimagingarchive.net/collection/lidc-idri/",
        "Setio, A.A.A., et al. Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: The LUNA16 challenge. Medical Image Analysis, 2017.",
        "Liu, Z., et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV, 2021.",
        "McWilliams, A., et al. Probability of cancer in pulmonary nodules detected on first screening CT. NEJM, 2013 (Brock/PanCan model context).",
        "PyTorch documentation; MONAI / SimpleITK / nibabel documentation for volumetric preprocessing and training.",
    ]
    for i, r in enumerate(refs, 1):
        add_para(doc, f"[{i}] {r}", first_line=False, space_after=6)

    # ---------- APPENDIX ----------
    add_page_break(doc)
    add_heading_custom(doc, "APPENDIX A — COMMANDS & REPRODUCIBILITY", level=1)
    add_para(doc, "Preprocess (WSL / CPU):", bold=True, first_line=False)
    add_para(
        doc,
        "python -m pulmoscan.data.preprocess_lidc --raw-root C:/lidc/raw --xml-root C:/lidc/tcia-lidc-xml "
        "--out C:/lidc/processed_v3 --fresh",
        size=10,
        first_line=False,
    )
    add_para(doc, "Train (Kubeflow GPU, 48³):", bold=True, first_line=False)
    add_para(
        doc,
        "python -m pulmoscan.classification.train --preset default --data-root /home/jovyan/data/processed_v3 "
        "--batch-size 2 --device cuda --output-dir /home/jovyan/artifacts/checkpoints_v3_48",
        size=10,
        first_line=False,
    )
    add_para(doc, "Evaluate test:", bold=True, first_line=False)
    add_para(
        doc,
        "python -m pulmoscan.evaluation.evaluate --data-root /home/jovyan/data/processed_v3 "
        "--checkpoint .../best.pt --split test --device cuda --out .../test_metrics.json",
        size=10,
        first_line=False,
    )
    add_para(
        doc,
        "Disclaimer: This project is for academic research and education only. It is not a certified medical device "
        "and must not be used as the sole basis for clinical diagnosis or treatment decisions.",
        italic=True,
        first_line=False,
        space_after=12,
    )
    add_para(doc, "— End of Report —", align="center", bold=True, first_line=False)
    return doc


def main():
    paths = {
        "overview": ASSET / "fig1_overview.png",
        "preprocess": ASSET / "fig2_preprocess.png",
        "model": ASSET / "fig3_model.png",
        "train": ASSET / "fig4_train.png",
        "stages": ASSET / "fig5_stages.png",
    }
    fig_system_overview(paths["overview"])
    fig_preprocess_flow(paths["preprocess"])
    fig_model_arch(paths["model"])
    fig_train_eval_flow(paths["train"])
    fig_stage_roadmap(paths["stages"])
    doc = build_doc(paths)
    doc.save(OUT)
    # Also copy to /workspace for Jupyter download convenience if present
    alt = Path("/workspace/dl_LCP_CNN_Project_Report.docx")
    try:
        alt.write_bytes(OUT.read_bytes())
    except Exception:
        pass
    print(f"Wrote {OUT}")
    print(f"Also: {alt if alt.exists() else 'no /workspace copy'}")


if __name__ == "__main__":
    main()
