# Product Requirements Document (PRD)
## PulmoScan AI — Hybrid CNN–Transformer Model for Lung Nodule Malignancy Risk Prediction

> **Working product name:** PulmoScan AI **[Assumption]** — the source documents describe the project as the "LCP-CNN Hybrid Model." A product-facing name is adopted here for clarity across the PRD. Rename freely.
>
> **Document type:** Industry-standard Product Requirements Document + Engineering Blueprint + Implementation Plan
> **Version:** 1.0
> **Date:** 2026-07-29
> **Status:** Draft for engineering hand-off
> **Prepared by:** Senior Product Manager / Solutions Architect / AI Engineer
> **Intended audience:** Full engineering team (ML, backend, frontend, DevOps, QA), building the product from scratch over a 6-month horizon.

---

### How to read this document

This PRD is deliberately exhaustive. It is written so that an engineering team — or another AI system — can build the product **without access to the original synopsis or review deck**. Every material fact is either sourced from the two project documents or, where the source is silent, inferred using industry best practice and explicitly tagged **[Assumption]**. Assumptions are engineering defaults; the team may override them, but they must remain internally consistent if changed.

The document is organized into six phases:

| Phase | Purpose | Primary reader |
|-------|---------|----------------|
| Phase 1 | Document Analysis — what the source material actually says | Everyone (onboarding) |
| Phase 2 | Product Understanding — why this product exists | Product, leadership, ML |
| Phase 3 | Full PRD (28 sections) — the buildable specification | All engineering |
| Phase 4 | Engineering Blueprint — repo, standards, ops | Backend, DevOps, ML infra |
| Phase 5 | Implementation Plan — build order and effort | Eng leads, PM |
| Phase 6 | Critical Improvements — going beyond the synopsis | Architects, research leads |

---

# PHASE 1 — DOCUMENT ANALYSIS

This phase reports what the two source documents (the Major Project Synopsis PDF, 8 pages, and the Major Project Review deck, 21 slides) actually contain. It is descriptive, not aspirational. Inferred content is tagged **[Assumption]**.

## 1.1 Project identity

| Attribute | Value | Source |
|-----------|-------|--------|
| Academic title | LCP-CNN Hybrid Model for Lung Nodule Malignancy Risk Prediction (beyond Brock / Lung-RADS) | Synopsis + Deck |
| Degree | B.E., Information Science & Engineering | Deck (student USNs) |
| Institution | RV Institute of Technology and Management, affiliated to VTU Belagavi | Deck **[Assumption on full name expansion]** |
| Course code | BIS685 (Major Project) | Deck slide 1 |
| Domain | AI in Healthcare | Deck slide 1 |
| Group | G17 | Deck slide 1 |
| Team | Dhruva P Patavardhan (1RF23IS030), H M Ajay (1RF23IS036), Joseph J Chalakkal (1RF23IS045), Shobith M (1RF23IS077) | Deck slide 1 |
| Guide | Prof. Pallavi K N | Deck slide 1 |

## 1.2 Core idea

Predict the **probability that a pulmonary nodule detected on a low-dose CT (LDCT) scan is malignant**, using a hybrid deep learning model that combines a **3D Convolutional Neural Network** (for local features — texture, density, margin, shape) with a **Transformer / multi-head self-attention encoder** (for global spatial context across the nodule volume). The model outputs a continuous malignancy risk score in the range 0–1, intended to outperform established clinical risk calculators such as the Brock University model and the Lung-RADS categorical system.

## 1.3 Problem statement (as stated in sources)

Lung cancer is the leading cause of cancer death (~1.8 million deaths/year cited). LDCT screening reduces mortality (the NLST trial showed ~20% lung-cancer mortality reduction), but screening produces large numbers of indeterminate nodules. Existing risk stratification tools have real limitations:

- **Brock model** relies on hand-picked clinical/radiographic variables; reported AUC ~0.74. It cannot exploit the full richness of the imaging data.
- **Lung-RADS** is a categorical rule-based system; coarse and reader-dependent.
- **Conventional CAD / 2D CNNs** underperform on difficult nodule subtypes — part-solid and ground-glass opacity (GGO) nodules — and 2D approaches discard volumetric context.

The gap: no widely deployed tool fuses full 3D volumetric features with global attention-based context to produce a calibrated, explainable malignancy probability that beats Brock/Lung-RADS.

## 1.4 Existing system

| Existing approach | What it does | Limitation (per sources) |
|-------------------|--------------|--------------------------|
| Brock University malignancy score | Logistic-regression risk from clinical + nodule variables | AUC ~0.74; manual variable entry; no raw-image learning |
| Lung-RADS | Categorical screening management tool | Coarse, rule-based, reader variability |
| LCP-CNN (prior art) | CNN-based malignancy prediction on LDCT | Basis/inspiration; primarily convolutional, limited global context |
| Traditional CAD | Rule/feature-engineered detection | Weak on part-solid / GGO nodules |

## 1.5 Proposed solution

A **two-stage pipeline**:

1. **Stage 1 — Detection & segmentation:** A 3D U-Net (or equivalent) locates and segments candidate nodules from the full CT volume and produces cropped nodule sub-volumes (VOIs).
2. **Stage 2 — Hybrid classification:** Each nodule VOI passes through a 3D CNN backbone producing a feature tensor of shape (B, C, D, H, W); the spatial grid is flattened into patch embeddings, augmented with positional encodings and a learnable **[CLS]** token, and processed by a Transformer encoder (multi-head self-attention). The [CLS] representation feeds a fully-connected layer → sigmoid → malignancy probability (0–1).

**Training strategy (per sources):** binary cross-entropy loss; class-weighted sampling to handle malignant/benign imbalance; data augmentation (rotation, flipping, intensity normalization); fine-tuning of a pre-trained CNN backbone; evaluation via AUC-ROC, sensitivity, specificity, confusion matrix; attention-map visualization for explainability.

## 1.6 Objectives (from synopsis)

1. Develop a hybrid 3D CNN + Transformer model for lung nodule malignancy risk prediction from LDCT.
2. Exceed the Brock model baseline (target AUC-ROC > 0.90 vs ~0.74).
3. Provide explainability through attention-map visualizations (XAI).
4. Produce a comparative benchmark and a survey/review contribution (40+ references).

## 1.7 Scope

**In scope (sources):** model design, training and evaluation on public datasets (LIDC-IDRI; NLST where accessible), benchmarking against Brock/Lung-RADS, attention-based explainability, comparative results table.

**Out of scope / not stated — [Assumption]:** clinical deployment, regulatory clearance (FDA/CE), PACS/EHR integration, prospective clinical trials, and a production web application are **not** claimed by the academic synopsis. This PRD, however, is asked to specify a *buildable product*, so Phases 3–6 extend the academic core into a full software product and clearly mark those extensions.

## 1.8 Key features (extracted + inferred)

| # | Feature | Source vs [Assumption] |
|---|---------|------------------------|
| F1 | DICOM/CT volume ingestion & preprocessing | Source (SimpleITK/pydicom) |
| F2 | 3D nodule detection & segmentation | Source (two-stage pipeline) |
| F3 | Hybrid CNN–Transformer malignancy classifier | Source (core) |
| F4 | Malignancy probability score 0–1 | Source |
| F5 | Attention-map / XAI visualization | Source |
| F6 | Benchmark vs Brock / Lung-RADS | Source |
| F7 | Web-based clinician UI for upload & report | [Assumption] |
| F8 | Structured report export (PDF) | [Assumption] |
| F9 | Model performance dashboard | [Assumption] |
| F10 | Audit logging & user management | [Assumption] |

## 1.9 Technologies (as stated) & the CUDA inconsistency

| Layer | Technology | Source |
|-------|-----------|--------|
| Language | Python 3.9+ | Both |
| DL framework | PyTorch 2.x | Both |
| CUDA | **11.8 (deck) vs 13.0 (PDF) — INCONSISTENCY, flagged** | Deck / PDF |
| Medical imaging | MONAI, SimpleITK, pydicom | Both |
| ML utils | scikit-learn | Both |
| Visualization | Matplotlib, Seaborn | Both |
| Compute | Google Colab Pro+ / Kaggle | Both |
| Hardware (min) | NVIDIA RTX 3080 or Tesla T4, ≥8 GB VRAM, 32 GB RAM, 500 GB SSD | Synopsis §7.1 |

> **⚠ Flag — resolve before build:** the PDF cites CUDA 13.0 while the deck cites CUDA 11.8. CUDA 13.0 did not exist at the time these are typical PyTorch 2.x builds; **recommendation: standardize on CUDA 12.1 with PyTorch 2.2+** for current GPU driver compatibility. **[Assumption]**

## 1.10 AI models, algorithms & datasets

- **Models:** 3D CNN backbone (e.g., 3D ResNet / DenseNet variant) **[Assumption on exact backbone]**; Transformer encoder (ViT-style, multi-head self-attention); 3D U-Net for detection/segmentation **[Assumption on U-Net specifically]**.
- **Loss:** binary cross-entropy (source); **[Assumption]** add focal loss option for imbalance.
- **Datasets:** LIDC IDRI dataset NLST (~6 TB, access-controlled).
- **Baselines:** Brock (AUC ~0.74), Lung-RADS.
- **Metrics:** AUC-ROC (primary), sensitivity, specificity, confusion matrix; **[Assumption]** add calibration (Brier score, reliability curve) and FROC for detection.

## 1.11 Architecture (from textual descriptions)

The synopsis §6 and deck slide 12–13 describe the two-stage pipeline of §1.5 above. The architecture figures (PDF p.5 "Proposed Architecture for 3D Splice"; deck slide 13 diagram) are **images and not text-extractable**; this PRD reconstructs the architecture from the surrounding textual descriptions and re-specifies it formally in Section 10.

## 1.12 Constraints

- Compute/VRAM limited (≥8 GB stated) — constrains 3D batch size and volume resolution.
- NLST access is credentialed and large (~6 TB) — storage and access-approval constraint.
- Academic timeline and team of four.
- **[Assumption]** No clinical/regulatory infrastructure; research-grade product only unless extended.

## 1.13 Requirements captured in sources

**Functional (synopsis §7 / deck slide 14):** ingest CT, preprocess, detect/segment nodules, classify malignancy, output probability, visualize attention, benchmark.
**Non-functional (inferred, §7.1 hardware implies):** must train within stated hardware; reproducibility; explainability. Formalized in Sections 8–9.

## 1.14 Referenced literature

Synopsis lists references [1]–[15]; deck lists references up to [25] and literature-survey tables covering ~20 papers (authors, advantages, disadvantages). Combined, the project targets a 40+ reference survey contribution. Consolidated in the Appendix (Section 28). **[Assumption]** full bibliographic details beyond what the tables show are not all extractable and are marked as such.

## 1.15 Future scope (from sources)

Survey/review paper with 40+ references; improved benchmark performance; explainable AI outputs. Extended substantially in Phase 6.

## 1.16 Missing-information register

| Gap | Handled by |
|-----|-----------|
| Exact CNN backbone / U-Net variant | [Assumption] in §10, Section 14 |
| Web/product layer, APIs, DB | Fully specified as [Assumption] in Sections 10–13 |
| Deployment, security, testing specifics | Phases 3–4 |
| CUDA version conflict | Flagged §1.9, resolved [Assumption] |
| Exact dataset split sizes | [Assumption] in Section 12 / Appendix |

---

# PHASE 2 — PRODUCT UNDERSTANDING (Team Onboarding)

This phase explains *why* PulmoScan AI exists, framed so a new engineer can grasp the mission in ten minutes.

## 2.1 Vision

A world where every indeterminate lung nodule found on a CT scan gets an accurate, explainable, image-native malignancy risk estimate at the moment of reading — so patients with early cancer are caught sooner and patients with benign nodules are spared unnecessary biopsies and anxiety.

## 2.2 Mission

Build a hybrid 3D CNN–Transformer model, and the software around it, that predicts lung nodule malignancy from LDCT with AUC-ROC > 0.90, beats the Brock model and Lung-RADS, explains its reasoning through attention maps, and is reproducible and extensible.

## 2.3 Purpose

Turn the rich volumetric information already present in a routine LDCT — which human-variable-based tools like Brock discard — into a calibrated probability and a visual explanation, closing the gap between what the imaging *contains* and what current risk tools *use*.

## 2.4 The business problem

Radiology departments and screening programs face rising CT volumes and a shortage of thoracic radiologists. Indeterminate nodules drive expensive downstream follow-up (repeat CT, PET, biopsy). A more accurate risk score reduces false-positive workups (cost, radiation, patient harm) and shortens time-to-diagnosis for true cancers, improving both economics and outcomes.

## 2.5 The user problem

- **Radiologist:** must stratify many nodules quickly; categorical tools are coarse and reader-dependent; wants a trustworthy, explainable second opinion.
- **Pulmonologist / MDT:** needs a defensible risk number to decide surveillance vs intervention.
- **Screening-program coordinator:** needs consistent, reproducible stratification across readers and sites.
- **ML researcher (this team):** needs a reproducible pipeline and honest benchmarks.

## 2.6 Current solutions & why they fall short

| Solution | Why it falls short |
|----------|--------------------|
| Brock model | Hand-picked variables; ignores full imaging; AUC ~0.74; manual entry |
| Lung-RADS | Categorical, coarse, reader variability |
| 2D CNN / conventional CAD | Discards 3D context; weak on part-solid & GGO nodules |
| Pure 3D CNN (e.g., LCP-CNN) | Strong local features but limited global spatial context |

## 2.7 Competitive gap

No accessible tool combines **full 3D volumetric feature extraction** with **global self-attention context** to yield a **calibrated, explainable** malignancy probability that demonstrably beats Brock/Lung-RADS on public benchmarks. That fusion is the whitespace.

## 2.8 Innovation

The core innovation is the **CNN→Transformer fusion on 3D nodule volumes**: the CNN captures fine local morphology (spiculation, density, margin), while the Transformer models long-range dependencies across the volume that convolutions with limited receptive fields miss — with attention weights doubling as the explainability signal.

## 2.9 Unique selling proposition (USP)

"Image-native, explainable lung-nodule malignancy scoring that beats Brock/Lung-RADS — combining 3D CNN detail with Transformer global context, and showing you *where* it looked."

## 2.10 Success criteria

| Criterion | Target |
|-----------|--------|
| Primary metric | AUC-ROC > 0.90 on LUNA16-derived test set |
| Beat baseline | Statistically better than Brock (~0.74) |
| Sensitivity @ fixed specificity | ≥ 0.90 sensitivity @ ≥ 0.80 specificity **[Assumption]** |
| Explainability | Attention maps localize to the nodule in ≥ 90% of cases (qualitative) **[Assumption]** |
| Reproducibility | Full training reproducible from repo + config + seed |
| Deliverable | Model, benchmark table, survey (40+ refs) |

## 2.11 Value proposition

For radiologists and screening programs who need faster, more consistent nodule risk stratification, PulmoScan AI is an explainable AI scoring tool that, unlike Brock/Lung-RADS, learns directly from the 3D image and shows its reasoning — reducing unnecessary workups and speeding true-cancer diagnosis.

## 2.12 Expected outcomes

A trained hybrid model exceeding the Brock baseline; a reproducible two-stage pipeline; attention-based explanations; a comparative benchmark table (proposed vs Brock vs Lung-RADS vs 2D/3D CNN baselines); and a survey/review contribution with 40+ references. Product extensions (web UI, reporting, deployment) are specified for teams that carry the research core toward a usable tool.

---

# PHASE 3 — FULL PRODUCT REQUIREMENTS DOCUMENT (28 Sections)

## Section 1 — Executive Summary

PulmoScan AI is a research-grade, extensible software product that predicts the malignancy risk of pulmonary nodules from low-dose CT scans using a hybrid 3D CNN–Transformer deep learning model. The system ingests CT volumes, detects and segments candidate nodules, and outputs a calibrated malignancy probability (0–1) accompanied by attention-based visual explanations. The headline goal is to exceed the Brock University malignancy model (AUC ~0.74) and the Lung-RADS categorical system, targeting AUC-ROC > 0.90 on LUNA16-derived data.

The academic core (model, training, benchmark, survey) is specified in full and then extended into a buildable product: a preprocessing pipeline, an inference API, a clinician-facing web UI, structured PDF reporting, and a model-performance dashboard. The build is scoped to six months across six sprints, executed by a small ML-focused team with backend, frontend, and DevOps support. This document is the single source of truth: it defines requirements, architecture, data, APIs, AI components, security posture, testing, timeline, risks, KPIs, and acceptance criteria in enough detail to build from scratch. All items not grounded in the source documents are tagged **[Assumption]** and represent overridable engineering defaults.

## Section 2 — Problem Statement

Lung cancer is the leading cause of cancer mortality (~1.8M deaths/year). LDCT screening reduces mortality (~20% in NLST) but generates large volumes of indeterminate nodules requiring risk stratification. The dominant tools are inadequate: the Brock model uses hand-selected variables and achieves only ~0.74 AUC while ignoring the full imaging signal; Lung-RADS is coarse, categorical, and reader-dependent; and conventional 2D CNN / CAD systems discard volumetric context and struggle with part-solid and ground-glass nodules. The result is over-investigation of benign nodules (cost, radiation, biopsy harm, patient anxiety) and delayed diagnosis of some malignancies. There is no widely accessible tool that (a) learns directly from the full 3D nodule volume, (b) models global spatial context via attention, (c) produces a calibrated probability that beats Brock/Lung-RADS, and (d) explains its output. PulmoScan AI targets exactly this gap.

## Section 3 — Product Vision & Roadmap

**Vision statement:** the image-native, explainable standard for lung-nodule malignancy risk.

**Roadmap (horizon view):**

| Horizon | Timeframe | Theme | Key deliverables |
|---------|-----------|-------|------------------|
| H1 — Research core | Months 1–4 | Prove the model | Data pipeline, hybrid model, AUC > 0.90, benchmark table |
| H2 — Usable product | Months 4–6 | Wrap the model | Inference API, web UI, PDF report, XAI overlays, dashboard |
| H3 — Extensions | Post-project **[Assumption]** | Toward clinic | PACS/DICOM-node integration, multi-site validation, calibration hardening, regulatory groundwork |
| H4 — Scale | Future **[Assumption]** | Platform | Multi-model registry, active learning, federated training |

## Section 4 — Stakeholders

| Stakeholder | Interest | Involvement |
|-------------|----------|-------------|
| Project team (4 students) | Build, train, evaluate, publish | Owners/engineers |
| Project guide (Prof. Pallavi K N) | Academic quality, milestones | Approver/reviewer |
| Examiners / academic panel | Evaluation against rubric | Reviewers |
| Radiologists (target users) | Accurate, explainable risk | Domain validators **[Assumption]** |
| Pulmonologists / MDT | Decision support | End users **[Assumption]** |
| Dataset providers (LUNA16/LIDC, NLST) | Correct, licensed use | Data source |
| ML/research community | Reproducibility, survey | Audience |
| (Future) regulatory/clinical ops | Safety, compliance | Out of current scope **[Assumption]** |

## Section 5 — User Personas

**Persona A — Dr. Anita Rao, Thoracic Radiologist (primary).** Reads 40–60 CTs/day; time-pressured; wants a trustworthy, explainable second opinion; distrusts black boxes; needs the attention overlay to confirm the model looked at the right nodule. Success = faster, defensible stratification.

**Persona B — Dr. Marcus Feld, Pulmonologist / MDT lead (secondary).** Uses the risk number to choose surveillance vs biopsy vs resection; wants calibration and a printable report for the tumor board.

**Persona C — Priya, Screening Program Coordinator (secondary).** Wants consistent, reproducible stratification across readers/sites; cares about throughput and audit trails.

**Persona D — Sam, ML Researcher / this team (internal).** Needs reproducible training, config-driven experiments, honest benchmarks, and clean data pipelines.

**Persona E — Jordan, DevOps/MLOps (internal).** Needs deployable, monitored, logged services with reproducible environments.

*(Personas B–E details are **[Assumption]** built on standard clinical/eng roles; Persona A/D map directly to the source's stated users.)*

## Section 6 — User Journey

**Clinical inference journey (target product):**
1. Radiologist uploads (or PACS pushes) a CT series into PulmoScan AI.
2. System validates the DICOM series and preprocesses (resample, normalize, lung segmentation).
3. Stage-1 detects/segments candidate nodules; each is cropped to a VOI.
4. Stage-2 hybrid model scores each nodule's malignancy probability.
5. UI displays nodules, scores, risk tiers, and attention overlays on the slices.
6. Radiologist reviews, optionally annotates, and exports a structured PDF report.
7. Result and audit record are persisted; dashboard metrics update.

**Researcher training journey (internal):**
1. Configure experiment (YAML) → 2. Pull/prepare dataset → 3. Preprocess & cache → 4. Train (logged to experiment tracker) → 5. Evaluate (AUC-ROC, sensitivity/specificity, calibration, FROC) → 6. Compare vs Brock/Lung-RADS → 7. Export model artifact + report.

## Section 7 — User Stories (100+)

Stories use: *As a [role], I want [capability], so that [value].* Grouped by epic. IDs are stable references for Sections 8, 22–23, 27.

### Epic A — Data ingestion & preprocessing
- US-A01: As a researcher, I want to download/register LUNA16 data, so that I can train reproducibly.
- US-A02: As a researcher, I want to load DICOM/MHD volumes via SimpleITK/pydicom, so that raw CTs become tensors.
- US-A03: As a researcher, I want to resample volumes to isotropic spacing, so that models see consistent geometry.
- US-A04: As a researcher, I want HU windowing/intensity normalization, so that inputs are standardized.
- US-A05: As a researcher, I want lung-field segmentation/masking, so that irrelevant anatomy is excluded.
- US-A06: As a researcher, I want cached preprocessed tensors, so that training is fast and repeatable.
- US-A07: As a researcher, I want data-integrity checks (shape, spacing, NaNs), so that bad scans are caught early.
- US-A08: As a researcher, I want a train/val/test split with no patient leakage, so that evaluation is honest.
- US-A09: As a researcher, I want configurable augmentation (rotate, flip, intensity jitter), so that the model generalizes.
- US-A10: As a clinician, I want to upload a CT series in the UI, so that I can get a score without coding.
- US-A11: As a clinician, I want the system to reject invalid/incomplete series with a clear message, so that I trust the input handling.
- US-A12: As a coordinator, I want batch ingestion of multiple studies, so that screening volumes are processed efficiently.

### Epic B — Nodule detection & segmentation (Stage 1)
- US-B01: As a researcher, I want a 3D detection model to propose candidate nodules, so that Stage 2 has VOIs.
- US-B02: As a researcher, I want segmentation masks per nodule, so that features focus on the lesion.
- US-B03: As a researcher, I want configurable VOI crop size around each nodule, so that context is controlled.
- US-B04: As a researcher, I want candidate false-positive reduction, so that Stage 2 isn't flooded.
- US-B05: As a researcher, I want FROC evaluation of detection, so that I quantify detection quality.
- US-B06: As a clinician, I want detected nodules listed with locations, so that I can review each.
- US-B07: As a clinician, I want to manually add a missed nodule ROI, so that I can score lesions the detector missed.
- US-B08: As a researcher, I want to persist detection outputs, so that Stage 2 is decoupled and re-runnable.

### Epic C — Malignancy classification (Stage 2, hybrid model)
- US-C01: As a researcher, I want a 3D CNN backbone producing (B,C,D,H,W) features, so that local morphology is captured.
- US-C02: As a researcher, I want patch embedding + positional encoding + [CLS] token, so that features feed a Transformer.
- US-C03: As a researcher, I want a multi-head self-attention encoder, so that global context is modeled.
- US-C04: As a researcher, I want a sigmoid classification head, so that output is a 0–1 probability.
- US-C05: As a researcher, I want BCE (and optional focal) loss with class weighting, so that imbalance is handled.
- US-C06: As a researcher, I want to fine-tune a pre-trained CNN backbone, so that training converges faster.
- US-C07: As a researcher, I want mixed-precision training, so that 3D models fit in ≥8 GB VRAM.
- US-C08: As a researcher, I want gradient checkpointing/accumulation, so that effective batch size is adequate.
- US-C09: As a researcher, I want configurable Transformer depth/heads/dim, so that I can tune capacity.
- US-C10: As a researcher, I want model checkpointing and resume, so that long runs survive interruptions.
- US-C11: As a researcher, I want deterministic seeding, so that results are reproducible.
- US-C12: As a clinician, I want a per-nodule probability and risk tier, so that I can act on it.

### Epic D — Explainability (XAI)
- US-D01: As a researcher, I want attention maps extracted per nodule, so that I can visualize model focus.
- US-D02: As a researcher, I want Grad-CAM (3D) as a secondary XAI method, so that I can cross-check attention. **[Assumption]**
- US-D03: As a clinician, I want attention overlays on CT slices, so that I can verify the model looked at the lesion.
- US-D04: As a clinician, I want a plain-language explanation of the risk tier, so that I can communicate to patients/MDT.
- US-D05: As a researcher, I want to quantify attention-to-nodule overlap, so that I can report XAI quality.

### Epic E — Evaluation & benchmarking
- US-E01: As a researcher, I want AUC-ROC computed on the test set, so that I measure the primary metric.
- US-E02: As a researcher, I want sensitivity/specificity at chosen thresholds, so that I report clinical operating points.
- US-E03: As a researcher, I want a confusion matrix, so that error types are visible.
- US-E04: As a researcher, I want calibration curves + Brier score, so that probabilities are trustworthy. **[Assumption]**
- US-E05: As a researcher, I want a Brock-model reimplementation baseline, so that I can compare fairly.
- US-E06: As a researcher, I want a Lung-RADS mapping baseline, so that I compare against categorical practice.
- US-E07: As a researcher, I want 2D-CNN and pure-3D-CNN ablation baselines, so that I isolate the Transformer's value.
- US-E08: As a researcher, I want bootstrap confidence intervals on AUC, so that comparisons are statistically grounded.
- US-E09: As a researcher, I want an auto-generated benchmark table, so that the comparative result is reproducible.
- US-E10: As a researcher, I want per-subtype metrics (solid / part-solid / GGO), so that I show gains on hard cases.

### Epic F — Clinician web application
- US-F01: As a clinician, I want to log in securely, so that access is controlled.
- US-F02: As a clinician, I want a study list/worklist, so that I can pick a case.
- US-F03: As a clinician, I want an image viewer with scroll/zoom/window controls, so that I can inspect slices.
- US-F04: As a clinician, I want nodule markers on the viewer, so that I can jump to each lesion.
- US-F05: As a clinician, I want to trigger inference on a study, so that I get scores on demand.
- US-F06: As a clinician, I want a results panel (score, tier, confidence), so that I can review quickly.
- US-F07: As a clinician, I want to export a PDF report, so that I can share with the MDT/record.
- US-F08: As a clinician, I want to add notes/agree/disagree, so that human oversight is recorded.
- US-F09: As a coordinator, I want a dashboard of throughput and score distributions, so that I monitor the program.
- US-F10: As an admin, I want user & role management, so that access is appropriate.

### Epic G — Inference API & integration
- US-G01: As a developer, I want a REST endpoint to submit a study for inference, so that other systems integrate.
- US-G02: As a developer, I want async job status polling, so that long inferences don't block.
- US-G03: As a developer, I want a results-retrieval endpoint, so that clients fetch scores + overlays.
- US-G04: As a developer, I want an OpenAPI/Swagger spec, so that integration is self-documenting.
- US-G05: As a developer, I want API auth (token/JWT), so that access is secured.
- US-G06: As a developer, I want rate limiting, so that the service is protected.
- US-G07: As a developer, I want a health/readiness endpoint, so that orchestration can monitor it.
- US-G08: As a developer, I want a model-version field in responses, so that results are traceable.

### Epic H — MLOps, monitoring & operations
- US-H01: As an MLOps engineer, I want experiment tracking (params/metrics/artifacts), so that runs are comparable.
- US-H02: As an MLOps engineer, I want a model registry with versions, so that deployments are traceable.
- US-H03: As an MLOps engineer, I want containerized services, so that environments are reproducible.
- US-H04: As an MLOps engineer, I want structured logs, so that issues are debuggable.
- US-H05: As an MLOps engineer, I want metrics/alerting (latency, errors, GPU), so that I catch problems.
- US-H06: As an MLOps engineer, I want data/prediction drift monitoring, so that model decay is detected. **[Assumption]**
- US-H07: As an MLOps engineer, I want automated CI (lint/test/build), so that quality is enforced.
- US-H08: As an MLOps engineer, I want reproducible training via pinned deps + seeds, so that results replicate.

### Epic I — Security, privacy & compliance
- US-I01: As a security lead, I want PHI/PII de-identification of DICOM, so that privacy is protected.
- US-I02: As a security lead, I want encryption in transit (TLS), so that data is safe on the wire.
- US-I03: As a security lead, I want encryption at rest, so that stored studies are protected.
- US-I04: As a security lead, I want audit logs of access & inference, so that actions are traceable.
- US-I05: As a security lead, I want RBAC, so that least privilege is enforced.
- US-I06: As a security lead, I want secrets stored outside code, so that credentials aren't leaked.
- US-I07: As a compliance owner, I want data-retention/deletion controls, so that policy is met. **[Assumption]**
- US-I08: As a clinician, I want a clear "research use, not a diagnosis" disclaimer, so that the tool is used safely.

### Epic J — Reporting, admin & quality-of-life
- US-J01: As a clinician, I want branded, structured PDF reports, so that outputs are professional.
- US-J02: As a researcher, I want exportable CSV of results, so that I can analyze externally.
- US-J03: As an admin, I want configuration via env/UI, so that ops don't require code changes.
- US-J04: As a user, I want responsive UI + error toasts, so that the app is pleasant and clear.
- US-J05: As a user, I want i18n-ready UI strings, so that localization is possible later. **[Assumption]**
- US-J06: As a researcher, I want a reproducible "one-command" demo, so that reviewers can run it.
- US-J07: As a team, I want documentation (README, model card, API docs), so that others can build on it.
- US-J08: As an examiner, I want a model card describing data, metrics, and limits, so that I can assess rigor. **[Assumption]**

*(Total: 100+ stories across 10 epics. Stories tagged **[Assumption]** extend beyond the academic synopsis into full-product territory.)*

## Section 8 — Functional Requirements

Each FR maps to user stories and has a priority (MoSCoW). "M"=Must, "S"=Should, "C"=Could.

| FR | Requirement | Priority | Stories |
|----|-------------|----------|---------|
| FR-01 | Ingest CT volumes from DICOM series and MHD/RAW (LUNA16) formats | M | A02, A10 |
| FR-02 | Validate series integrity and reject malformed input with clear errors | M | A07, A11 |
| FR-03 | Preprocess: resample to isotropic spacing, HU windowing, intensity normalization | M | A03, A04 |
| FR-04 | Segment lung fields and mask non-lung anatomy | S | A05 |
| FR-05 | Cache preprocessed tensors keyed by config hash | S | A06 |
| FR-06 | Patient-level train/val/test split with no leakage | M | A08 |
| FR-07 | Configurable augmentation pipeline | S | A09 |
| FR-08 | Detect & segment candidate nodules (Stage 1) and emit VOIs | M | B01–B04 |
| FR-09 | Reduce detection false positives | S | B04 |
| FR-10 | Classify each nodule via hybrid 3D CNN–Transformer → probability 0–1 | M | C01–C04 |
| FR-11 | Support class-weighted BCE (and optional focal) loss | M | C05 |
| FR-12 | Fine-tune pre-trained CNN backbone | S | C06 |
| FR-13 | Mixed precision + gradient checkpointing/accumulation | M | C07, C08 |
| FR-14 | Checkpoint, resume, deterministic seeding | M | C10, C11 |
| FR-15 | Extract & render attention maps per nodule | M | D01, D03 |
| FR-16 | Provide secondary XAI (3D Grad-CAM) | C | D02 |
| FR-17 | Compute AUC-ROC, sensitivity, specificity, confusion matrix | M | E01–E03 |
| FR-18 | Compute calibration (Brier, reliability curve) | S | E04 |
| FR-19 | Provide Brock and Lung-RADS baselines + CNN ablations | M | E05–E07 |
| FR-20 | Bootstrap CIs and auto-generate benchmark table | S | E08, E09 |
| FR-21 | Per-subtype metrics (solid/part-solid/GGO) | S | E10 |
| FR-22 | Web UI: auth, worklist, viewer, nodule markers, results panel | S | F01–F06 |
| FR-23 | Trigger inference from UI and via REST API (async) | M | F05, G01–G03 |
| FR-24 | Export structured PDF report and CSV | S | F07, J01, J02 |
| FR-25 | Capture clinician agree/disagree + notes | S | F08 |
| FR-26 | Performance dashboard (throughput, score distribution) | C | F09 |
| FR-27 | RBAC + user/admin management | S | F10, I05 |
| FR-28 | OpenAPI spec, API auth, rate limiting, health checks, model-version in response | M | G04–G08 |
| FR-29 | Experiment tracking + model registry | M | H01, H02 |
| FR-30 | Containerized services + CI pipeline | S | H03, H07 |
| FR-31 | Structured logging, metrics, alerting | S | H04, H05 |
| FR-32 | Drift monitoring | C | H06 |
| FR-33 | DICOM de-identification | M | I01 |
| FR-34 | Encryption in transit + at rest | M | I02, I03 |
| FR-35 | Audit logging of access & inference | M | I04 |
| FR-36 | Secrets managed outside code | M | I06 |
| FR-37 | Retention/deletion controls | C | I07 |
| FR-38 | "Research use, not a diagnosis" disclaimer everywhere results appear | M | I08 |
| FR-39 | Model card + full documentation | S | J07, J08 |
| FR-40 | One-command reproducible demo | S | J06 |

## Section 9 — Non-Functional Requirements

| NFR | Category | Requirement | Target |
|-----|----------|-------------|--------|
| NFR-01 | Accuracy | Primary model quality | AUC-ROC > 0.90 on LUNA16 test |
| NFR-02 | Accuracy | Clinical operating point | ≥0.90 sensitivity @ ≥0.80 specificity **[Assumption]** |
| NFR-03 | Calibration | Probability reliability | Brier ≤ 0.15 **[Assumption]** |
| NFR-04 | Performance | Single-study inference (GPU) | ≤ 60 s end-to-end **[Assumption]** |
| NFR-05 | Performance | Training within stated HW | Fits ≥8 GB VRAM via AMP/checkpointing |
| NFR-06 | Scalability | Concurrent inference jobs | ≥ 10 queued, horizontally scalable workers **[Assumption]** |
| NFR-07 | Availability | Service uptime (product mode) | 99% **[Assumption]** |
| NFR-08 | Reproducibility | Re-run yields same metrics | ±0.005 AUC with fixed seed |
| NFR-09 | Security | Encryption | TLS 1.2+ in transit, AES-256 at rest **[Assumption]** |
| NFR-10 | Privacy | De-identification | No PHI persisted post-ingest **[Assumption]** |
| NFR-11 | Usability | Time-to-result in UI | ≤ 3 clicks from worklist to score **[Assumption]** |
| NFR-12 | Explainability | Attention overlay available | For 100% of scored nodules |
| NFR-13 | Maintainability | Test coverage | ≥ 80% on core pipeline **[Assumption]** |
| NFR-14 | Portability | Runs on Colab/Kaggle + local GPU + container | Verified on all three |
| NFR-15 | Observability | Logs/metrics/traces | Structured, centralized |
| NFR-16 | Compliance | Clear research-use disclaimer, no clinical claims | Present throughout |
| NFR-17 | Cost | Training within academic budget | Colab Pro+/Kaggle free/low tiers |
| NFR-18 | Accessibility | UI meets WCAG 2.1 AA basics | Keyboard nav, contrast **[Assumption]** |

## Section 10 — System Architecture

**Logical view (two-stage ML pipeline inside a service architecture):**

```
                        ┌─────────────────────────────────────────────┐
   Clinician / PACS ───▶│  Web UI (React)  │  REST API (FastAPI)       │
                        └───────┬──────────────────────┬──────────────┘
                                │                       │
                                ▼                       ▼
                      ┌───────────────────┐   ┌────────────────────────┐
                      │ Ingestion &       │   │ Job Queue (Celery/RQ)  │
                      │ De-identification │   │ + Redis broker         │
                      └─────────┬─────────┘   └───────────┬────────────┘
                                ▼                         ▼
                      ┌───────────────────┐   ┌────────────────────────┐
                      │ Preprocessing     │   │ Inference Worker (GPU)  │
                      │ (resample, HU,    │   │  Stage 1: 3D detect/seg │
                      │  normalize, mask) │──▶│  Stage 2: CNN+Transformer│
                      └───────────────────┘   │  + XAI (attention/CAM)  │
                                              └───────────┬────────────┘
                                                          ▼
                      ┌──────────────────────────────────────────────┐
                      │ Storage: Object store (volumes, overlays),    │
                      │ Postgres (metadata, results, audit),          │
                      │ Model registry + Experiment tracker           │
                      └──────────────────────────────────────────────┘
```

**ML pipeline detail (Stage 2 — the core innovation):**

```
Nodule VOI (1×D×H×W)
      │
      ▼
3D CNN backbone (e.g., 3D ResNet-ish)  ──► feature tensor (B, C, D', H', W')
      │
      ▼  flatten spatial grid → N patches of dim C
Patch embeddings + positional encoding + [CLS] token
      │
      ▼
Transformer encoder ×L  (multi-head self-attention + MLP + residual/LayerNorm)
      │
      ▼  take [CLS] representation
Fully-connected head → sigmoid → P(malignant) ∈ [0,1]
      │
      └──► attention weights → XAI overlay
```

**Architectural style:** modular monorepo with clear service boundaries (UI, API, worker, ML core, storage). Stage 1 and Stage 2 are decoupled so each can be developed, evaluated, and swapped independently. The ML core is a standalone Python package importable both by the training scripts and the inference worker — this guarantees train/serve parity.

> The synopsis figure "Proposed Architecture for 3D Splice" (PDF p.5) and deck slide 13 diagram are images; the above is reconstructed from their textual descriptions and is the authoritative spec for the build. **[Assumption on service-layer topology]**

## Section 11 — Technical Stack

| Layer | Choice | Notes / source |
|-------|--------|----------------|
| Language | Python 3.9+ (target 3.11 **[Assumption]**) | Source |
| DL framework | PyTorch 2.2+ | Source (2.x) |
| CUDA | **12.1** (resolves 11.8/13.0 conflict) **[Assumption]** | Flagged in §1.9 |
| Medical imaging | MONAI, SimpleITK, pydicom | Source |
| Classic ML | scikit-learn | Source |
| Viz | Matplotlib, Seaborn | Source |
| Experiment tracking | MLflow (or Weights & Biases) **[Assumption]** | — |
| API | FastAPI + Uvicorn **[Assumption]** | — |
| Async jobs | Celery or RQ + Redis **[Assumption]** | — |
| Frontend | React + TypeScript + Vite; Cornerstone.js for DICOM viewing **[Assumption]** | — |
| PDF reporting | ReportLab / WeasyPrint **[Assumption]** | — |
| Database | PostgreSQL **[Assumption]** | — |
| Object storage | S3-compatible (MinIO locally) **[Assumption]** | — |
| Containerization | Docker + docker-compose; optional Kubernetes **[Assumption]** | — |
| CI/CD | GitHub Actions **[Assumption]** | — |
| Compute (training) | Google Colab Pro+ / Kaggle; local RTX 3080 / Tesla T4 | Source |

## Section 12 — Database Design

Relational schema (PostgreSQL). Image binaries live in object storage; the DB holds metadata, results, and audit. **[Assumption — entire product DB layer; the academic core needs only file-based storage.]**

**Core entities:**

| Table | Key columns | Purpose |
|-------|-------------|---------|
| `users` | id, email, password_hash, role, created_at | Auth & RBAC |
| `studies` | id, anon_patient_id, modality, series_uid, source, status, uploaded_by, created_at | One CT study |
| `volumes` | id, study_id (FK), object_key, spacing, shape, hu_range, checksum | Preprocessed volume ref |
| `nodules` | id, study_id (FK), centroid_xyz, diameter_mm, subtype, mask_key | Detected nodule |
| `predictions` | id, nodule_id (FK), model_version, malignancy_prob, risk_tier, threshold, created_at | Stage-2 output |
| `attention_maps` | id, prediction_id (FK), overlay_key, method | XAI artifact ref |
| `reports` | id, study_id (FK), pdf_key, generated_by, generated_at | Exported report |
| `annotations` | id, nodule_id (FK), user_id (FK), agree, note, created_at | Human oversight |
| `experiments` | id, name, config_hash, dataset_version, metrics_json, model_uri, created_at | Training runs |
| `audit_log` | id, user_id, action, entity, entity_id, ip, timestamp | Security/audit |

**Relationships:** a `study` has many `nodules`; a `nodule` has many `predictions` (one per model version); a `prediction` has one `attention_map`; a `study` has many `reports` and (via nodules) `annotations`. Foreign keys enforce integrity; `anon_patient_id` guarantees no raw PHI in the relational store.

**Dataset split table (training):** `dataset_splits(id, dataset, patient_id, split ∈ {train,val,test}, fold)` ensures patient-level, leak-free splits (FR-06). **[Assumption on exact split ratios: 70/15/15 patient-level.]**

## Section 13 — API Design

REST/JSON over HTTPS; OpenAPI 3 documented; JWT auth; async inference via job queue. **[Assumption — entire API layer.]**

| Method | Path | Purpose | Auth |
|--------|------|---------|------|
| POST | `/api/v1/auth/login` | Obtain JWT | public |
| POST | `/api/v1/studies` | Upload/register a CT series (multipart or PACS ref) | user |
| GET | `/api/v1/studies` | List studies (worklist, paginated, filterable) | user |
| GET | `/api/v1/studies/{id}` | Study detail + nodules | user |
| POST | `/api/v1/studies/{id}/infer` | Enqueue inference job → returns job_id | user |
| GET | `/api/v1/jobs/{job_id}` | Poll job status (queued/running/done/failed) | user |
| GET | `/api/v1/studies/{id}/results` | Predictions + risk tiers + overlay URLs + model_version | user |
| GET | `/api/v1/nodules/{id}/attention` | Attention/CAM overlay artifact | user |
| POST | `/api/v1/studies/{id}/report` | Generate PDF report → report URL | user |
| POST | `/api/v1/nodules/{id}/annotations` | Record agree/disagree + note | user |
| GET | `/api/v1/dashboard/metrics` | Aggregate throughput/score stats | coordinator |
| POST | `/api/v1/admin/users` | Create/manage users | admin |
| GET | `/api/v1/health` / `/api/v1/ready` | Liveness/readiness | public |

**Standards:** consistent error envelope `{error: {code, message, details}}`; pagination via `?page&size`; every prediction response carries `model_version` and a `disclaimer` field (FR-38); rate limiting per token; idempotency key on `POST /infer`.

**Example — results response:**
```json
{
  "study_id": "st_123",
  "model_version": "hybrid-cnn-tr-v1.3.0",
  "disclaimer": "Research use only. Not a medical diagnosis.",
  "nodules": [
    { "nodule_id": "nd_1", "malignancy_prob": 0.87, "risk_tier": "high",
      "threshold": 0.5, "subtype": "part-solid",
      "attention_overlay_url": "/api/v1/nodules/nd_1/attention" }
  ]
}
```

## Section 14 — AI Components (Detailed Specification)

### 14.1 Stage 1 — Detection & Segmentation
- **Model:** 3D U-Net (MONAI implementation) for voxel-wise nodule segmentation + candidate generation. **[Assumption on U-Net; source states two-stage detection→classification generically.]**
- **Input:** preprocessed full-lung volume (or sliding-window 3D patches, e.g. 128³ with overlap).
- **Output:** segmentation mask → connected-component candidates → centroids + bounding VOIs.
- **False-positive reduction:** lightweight 3D CNN classifier on candidates (optional second pass). **[Assumption]**
- **Loss:** Dice + BCE (segmentation). **[Assumption]**
- **Eval:** FROC (sensitivity vs false positives/scan), the LUNA16 standard.

### 14.2 Stage 2 — Hybrid CNN–Transformer Classifier (core)
- **Backbone:** 3D CNN (3D ResNet-18/34-style or DenseNet3D), optionally pre-trained (e.g., on medical 3D data / self-supervised). Produces feature map (B, C, D', H', W'). **[Assumption on exact backbone; source specifies "3D CNN backbone".]**
- **Tokenization:** flatten spatial grid → N = D'·H'·W' tokens of dim C; linear patch-embedding projection to model dim d_model; add learnable positional encodings + prepend a [CLS] token. *(Directly from synopsis §6.2.)*
- **Encoder:** L Transformer blocks, each = multi-head self-attention (h heads) + MLP, with residual connections and LayerNorm. **[Assumption defaults: d_model=384, L=6, h=6, MLP ratio 4, dropout 0.1.]**
- **Head:** [CLS] token → LayerNorm → FC → sigmoid → P(malignant). *(From synopsis §6.2.)*
- **Loss:** class-weighted BCE (source); optional focal loss (γ=2) for imbalance. **[Assumption on focal.]**
- **Training tricks:** mixed precision (AMP), gradient accumulation, gradient checkpointing (fit ≥8 GB VRAM), cosine LR schedule with warmup, early stopping on val AUC. **[Assumption on schedule.]**
- **Augmentation:** random rotation, flips, intensity normalization/jitter, random crop within VOI (source lists rotation/flip/intensity normalization).

### 14.3 Explainability
- **Primary:** attention rollout / [CLS]-to-patch attention weights mapped back to voxel space → heatmap overlay. *(Source: attention-map visualization.)*
- **Secondary:** 3D Grad-CAM on the CNN backbone for cross-validation of focus. **[Assumption]**
- **Quality metric:** attention-mass-inside-nodule-mask ratio. **[Assumption]**

### 14.4 Baselines (for benchmarking)
Brock model reimplementation (logistic regression on nodule features), Lung-RADS categorical mapping, 2D-CNN slice model, pure-3D-CNN (no Transformer) — the last isolates the Transformer's contribution (ablation). *(Brock/Lung-RADS from source; ablations [Assumption].)*

### 14.5 Model card (required deliverable)
Each released model ships a model card: intended use, training data + version, metrics (AUC/sensitivity/specificity/calibration, per-subtype), known limitations, and the research-use disclaimer. **[Assumption]**

## Section 15 — Module Breakdown

| Module | Responsibility | Key deps |
|--------|----------------|----------|
| `data/` | Download, register, split (patient-level), integrity checks | SimpleITK, pydicom, pandas |
| `preprocess/` | Resample, HU window, normalize, lung mask, VOI crop, cache | MONAI, SimpleITK, numpy |
| `detect/` (Stage 1) | 3D U-Net detection/segmentation + FP reduction | MONAI, PyTorch |
| `model/` (Stage 2) | CNN backbone, tokenizer, Transformer encoder, head | PyTorch |
| `train/` | Training loop, AMP, checkpointing, schedules, logging | PyTorch, MLflow |
| `eval/` | Metrics, calibration, FROC, bootstrap CIs, benchmark table | scikit-learn, numpy |
| `xai/` | Attention rollout, 3D Grad-CAM, overlay rendering | PyTorch, Matplotlib |
| `baselines/` | Brock, Lung-RADS, 2D/3D-CNN ablations | scikit-learn |
| `api/` | FastAPI app, routes, auth, schemas | FastAPI, pydantic |
| `worker/` | Async inference jobs, GPU orchestration | Celery/RQ, Redis |
| `web/` | React clinician UI, viewer, dashboard | React, TS, Cornerstone.js |
| `report/` | PDF/CSV generation | ReportLab |
| `common/` | Config, logging, DB models, storage clients, errors | pydantic, SQLAlchemy |
| `infra/` | Dockerfiles, compose, CI, IaC | Docker, GH Actions |

Modules `data`→`preprocess`→`detect`→`model`→`train`→`eval`→`xai` are the research core (buildable/gradable independent of the product layer). `api`/`worker`/`web`/`report` are the product layer. **[Assumption on product-layer modules.]**

## Section 16 — UI/UX Requirements

- **Worklist screen:** sortable/filterable table of studies (status, date, uploader, #nodules); one-click open.
- **Viewer screen:** DICOM slice viewer with scroll, zoom, pan, window/level presets (lung/mediastinum); nodule markers; slider to step through slices; toggle for attention overlay (opacity slider).
- **Results panel:** per-nodule card — probability (with visual gauge), risk tier (color-coded low/medium/high), subtype, confidence, model version, and a prominent research-use disclaimer banner.
- **Report action:** "Generate report" → preview → download PDF; "Export CSV".
- **Oversight:** agree/disagree toggle + free-text note per nodule (captured for audit and future active learning).
- **Dashboard:** throughput over time, score distribution histogram, agree/disagree rate, per-subtype counts.
- **Admin:** user/role management.
- **UX principles:** ≤3 clicks worklist→score (NFR-11); non-blocking async with progress; clear empty/error/loading states; keyboard navigation and adequate contrast (WCAG 2.1 AA basics, NFR-18); consistent color semantics for risk tiers; no medical claims in copy. **[Assumption — entire UI/UX layer; not in academic scope.]**

## Section 17 — Wireframe Suggestions

Text wireframes (to be realized in Figma/React). **[Assumption]**

**Worklist**
```
┌───────────────────────────────────────────────────────────┐
│ PulmoScan AI   ▸ Worklist   Dashboard   Admin      [user ▾]│
├───────────────────────────────────────────────────────────┤
│ [Search…]        [Status ▾] [Date ▾]         [+ Upload CT] │
│ ┌─────┬───────────┬──────────┬────────┬────────┬─────────┐ │
│ │ ID  │ Patient*  │ Date     │ Status │Nodules │ Action  │ │
│ │ 123 │ anon_ab12 │ 07-29    │ Done   │  3     │ [Open]  │ │
│ │ 124 │ anon_cd34 │ 07-29    │ Running│  –     │  …      │ │
│ └─────┴───────────┴──────────┴────────┴────────┴─────────┘ │
│ * de-identified id — no PHI                                 │
└───────────────────────────────────────────────────────────┘
```

**Viewer + Results**
```
┌──────────────────────────────┬────────────────────────────┐
│  [Axial slice image]         │  Nodule 1  ● HIGH           │
│                              │  Malignancy: 0.87  ▓▓▓▓▓▓�. │
│   ○ nodule marker            │  Subtype: part-solid        │
│   [attention overlay ▢on]    │  Model: v1.3.0              │
│   Slice  ◀ 128/220 ▶         │  ─────────────────────────  │
│   W/L: [Lung ▾]  Zoom [＋][－]│  Nodule 2  ● LOW  0.12      │
│                              │  [ Agree ] [ Disagree ] 📝  │
│                              │  ⚠ Research use only.        │
│                              │  [ Generate report ] [CSV]  │
└──────────────────────────────┴────────────────────────────┘
```

## Section 18 — Security Requirements

| Area | Requirement |
|------|-------------|
| Authentication | JWT with short-lived access + refresh tokens; bcrypt/argon2 password hashing |
| Authorization | RBAC (admin / clinician / coordinator / researcher); least privilege |
| PHI/PII | DICOM de-identification on ingest (strip patient tags); only `anon_patient_id` persisted |
| Encryption | TLS 1.2+ in transit; AES-256 at rest (DB + object store) |
| Secrets | Env/secret manager (never in code or git); rotation policy |
| Audit | Immutable audit log of login, access, inference, export, deletion |
| Input safety | Validate/scan uploads; size limits; content-type checks; guard against zip/DICOM bombs |
| API hardening | Rate limiting, CORS allowlist, security headers, idempotency keys |
| Dependencies | Pinned versions; automated vulnerability scanning (e.g., pip-audit) |
| Disclaimer | "Research use, not a diagnosis" surfaced on all result surfaces (FR-38) |
| Data governance | Retention/deletion controls; documented data-use agreement for NLST/LUNA16 |

> **Security note:** the clinician web app and API are network-exposed services handling medical images. They **must not** be deployed without authentication, TLS, and de-identification in place. Absent these, the system is research-only on a local, isolated machine. **[Assumption — the academic synopsis has no product security layer; this is added per product-build mandate.]**

## Section 19 — Performance Requirements

| Metric | Target | Rationale |
|--------|--------|-----------|
| End-to-end single-study inference (GPU) | ≤ 60 s | Fits clinical review flow **[Assumption]** |
| Stage-2 per-nodule classification | ≤ 500 ms (GPU) | Interactive **[Assumption]** |
| Training epoch (LUNA16 subset, T4) | practical within Colab session limits via AMP + caching | Source HW constraint |
| Peak VRAM (training) | ≤ 8 GB via AMP + checkpointing + accumulation | Source min spec |
| API p95 latency (non-inference endpoints) | ≤ 300 ms | Snappy UI **[Assumption]** |
| Concurrent inference workers | horizontally scalable; queue depth ≥ 10 | Screening volume **[Assumption]** |
| Preprocessing cache hit | > 90% on repeat runs | Fast iteration |

Optimization levers: preprocessed-tensor caching, mixed precision, ONNX/TorchScript export for inference, batch nodule scoring, and GPU worker autoscaling. **[Assumption on ONNX/autoscale.]**

## Section 20 — Deployment Architecture

**Environments:** local dev (docker-compose) → training (Colab Pro+/Kaggle/local GPU) → staging → production. **[Assumption — product deployment; academic core runs in notebooks/local.]**

```
                 ┌──────────── Ingress (TLS, reverse proxy: nginx/Traefik) ────────────┐
                 │                                                                      │
        ┌────────▼────────┐   ┌───────────────┐   ┌───────────────────────────────┐    │
        │ Web (React/SPA) │   │ API (FastAPI) │   │ Redis (broker + cache)         │    │
        └─────────────────┘   └───────┬───────┘   └───────────────┬───────────────┘    │
                                       │                           │                    │
                         ┌─────────────▼─────────────┐   ┌─────────▼───────────────┐    │
                         │ PostgreSQL (metadata)     │   │ GPU Inference Worker(s)  │    │
                         └───────────────────────────┘   │ (Celery/RQ, model svc)  │    │
                         ┌───────────────────────────┐   └─────────┬───────────────┘    │
                         │ Object store (MinIO/S3)   │◀────────────┘                    │
                         │ volumes, overlays, reports│                                  │
                         └───────────────────────────┘  Model registry + MLflow ────────┘
```

Containerized with Docker; orchestrated via docker-compose for the project, with a documented path to Kubernetes for scale. CI builds/tests/pushes images; CD deploys to staging on merge, production on tag. Model artifacts pulled from the registry by version at worker startup. **[Assumption.]**

## Section 21 — Testing Strategy

| Level | Scope | Tools |
|-------|-------|-------|
| Unit | Preprocessing transforms, tokenizer shapes, loss, metric math, API schema validation | pytest |
| Integration | data→preprocess→detect→model→eval pipeline on tiny fixture volumes | pytest + fixtures |
| Model/ML tests | Overfit-on-one-batch sanity, shape/gradient tests, deterministic seed test, metric-threshold gate (AUC on held-out fixture) | pytest + torch |
| Data validation | Split-leakage check, spacing/shape/NaN checks, label balance report | Great Expectations-style **[Assumption]** |
| API | Endpoint contract tests against OpenAPI, auth, rate-limit, error envelope | pytest + httpx |
| E2E | Upload→infer→results→report happy path (staging) | Playwright **[Assumption]** |
| Security | Dependency scan, auth bypass, upload safety | pip-audit, custom |
| Performance | Inference latency + VRAM budget regression | pytest-benchmark **[Assumption]** |
| Reproducibility | Re-run seeded training → AUC within ±0.005 | CI nightly **[Assumption]** |

Coverage target ≥ 80% on the core pipeline (NFR-13). CI runs unit+integration+ML-sanity on every PR; heavier E2E/repro nightly. A **metric gate** blocks merges that drop held-out AUC below threshold.

## Section 22 — Project Timeline (6 Months)

| Month | Milestone | Exit criteria |
|-------|-----------|---------------|
| 1 | Foundations & data | Repo, env, LUNA16 ingested, preprocessing + patient-level splits, EDA |
| 2 | Stage 1 detection | 3D U-Net detection/segmentation, VOIs generated, FROC baseline |
| 3 | Stage 2 model v0 | Hybrid CNN–Transformer trains end-to-end, first AUC on val |
| 4 | Optimize & benchmark | AUC > 0.90 target, baselines (Brock/Lung-RADS/ablations), calibration, XAI |
| 5 | Product layer | API, worker, web UI, PDF report, dashboard, security basics |
| 6 | Harden, document, deliver | Tests ≥80%, deployment, model card, survey (40+ refs), final report/demo |

Critical path: data (M1) → detection (M2) → classifier (M3) → target metric (M4). Product layer (M5) can partially parallelize once the model API contract is fixed at end of M3. **[Assumption on product-layer timing.]**

## Section 23 — Sprint Planning (6 Sprints ≈ 1 month each)

**Sprint 1 — Foundations.** Repo + CI skeleton, env pinning, config system, data download/registration (US-A01/02), preprocessing (US-A03/04/05), caching (US-A06), integrity checks (US-A07), patient-level split (US-A08), EDA notebook. *Demo: preprocessed tensor + split report.*

**Sprint 2 — Detection (Stage 1).** 3D U-Net (US-B01/02), VOI cropping (US-B03), FP reduction (US-B04), FROC eval (US-B05), persist detections (US-B08), augmentation (US-A09). *Demo: nodule candidates + FROC curve.*

**Sprint 3 — Classifier v0 (Stage 2).** CNN backbone (US-C01), tokenizer + [CLS] + positional (US-C02), Transformer encoder (US-C03), sigmoid head (US-C04), weighted BCE (US-C05), AMP + checkpointing (US-C07/08), seeding/resume (US-C10/11). *Demo: first AUC on val.*

**Sprint 4 — Optimize, XAI & benchmark.** Fine-tuning (US-C06), config sweeps (US-C09), attention maps (US-D01/03), Grad-CAM (US-D02), metrics/calibration (US-E01–E04), Brock/Lung-RADS + ablations (US-E05–E07), CIs + benchmark table (US-E08/09), per-subtype (US-E10). *Demo: AUC>0.90 + benchmark table + overlays.*

**Sprint 5 — Product layer.** API + auth (US-G01–G05, US-F01), worker/async (US-G02), viewer + worklist + results (US-F02–F06), report/CSV (US-F07, US-J01/02), oversight (US-F08), de-identification + TLS + audit (US-I01–I05), model registry/tracking (US-H01/02), containers (US-H03). *Demo: upload→score→report in UI.*

**Sprint 6 — Harden & deliver.** Tests to ≥80% (US-J*), monitoring/logging (US-H04/05), drift hook (US-H06), health/rate-limit (US-G06/07), dashboard (US-F09), admin (US-F10), docs + model card (US-J07/08), one-command demo (US-J06), deployment, survey paper. *Demo: full E2E + docs + benchmark + survey.*

## Section 24 — Risks & Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | ≥8 GB VRAM insufficient for 3D + Transformer | High | High | AMP, gradient checkpointing/accumulation, smaller VOIs, reduce d_model/L, patch-based |
| R2 | Class imbalance hurts sensitivity | High | High | Class-weighted/focal loss, weighted sampling, threshold tuning, augmentation |
| R3 | NLST access (~6 TB, credentialed) delayed/unavailable | Med | Med | Proceed on LUNA16/LIDC; treat NLST as stretch validation |
| R4 | AUC>0.90 not reached | Med | High | Ablations to find gains, better backbone/pretraining, ensembling, more augmentation |
| R5 | Overfitting on small data | High | Med | Augmentation, regularization, cross-validation, early stopping |
| R6 | Detection misses hard subtypes (GGO/part-solid) | Med | High | Subtype-aware sampling, per-subtype eval, tuned detection thresholds |
| R7 | Poor calibration despite good AUC | Med | Med | Temperature scaling / Platt; report Brier & reliability |
| R8 | XAI overlays don't localize to nodule | Med | Med | Attention rollout + Grad-CAM cross-check; overlap metric |
| R9 | CUDA/dep version conflicts (11.8 vs 13.0 flag) | Med | Med | Standardize CUDA 12.1 + pinned envs + container |
| R10 | Scope creep from product layer vs academic timeline | High | Med | Research core first (M1–M4); product layer is additive (M5–M6) |
| R11 | PHI leakage in a real-data product | Low | High | De-identification on ingest, no raw PHI persisted, audit |
| R12 | Reproducibility drift | Med | Med | Seeds, pinned deps, config hashing, CI repro test |

## Section 25 — Future Enhancements

See Phase 6 for the full treatment. Headline items: self-supervised/foundation-model pretraining on unlabeled CT; multi-task learning (segmentation + malignancy + subtype); longitudinal (prior vs current scan) modeling of nodule growth; PACS/DICOM-node integration; multi-site external validation; uncertainty estimation (MC-dropout/ensembles); active learning from clinician agree/disagree; report auto-narration; regulatory groundmapping (SaMD).

## Section 26 — Key Performance Indicators (KPIs)

| KPI | Definition | Target |
|-----|------------|--------|
| Model AUC-ROC | On LUNA16 test | > 0.90 |
| Δ over Brock | AUC improvement vs Brock (~0.74) | > +0.15, statistically significant |
| Sensitivity @ ≥0.80 specificity | Clinical operating point | ≥ 0.90 **[Assumption]** |
| Calibration (Brier) | Probability reliability | ≤ 0.15 **[Assumption]** |
| GGO/part-solid AUC | Hard-subtype performance | > 0.85 **[Assumption]** |
| Inference latency (p95) | Per study, GPU | ≤ 60 s **[Assumption]** |
| XAI localization | Attention-in-nodule ratio | ≥ 0.90 of cases **[Assumption]** |
| Test coverage | Core pipeline | ≥ 80% |
| Reproducibility delta | Re-run AUC variance | ≤ ±0.005 |
| Survey references | Literature covered | ≥ 40 |
| Clinician agreement | Agree rate on high-tier (product) | tracked, ≥ 80% **[Assumption]** |

## Section 27 — Acceptance Criteria

**Research core (must-pass for academic deliverable):**
- AC-1: Pipeline runs end-to-end from raw LUNA16 to malignancy score reproducibly (US-A*, B*, C*).
- AC-2: Held-out test AUC-ROC > 0.90 with reported CI (FR-17, NFR-01).
- AC-3: Sensitivity, specificity, confusion matrix, and calibration reported (FR-17/18).
- AC-4: Benchmark table vs Brock, Lung-RADS, and ≥1 CNN ablation, with significance (FR-19/20).
- AC-5: Attention overlays produced for scored nodules and qualitatively localize to lesions (FR-15).
- AC-6: Patient-level split verified leak-free (FR-06).
- AC-7: Survey with ≥40 references delivered (Objective 4).
- AC-8: Model card + reproducible one-command demo (FR-39/40).

**Product layer (must-pass if built):**
- AC-9: Clinician can upload a study, run inference, view scored nodules with overlays, and export a PDF report via the UI (US-F*).
- AC-10: API matches OpenAPI spec; auth, rate limiting, health checks pass (FR-28).
- AC-11: DICOM de-identification verified; TLS + at-rest encryption + audit logging in place (FR-33/34/35).
- AC-12: Research-use disclaimer present on all result surfaces (FR-38).
- AC-13: Core-pipeline test coverage ≥ 80%; CI green including metric gate (Section 21).
- AC-14: Services deploy via containers to staging with monitoring/logging (Sections 20–21).

## Section 28 — Appendix

**A. Glossary.** LDCT — low-dose CT; VOI — volume of interest; GGO — ground-glass opacity; AUC-ROC — area under receiver-operating curve; FROC — free-response ROC; XAI — explainable AI; [CLS] — classification token; SaMD — software as a medical device; PHI — protected health information.

**B. Datasets.** LUNA16 (~125 GB; derived from LIDC-IDRI; nodule annotations; standard FROC eval) — primary. NLST (~6 TB; credentialed access) — stretch external validation. Split: patient-level 70/15/15 **[Assumption]**.

**C. Baselines.** Brock University malignancy model (AUC ~0.74); Lung-RADS categorical management; 2D-CNN and pure-3D-CNN ablations.

**D. Source-document trace.** Synopsis (8 pp): Intro, Literature Survey [1]–[15], Problem Statement, Objectives (4), Design §6.1–6.3, Functional Requirements §7.1 (HW: RTX 3080/Tesla T4 ≥8 GB VRAM, 32 GB RAM, 500 GB SSD) / §7.2 (SW), Expected Outcome (AUC>0.90, 40+ ref survey, attention maps, benchmark), References. Review deck (21 slides): Title/team (BIS685, Group G17, Guide Prof. Pallavi K N), Agenda, Intro, Objectives, Problem Statement, Literature Survey tables (~20 papers), Outcome of survey, System Design, Architecture diagram (image), Functional Requirements, Applications (7), References (1–25), Q&A.

**E. Open items to confirm with stakeholders.** (1) CUDA version (11.8 vs 13.0 → recommend 12.1). (2) Exact CNN backbone & whether pretrained weights are permitted. (3) NLST access status. (4) Whether the product layer (UI/API/deploy) is in academic scope or a stretch goal. (5) Exact clinical operating-point targets.

**F. References.** Consolidated from synopsis [1]–[15] and deck [1]–[25]; the project targets ≥40 curated references for the survey contribution. Full bibliographic details for entries beyond the extractable table rows are **[Assumption — to be completed from the source reference lists]**.

---

# PHASE 4 — ENGINEERING BLUEPRINT

> This phase turns the PRD into a concrete engineering setup: how the repository is laid out, how code is written and named, how the team collaborates through Git, how the system is configured across environments, and how it is deployed, logged, monitored, and hardened against failure. Everything in Phase 4 beyond the model-training core is tagged **[Assumption]**, since the source synopsis and deck describe the research artifact, not the surrounding product engineering. These assumptions follow mainstream 2025-era MLOps and web-service best practices and are chosen to be internally consistent with the stack in Section 11.

## 4.1 Repository Strategy

**[Assumption]** A **monorepo** is used for the first 6 months. Rationale: a single small team (4 students + guide) building tightly coupled components (shared data schemas, shared model artifacts, shared types between API and frontend) benefits from atomic cross-cutting commits and a single CI pipeline. A polyrepo split is deferred to Future Enhancements (only worth it once separate teams own separate services).

Repository host: **GitHub** (private repo), with GitHub Actions for CI/CD and GitHub Releases for tagged model/app versions.

| Concern | Decision | Rationale |
|---|---|---|
| Layout | Monorepo | Small team, coupled components, atomic changes |
| Package boundary | Python packages under `src/`, JS workspace under `web/` | Clear language separation |
| Large files | Git LFS for sample data/figures; **models & datasets NOT in Git** | Datasets are 125 GB–6 TB; store in object storage + registry |
| Env reproducibility | `pyproject.toml` (uv/pip) + `package.json` (pnpm) + pinned lockfiles | Deterministic installs |

## 4.2 Folder & Repository Structure

```
pulmoscan-ai/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example                # documented, non-secret placeholders
├── docker-compose.yml          # local dev: api, worker, redis, postgres, minio, mlflow
├── docker-compose.prod.yml
├── Makefile                    # make setup | test | lint | train | serve
├── pyproject.toml              # python deps (uv/pip), tooling config
├── uv.lock
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       ├── ci.yml              # lint + type + unit/integration tests
│       ├── model-eval.yml      # runs eval gate on model PRs
│       └── deploy.yml          # build+push images, deploy on tag
├── docs/
│   ├── PRD_LungNoduleAI.md
│   ├── architecture/           # diagrams, ADRs (Architecture Decision Records)
│   ├── model_card.md
│   └── runbooks/               # on-call / incident procedures
├── configs/                    # Hydra/YAML configs (data, model, train, infer)
│   ├── data/
│   ├── model/
│   ├── train/
│   └── infer/
├── data/                       # NOT committed (gitignored); local mount point
│   ├── raw/                    # LUNA16 / LIDC-IDRI / NLST (credentialed)
│   ├── interim/                # resampled, windowed volumes
│   ├── processed/              # cached VOIs, splits
│   └── external/
├── src/
│   └── pulmoscan/
│       ├── __init__.py
│       ├── common/             # logging, config loader, types, errors
│       ├── io/                 # DICOM/NIfTI readers, MinIO/S3 client
│       ├── preprocessing/      # resample, HU windowing, lung-mask, augment
│       ├── detection/          # Stage-1 3D U-Net (model, train, infer)
│       ├── classification/     # Stage-2 hybrid CNN–Transformer
│       │   ├── backbone_cnn.py
│       │   ├── transformer_encoder.py
│       │   ├── model.py        # fusion head → sigmoid
│       │   ├── losses.py       # weighted BCE / focal
│       │   └── train.py
│       ├── xai/                # attention rollout, 3D Grad-CAM
│       ├── evaluation/         # AUC, sens/spec, FROC, calibration, bootstrap CI
│       ├── inference/          # end-to-end pipeline orchestration
│       ├── serving/            # FastAPI app, routers, schemas, deps
│       │   ├── main.py
│       │   ├── routers/
│       │   ├── schemas/        # pydantic request/response models
│       │   └── security/       # auth, RBAC, audit
│       ├── tasks/              # Celery tasks (async inference, report gen)
│       ├── reporting/          # ReportLab PDF report builder
│       └── db/                 # SQLAlchemy models, alembic migrations
│           └── migrations/
├── web/                        # React + TypeScript frontend (pnpm workspace)
│   ├── src/
│   │   ├── components/
│   │   ├── features/           # worklist, viewer, results, admin
│   │   ├── viewer/             # Cornerstone.js integration
│   │   ├── api/                # generated API client (from OpenAPI)
│   │   ├── hooks/
│   │   └── types/
│   ├── public/
│   └── package.json
├── notebooks/                  # EDA & experiments (not imported by src)
├── scripts/                    # one-off ops (download data, seed db, export)
└── tests/
    ├── unit/
    ├── integration/
    ├── e2e/
    └── fixtures/               # tiny synthetic volumes, mock DICOM
```

**[Assumption]** `data/`, `models/`, and any credentialed dataset live outside Git. Model artifacts are tracked in **MLflow Model Registry**; raw/processed data live in **MinIO/S3** with a manifest (checksums + split assignment) committed under `configs/data/`.

## 4.3 Coding Standards

**Python (research + backend)**
- Style/format: **Ruff** (lint + format, Black-compatible), line length 100.
- Typing: full type hints on public functions; **mypy** in `--strict` for `src/pulmoscan/serving`, `db`, `common`; relaxed for research modules.
- Docstrings: Google style; every public function documents shapes/units (e.g., "volume: float32 tensor `[B,1,D,H,W]`, HU-windowed to [-1000,400]").
- No hard-coded paths, magic numbers, or thresholds — all via `configs/`.
- Tensor shape/units annotated in comments at boundaries.
- Reproducibility: single `set_seed(seed)` utility seeding Python/NumPy/PyTorch + `cudnn.deterministic` for eval runs.

**TypeScript (frontend)**
- **ESLint** + **Prettier**; `strict: true` in `tsconfig`.
- API types generated from backend **OpenAPI** schema (single source of truth) — no hand-written response types.
- Component structure: feature-first folders; presentational vs container separation; no business logic in JSX.

**General**
- Pre-commit hooks run Ruff, mypy (scoped), ESLint, Prettier, and secret scanning (`detect-secrets`/gitleaks) on every commit.
- No commented-out code on `main`; use Git history instead.
- Every threshold that affects a clinical output (e.g., malignancy operating point) is a named config value, never inline.

## 4.4 Naming Conventions

| Artifact | Convention | Example |
|---|---|---|
| Python modules/packages | `snake_case` | `transformer_encoder.py` |
| Python classes | `PascalCase` | `HybridClassifier` |
| Python functions/vars | `snake_case` | `compute_auc()` |
| Constants | `UPPER_SNAKE` | `DEFAULT_HU_WINDOW` |
| React components | `PascalCase` | `NoduleViewer.tsx` |
| React hooks | `useCamelCase` | `useStudyResults` |
| DB tables | `snake_case` plural | `attention_maps` |
| DB columns | `snake_case` | `malignancy_prob` |
| API routes | `kebab-case`, plural nouns | `/api/v1/studies/{id}/nodules` |
| Env variables | `UPPER_SNAKE`, prefixed | `PULMOSCAN_DB_URL` |
| Docker images | `pulmoscan/<service>:<semver>` | `pulmoscan/api:1.2.0` |
| Model artifacts | `<stage>_<arch>_v<major.minor>` | `stage2_hybrid_v0.3` |
| Experiments (MLflow) | `<component>/<yyyymmdd>/<short-desc>` | `stage2/20260812/focal-loss` |
| Git branches | see 4.5 | `feat/stage2-attention-fusion` |

## 4.5 Git Branching Strategy

**[Assumption]** A trunk-based flow with short-lived feature branches (simpler than Git Flow, appropriate for a small team on a 6-month timeline).

| Branch | Purpose | Rules |
|---|---|---|
| `main` | Always releasable | Protected; no direct pushes; PR + 1 review + green CI required |
| `feat/<slug>` | New feature | Branch from `main`; squash-merge back |
| `fix/<slug>` | Bug fix | Same as feat |
| `exp/<slug>` | Research experiment | May be messy; not merged unless promoted |
| `release/<semver>` | Release stabilization (optional) | Cut near sprint end if needed |

- **Commit style:** Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `perf:`). Enables automated changelog + semver bumps.
- **PR requirements:** description + linked user story (US-x) / FR-x, screenshots for UI, and for model changes a link to the MLflow run + eval-gate result.
- **Model changes** trigger `model-eval.yml`: PR cannot merge if primary metric (AUC-ROC on the frozen validation split) regresses beyond a configured tolerance.

## 4.6 Versioning

**[Assumption]** Three independently versioned artifacts, all SemVer (`MAJOR.MINOR.PATCH`):

| Artifact | Scheme | Bump rules |
|---|---|---|
| Application (API + web) | SemVer | MAJOR = breaking API; MINOR = feature; PATCH = fix |
| API contract | Path version `/api/v1` + OpenAPI SemVer | New version path only on breaking change |
| Model | SemVer, registered in MLflow | MAJOR = architecture change; MINOR = retrain w/ new data; PATCH = threshold/calibration tweak |

- Every inference response embeds `model_version` and `app_version` (traceability; ties to audit_log). This was already reflected in the Section 13 API results example.
- Datasets/splits versioned by manifest hash (checksum of file list + split map) so any result is reproducible to an exact split.
- Git tags `v<app-semver>` create GitHub Releases; images tagged with the same semver + git SHA.

## 4.7 Environment Variables & Configuration

**[Assumption]** Config precedence: **defaults in code < YAML config (`configs/`) < environment variables < CLI overrides**. Runtime secrets and connection strings come only from env vars (never YAML, never Git). `.env.example` documents every variable with a safe placeholder.

| Variable | Purpose | Example (placeholder) |
|---|---|---|
| `PULMOSCAN_ENV` | dev / staging / prod | `dev` |
| `PULMOSCAN_DB_URL` | PostgreSQL DSN | `postgresql+psycopg://user:***@db:5432/pulmoscan` |
| `PULMOSCAN_REDIS_URL` | Celery broker/result | `redis://redis:6379/0` |
| `PULMOSCAN_S3_ENDPOINT` | MinIO/S3 endpoint | `http://minio:9000` |
| `PULMOSCAN_S3_BUCKET` | Volume/artifact bucket | `pulmoscan-studies` |
| `PULMOSCAN_S3_ACCESS_KEY` / `_SECRET_KEY` | Object store creds | `***` |
| `PULMOSCAN_MLFLOW_URI` | Tracking/registry | `http://mlflow:5000` |
| `PULMOSCAN_JWT_SECRET` | Token signing | `***` |
| `PULMOSCAN_JWT_TTL_MIN` | Access token TTL | `30` |
| `PULMOSCAN_MODEL_STAGE` | Registry stage to load | `Production` |
| `PULMOSCAN_LOG_LEVEL` | Logging verbosity | `INFO` |
| `PULMOSCAN_CUDA_VISIBLE_DEVICES` | GPU pinning | `0` |
| `PULMOSCAN_SENTRY_DSN` | Error tracking (optional) | `***` |

- The **CUDA target is standardized at 12.1 with PyTorch 2.2+ [Assumption]** (resolving the synopsis "13.0" vs deck "11.8" inconsistency flagged in Phase 1); this is enforced by the base Docker image, not an env var.
- Config validation on startup via **pydantic-settings** — the service refuses to boot with a missing/invalid required variable (fail-fast).

## 4.8 Secrets Management

**[Assumption]**
- **Local dev:** `.env` file (gitignored); only `.env.example` is committed.
- **CI:** GitHub Actions encrypted secrets/OIDC; never echoed to logs.
- **Staging/prod:** a dedicated secrets manager (**AWS Secrets Manager / Vault**), injected at runtime; no secret ever baked into an image or committed.
- **Scanning:** `gitleaks`/`detect-secrets` in pre-commit and CI; a hit fails the build.
- **Rotation:** JWT secret, DB and object-store credentials rotatable without code change (env-injected). Rotation runbook in `docs/runbooks/`.
- **Credentialed datasets (NLST):** access keys treated as secrets; the data-use agreement and de-identification status recorded in `docs/` and never stored in the repo. **[Assumption]**

## 4.9 Deployment Strategy

**[Assumption]**
- **Containerized** everything; `docker-compose` for local/staging, migrate to Kubernetes for prod scale (Future Enhancement, R-linked).
- **Two image families:** (a) CPU image for API/web/worker-light; (b) CUDA 12.1 GPU image for the inference worker running Stage-1 + Stage-2.
- **Release flow:** tag `vX.Y.Z` → `deploy.yml` builds+pushes images → applies Alembic migrations (guarded, backward-compatible) → rolling update of API, then workers.
- **Model deployment decoupled from app deployment:** promoting a model in the MLflow registry to `Production` + a config flip triggers workers to hot-load the new version; no app redeploy needed. Enables safe rollback (repoint to prior model version).
- **Rollback:** app = redeploy previous image tag; model = repoint registry stage; DB = only backward-compatible migrations so prior app version still runs.
- **Environments:** `dev` (compose, synthetic data) → `staging` (compose/K8s, de-identified subset, full pipeline) → `prod` (K8s, GPU node pool). Clinical/production use is gated behind the regulatory disclaimer already stated in the PRD (research-use / not-for-diagnosis in the current scope).

## 4.10 Logging, Monitoring & Error Handling

**Logging [Assumption]**
- Structured **JSON logs** (one event per line) via `structlog`, shipped to a central store (Loki/CloudWatch).
- Every request/inference carries a **correlation/trace ID** (`study_id` + request UUID) threaded API → Celery task → model → DB → report.
- **No PHI in logs** — log study IDs and hashes, never patient identifiers or raw pixel data (ties to Security section).
- Log levels: `DEBUG` (dev only), `INFO` (lifecycle/inference start-end), `WARNING` (degraded/fallback), `ERROR` (handled failure), `CRITICAL` (pipeline down).

**Monitoring [Assumption]**
- **Metrics** (Prometheus + Grafana): request latency/throughput, queue depth, GPU utilization/memory, inference time per stage, error rate, model confidence distribution.
- **Model/data drift:** track input distribution (HU stats, spacing, nodule size histogram) and output score distribution vs the validation baseline; alert on drift beyond threshold — an early signal to retrain.
- **Alerting:** paging on queue backlog, GPU OOM, error-rate spike, or eval-gate failure in CI. Runbooks in `docs/runbooks/`.
- **Audit vs telemetry:** clinical/auditable events go to the `audit_log` table (immutable, Section 12); operational telemetry goes to metrics/logs. Kept separate by design.

**Error Handling [Assumption]**
- Typed exception hierarchy (`PulmoscanError` → `DataError`, `ModelError`, `InferenceTimeout`, `AuthError`, …); mapped to stable API error codes + safe messages (no stack traces to clients).
- **Fail-safe over fail-silent:** an inference failure marks the study `failed` with a reason, never emits a fabricated or partial malignancy score. A missing/low-quality volume yields an explicit "cannot assess" state, not a default probability.
- **Retries:** transient I/O (object store, DB) retried with exponential backoff; non-transient (corrupt DICOM) fails fast with a clear message.
- **Idempotency:** re-submitting the same study is idempotent on `study_id` (no duplicate predictions).
- **Graceful degradation:** if XAI (attention/Grad-CAM) generation fails, the prediction still returns with `explainability_available=false` rather than failing the whole request.

---

# PHASE 5 — IMPLEMENTATION PLAN

> This phase sequences the build. It defines the order modules are constructed, why that order (dependency-driven, de-risking research uncertainty first), what each module depends on, estimated effort, complexity, and the blockers most likely to bite. Effort is expressed in **engineer-weeks (EW)** for the 4-person team over the 6-month (~26-week) window from the Section 22 timeline. Effort figures beyond the model core are **[Assumption]**, calibrated to a student team ramping on unfamiliar tooling (add ~20% learning tax vs a seasoned team).

## 5.1 Guiding Principles for Build Order

1. **Data before models.** Nothing can be trained or evaluated until DICOM/NIfTI ingestion, HU windowing, resampling, and split management are trustworthy. Data bugs masquerade as model bugs — fix the foundation first.
2. **De-risk the research core early.** The hybrid CNN–Transformer classifier (the graded innovation and the highest technical uncertainty) must reach a measurable baseline as soon as data + a detector exist, so there is runway to iterate toward the AUC > 0.90 target.
3. **Evaluation harness is not last.** Build metrics (AUC, sensitivity/specificity, calibration, FROC, bootstrap CIs) alongside the first model so every experiment is judged the same way from day one.
4. **Product layer wraps a working model.** API, DB, frontend, reporting come after the model produces trustworthy scores — they are [Assumption] scope and must never delay the core deliverable (mitigates scope-creep risk R10).
5. **Vertical slice early.** As soon as a minimal end-to-end path exists (upload → detect → classify → score → view), keep it working continuously rather than integrating big-bang at the end.

## 5.2 Module Build Order

| # | Module | Depends on | Effort (EW) | Complexity | Sprint |
|---|---|---|---|---|---|
| 1 | Repo scaffold, CI, docker-compose, config/logging | — | 2 | Low | S1 |
| 2 | Data I/O (DICOM/NIfTI readers, MinIO client, manifest/splits) | 1 | 3 | Medium | S1 |
| 3 | Preprocessing (resample, HU windowing, lung mask, augmentation) | 2 | 3 | Medium–High | S1–S2 |
| 4 | Evaluation harness (AUC, sens/spec, calibration, FROC, bootstrap CI) | 2 | 2 | Medium | S2 |
| 5 | Stage-1 detector (3D U-Net) → VOIs | 3, 4 | 4 | High | S2–S3 |
| 6 | Stage-2 hybrid CNN–Transformer classifier + losses | 3, 4 | 6 | **Very High** | S3–S4 |
| 7 | XAI (attention rollout, 3D Grad-CAM) | 6 | 2.5 | High | S4 |
| 8 | Inference pipeline orchestration (E2E: vol→detect→classify→score) | 5, 6 | 2 | Medium | S4 |
| 9 | Database schema + migrations (Alembic) | 1 | 1.5 | Low–Medium | S3 |
| 10 | FastAPI service (auth, RBAC, endpoints, schemas, audit) | 8, 9 | 4 | Medium–High | S4–S5 |
| 11 | Async tasks (Celery/Redis) for long inference + report gen | 8, 10 | 2 | Medium | S5 |
| 12 | Frontend (worklist, Cornerstone viewer, results, XAI overlays) | 10 | 5 | High | S5 |
| 13 | Reporting (ReportLab PDF w/ score, overlays, disclaimer) | 8, 10 | 2 | Medium | S5 |
| 14 | MLOps (MLflow tracking/registry, eval-gate CI, model hot-load) | 4, 6, 10 | 2.5 | Medium | S3–S6 |
| 15 | Security hardening, monitoring, drift, runbooks | 10, 11, 12 | 3 | Medium–High | S6 |
| 16 | System test, calibration finalize, documentation, model card | all | 3 | Medium | S6 |

Modules 1–8 + 14 constitute the **research-core critical path**; modules 9–13, 15 are the **[Assumption] product layer** running partly in parallel once the model is trustworthy. The two tracks are staffed concurrently (ML pair on 3/5/6/7; platform pair on 1/2/9/10/12) after the shared data foundation is up.

## 5.3 Dependency Graph (textual)

```
1 scaffold
 ├─> 2 data I/O ──> 3 preprocessing ──┬─> 5 detector ──┐
 │                                    │                ├─> 8 inference E2E ──> 11 tasks
 │                                    ├─> 6 classifier ┘         │
 │                 4 eval harness <───┘        └─> 7 XAI ────────┤
 ├─> 9 database ───────────────────────────────────────> 10 API ─┴─> 12 frontend
 4,6,10 ─> 14 MLOps                                     10 ─> 13 reporting
 10,11,12 ─> 15 hardening/monitoring ─> 16 system test & docs
```

Critical path (longest chain): **1 → 2 → 3 → 6 → 8 → 10 → 12 → 15 → 16**. The single longest-pole node is **Module 6 (hybrid classifier)** — it holds the most schedule risk and the graded innovation, which is why it starts the moment a detector + eval harness exist rather than after the full detector is polished.

## 5.4 Effort & Complexity Rationale (selected)

- **Module 6 — Hybrid CNN–Transformer (Very High, 6 EW):** the core research contribution. Risk drivers: 3D data is memory-hungry (drives AMP + gradient checkpointing/accumulation), class imbalance (weighted BCE / focal), Transformer data-hunger on limited labeled volumes (mitigate with pretrained CNN backbone + heavy augmentation + fine-tuning), and hyperparameter sensitivity (d_model=384, L=6, h=6 as starting defaults). Highest chance of needing multiple iterations to hit AUC > 0.90.
- **Module 5 — 3D U-Net detector (High, 4 EW):** memory and FROC tuning; false-positive reduction is the classic hard part. A weak detector caps end-to-end sensitivity regardless of the classifier.
- **Module 3 — Preprocessing (Medium–High, 3 EW):** deceptively hard; inconsistent slice spacing, HU calibration, and lung-mask edge cases silently degrade everything downstream. Worth over-investing in tests here.
- **Module 12 — Frontend + Cornerstone viewer (High, 5 EW):** DICOM viewing in-browser, overlaying attention/Grad-CAM on 3D volumes, and a usable worklist are non-trivial UI work; student team likely new to Cornerstone.
- **Module 4 — Eval harness (Medium, 2 EW):** modest code, but correctness is critical — a subtly wrong AUC or leaky split invalidates the whole project. Built early and frozen.
- **Modules 1, 9 (Low):** well-trodden scaffolding; main cost is the team's tooling ramp, already absorbed in the learning tax.

## 5.5 Expected Blockers & Mitigations

| Blocker | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GPU memory limits on 3D volumes | High | High | AMP, gradient checkpointing + accumulation, patch/VOI-based training, smaller batch; secure adequate GPU early |
| NLST access delay (credentialed, ~6 TB) | High | High | Start on LUNA16 (open); treat NLST as stretch data; begin data-use application in Sprint 1 |
| Class imbalance → poor sensitivity | High | High | Weighted/focal loss, resampling, threshold tuning on validation, report operating point explicitly |
| Transformer underfits/overfits on limited data | Medium–High | High | Pretrained CNN backbone, strong augmentation, regularization, consider smaller d_model/L before scaling up |
| Data leakage across train/val/test splits | Medium | Critical | Patient-level splitting (never slice-level), frozen split manifest, leakage checks in eval harness |
| Slice-spacing / HU inconsistency across sources | Medium | Medium | Robust resampling + calibration in preprocessing, extensive fixtures/tests |
| Cornerstone/3D overlay integration friction | Medium | Medium | Spike early in S5, fall back to 2D key-slice overlay if 3D overlay slips |
| Long inference blocking API | Medium | Medium | Async Celery from the start for inference + report gen |
| Scope creep from product layer | Medium | High | Research core is the gated deliverable; product-layer stories are Could/Won't-negotiable (R10) |
| Reproducibility drift (results not repeatable) | Medium | High | Deterministic seeding, pinned deps, MLflow logging of config+data hash from first experiment |
| Small team bandwidth / exam periods | Medium | Medium | Buffer in S6, parallel tracks, keep vertical slice always-green to avoid big-bang integration |

---

# PHASE 6 — CRITICAL IMPROVEMENTS

> This phase steps back from documenting the project as-scoped and asks: where can it be made materially better? These are recommendations, not requirements — each is tagged with a priority (**P0** do-this-or-results-suffer, **P1** high-value, **P2** nice-to-have) and, where relevant, whether it belongs in the graded research core or the [Assumption] product layer. The intent is to raise scientific validity, engineering quality, and product value without changing the project's identity.

## 6.1 Missing Features / Gaps Worth Closing

| # | Gap | Why it matters | Priority |
|---|---|---|---|
| 1 | **Uncertainty quantification** on each prediction (e.g., MC-dropout or deep-ensemble variance) | A malignancy probability without a confidence interval is clinically thin; "0.62 ± wide" should be flagged for human review | P1 (core) |
| 2 | **Model calibration as a first-class output** (temperature scaling + reliability curve) | A probability used for triage must be calibrated, not just discriminative; AUC says nothing about whether "0.7" means 70% | **P0 (core)** |
| 3 | **Nodule tracking across prior scans** (growth/doubling-time) | Malignancy risk is strongly driven by interval growth; single-timepoint scoring ignores the strongest real-world signal | P1 |
| 4 | **Human-in-the-loop feedback capture** (radiologist agree/disagree + correction) | Creates a labeled improvement flywheel and supports post-market monitoring | P1 |
| 5 | **Rejection / "cannot assess" pathway** for out-of-distribution or low-quality scans | Prevents confident nonsense on inputs unlike training data | **P0 (core)** |
| 6 | **Explicit operating-point selector** (high-sensitivity screening vs balanced) | One threshold does not fit both screening and follow-up contexts | P1 |
| 7 | **Report versioning & sign-off workflow** | Clinical reports need an auditable "who approved what, when" | P2 (product) |

## 6.2 Better Architecture

- **P1 —** Introduce a thin **inference-service abstraction** between the API and the model workers so Stage-1 and Stage-2 can scale independently and models can be swapped (or A/B tested) without touching the API. Already partly enabled by the MLflow hot-load design; formalize it as a versioned internal contract.
- **P1 —** Adopt an **event/queue-first pipeline** (already using Celery/Redis) but model each stage (ingest → preprocess → detect → classify → explain → report) as an explicit step with its own status in the DB, so partial progress is visible and any stage is independently retryable.
- **P2 —** Prepare a **Kubernetes + GPU node-pool** target with horizontal pod autoscaling for the classifier worker; compose is fine for the 6-month build but won't scale to real throughput.
- **P2 —** Consider **ONNX/TensorRT export** of the trained models for faster, cheaper inference and portability off the training framework.

## 6.3 Better AI / Modeling

- **P0 (core) — Rigorous, leakage-proof evaluation:** patient-level splits, external validation on a dataset not used in training (e.g., train LUNA16 / validate on an NLST subset or vice-versa), and report AUC with bootstrap CIs. External validation is the single biggest credibility multiplier for this kind of work.
- **P1 — Self-supervised / transfer pretraining** of the 3D CNN backbone on unlabeled CT (or medical-imaging foundation-model weights) to counter the Transformer's data hunger on limited labeled volumes — directly de-risks Module 6.
- **P1 — Ensemble or test-time augmentation** to squeeze the last few AUC points toward the >0.90 target and to feed the uncertainty estimate (6.1 #1).
- **P1 — Class-imbalance strategy stated and measured:** compare weighted BCE vs focal loss vs resampling empirically, not by assumption; report sensitivity at a fixed clinically meaningful specificity.
- **P2 — Multi-task learning:** jointly predict nodule attributes (spiculation, texture, size) alongside malignancy; auxiliary supervision often improves the main task and enriches explanations.
- **P2 — Compare against the Brock model head-to-head** on the same test set (not just cite its AUC ~0.74) to make the improvement claim concrete and defensible.

## 6.4 Better UX

- **P1 —** Anchor explainability in the workflow: show the **attention/Grad-CAM overlay directly on the key slice next to the score**, with a one-line plain-language rationale, rather than as a separate tab. Trust is the adoption bottleneck for clinical AI.
- **P1 —** Make the **"research use / not for diagnosis" disclaimer and the model version persistently visible** on every result and report (already in the data model; enforce in UI).
- **P2 —** Worklist **triage sorting** by malignancy probability + uncertainty so the highest-risk / most-uncertain cases surface first.
- **P2 —** **Keyboard-driven viewer** and sensible defaults (auto lung-window, auto-scroll to detected nodule) to respect radiologist time.
- **P2 —** Accessibility: WCAG-conscious color choices for overlays (avoid red/green-only encodings). Full WCAG conformance requires manual assistive-technology testing and expert review.

## 6.5 Better Database

- **P1 —** Store predictions as **immutable, append-only, model-version-stamped** rows (never overwrite) so historical results remain reproducible and auditable — critical when a model is updated.
- **P1 —** Add a **`study_status` state machine** column (received → preprocessing → detected → classified → explained → reported → failed) to make pipeline progress and stuck jobs queryable.
- **P2 —** Keep **feature/embedding caches** (e.g., VOI tensors, backbone features) in object storage keyed by content hash to avoid recomputation across experiments.
- **P2 —** Partition/index large tables (`predictions`, `attention_maps`, `audit_log`) by time for retention and performance; define a data-retention policy for de-identified imaging.

## 6.6 Performance, Scalability & Cost

- **P0 (core) —** During training: **AMP + gradient checkpointing/accumulation** (already planned) and VOI-based training to fit 3D volumes in GPU memory; this is the difference between the project training at all and not.
- **P1 —** **Cache preprocessed volumes and detector VOIs** so re-runs and experiments don't repay preprocessing cost every time.
- **P1 —** **Batch and async** inference (Celery) with a bounded queue; separate GPU workers from the CPU API so a burst of uploads never starves request handling.
- **P2 —** **Spot/preemptible GPU instances** for training and **scale-to-zero GPU workers** for inference to cut cloud cost; export to ONNX/TensorRT (6.2) for cheaper serving.
- **P2 —** Set explicit latency budgets (already in NFRs) and load-test against them before claiming throughput.

## 6.7 Security & Privacy

- **P0 —** **De-identification verification** of all imaging (strip DICOM PHI tags, burned-in-annotation check) at ingest, with the result recorded — non-negotiable for medical data.
- **P0 —** **No PHI in logs, URLs, or error messages** (reinforced from Phase 4); enforce with automated log scanning in CI.
- **P1 —** **RBAC + full audit trail** on every access to a study or prediction (already modeled); add "break-glass" access logging.
- **P1 —** **Encryption in transit (TLS) and at rest** for object storage and DB; document key management.
- **P1 —** **Data-use-agreement compliance** for NLST/LIDC recorded and enforced; access scoped to credentialed team members only.
- **P2 —** Threat-model the network-exposed API (authn/z, rate limiting, input validation on uploads) — flagged earlier as a service exposed to the network; do not ship it unauthenticated even in staging.

## 6.8 Research & Innovation Improvements

- **P0 (core) —** **External validation + calibrated, CI-reported metrics** (restated because it is the highest-leverage scientific improvement and what distinguishes a class project from a publishable result).
- **P1 —** **Ablation study** isolating the Transformer's contribution: CNN-only vs CNN+Transformer fusion, quantifying the innovation's actual lift — this is the evidence the graded contribution rests on.
- **P1 —** **Prospective-style reader study or comparison to Brock/Lung-RADS** on the same cohort to translate AUC into a clinically legible improvement.
- **P2 —** Explore **foundation-model / self-supervised pretraining** and **cross-dataset generalization** as the natural publication and future-work angle (ties to Future Enhancements).
- **P2 —** Publish a **reproducibility package** (frozen splits, configs, seeds, model card) — increasingly expected and a strong differentiator for an academic project.

## 6.9 Summary of Top Recommendations

If only a handful are actioned, prioritize the **P0** items, all in the research core: (1) **calibration as a first-class, reported output**, (2) **leakage-proof, externally-validated, CI-reported metrics**, (3) a **rejection / "cannot assess" pathway** for out-of-distribution scans, (4) **AMP + checkpointing** to make 3D training feasible, and (5) **de-identification verification + no-PHI-in-logs** for data safety. These raise scientific credibility and safety the most for the least scope change, and none of them alter the project's core identity as a hybrid CNN–Transformer lung-nodule malignancy predictor.

---

## Document Control

| Field | Value |
|---|---|
| Document | Product Requirements Document — PulmoScan AI [Assumption: product name] |
| Version | 1.0 |
| Date | 2026-07-29 |
| Status | Complete — Phases 1–6, all 28 PRD sections |
| Source basis | Major Project Synopsis (PDF, 8 pp.) + Major Project Review (PPTX, 21 slides) |
| Provenance rule | Content beyond the academic synopsis is tagged **[Assumption]** |
| Open inconsistency resolved | CUDA 11.8 (deck) vs 13.0 (synopsis) → standardized on CUDA 12.1 + PyTorch 2.2+ [Assumption] |

*End of Product Requirements Document.*





