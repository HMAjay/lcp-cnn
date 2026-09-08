# Preprocessing — LLM Handoff Documentation

## Purpose

This document describes the preprocessing work completed for the pulmonary nodule detection project.

**Audience:** LLMs, developers, and team members who need to understand the existing preprocessing pipeline without reverse-engineering the entire codebase.

**Scope:** This document focuses on the preprocessing work and the contents of `data/processed/`. It does not describe CNN/Transformer training.

---

# 1. Dataset Overview

The current processed dataset contains:

- **25 patients**
- **25 CT scans**
- **820 reader-level nodule observations**
- **338 physical nodules**
- **30 ambiguous observations**
- **0 unmatched observations**
- **17 training patients**
- **4 validation patients**
- **4 test patients**
- **Final dataset QC: PASS**

The source is a locally downloaded subset of LIDC-IDRI data containing DICOM CT series and LIDC annotations.

The raw DICOM data is treated as immutable source data. The preprocessing pipeline creates derived artifacts under `data/processed/`.

---

# 2. Processed Data Directory

```text
data/processed/
├── volumes/
├── masks/
├── metadata/
│   ├── lidc/
│   └── physical_nodules/
├── lung_masks/
├── lung_qc/
└── manifests/
    ├── patient_splits.json
    └── dataset_manifest.json
```

Each directory has a different role.

---

# 3. `volumes/`

```text
data/processed/volumes/
└── <SeriesInstanceUID>.nii.gz
```

## Purpose

Contains the final processed 3D CT volumes.

There is one processed NIfTI volume for each CT series.

The filename is the original DICOM `SeriesInstanceUID`.

## Processing applied

The original DICOM slices were:

1. identified as CT,
2. validated for geometry,
3. sorted using physical slice position,
4. converted from stored pixel values to Hounsfield Units,
5. reconstructed into a 3D volume,
6. represented with physical image geometry,
7. resampled to **1 × 1 × 1 mm isotropic spacing**,
8. saved as compressed NIfTI (`.nii.gz`).

## Important

The original DICOM files were **not modified or overwritten**.

The NIfTI files are derived representations of the original CT series.

---

# 4. `masks/`

```text
data/processed/masks/
└── <SeriesInstanceUID>/
    ├── session_0/
    ├── session_1/
    ├── session_2/
    └── session_3/
```

## Purpose

Contains reader-specific **nodule segmentation masks** generated from LIDC polygon ROIs.

LIDC may contain multiple reading sessions for the same CT. A mask under a session represents that reader's polygon annotation.

## Polygon vs point annotations

LIDC annotations may contain:

- polygon ROIs with multiple points
- point-only ROIs

Polygon ROIs can be rasterized into masks.

Point-only ROIs are **not converted into artificial polygons**. Their location remains represented in the metadata.

## Mask geometry

Nodule masks are generated in the processed/resampled coordinate system and are intended to align with the corresponding processed CT volume.

The preprocessing validates compatible:

- dimensions
- spacing
- origin
- direction

---

# 5. `metadata/lidc/`

```text
data/processed/metadata/lidc/
└── <parsed annotation>.json
```

## Purpose

Contains parsed LIDC annotation data.

The original LIDC XML contains reader/session/nodule/ROI information. This directory stores a JSON representation so later stages do not need to repeatedly parse XML.

The metadata preserves information such as:

- series UID
- study UID
- reading sessions
- radiologist ID
- reader-specific nodule ID
- ROI slice information
- SOP Instance UID
- ROI points
- inclusion/exclusion information
- nodule characteristics

This is the reader-level annotation representation.

---

# 6. `metadata/physical_nodules/`

```text
data/processed/metadata/physical_nodules/
└── <SeriesInstanceUID>.json
```

## Purpose

Represents the result of matching multiple reader annotations into groups corresponding to the same physical nodule.

> A reader annotation is not automatically a physical nodule.

For example:

```text
Reader 0 → nodule A
Reader 1 → nodule B
Reader 2 → nodule C
```

may represent one physical nodule observed by three readers.

The physical-nodule metadata groups those observations.

## Stored information

Each physical nodule can contain:

- `physical_id`
- `confidence`
- `match_methods`
- `match_evidence`
- `num_observations`
- `representative_centroid_patient_mm`
- reader observations

Each observation preserves:

- session index
- radiologist ID
- reader-specific nodule ID
- centroid
- number of ROIs
- number of polygon ROIs
- number of point ROIs

## Matching methods

The matching implementation uses:

- spatial distance
- mask overlap

Mask-overlap evidence is available when polygon masks exist.

These are **matching methods**, not disease labels or model classes.

## Ambiguous observations

Observations for which the matching process could not confidently determine the physical nodule remain explicitly stored as:

```text
ambiguous_observations
```

They are not silently assigned to a physical nodule.

## Unmatched observations

Observations that could not be matched are explicitly stored as:

```text
unmatched_observations
```

The current dataset has **0 unmatched observations**.

---

# 7. `lung_masks/`

```text
data/processed/lung_masks/
└── <SeriesInstanceUID>.nii.gz
```

## Purpose

Contains a 3D binary lung segmentation mask for each processed CT volume.

The mask identifies the lung region and is aligned with the corresponding processed CT.

This is separate from the nodule masks.

```text
lung_masks/
    → lung region

masks/
    → individual LIDC nodule regions
```

## Lung segmentation approach

The implemented segmentation uses image intensity and morphological/connected-component processing, with an automatically determined thoracic region.

Broadly:

1. identify air-like voxels using an HU threshold,
2. estimate the body region,
3. perform morphological processing,
4. identify the thoracic region from the distribution of air,
5. restrict candidate lung regions to the thorax,
6. identify connected components,
7. retain the relevant lung components,
8. perform final morphological cleanup.

The exact implementation is in:

```text
src/sadj/preprocessing/lung_segmentation.py
```

If exact algorithmic behavior is required, inspect the source code rather than relying on this summary.

---

# 8. `lung_qc/`

```text
data/processed/lung_qc/
└── <SeriesInstanceUID>.json
```

## Purpose

Contains quality-control results for lung masks.

QC evaluates properties such as:

- lung fraction
- number of connected components
- largest-component fraction
- border fraction
- CT intensity statistics
- overall pass/fail status

The final dataset has:

```text
25 / 25 lung masks passing QC
```

These files are primarily preprocessing/QC artifacts. They are not necessarily required by the training loop.

---

# 9. `manifests/patient_splits.json`

Defines the patient-level train/validation/test split.

Current split:

```text
Train:       17 patients
Validation:   4 patients
Test:         4 patients
```

Splitting is performed at the **patient level**. A patient's data cannot appear in multiple splits.

The split contains no patient leakage and is deterministic.

---

# 10. `manifests/dataset_manifest.json`

This is the main index connecting processed artifacts.

Conceptually:

```text
Series UID
    ↓
Patient
    ↓
Train / Validation / Test
    ↓
CT volume
    ↓
Lung mask
    ↓
LIDC metadata
    ↓
Physical-nodule metadata
    ↓
Nodule masks
```

The current manifest describes:

```text
25 series
25 patients

train       17
validation   4
test         4
unassigned   0
```

Training code should preferably use the manifest rather than independently discovering files.

---

# 11. What Happened to the Original DICOM Data?

The raw DICOM data was **not modified or overwritten**.

The transformation was:

```text
Original DICOM CT series
        ↓
CT series identification
        ↓
DICOM geometry validation
        ↓
slice ordering
        ↓
HU conversion
        ↓
3D volume construction
        ↓
1 mm isotropic resampling
        ↓
NIfTI `.nii.gz`
```

The original DICOM remains the source/reference data.

The processed NIfTI volume is a standardized representation intended for downstream processing and model development.

---

# 12. DICOM Geometry Handling

A major part of preprocessing was preserving the physical relationship between voxels.

DICOM metadata used includes:

- `ImagePositionPatient`
- `ImageOrientationPatient`
- `PixelSpacing`
- `SliceThickness`
- `SOPInstanceUID`
- `SeriesInstanceUID`

Slices were sorted according to physical position rather than filenames.

The resulting volume therefore has a meaningful physical coordinate system.

---

# 13. HU Conversion

Stored DICOM pixel values were converted to Hounsfield Units using:

```text
HU = pixel_value × RescaleSlope + RescaleIntercept
```

The resulting values represent CT attenuation in HU.

---

# 14. Resampling

The original scans had different voxel spacings.

The preprocessing standardized them to:

```text
1.0 mm × 1.0 mm × 1.0 mm
```

isotropic spacing.

The resampling preserves the image's physical origin and orientation.

---

# 15. LIDC Coordinate Transformation

LIDC ROI coordinates are associated with specific DICOM slices.

The preprocessing uses `SOPInstanceUID` to identify the exact DICOM slice.

The transformation is:

```text
LIDC XML ROI
      ↓
DICOM pixel coordinate
      ↓
Patient physical coordinate (mm)
      ↓
Processed/resampled voxel coordinate
```

The system does **not** simply multiply old voxel coordinates by a scaling factor.

This is important because DICOM images have physical orientation and spacing.

---

# 16. ROI and Nodule Mask Processing

For a polygon ROI:

```text
LIDC polygon points
        ↓
2D polygon rasterization
        ↓
2D binary mask
        ↓
correct processed slice
        ↓
3D nodule mask
```

For an exclusion ROI, the excluded region is removed from the corresponding nodule mask.

For a point-only ROI:

```text
LIDC point
        ↓
preserved as point annotation
```

No artificial segmentation mask is created.

---

# 17. Reader-Level vs Physical-Level Annotations

There are three related concepts:

### Reader observation

One radiologist's annotation of a nodule.

### Physical nodule

A grouping of observations believed to represent the same physical nodule.

### Ambiguous observation

An observation for which the matching process could not confidently determine the physical nodule.

Therefore:

```text
Reader observations ≠ Physical nodules
```

Current counts:

```text
Reader observations:     820
Physical nodules:        338
Ambiguous observations:   30
Unmatched observations:    0
```

The 820 observations consist of:

```text
790 observations assigned to physical-nodule groups
30 ambiguous observations
0 unmatched observations
```

---

# 18. Polygon and Point-Only Statistics

Current dataset:

```text
Polygon observations:      294
Point-only observations:   496
```

A polygon observation can have a corresponding reader-specific nodule mask.

A point-only observation does not have a segmentation mask generated from it.

This distinction is intentional.

---

# 19. Physical Nodule Matching Statistics

Current physical-nodule confidence:

```text
High:       261
Medium:      22
Low:         55
```

Matching methods:

```text
Mask-overlap groups:       113
Spatial-distance groups:   197
```

Mask evidence:

```text
Mask-overlap evidence entries:   146
Unique mask-overlap pairs:       137
```

These numbers represent different concepts and should not be conflated.

In particular:

> 146 mask-overlap evidence entries does NOT mean there are 146 physical nodules with masks.

---

# 20. Final Dataset Statistics

```text
PROCESSED DATASET
==============================

Patients:                    25
CT scans:                    25

Reader observations:        820
Physical nodules:           338
Ambiguous observations:      30
Unmatched observations:        0

Polygon observations:       294
Point-only observations:    496

Physical nodule confidence:
    High:                   261
    Medium:                  22
    Low:                     55

Patient splits:
    Train:                   17
    Validation:               4
    Test:                     4

Lung masks:
    25 / 25 passed QC

Final dataset QC:
    PASS
```

---

# 21. Important Rules for Future LLMs / Developers

When modifying or extending the preprocessing pipeline:

1. **Do not modify raw DICOM data.**
2. Treat `data/processed/` as derived data.
3. Do not assume filenames determine DICOM slice order.
4. Use DICOM physical geometry.
5. Use SOP Instance UID to associate LIDC ROIs with exact slices.
6. Do not use naive coordinate scaling when converting annotations.
7. Preserve the distinction between reader observations and physical nodules.
8. Do not silently convert point-only annotations into segmentation masks.
9. Do not silently discard ambiguous observations.
10. Do not silently discard unmatched observations.
11. Keep patient-level train/validation/test separation.
12. Do not introduce patient leakage.
13. Preserve NIfTI dimensions, spacing, origin, and direction when generating aligned masks.
14. Do not change preprocessing parameters without understanding their effect on existing metadata and masks.
15. Prefer extending existing functions rather than duplicating preprocessing logic.
16. Run the existing tests and dataset QC after modifying preprocessing code.

---

# 22. Relevant functionality includes:

- DICOM geometry
- DICOM → HU volume loading
- CT resampling
- LIDC XML parsing
- coordinate conversion
- ROI rasterization
- nodule matching
- mask comparison
- physical-nodule metadata
- lung segmentation
- patient splitting
- dataset manifest

---

# 23. Current State

The preprocessing pipeline has been run successfully on the current 25-patient dataset.

The processed data is considered the current canonical preprocessing output.

The next stage is downstream model/data loading and training. Training-specific decisions such as patch sampling, target construction, augmentation, and CNN/Transformer architecture are outside the scope of this preprocessing documentation.

If the dataset is expanded, the same preprocessing stages should be applied consistently and final dataset QC should be rerun.
