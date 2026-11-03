# Verification and benchmark analyses

These are the scripts and results behind the geometry verification, the size benchmark, the added-value analysis and the centering-tolerance analysis. They use only public LIDC-IDRI data (TCIA, https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX, CC BY 3.0), synthetic phantoms, and predictions from the final Stage 2 checkpoint. Run them from this folder with the package installed or with `PYTHONPATH=../src`.

| Folder or file | What it holds |
|---|---|
| `phantom/` | 220 synthetic DICOM CT phantoms with LIDC-style reader XML. Both the corrected mapping in `src/pulmoscan/data/preprocess_lidc.py` and the defective mapping that preceded commit d87e6d0 (`preprocess_lidc_defective.py`, from `git show d87e6d0^:src/pulmoscan/data/preprocess_lidc.py`) run on identical inputs. Per-nodule results are in `results/phantom_results.csv`. |
| `lidc_audit/` | Both mappings applied to every LIDC-IDRI annotation (`audit_xml.py` → `nodules.csv`, using one header per series from `fetch_headers.py` → `headers.json`). Also: the real-CT check against each annotated slice's own header (`audit_images.py`, `image_summary.py`); reconstruction of the 153-patient cohort and its split (`find_split.py`, `cohort_153.py`); and the diameter-only benchmark (`analyze_cohort.py`, `benchmark_extras.py`, `size_bands.py`). |
| `added_value/` | Output of `../tolerance_study/added_value.py` on the final checkpoint. `run_output.txt` is the printed output; `added_value.json` is transcribed from it by `transcribe.py`, at 3 decimals (P values at 4). |
| `tolerance/` | `tolerance_summary.json`, written by `../tolerance_study/analyze_tolerance.py` from the predictions of `../tolerance_study/shift_eval.py` on the final checkpoint. |
| `stats_from_counts.py`, `stats.json` | Confidence intervals for the recorded test metrics, derived from their counts. |
| `phantom_summary.py`, `phantom_summary.json` | Summary of the phantom results. |
| `figures.py` | Paper figures, written to `figures_out/` (needs Arial). |

## Rerunning

Everything below runs on a CPU, without patient images:

```bash
python phantom/phantom_geometry_validation.py --repo .. --out phantom/results   # about 10 minutes
python phantom_summary.py phantom/results/phantom_results.csv phantom_summary.json
python lidc_audit/cohort_153.py lidc_audit/nodules.csv lidc_audit/headers.json lidc_audit/cohort153_summary.json
python lidc_audit/benchmark_extras.py lidc_audit/cohort153_summary.json lidc_audit/benchmark_extras.json
python lidc_audit/size_bands.py lidc_audit/nodules.csv lidc_audit/headers.json lidc_audit/size_bands.json
python figures.py
```

Some steps need data that isn't stored here:
- `audit_xml.py` needs the LIDC-IDRI XML files.
- `fetch_headers.py` and `audit_images.py` download from the public TCIA API.
- `../tolerance_study/shift_eval.py` needs the processed volumes and the trained checkpoint. The checkpoint is not in the repository.

The per-nodule predictions from that run (`tolerance_preds.csv`) are not included yet.

`lidc_audit/real_example.npz` holds one de-identified LIDC-IDRI slice (LIDC-IDRI-0568), used under CC BY 3.0 for the real-CT figure.
