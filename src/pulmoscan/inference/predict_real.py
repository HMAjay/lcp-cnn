"""Run inference on real preprocessed LIDC VOIs and save attention overlays."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pulmoscan.classification.model import build_model
from pulmoscan.common.utils import ensure_dir, get_device, load_yaml
from pulmoscan.data.dataset import make_datasets
from pulmoscan.xai.attention import save_attention_overlay, upsample_attention


def main() -> None:
    parser = argparse.ArgumentParser(description="Score real processed LIDC nodules")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--data-config", default="configs/data/lidc_processed.yaml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="./artifacts/real_inference")
    args = parser.parse_args()

    device = get_device(args.device)
    data_cfg = load_yaml(args.data_config)
    datasets = make_datasets(args.data_root, data_cfg)
    if args.split not in datasets:
        raise SystemExit(f"No samples for split={args.split}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt.get("model_cfg") or load_yaml("configs/model/hybrid_cnn_swin.yaml")
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = ensure_dir(args.out_dir)
    overlay_dir = ensure_dir(out_dir / "overlays")
    loader = DataLoader(datasets[args.split], batch_size=args.batch_size, shuffle=False)

    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            probs = model(x).detach().float().cpu().numpy()
            attn = model.attention_map(x)
            attn_up = upsample_attention(attn, tuple(x.shape[-3:])).cpu().numpy()

            for i in range(x.shape[0]):
                voi = x[i, 0].cpu().numpy()
                prob = float(probs[i])
                label = int(batch["label"][i].item())
                series = str(batch["series_uid"][i])
                pid = str(batch["physical_id"][i])
                patient = str(batch["patient_id"][i])
                truth = "malignant" if label == 1 else "benign"
                tier = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.3 else "LOW"
                pred = "malignant" if prob >= 0.5 else "benign"
                correct = pred == truth

                overlay_name = f"{patient}_{pid}_{prob:.3f}.png".replace("/", "_")
                overlay_path = overlay_dir / overlay_name
                save_attention_overlay(
                    voi,
                    attn_up[i],
                    overlay_path,
                    title=f"{patient} {pid} P={prob:.3f} truth={truth}",
                )

                row = {
                    "patient_id": patient,
                    "series_uid": series,
                    "physical_id": pid,
                    "split": args.split,
                    "truth_label": label,
                    "truth": truth,
                    "malignancy_prob": prob,
                    "risk_tier": tier,
                    "pred_at_0.5": pred,
                    "correct_at_0.5": correct,
                    "malignancy_mean_lidc": float(batch["malignancy_mean"][i]),
                    "overlay": str(overlay_path),
                }
                rows.append(row)
                print(
                    f"{patient} {pid}: truth={truth}  P(mal)={prob:.3f}  "
                    f"tier={tier}  pred={pred}  correct={correct}"
                )

    csv_path = out_dir / f"{args.split}_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["patient_id"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "split": args.split,
        "n": len(rows),
        "n_correct_at_0.5": sum(1 for r in rows if r["correct_at_0.5"]),
        "accuracy_at_0.5": (
            float(np.mean([r["correct_at_0.5"] for r in rows])) if rows else 0.0
        ),
        "checkpoint": args.checkpoint,
        "csv": str(csv_path),
        "overlays_dir": str(overlay_dir),
        "disclaimer": "Research use only. Not a medical diagnosis.",
    }
    summary_path = out_dir / f"{args.split}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
