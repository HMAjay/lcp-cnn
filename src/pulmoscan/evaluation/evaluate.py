"""Evaluation entrypoint for held-out splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pulmoscan.classification.model import build_model
from pulmoscan.classification.train import evaluate
from pulmoscan.common.utils import get_device, load_yaml
from pulmoscan.data.dataset import make_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PulmoScan checkpoint")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test", "train"])
    parser.add_argument("--data-config", default="configs/data/lidc_processed.yaml")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="./artifacts/eval_metrics.json")
    args = parser.parse_args()

    data_cfg = load_yaml(args.data_config)
    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt.get("model_cfg") or load_yaml("configs/model/hybrid_cnn_swin.yaml")
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    datasets = make_datasets(args.data_root, data_cfg)
    if args.split not in datasets:
        raise SystemExit(f"No samples for split={args.split}")
    loader = DataLoader(datasets[args.split], batch_size=args.batch_size, shuffle=False)
    metrics = evaluate(model, loader, device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"split": args.split, "checkpoint": args.checkpoint, "metrics": metrics}, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
