#!/usr/bin/env python3
"""Local (non-Colab) end-to-end runner for PulmoScan / LCP-CNN.

Examples:
  # verify labels
  python scripts/run_local.py verify --data-root /path/to/processed

  # train
  python scripts/run_local.py train --data-root /path/to/processed --output-dir ./artifacts/checkpoints_v2

  # evaluate + optional download-style copy
  python scripts/run_local.py eval --data-root /path/to/processed --checkpoint ./artifacts/checkpoints_v2/best.pt

  # real test inference + overlays
  python scripts/run_local.py predict --data-root /path/to/processed --checkpoint ./artifacts/checkpoints_v2/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_path() -> None:
    root = _repo_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def cmd_verify(args: argparse.Namespace) -> None:
    _ensure_src_path()
    from pulmoscan.data.labels import build_samples, summarize_samples

    samples, stats = build_samples(args.data_root, return_stats=True)
    print(json.dumps(stats, indent=2))
    print(json.dumps(summarize_samples(samples), indent=2))
    print("n=", len(samples))
    if len(samples) == 0:
        raise SystemExit("No labeled samples. Check metadata/lidc linking and manifests.")
    if not any(s.split == "train" for s in samples):
        raise SystemExit("No train-split samples.")


def cmd_build_manifests(args: argparse.Namespace) -> None:
    _ensure_src_path()
    sys.path.insert(0, str(_repo_root() / "scripts"))
    from build_manifests_from_processed import build_manifests

    info = build_manifests(args.data_root, seed=args.seed)
    print(json.dumps(info, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", device)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pulmoscan.classification.train",
        "--data-root",
        args.data_root,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        device,
        "--output-dir",
        str(out),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(_repo_root()), env={**os.environ, "PYTHONPATH": str(_repo_root() / "src")})


def cmd_eval(args: argparse.Namespace) -> None:
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", device)

    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pulmoscan.evaluation.evaluate",
        "--data-root",
        args.data_root,
        "--checkpoint",
        args.checkpoint,
        "--split",
        args.split,
        "--batch-size",
        str(args.batch_size),
        "--device",
        device,
        "--out",
        str(out_json),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(_repo_root()), env={**os.environ, "PYTHONPATH": str(_repo_root() / "src")})
    print(Path(out_json).read_text())

    if args.copy_to:
        dest = Path(args.copy_to)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.checkpoint, dest / Path(args.checkpoint).name)
        shutil.copy2(out_json, dest / out_json.name)
        print("Copied checkpoint + metrics to", dest)


def cmd_predict(args: argparse.Namespace) -> None:
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pulmoscan.inference.predict_real",
        "--data-root",
        args.data_root,
        "--checkpoint",
        args.checkpoint,
        "--split",
        args.split,
        "--batch-size",
        "1",
        "--device",
        device,
        "--out-dir",
        str(out_dir),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(_repo_root()), env={**os.environ, "PYTHONPATH": str(_repo_root() / "src")})


def cmd_demo(args: argparse.Namespace) -> None:
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    cmd = [
        sys.executable,
        "-m",
        "pulmoscan.inference.demo_app",
        "--checkpoint",
        args.checkpoint,
        "--device",
        device,
        "--port",
        str(args.port),
        "--host",
        "127.0.0.1",
    ]
    print("Running:", " ".join(cmd))
    print("Open http://127.0.0.1:%d" % args.port)
    subprocess.check_call(cmd, cwd=str(_repo_root()), env={**os.environ, "PYTHONPATH": str(_repo_root() / "src")})


def main() -> None:
    parser = argparse.ArgumentParser(description="Local PulmoScan runner (no Colab)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("verify", help="Count labeled samples")
    p.add_argument("--data-root", required=True)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("build-manifests", help="Create manifests if missing")
    p.add_argument("--data-root", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_build_manifests)

    p = sub.add_parser("train", help="Train hybrid CNN+Swin")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", default="./artifacts/checkpoints_v2")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="Evaluate checkpoint")
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out", default="./artifacts/checkpoints_v2/test_metrics.json")
    p.add_argument("--copy-to", default="", help="Optional folder to copy best.pt + metrics")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("predict", help="Per-nodule predictions + overlays on real data")
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-dir", default="./artifacts/real_inference")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("demo", help="Gradio synthetic demo")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--port", type=int, default=7865)
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
