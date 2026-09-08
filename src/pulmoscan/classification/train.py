"""Training loop for Hybrid CNN–Swin malignancy classifier."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.amp.autocast("cpu", enabled=False)

from pulmoscan.classification.losses import build_loss
from pulmoscan.classification.model import build_model
from pulmoscan.common.utils import ensure_dir, get_device, load_yaml, set_seed
from pulmoscan.data.dataset import make_datasets
from pulmoscan.data.labels import summarize_samples
from pulmoscan.evaluation.metrics import compute_binary_metrics


def _estimate_pos_weight(loader: DataLoader) -> float:
    n_pos = 0
    n = 0
    for batch in loader:
        y = batch["label"]
        n_pos += int((y > 0.5).sum().item())
        n += int(y.numel())
    n_neg = max(n - n_pos, 1)
    return float(n_neg / max(n_pos, 1))


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    probs: list[float] = []
    labels: list[float] = []
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].cpu().numpy().tolist()
        with _autocast(device, enabled=device.type == "cuda"):
            p = model(x).detach().float().cpu().numpy().tolist()
        probs.extend(p if isinstance(p, list) else [p])
        labels.extend(y if isinstance(y, list) else [y])
    return compute_binary_metrics(labels, probs)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | torch.amp.GradScaler | None,
    grad_accum: int,
    max_grad_norm: float,
) -> float:
    model.train()
    total = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)
    use_amp = scaler is not None and device.type == "cuda"
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        with _autocast(device, enabled=use_amp):
            logits = model(x, return_logits=True)
            loss = loss_fn(logits, y) / grad_accum
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step + 1) % grad_accum == 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total += float(loss.item()) * grad_accum * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def run_training(
    data_root: str | Path,
    model_cfg_path: str | Path = "configs/model/hybrid_cnn_swin.yaml",
    data_cfg_path: str | Path = "configs/data/lidc_processed.yaml",
    train_cfg_path: str | Path = "configs/train/default.yaml",
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    device_pref: str = "auto",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    model_cfg = load_yaml(model_cfg_path)
    data_cfg = load_yaml(data_cfg_path)
    train_cfg = load_yaml(train_cfg_path)

    set_seed(int(train_cfg.get("seed", 42)))
    device = get_device(device_pref)

    if epochs is not None:
        train_cfg["epochs"] = epochs
    if batch_size is not None:
        train_cfg["batch_size"] = batch_size

    datasets = make_datasets(data_root, data_cfg)
    if "train" not in datasets:
        raise RuntimeError(f"No train samples under {data_root}")

    summary = {
        split: {
            "n": len(ds),
            "malignant": sum(1 for s in ds.samples if s.label == 1),
            "benign": sum(1 for s in ds.samples if s.label == 0),
        }
        for split, ds in datasets.items()
    }
    print("Dataset summary:", json.dumps(summary, indent=2))

    bs = int(train_cfg.get("batch_size", 4))
    nw = int(data_cfg.get("num_workers", 2))
    train_loader = DataLoader(
        datasets["train"],
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if "validation" in datasets:
        val_loader = DataLoader(
            datasets["validation"],
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=device.type == "cuda",
        )

    model = build_model(model_cfg).to(device)
    pos_weight = train_cfg.get("loss", {}).get("pos_weight")
    if pos_weight is None:
        pos_weight = _estimate_pos_weight(train_loader)
    loss_fn = build_loss(
        train_cfg.get("loss", {}).get("name", "weighted_bce"),
        pos_weight=float(pos_weight),
        gamma=float(train_cfg.get("loss", {}).get("focal_gamma", 2.0)),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    n_epochs = int(train_cfg.get("epochs", 30))
    warmup = int(train_cfg.get("warmup_epochs", 2))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return float(epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(n_epochs - warmup, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_cuda_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    out_dir = ensure_dir(output_dir or train_cfg.get("checkpoint_dir", "./artifacts/checkpoints"))
    best_auc = -1.0
    best_path = out_dir / "best.pt"
    history: list[dict[str, Any]] = []
    patience = int(train_cfg.get("early_stopping_patience", 8))
    stale = 0

    for epoch in range(n_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            scaler if scaler.is_enabled() else None,
            grad_accum=int(train_cfg.get("grad_accum_steps", 1)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        )
        scheduler.step()
        row: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss, "seconds": time.time() - t0}
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device)
            row.update({f"val_{k}": v for k, v in metrics.items()})
            auc = float(metrics.get("auc", 0.0))
            print(
                f"epoch {epoch+1}/{n_epochs} loss={train_loss:.4f} "
                f"val_auc={auc:.4f} val_acc={metrics.get('accuracy', 0):.3f}"
            )
            if auc > best_auc:
                best_auc = auc
                stale = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_cfg": model_cfg,
                        "epoch": epoch,
                        "metrics": metrics,
                        "data_summary": summary,
                    },
                    best_path,
                )
            else:
                stale += 1
                if stale >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    history.append(row)
                    break
        else:
            print(f"epoch {epoch+1}/{n_epochs} loss={train_loss:.4f}")
            torch.save(
                {"model_state": model.state_dict(), "model_cfg": model_cfg, "epoch": epoch},
                best_path,
            )
        history.append(row)
        if (epoch + 1) % int(train_cfg.get("save_every", 5)) == 0:
            torch.save(
                {"model_state": model.state_dict(), "model_cfg": model_cfg, "epoch": epoch},
                out_dir / f"epoch_{epoch+1:03d}.pt",
            )

    hist_path = out_dir / "history.json"
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {"best_path": str(best_path), "best_auc": best_auc, "history": history, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PulmoScan hybrid CNN–Swin")
    parser.add_argument("--data-root", default=None, help="Processed LIDC root (or synthetic)")
    parser.add_argument("--model-config", default="configs/model/hybrid_cnn_swin.yaml")
    parser.add_argument("--data-config", default="configs/data/lidc_processed.yaml")
    parser.add_argument("--train-config", default="configs/train/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="./artifacts/checkpoints")
    args = parser.parse_args()

    data_root = args.data_root
    if data_root is None:
        import os

        data_root = os.environ.get("PULMOSCAN_DATA_ROOT", "./data/processed")

    result = run_training(
        data_root,
        args.model_config,
        args.data_config,
        args.train_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device_pref=args.device,
        output_dir=args.output_dir,
    )
    print(json.dumps({k: result[k] for k in ("best_path", "best_auc", "summary")}, indent=2))


if __name__ == "__main__":
    main()
