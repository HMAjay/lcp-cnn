"""Training loop for Hybrid CNN–Swin malignancy classifier."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pulmoscan.classification.losses import build_loss
from pulmoscan.classification.model import TGHybridCNNSwin, build_model
from pulmoscan.common.utils import ensure_dir, get_device, load_yaml, set_seed
from pulmoscan.data.dataset import make_datasets
from pulmoscan.evaluation.metrics import compute_binary_metrics


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda":
        try:
            return torch.amp.autocast("cuda", enabled=enabled)
        except (TypeError, AttributeError):
            return torch.cuda.amp.autocast(enabled=enabled)
    try:
        return torch.amp.autocast("cpu", enabled=False)
    except (TypeError, AttributeError):
        return torch.cuda.amp.autocast(enabled=False)


def _configure_runtime(train_cfg: dict[str, Any], device: torch.device) -> None:
    """Keep startup lean: few CPU threads, no DataLoader worker farms."""
    nthreads = int(train_cfg.get("torch_num_threads", 2))
    torch.set_num_threads(max(1, nthreads))
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, nthreads)))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, nthreads)))
    # Avoid fork/worker storms on Colab / 4GB hosts
    try:
        torch.multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()


def _estimate_pos_weight_from_dataset(dataset) -> float:
    """Compute pos_weight from definite labels only (no image I/O / DataLoader workers)."""
    n_pos = sum(1 for s in dataset.samples if int(s.label) == 1)
    n_neg = sum(1 for s in dataset.samples if int(s.label) == 0)
    return float(max(n_neg, 1) / max(n_pos, 1))


def _majority_baseline(dataset) -> float:
    n_pos = sum(1 for s in dataset.samples if int(s.label) == 1)
    n_neg = sum(1 for s in dataset.samples if int(s.label) == 0)
    n = max(n_pos + n_neg, 1)
    return float(max(n_pos, n_neg) / n)


class ModelEma:
    def __init__(self, model: torch.nn.Module, decay: float = 0.998) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            if not v.is_floating_point():
                self.shadow[k] = v.detach().clone()
                continue
            self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


def _mixup_batch(batch: dict[str, Any], alpha: float) -> dict[str, Any]:
    x = batch["image"]
    if alpha <= 0.0 or x.size(0) < 2:
        return batch
    # Mixing an uncertain nodule into a definite one corrupts the malignancy target.
    unc = batch.get("uncertain")
    if unc is not None and torch.is_tensor(unc) and float(unc.max()) > 0.5:
        return batch
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = dict(batch)
    mixed["image"] = lam * x + (1.0 - lam) * x[idx]
    for key in (
        "label",
        "malignancy_target",
        "malignancy_weight",
        "eval_mask",
        "concepts",
        "attr_mask",
        "subsolid",
        "uncertain",
    ):
        if key in batch and torch.is_tensor(batch[key]) and batch[key].is_floating_point():
            mixed[key] = lam * batch[key] + (1.0 - lam) * batch[key][idx]
    return mixed


def _balanced_sampler(dataset, uncertain_weight: float = 0.25):
    from torch.utils.data import WeightedRandomSampler

    weights: list[float] = []
    n_pos = max(sum(1 for s in dataset.samples if int(s.label) == 1), 1)
    n_neg = max(sum(1 for s in dataset.samples if int(s.label) == 0), 1)
    n_unc = max(sum(1 for s in dataset.samples if int(s.label) == -1), 1)
    unc_w = max(float(uncertain_weight), 0.0)
    for s in dataset.samples:
        if int(s.label) == 1:
            weights.append(1.0 / n_pos)
        elif int(s.label) == 0:
            weights.append(1.0 / n_neg)
        else:
            weights.append(unc_w / n_unc)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key, val in batch.items():
        if torch.is_tensor(val):
            out[key] = val.to(device, non_blocking=False)
    return out


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    tta: bool = False,
) -> dict[str, float]:
    model.eval()
    probs: list[float] = []
    labels: list[float] = []
    abstain_probs: list[float] = []
    uncertain: list[float] = []
    use_aux = isinstance(model, TGHybridCNNSwin)
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"]
        w = batch.get("malignancy_weight")
        with _autocast(device, enabled=device.type == "cuda"):
            if use_aux:
                aux = model(x, return_aux=True)
                assert isinstance(aux, dict)
                p = aux["prob"]
                if tta:
                    aux_f = model(torch.flip(x, dims=[-1]), return_aux=True)
                    assert isinstance(aux_f, dict)
                    p = 0.5 * (p + aux_f["prob"])
                a = torch.sigmoid(aux["abstain_logits"])
            else:
                p = model(x)
                if tta:
                    p = 0.5 * (p + model(torch.flip(x, dims=[-1])))
                a = None
        p_list = p.detach().float().cpu().numpy().reshape(-1)
        y_list = y.detach().float().cpu().numpy().reshape(-1)
        ev = batch.get("eval_mask")
        u = batch.get("uncertain")
        if ev is not None:
            keep = ev.detach().float().cpu().numpy().reshape(-1) > 0.5
        elif u is not None:
            keep = u.detach().float().cpu().numpy().reshape(-1) < 0.5
        elif w is None:
            keep = np.ones(len(y_list), dtype=bool)
        else:
            keep = w.detach().float().cpu().numpy().reshape(-1) > 0.5
        probs.extend(p_list[keep].tolist())
        labels.extend(y_list[keep].tolist())
        if a is not None:
            abstain_probs.extend(a.detach().float().cpu().numpy().reshape(-1).tolist())
            u = batch.get("uncertain")
            if u is not None:
                uncertain.extend(u.detach().float().cpu().numpy().reshape(-1).tolist())

    metrics = compute_binary_metrics(labels, probs)
    metrics["n_eval"] = float(len(labels))
    thr = float(metrics.get("best_threshold", 0.5))
    tuned = compute_binary_metrics(labels, probs, threshold=thr)
    metrics["accuracy_tuned"] = tuned["accuracy"]
    metrics["sensitivity_tuned"] = tuned["sensitivity"]
    metrics["specificity_tuned"] = tuned["specificity"]
    if abstain_probs and uncertain:
        pred_abs = (np.asarray(abstain_probs) >= 0.5).astype(np.int32)
        u_int = np.asarray(uncertain) >= 0.5
        metrics["abstain_accuracy"] = float((pred_abs == u_int.astype(np.int32)).mean())
    return metrics


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | torch.amp.GradScaler | None,
    grad_accum: int,
    max_grad_norm: float,
    *,
    empty_cache_each_step: bool = False,
    mixup_alpha: float = 0.0,
    ema: ModelEma | None = None,
) -> float:
    model.train()
    total = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)
    use_amp = scaler is not None and device.type == "cuda"
    use_aux = isinstance(model, TGHybridCNNSwin)
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        batch = _move_batch(batch, device)
        batch = _mixup_batch(batch, mixup_alpha)
        x = batch["image"]
        y = batch["label"]
        with _autocast(device, enabled=use_amp):
            if use_aux:
                aux = model(x, return_aux=True)
                assert isinstance(aux, dict)
                loss = loss_fn(aux, batch) / grad_accum
            else:
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
            if ema is not None:
                ema.update(model)
            if empty_cache_each_step and device.type == "cuda":
                torch.cuda.empty_cache()
        total += float(loss.item()) * grad_accum * x.size(0)
        n += x.size(0)
        del x, y, loss
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
    tg_hybrid: bool | None = None,
) -> dict[str, Any]:
    model_cfg = load_yaml(model_cfg_path)
    data_cfg = load_yaml(data_cfg_path)
    train_cfg = load_yaml(train_cfg_path)

    set_seed(int(train_cfg.get("seed", 42)))
    device = get_device(device_pref)
    _configure_runtime(train_cfg, device)
    print(
        f"Runtime: device={device}, cuda_mem_gb="
        f"{(torch.cuda.get_device_properties(0).total_memory/1e9):.2f}"
        if device.type == "cuda"
        else f"Runtime: device={device}"
    )

    if epochs is not None:
        train_cfg["epochs"] = epochs
    if batch_size is not None:
        train_cfg["batch_size"] = batch_size

    # Align model declared VOI with data crop size when present
    if "voi_size" in data_cfg:
        model_cfg["voi_size"] = data_cfg["voi_size"]

    if tg_hybrid is not None:
        model_cfg["tg_hybrid"] = bool(tg_hybrid)
    use_tg = bool(model_cfg.get("tg_hybrid", False))
    labeling = data_cfg.setdefault("labeling", {})
    if use_tg:
        labeling.setdefault("include_uncertain", True)

    print("Building datasets (metadata only; volumes load lazily per sample)...")
    datasets = make_datasets(data_root, data_cfg)
    if "train" not in datasets:
        # Print label-build stats to help debug empty datasets
        try:
            from pulmoscan.data.labels import build_samples

            labeling = data_cfg.get("labeling", {})
            _, stats = build_samples(
                data_root,
                malignant_min=float(labeling.get("malignant_min", 4.0)),
                benign_max=float(labeling.get("benign_max", 2.0)),
                min_confidence=labeling.get("min_confidence", "low"),
                use_physical_nodules=bool(labeling.get("use_physical_nodules", True)),
                include_uncertain=bool(labeling.get("include_uncertain", False)),
                return_stats=True,
            )
            print("Label build stats:", json.dumps(stats, indent=2))
        except Exception as exc:  # noqa: BLE001
            print("Could not collect label stats:", exc)
        raise RuntimeError(f"No train samples under {data_root}")

    summary = {
        split: {
            "n": len(ds),
            "malignant": sum(1 for s in ds.samples if s.label == 1),
            "benign": sum(1 for s in ds.samples if s.label == 0),
            "uncertain": sum(1 for s in ds.samples if s.label == -1),
            "subsolid": sum(1 for s in ds.samples if s.subsolid == 1),
            "majority_baseline_acc": _majority_baseline(ds),
        }
        for split, ds in datasets.items()
    }
    print("Dataset summary:", json.dumps(summary, indent=2))
    print(
        "Note: ~0.65 accuracy at threshold 0.5 often matches the majority-class "
        "baseline; use val_auc and val_accuracy_tuned (Youden) as the real metrics."
    )

    bs = int(train_cfg.get("batch_size", 1))
    # Hard-cap workers: lazy single-process load (no thread/process farm at start)
    nw = int(data_cfg.get("num_workers", 0))
    if nw != 0:
        print(f"Warning: forcing num_workers 0 (was {nw}) for lazy 4GB-safe loading")
        nw = 0
    loader_kwargs: dict[str, Any] = {
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    }
    use_balanced = bool(train_cfg.get("balanced_sampler", True))
    train_loader_kwargs = dict(loader_kwargs)
    if use_balanced:
        train_loader_kwargs["sampler"] = _balanced_sampler(
            datasets["train"],
            uncertain_weight=float(train_cfg.get("uncertain_sample_weight", 0.25)),
        )
        train_loader_kwargs["shuffle"] = False
    else:
        train_loader_kwargs["shuffle"] = True
    train_loader = DataLoader(
        datasets["train"],
        batch_size=bs,
        **train_loader_kwargs,
    )
    val_loader = None
    if "validation" in datasets:
        val_loader = DataLoader(
            datasets["validation"],
            batch_size=bs,
            shuffle=False,
            **loader_kwargs,
        )

    print("Building model...")
    model = build_model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_params/1e6:.2f}M  tg_hybrid={use_tg}")
    pos_weight = train_cfg.get("loss", {}).get("pos_weight")
    if pos_weight is None:
        pos_weight = _estimate_pos_weight_from_dataset(datasets["train"])
    loss_cfg = train_cfg.get("loss", {})
    loss_name = str(loss_cfg.get("name", "weighted_bce"))
    if use_tg and loss_name in {"weighted_bce", "tg_hybrid", "tg-hybrid"}:
        loss_name = "tg_hybrid"
    elif not use_tg and loss_name in {"tg_hybrid", "tg-hybrid"}:
        loss_name = "weighted_bce"
    tg_kwargs = {
        k: float(loss_cfg[k])
        for k in (
            "lambda_mal",
            "lambda_concept",
            "lambda_gate",
            "lambda_expert",
            "lambda_abstain",
        )
        if k in loss_cfg
    }
    loss_fn = build_loss(
        loss_name,
        pos_weight=float(pos_weight),
        gamma=float(loss_cfg.get("focal_gamma", 2.0)),
        **tg_kwargs,
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
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)
    mixup_alpha = float(train_cfg.get("mixup_alpha", 0.2 if bs >= 2 else 0.0))
    ema = ModelEma(model, decay=float(train_cfg.get("ema_decay", 0.998))) if bool(
        train_cfg.get("ema", True)
    ) else None

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
            empty_cache_each_step=bool(train_cfg.get("empty_cache_each_step", False)),
            mixup_alpha=mixup_alpha,
            ema=ema,
        )
        scheduler.step()
        row: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss, "seconds": time.time() - t0}
        if val_loader is not None:
            backup = None
            if ema is not None:
                backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
                ema.copy_to(model)
            metrics = evaluate(
                model,
                val_loader,
                device,
                tta=bool(train_cfg.get("tta", False)),
            )
            if backup is not None:
                model.load_state_dict(backup, strict=True)
            row.update({f"val_{k}": v for k, v in metrics.items()})
            auc = float(metrics.get("auc", 0.0))
            print(
                f"epoch {epoch+1}/{n_epochs} loss={train_loss:.4f} "
                f"val_auc={auc:.4f} val_acc@0.5={metrics.get('accuracy', 0):.3f} "
                f"val_acc_tuned={metrics.get('accuracy_tuned', 0):.3f} "
                f"n_eval={int(metrics.get('n_eval', 0))}"
            )
            if auc > best_auc:
                best_auc = auc
                stale = 0
                state = ema.shadow if ema is not None else model.state_dict()
                torch.save(
                    {
                        "model_state": state,
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
        if bool(train_cfg.get("empty_cache_each_epoch", True)) and device.type == "cuda":
            torch.cuda.empty_cache()

    hist_path = out_dir / "history.json"
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {"best_path": str(best_path), "best_auc": best_auc, "history": history, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PulmoScan hybrid CNN–Swin")
    parser.add_argument("--data-root", default=None, help="Processed LIDC root (or synthetic)")
    parser.add_argument("--model-config", default="configs/model/hybrid_cnn_swin_4gb.yaml")
    parser.add_argument("--data-config", default="configs/data/lidc_4gb.yaml")
    parser.add_argument("--train-config", default="configs/train/colab_4gb.yaml")
    parser.add_argument(
        "--preset",
        choices=["4gb", "default"],
        default="4gb",
        help="Config preset (4gb = lazy load + small model for <=4GB VRAM)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="./artifacts/checkpoints_4gb")
    parser.add_argument(
        "--tg-hybrid",
        dest="tg_hybrid",
        action="store_true",
        default=None,
        help="Enable TG-Hybrid heads (default: value in model yaml)",
    )
    parser.add_argument(
        "--no-tg-hybrid",
        dest="tg_hybrid",
        action="store_false",
        help="Disable TG-Hybrid heads",
    )
    args = parser.parse_args()

    if args.preset == "default":
        if args.model_config.endswith("hybrid_cnn_swin_4gb.yaml"):
            args.model_config = "configs/model/hybrid_cnn_swin.yaml"
        if args.data_config.endswith("lidc_4gb.yaml"):
            args.data_config = "configs/data/lidc_processed.yaml"
        if args.train_config.endswith("colab_4gb.yaml"):
            args.train_config = "configs/train/default.yaml"

    data_root = args.data_root
    if data_root is None:
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
        tg_hybrid=args.tg_hybrid,
    )
    print(json.dumps({k: result[k] for k in ("best_path", "best_auc", "summary")}, indent=2))


if __name__ == "__main__":
    main()
