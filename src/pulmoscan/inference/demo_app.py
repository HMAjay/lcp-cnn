"""Gradio demo: score a nodule VOI (synthetic or uploaded NIfTI crop)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from pulmoscan.classification.model import build_model
from pulmoscan.common.utils import get_device, load_yaml
from pulmoscan.xai.attention import save_attention_overlay, upsample_attention

DISCLAIMER = "Research use only. Not a medical diagnosis."


def load_checkpoint(path: str | Path, device: torch.device):
    path = Path(path)
    if not path.exists():
        model = build_model(load_yaml("configs/model/hybrid_cnn_swin.yaml")).to(device)
        model.eval()
        return model, None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_cfg") or load_yaml("configs/model/hybrid_cnn_swin.yaml")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt.get("metrics")


def _random_voi(malignant_bias: bool = False, size: int = 48) -> np.ndarray:
    rng = np.random.default_rng()
    voi = rng.normal(0.25, 0.08, size=(size, size, size)).astype(np.float32)
    c = size // 2
    zz, yy, xx = np.ogrid[:size, :size, :size]
    r = 7 if malignant_bias else 5
    mask = (zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2 <= r**2
    voi[mask] = rng.normal(0.7 if malignant_bias else 0.45, 0.05, size=int(mask.sum()))
    return np.clip(voi, 0, 1)


def predict_voi(model, device, voi: np.ndarray, overlay_dir: Path) -> tuple[str, str | None]:
    x = torch.from_numpy(voi).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = float(model(x).item())
        attn = model.attention_map(x)
        attn_up = upsample_attention(attn, voi.shape).cpu().numpy()[0]
    tier = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.3 else "LOW"
    overlay = save_attention_overlay(
        voi,
        attn_up,
        overlay_dir / "demo_overlay.png",
        title=f"P(malignant)={prob:.3f}",
    )
    text = (
        f"**Maligancy probability:** {prob:.3f}\n\n"
        f"**Risk tier:** {tier}\n\n"
        f"*{DISCLAIMER}*"
    )
    return text, str(overlay)


def build_app(checkpoint: str, device_pref: str = "auto") -> gr.Blocks:
    device = get_device(device_pref)
    model, _ = load_checkpoint(checkpoint, device)
    overlay_dir = Path("./artifacts/demo")
    overlay_dir.mkdir(parents=True, exist_ok=True)

    def on_synthetic(kind: str):
        voi = _random_voi(malignant_bias=(kind == "Suspicious pattern"))
        return predict_voi(model, device, voi, overlay_dir)

    with gr.Blocks(title="PulmoScan AI") as demo:
        gr.Markdown(
            "# PulmoScan AI\n"
            "Hybrid **3D CNN + Swin Transformer** malignancy risk demo on nodule VOIs.\n\n"
            f"*{DISCLAIMER}*"
        )
        with gr.Row():
            kind = gr.Radio(
                ["Benign-like pattern", "Suspicious pattern"],
                value="Suspicious pattern",
                label="Synthetic VOI",
            )
            btn = gr.Button("Score synthetic VOI", variant="primary")
        out_md = gr.Markdown()
        out_img = gr.Image(label="Attention overlay (mid-slice)", type="filepath")
        btn.click(on_synthetic, inputs=[kind], outputs=[out_md, out_img])
        gr.Markdown(
            "### Train on Colab\n"
            "Use `notebooks/PulmoScan_Colab_Train.ipynb` with your ~3 GB processed LIDC dataset "
            "and a GPU runtime, then point this demo at the downloaded `best.pt` checkpoint."
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.environ.get("PULMOSCAN_CHECKPOINT", "./artifacts/checkpoints/best.pt"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PULMOSCAN_DEMO_PORT", "7865")))
    parser.add_argument("--device", default=os.environ.get("PULMOSCAN_DEVICE", "auto"))
    args = parser.parse_args()
    app = build_app(args.checkpoint, args.device)
    app.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
