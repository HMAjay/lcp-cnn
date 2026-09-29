# Model Architecture — Hybrid 3D CNN + Swin

Research use only. Not a medical diagnosis.

## Downloadable images

- Mermaid render: [`docs/images/hybrid_cnn_swin_architecture.png`](images/hybrid_cnn_swin_architecture.png)
- SVG: [`docs/images/hybrid_cnn_swin_architecture.svg`](images/hybrid_cnn_swin_architecture.svg)
- Report-style poster: [`docs/images/pulmoscan-hybrid-cnn-swin-architecture.png`](images/pulmoscan-hybrid-cnn-swin-architecture.png)

![Hybrid CNN + Swin architecture](images/pulmoscan-hybrid-cnn-swin-architecture.png)

## Diagram (Mermaid source)

```mermaid
flowchart TB
    subgraph INPUT["Input"]
        VOI["Nodule VOI<br/>(B, 1, 48, 48, 48)"]
    end

    subgraph CNN["3D CNN Backbone — ResNet3D18"]
        STEM["Stem: Conv3d 7×7×7 → BN → ReLU → MaxPool3d"]
        L1["Residual blocks ×2"]
        L2["Downsample + Residual blocks ×2"]
        L3["Downsample + Residual blocks ×2"]
        PROJ["1×1×1 Conv project → C channels"]
        FEAT["Feature map<br/>(B, C, D′, H′, W′)"]
        STEM --> L1 --> L2 --> L3 --> PROJ --> FEAT
    end

    subgraph SWIN["3D Swin Transformer Encoder"]
        TOK["Flatten spatial grid → tokens<br/>+ linear embed"]
        S1["Swin blocks<br/>(window + shifted-window attention)"]
        PM["Patch merging"]
        S2["Swin blocks<br/>(window + shifted-window attention)"]
        CLS["CLS query × Multi-head attention<br/>over tokens"]
        TOK --> S1 --> PM --> S2 --> CLS
    end

    subgraph HEAD["Classification Head"]
        LN["LayerNorm"]
        FC1["Linear → GELU → Dropout"]
        FC2["Linear → 1 logit"]
        SIG["Sigmoid"]
        OUT["P(malignant) ∈ [0, 1]"]
        LN --> FC1 --> FC2 --> SIG --> OUT
    end

    subgraph XAI["Explainability"]
        ATTN["CLS → token attention weights"]
        UP["Upsample to VOI grid"]
        OV["Attention overlay on mid-slice"]
        ATTN --> UP --> OV
    end

    VOI --> STEM
    FEAT --> TOK
    CLS --> LN
    CLS -.-> ATTN
```

## Pipeline (system view)

```mermaid
flowchart LR
    A["Preprocessed LIDC<br/>volumes + manifests"] --> B["labels.py<br/>malignancy ≥4 / ≤2"]
    B --> C["dataset.py<br/>crop 48³ VOI"]
    C --> D["HybridCNNSwin"]
    D --> E["P(malignant)"]
    D --> F["Attention overlay"]
```

## Code map

| Block | File |
|-------|------|
| 3D CNN | `src/pulmoscan/classification/backbone_cnn.py` |
| Swin | `src/pulmoscan/classification/swin3d.py` |
| Hybrid glue + head | `src/pulmoscan/classification/model.py` |
| Config | `configs/model/hybrid_cnn_swin.yaml` |
