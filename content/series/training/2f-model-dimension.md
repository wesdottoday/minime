---
title: "Model dimension (d_model): 8,192"
date: 2026-05-14
draft: false
node_id: "2f"
tier: 2
parent: "1b"
children: []
siblings:
  - "2d"
  - "2e"
  - "2g"
  - "2h"
  - "2i"
  - "2j"
url: "/series/training/train-from-scratch/model-architecture/model-dimension/"
summary: "This is the width of the highway. Every token at every layer is a vector of exactly 8,192 floating-point numbers."
series: "training"
mermaid: true
weight: 212
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture]:::hl --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click B "/series/training/train-from-scratch/model-architecture/"
```

This is the width of the highway. Every token at every layer is a vector of exactly 8,192 floating-point numbers. We covered in the [LLMs series](/series/inference/what-happens/vectors/) how these vectors encode meaning — similar concepts cluster in similar regions of this 8,192-dimensional space, directions encode relationships.

**Why 8,192?** It's a power of 2 (2^13), which matters because GPU tensor cores operate most efficiently on dimensions that are multiples of 8, 16, 64, or 128. Powers of 2 give you clean division across all of those. Beyond that, it's tied to the parameter budget: once you fix the layer count at 80, d_model is mostly determined by how many parameters you have left to spend.

**What changes if you go wider?** More capacity per layer — each token's representation can encode more features. But the compute cost of every matrix multiplication in every layer scales with d_model², so doubling the width roughly quadruples the per-layer compute. Going from 8,192 to 16,384 wouldn't just double the model — it would roughly 4x the FLOP cost per layer.

### Performance Profile
- **Compute scaling:** Matrix multiplications in attention and FFN scale with d_model². Doubling d_model from 8,192 to 16,384 ≈ 4x FLOP cost per layer. This is the single biggest lever on training and inference compute cost.
- **Memory bandwidth:** Every layer reads and writes vectors of size d_model for every token. At 8,192 × 2 bytes (FP16) = 16KB per token per layer. Across 80 layers and a batch of sequences, this determines whether operations are compute-bound or memory-bandwidth-bound.
- **Tensor core alignment:** 8,192 = 2^13. GPU tensor cores operate most efficiently on dimensions that are multiples of 8, 16, 64, or 128. Powers of 2 guarantee clean alignment at every level, avoiding wasted cycles on padding.
