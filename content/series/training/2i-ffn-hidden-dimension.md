---
title: "FFN hidden dimension: 28,672"
date: 2026-05-14
draft: false
node_id: "2i"
tier: 2
parent: "1b"
children:
  - "3a"
  - "3e"
siblings: ["2d", "2e", "2f", "2g", "2h", "2j"]
url: "/series/training/train-from-scratch/model-architecture/ffn-hidden-dimension/"
summary: "Each layer has a feed-forward network that takes the 8,192-dimensional vector, expands it to 28,672 dimensions, applies SiLU activation, then projects it back."
series: "training"
mermaid: true
weight: 215
tags: [llm-training]
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture]:::hl --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click B "/series/training/train-from-scratch/model-architecture/"
```

Each layer has a feed-forward network that takes the 8,192-dimensional vector, expands it to 28,672 dimensions, applies SiLU activation, then projects it back to 8,192. We covered [FFNs](/series/inference/what-happens/embeddings/model-layers/ffn-deep-dive/) in the LLMs series — they're the "thinking" step where the model processes each token independently after attention has mixed information between tokens.

**Why 28,672?** Convention says the FFN hidden dimension is roughly 3.5x d_model. The original transformer used 4x (d_model=512, FFN=2,048). Llama 3 uses a SwiGLU variant (SiLU + gating), which uses three projection matrices instead of two, so the effective multiplier drops to ~3.5x to keep the parameter count comparable to a standard 4x FFN. 8,192 × 3.5 = 28,672.

**Why not exactly 4x (32,768)?** The gated architecture (SwiGLU) adds a third weight matrix for the gating mechanism. If you kept the hidden dim at 4x, the FFN would be ~50% more parameters than a standard FFN. Reducing to ~3.5x keeps the FFN parameter count in line with previous architectures while getting the quality benefit of the gating mechanism.

**Where the parameters live:** The FFN is the majority of each layer's parameters. Each FFN has three matrices of size 8,192 × 28,672, totaling ~707M parameters per layer. Across 80 layers, that's ~56.5B parameters — roughly 80% of the entire model. Attention is the star of the show conceptually, but the FFN is where most of the weights live.

### Performance Profile
- **Compute dominance:** The FFN's three matrix multiplications (8,192 × 28,672 each) account for the majority of FLOP per layer. These are large, regular matmuls — exactly what GPU tensor cores are optimized for. Compute-bound, not memory-bound.
- **Parameter dominance:** ~707M params per layer × 80 layers = ~56.5B FFN parameters = ~80% of the model. Any optimization that reduces FFN cost (MoE, pruning, quantization) has outsized impact on total model cost.
- **Memory access pattern:** Three large weight matrices must be read from HBM for every token at every layer. At FP16: 3 × 8,192 × 28,672 × 2 bytes = ~1.3GB per layer. Across 80 layers, ~107GB just for FFN weights — the bulk of the 140GB model weight footprint.
- **SwiGLU cost:** The gated variant adds a third matrix (vs. standard FFN's two) but reduces hidden dim from 4x to ~3.5x. Net result: ~same total FLOP as standard FFN, marginally better quality.
