---
title: "What is the model architecture?"
date: 2026-05-14
draft: false
node_id: "1b"
tier: 1
parent: "0"
children:
  - "2d"
  - "2e"
  - "2f"
  - "2g"
  - "2h"
  - "2i"
  - "2j"
siblings:
  - "1a"
  - "1c"
  - "1d"
  - "1e"
  - "1f"
  - "1g"
url: "/series/training/train-from-scratch/model-architecture/"
summary: "A model architecture is a blueprint — it defines how information flows from input tokens to output predictions, without saying anything about what the model knows."
series: "training"
mermaid: true
weight: 110
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture]:::hl --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click A "/series/training/train-from-scratch/training-data/"
    click B "/series/training/train-from-scratch/model-architecture/"
    click C "/series/training/train-from-scratch/hardware-and-scale/"
    click D "/series/training/train-from-scratch/training-step/"
    click E "/series/training/train-from-scratch/training-loop/"
    click F "/series/training/train-from-scratch/post-training/"
    click G "/series/training/train-from-scratch/evaluation/"
```

A model architecture is a blueprint — it defines how information flows from input tokens to output predictions, without saying anything about what the model knows. The knowledge comes from training. The architecture just defines the shape of the container.

Llama 3 70B is a decoder-only transformer. That means it's a stack of identical layers, each containing the same two operations: an attention mechanism (which lets every token look at every token that came before it) and a feed-forward network (which processes each token's representation independently). Llama 3 70B has 80 of these layers stacked on top of each other. A token enters at the bottom as a raw embedding vector, and by the time it exits the top, it's been transformed 80 times — each layer refining what the model "thinks" should come next.

The key architectural choices that define Llama 3 70B:

- **Vocabulary size:** 128,000 tokens — the lookup table that maps token IDs to embedding vectors
- **Model dimension (d_model):** 8,192 — every token is represented as a vector of 8,192 numbers throughout the entire model
- **Layers:** 80 — the depth of the stack
- **Attention heads:** 64 — each layer's attention mechanism runs 64 parallel attention computations, each focusing on a different 128-dimensional slice of the 8,192-dimensional vector
- **Key-value heads:** 8 — Llama 3 uses Grouped Query Attention (GQA), where 8 groups of 8 attention heads share key-value projections to save memory
- **FFN hidden dimension:** 28,672 — the feed-forward network inside each layer expands the 8,192-dimensional vector to 28,672 dimensions, applies a nonlinearity, then projects it back down
- **Normalization:** RMSNorm (before each sub-layer, not after — "pre-norm" style)
- **Position encoding:** Rotary Position Embeddings (RoPE) — injected into the attention computation so the model knows token order
- **Activation function:** SiLU (Sigmoid Linear Unit) in the feed-forward networks

Every one of these is a choice someone made. Different choices produce different models — GPT-4, Claude, Gemini all use decoder-only transformers but differ in these specifics. The architecture defines the parameter count: how many total numbers need to be trained. For Llama 3 70B, those choices add up to roughly 70 billion trainable parameters.

### Performance Profile
- **Compute cost:** Scales with depth (layers) × width (d_model) × sequence length — more of any means more matrix multiplications per token
- **Memory cost:** Parameter count drives the memory floor — 70B params × 2 bytes (FP16) = 140GB just for weights, before gradients and optimizer states
- **Architecture vs. training:** The architecture determines the compute-per-token cost; the training duration (how many tokens you run through it) determines total cost. A wider, deeper model costs more per step but may need fewer steps to reach the same quality — or it may reach quality a narrower model never could
