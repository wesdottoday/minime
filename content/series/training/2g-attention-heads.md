---
title: "Attention heads: 64"
date: 2026-05-14
draft: false
node_id: "2g"
tier: 2
parent: "1b"
children: []
siblings: ["2d", "2e", "2f", "2h", "2i", "2j"]
url: "/series/training/train-from-scratch/model-architecture/attention-heads/"
summary: "Each layer's attention mechanism runs 64 independent attention computations in parallel."
series: "training"
mermaid: true
weight: 213
tags: [llm-training]
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture]:::hl --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click B "/series/training/train-from-scratch/model-architecture/"
```

Each layer's attention mechanism runs 64 independent attention computations in parallel. We covered [multi-head attention](/series/inference/what-happens/embeddings/model-layers/attention-deep-dive/multi-head-attention/) in the LLMs series — the key insight is that each head attends to different aspects of the input. One head might track syntactic relationships (subject-verb), another tracks co-reference (what "it" refers to), another tracks proximity.

**How is 64 chosen?** It's d_model divided by the head dimension. The head dimension for Llama 3 is 128 (a design choice). So: 8,192 / 128 = 64 heads. The head dimension of 128 is itself a convention — the original transformer paper used 64, most modern models use 64 or 128. Larger head dimensions give each head more expressiveness per head. Smaller head dimensions give you more heads (more parallel "perspectives") for the same d_model.

**The tradeoff:** More heads with smaller dimensions = more diverse attention patterns but each pattern is lower-resolution. Fewer heads with larger dimensions = fewer patterns but each one is richer. 128-dimensional heads at 64 heads has become the sweet spot at this scale.

**Compute note:** Multi-head attention is the same total FLOP cost regardless of how you split the heads — 64 heads × 128 dims = 8 heads × 1024 dims in total compute. The difference is in what the model *learns* to attend to, not how fast it runs.

### Performance Profile
- **Training FLOP cost:** Identical regardless of head count split — 64 × 128 = 8,192 total attention dimensions either way. Head configuration doesn't affect training speed.
- **Parallelism:** Attention heads are embarrassingly parallel — each head computes independently, then results are concatenated. 64 heads map naturally to GPU parallelism (64 is divisible by 1, 2, 4, 8, 16, 32, 64 GPUs in tensor parallelism).
- **Per-head memory:** Each head operates on 128-dimensional slices. Q, K, V projections per head per token = 3 × 128 × 2 bytes = 768 bytes. Small individually, but multiplied across 64 heads × sequence length × batch size × 80 layers, it's the bulk of attention's memory footprint.
