---
title: "Key-value heads: 8 (Grouped Query Attention)"
date: 2026-05-14
draft: false
node_id: "2h"
tier: 2
parent: "1b"
children: []
siblings: ["2d", "2e", "2f", "2g", "2i", "2j"]
url: "/series/training/train-from-scratch/model-architecture/gqa/"
summary: "GQA shares Key-Value projections across groups of heads. Llama 3 has 8 KV heads — cutting the KV cache size by 8x compared to full MHA."
series: "training"
mermaid: true
weight: 214
tags: [llm-training]
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture]:::hl --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click B "/series/training/train-from-scratch/model-architecture/"
```

This is where it gets interesting and directly connects to inference cost. Standard multi-head attention (MHA) gives each of the 64 heads its own Key and Value projections. That means during inference, the KV cache stores 64 separate K and V vectors per token per layer. We covered in the LLMs series how the [KV cache](/series/inference/what-happens/embeddings/kv-cache/) is often the memory bottleneck during inference.

Grouped Query Attention (GQA) shares Key-Value projections across groups of heads. Llama 3 has 8 KV heads, so every 8 query heads share one KV head (64 / 8 = 8 heads per group). This cuts the [KV cache](/series/inference/what-happens/embeddings/kv-cache/mqa-gqa/) size by 8x compared to full MHA.

**Why 8 KV heads specifically?** It's a spectrum:
- **MHA (64 KV heads):** Maximum expressiveness, maximum KV cache memory
- **GQA (8 KV heads):** Minor quality loss, 8x KV cache reduction
- **MQA (1 KV head):** All heads share one KV projection. Maximum compression but measurable quality degradation on complex tasks

Meta tested these ratios and found 8 KV heads preserved nearly all of MHA's quality while dramatically reducing inference memory. This is a *training-time decision that optimizes for inference cost* — a recurring theme. You're choosing the architecture now, but the thing you're optimizing for is how efficiently this model will serve millions of requests later.

**The math:** At full MHA, KV cache for one 8K-token sequence across 80 layers at FP16: `2 × 64 × 128 × 8192 × 80 × 2 bytes = ~21GB`. With GQA at 8 KV heads: `2 × 8 × 128 × 8192 × 80 × 2 bytes = ~2.6GB`. That's the difference between serving 1 user per GPU and serving 8.

### Performance Profile
- **KV cache reduction:** 8x reduction vs. full MHA (21GB → 2.6GB per 8K sequence). This is the dominant factor in inference serving density — more concurrent users per GPU.
- **Training cost:** GQA is slightly cheaper to train than full MHA because the KV projections have fewer parameters (8 KV heads vs. 64). Minor savings during training, massive savings during inference.
- **Quality impact:** Meta's ablations showed GQA with 8 KV heads preserved >99% of full MHA quality on standard benchmarks. The quality gap widens at very long contexts and complex multi-hop reasoning tasks but remains small.
- **Architecture decision with inference consequences:** GQA is chosen at architecture design time (before training) specifically to optimize inference. A training-time decision that pays dividends at serving time.
