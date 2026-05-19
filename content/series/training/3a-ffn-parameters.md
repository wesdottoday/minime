---
title: "Why does the FFN hold 80% of the parameters?"
date: 2026-05-19
draft: false
node_id: "3a"
tier: 3
parent: "2i"
children: []
siblings:
  - "3e"
url: "/series/training/train-from-scratch/model-architecture/ffn-hidden-dimension/ffn-parameters/"
summary: "The FFN accounts for 82% of Llama 3 70B's parameters — 56.4 billion out of 69.5 billion. SwiGLU added a third matrix while GQA shrank attention. The gap widened from both directions."
series: "training"
mermaid: true
weight: 300
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

Let's do the math. In each of Llama 3 70B's 80 layers, there are two components: attention and FFN. Count their parameters separately.

**Attention parameters per layer:**
- Q projection: `d_model × d_model` = 8,192 × 8,192 = 67.1M
- K projection: `d_model × (kv_heads × head_dim)` = 8,192 × (8 × 128) = 8,192 × 1,024 = 8.4M
- V projection: same as K = 8.4M
- Output projection: `d_model × d_model` = 67.1M
- **Total attention per layer: ~151M parameters**

(Note: K and V are smaller because of [GQA](/series/training/train-from-scratch/model-architecture/gqa/) — only 8 KV heads instead of 64. With full MHA, attention would be 4 × 67.1M = 268M per layer.)

**FFN parameters per layer:**
Llama 3 uses SwiGLU, which has three weight matrices instead of the standard two:
- Gate projection: `d_model × ffn_dim` = 8,192 × 28,672 = 234.9M
- Up projection: `d_model × ffn_dim` = 8,192 × 28,672 = 234.9M
- Down projection: `ffn_dim × d_model` = 28,672 × 8,192 = 234.9M
- **Total FFN per layer: ~704.7M parameters**

**The ratio:** FFN is 704.7M vs attention's 151M per layer. That's 82% FFN, 18% attention. Across 80 layers: ~56.4B in FFN, ~12.1B in attention, plus ~1B for embeddings and the output head. Total: ~69.5B. The math checks out.

**Why is this the case?** It comes down to the expansion ratio. The FFN expands from 8,192 to 28,672 — a 3.5x blowup — and it does this with *three* large matrices because of the gating mechanism. Attention, meanwhile, got *cheaper* with GQA: the K and V projections shrank by 8x compared to full MHA. So the FFN grew (SwiGLU added a third matrix) while attention shrank (GQA removed redundant KV projections). The gap widened from both directions.

**What does this mean practically?** Much factual association appears to be mediated by FFN weights, though knowledge is distributed across the full network rather than stored in a single component. Research on model editing and knowledge localization finds that specific facts ("The capital of France is Paris") are strongly associated with FFN weights, while attention handles relational reasoning ("given these facts, what follows?"). Attention is the routing mechanism — it figures out which stored knowledge is relevant to the current context and pulls it forward. It needs far fewer parameters to do that job.

**The training implication:** 80% of your gradient updates, 80% of your optimizer states, 80% of the memory pressure during training — it's all FFN. When people talk about the memory wall in training, they're mostly talking about the FFN. When tensor parallelism splits a model across GPUs, the FFN matrices are the ones that dominate the split.

## Performance Profile

- **Memory dominance:** FFN weights = ~107GB at FP16 (56.4B × 2 bytes). With Adam optimizer states (2 additional copies per parameter at FP32): ~450GB for FFN alone. This is the primary driver of the multi-GPU memory requirement.
- **Compute dominance:** Three large matmuls per layer (8,192 × 28,672 each), 80 layers, forward + backward = ~80% of total training FLOP per step. These are the operations that keep tensor cores saturated.
- **Tensor parallelism split:** When splitting across GPUs, the FFN's 28,672-wide hidden dimension divides across GPUs (e.g., 4-way TP = 7,168 per GPU). The split is clean because 28,672 = 7,168 × 4. Each GPU computes its slice independently, then results are all-reduced. The FFN split is where most of the inter-GPU communication in tensor parallelism originates.
- **Knowledge localization:** Much factual association appears to be mediated by FFN weights, though knowledge is distributed across the full network. This has implications for model editing, fine-tuning efficiency (LoRA targets attention but FFN carries much of the factual association), and quantization sensitivity (aggressive FFN quantization degrades factual recall).
