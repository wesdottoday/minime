---
title: "Activation checkpointing"
date: 2026-05-19
draft: false
node_id: "3g"
tier: 3
parent: "2p"
children: []
siblings: []
url: "/series/training/train-from-scratch/training-step/backward-pass/activation-checkpointing/"
summary: "Throw away most activations during the forward pass, recompute them during the backward pass. The universal answer to the activation memory wall — trading compute for memory at a ratio that makes large-scale training possible."
series: "training"
mermaid: true
weight: 330
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture] --> C[Hardware & Scale] --> D[Training Step]:::hl --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click D "/series/training/train-from-scratch/training-step/"
```

During the forward pass, every layer produces intermediate outputs — activations — that the backward pass needs to compute gradients. For Llama 3 70B, storing all activations for a single micro-batch across all 80 layers requires ~43GB per GPU. That's on top of model weights (~140GB), gradients (~140GB), and optimizer states (~560GB). Something has to give. Activation checkpointing is the universal answer: throw away most activations during the forward pass, and recompute them during the backward pass when you need them.

**The core tradeoff.** Without checkpointing, the backward pass can immediately compute gradients at each layer because the forward activations are sitting in memory. With checkpointing, you only keep activations at designated "checkpoint" layers. When the backward pass reaches a non-checkpointed layer, it re-runs the forward pass from the nearest checkpoint to regenerate the missing activations, computes the gradients, then discards the regenerated activations. You're trading compute (extra forward passes) for memory (fewer stored activations).

## Checkpoint interval strategies

The simplest approach: keep activations at every Nth layer and discard the rest.

- **Checkpoint every layer (N=1):** No checkpointing. Full activation memory. ~43GB per micro-batch. Fastest compute, highest memory.
- **Checkpoint every 4th layer (N=4):** Store activations at layers 0, 4, 8, ..., 76. When the backward pass needs layer 6's activations, it re-runs the forward pass from layer 4's checkpoint through layers 5 and 6. Activation memory drops ~4x to ~11GB. Compute overhead: recomputing an average of 1.5 layers per backward layer = ~37% overhead on the forward pass, or ~19% overhead on total forward+backward.
- **Checkpoint every 8th layer (N=8):** Activation memory drops ~8x to ~5.4GB. Compute overhead: recomputing an average of 3.5 layers per backward layer = ~87% overhead on the forward pass, or ~29% overhead on total forward+backward.
- **Checkpoint only the first layer (N=80):** Minimum memory — only one set of activations stored. Maximum recompute — the entire forward pass runs again during backward. Compute overhead: ~100% forward pass overhead = ~33% of total. Rarely used because the memory savings from N=4 or N=8 are usually sufficient.

The standard for Llama 3-scale training is **selective checkpointing** — a smarter version described below.

## Selective checkpointing

Not all operations are equally expensive to recompute. The FFN's three matrix multiplications (8,192 x 28,672 each) are compute-heavy. RMSNorm, SiLU, and dropout are nearly free — element-wise operations that take microseconds. Selective checkpointing keeps the expensive-to-recompute activations and discards the cheap-to-recompute ones.

In practice, this means:
- **Keep:** Outputs of major matrix multiplications (attention QKV projections, FFN up/gate/down projections). These are expensive to recompute because they involve large matmuls.
- **Discard:** Normalization outputs (RMSNorm), activation function outputs (SiLU), dropout masks, attention softmax outputs. These are cheap to recompute — element-wise operations that are bandwidth-bound, not compute-bound.

Selective checkpointing achieves most of the memory savings of full checkpointing (discarding 60-70% of activation memory) while only adding 10-15% compute overhead, versus 20-35% for blanket layer-level checkpointing. This is what production training runs actually use.

## The activation memory breakdown

For one micro-batch (4 sequences x 8,192 tokens) through one layer of Llama 3 70B at BF16:

| Activation | Size |
|-----------|------|
| Attention input (post-RMSNorm) | 537MB |
| Q, K, V projections | 1.6GB |
| Attention scores (post-softmax) | 34GB* |
| Attention output | 537MB |
| FFN input (post-RMSNorm) | 537MB |
| FFN gate/up outputs | 3.75GB |
| FFN SiLU output | 1.88GB |

*The attention score matrix is 34GB per layer — this is why FlashAttention never materializes it. FlashAttention computes attention in tiles, so this 34GB tensor never exists in memory. Without FlashAttention, attention memory alone would blow out HBM.

With FlashAttention and selective checkpointing, effective activation memory per layer per micro-batch is roughly 2-3GB (keeping Q/K/V outputs, discarding everything recomputable). Across 80 layers: ~160-240GB per micro-batch — which still needs to be divided across pipeline and tensor parallel groups.

## Interaction with pipeline parallelism

In pipeline parallelism, each stage owns a subset of layers (e.g., 20 layers per stage in a 4-stage pipeline). Each stage only stores activations for its own layers. At stage boundaries, the output activation is sent to the next stage and can be discarded locally. This means pipeline parallelism naturally reduces per-GPU activation memory by the pipeline degree, independent of checkpointing.

The two optimizations stack: pipeline parallelism reduces the number of layers per GPU, and checkpointing reduces the activation memory per layer. A 4-stage pipeline with selective checkpointing might bring per-GPU activation memory from 43GB (no optimization) down to ~5-10GB.

## CPU offloading as an alternative

On Grace Blackwell, each GPU has access to 512GB of CPU LPDDR5X via NVLink-C2C at ~900GB/s. Instead of recomputing activations, you can offload them to CPU memory during the forward pass and reload them during the backward pass. The bandwidth cost: 3GB per layer x 80 layers x 2 (offload + reload) = ~480GB of data transferred, at ~900GB/s = ~0.5 seconds. Whether this is faster than recomputation depends on the layer: offloading is better for compute-heavy activations (FFN outputs), while recomputation is better for cheap activations (normalization, element-wise ops).

In practice, production training uses a hybrid: selective checkpointing for cheap activations (recompute them) combined with CPU offloading for expensive activations (store them in LPDDR5X). This minimizes both compute overhead and GPU memory usage.

## Performance Profile

- **Memory savings:** Selective checkpointing reduces activation memory by 60-70% with 10-15% compute overhead. Full layer checkpointing (every 4th layer) saves ~75% with 20-35% overhead.
- **FlashAttention interaction:** Eliminates the 34GB/layer attention score matrix — the single largest activation. Without FlashAttention, checkpointing alone isn't enough to fit large-scale training in memory.
- **Pipeline stacking:** 4-stage pipeline x selective checkpointing = ~8-10x reduction in per-GPU activation memory compared to no optimization.
- **CPU offloading bandwidth:** NVLink-C2C at ~900GB/s enables offloading ~480GB of activations in ~0.5s. Competitive with recomputation for compute-heavy activations.
- **The real constraint:** Activation memory is the swing factor in batch size selection. Smaller batch = less activation memory = less checkpointing needed = faster per-step. But smaller batch = worse GPU utilization and more gradient noise. Checkpointing lets you run larger batches at the cost of compute overhead.
