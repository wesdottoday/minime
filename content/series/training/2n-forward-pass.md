---
title: "Phase 2 — Forward pass"
date: 2026-05-14
draft: false
node_id: "2n"
tier: 2
parent: "1d"
children:
  - "3b"
siblings:
  - "2m"
  - "2n"
  - "2o"
  - "2p"
  - "2q"
  - "2r"
  - "2s"
url: "/series/training/train-from-scratch/training-step/forward-pass/"
summary: "The forward pass pushes a batch of tokens through all 80 layers to produce predictions. At 16,384 GPUs, the model isn't sitting on one GPU — it's sliced up across thousands of them using three types of parallelism simultaneously."
series: "training"
mermaid: true
weight: 231
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

The forward pass pushes a batch of tokens through all 80 layers to produce predictions. At 16,384 GPUs, the model isn't sitting on one GPU — it's sliced up across thousands of them using three types of parallelism simultaneously.

**Data parallelism** is the simplest. The [global batch](/series/training/train-from-scratch/training-step/training-units/) of ~4 million tokens is split across 512 data-parallel replicas. Each replica gets different data but runs the same model (distributed across its 32 GPUs via tensor and pipeline parallelism). They operate independently during the forward pass and only need to talk during gradient sync (Phase 5).

**Tensor parallelism** splits individual layers across GPUs within a single NVLink domain. The massive FFN matrices (8,192 × 28,672) are sliced column-wise or row-wise across, say, 8 GPUs. Each GPU does its portion of the matrix multiplication, then they combine results over NVLink. This happens *inside every layer, every step, multiple times*. It must be on NVLink — if this went over InfiniBand, you'd lose 30-50% throughput.

**Pipeline parallelism** assigns chunks of the 80 layers to different GPU groups. GPUs 1-8 handle layers 1-20, GPUs 9-16 handle layers 21-40, and so on. The forward pass becomes a literal pipeline — each group processes its layers and ships the output activations to the next group. The activation tensor per micro-batch is roughly `batch_size × seq_len × d_model × 2 bytes` — at 4 × 8,192 × 8,192 × 2 = ~537MB per transfer over InfiniBand.

**Pipeline bubbles** are the main weakness. When group 1 is processing, groups 2-4 idle waiting. When group 4 finishes, groups 1-3 have nothing to do. At 4 pipeline stages, ~25% of compute is lost to bubbles. Multiple micro-batches in flight reduce this (while group 1 works on micro-batch 2, group 2 works on micro-batch 1's output), but you never eliminate it entirely.

**The topology mapping is everything:** Tensor parallelism goes inside the NVLink domain (chatty, high-bandwidth). Pipeline parallelism goes across racks over InfiniBand (less chatty, bigger transfers). Data parallelism is the outermost ring — replicas run independently until gradient sync. Getting this mapping wrong (putting tensor-parallel communication on InfiniBand, for instance) can cut throughput in half.

**Activation memory** is the other pressure point. Each GPU stores its forward pass activations because the backward pass needs them. At 8,192 dimensions × 8,192 sequence length across the layers each GPU owns, this is tens of GB per GPU on top of model weights and optimizer states. [Activation checkpointing](/series/training/train-from-scratch/training-step/backward-pass/activation-checkpointing/) trades compute for memory — discard some activations and recompute them during the backward pass. More on this in Phase 4.

### Performance Profile
- **Compute-bound:** Matrix multiplications through 80 layers — this is where the GPUs actually earn their keep
- **Network-bound:** Tensor-parallel all-reduce within NVLink domain (every layer), pipeline activation transfers across InfiniBand (at stage boundaries)
- **Memory-bound:** Activation storage grows linearly with sequence length and the number of layers each GPU owns
- **Weak points:** Pipeline bubbles waste 10-25% of compute; a slow InfiniBand link at a pipeline stage boundary stalls everything downstream; activation memory competes with weights and optimizer states for limited HBM
