---
title: "Batch size & gradient accumulation"
date: 2026-05-19
draft: false
node_id: "2u"
tier: 2
parent: "1e"
children: []
siblings:
  - "2t"
  - "2v"
url: "/series/training/train-from-scratch/training-loop/batch-size/"
summary: "Global batch size is the total number of tokens processed before the optimizer updates the weights. For Llama 3 70B, Meta reportedly used ~4 million tokens per step — roughly 500 sequences of 8,192 tokens."
series: "training"
mermaid: true
weight: 241
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture] --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop]:::hl --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click E "/series/training/train-from-scratch/training-loop/"
```

**Global batch size** is the total number of tokens processed before the optimizer updates the weights. For Llama 3 70B, Meta reportedly used a global batch size of ~4 million tokens per step (~500 sequences of 8,192 tokens). Across 15T total tokens, that's roughly 3.75 million training steps. Each step touches all 70B parameters once.

## Why large batches?

Two reasons:

1. **Gradient noise.** One sequence gives you one gradient — a noisy estimate of which direction to push the weights. Average the gradients from 500 sequences and the noise cancels out. The update is more stable, more representative. Like polling 500 people instead of one.

2. **GPU utilization.** A single sequence doesn't come close to saturating the compute capacity of 72 GPUs. Large batches keep the matrix multiplication units busy. Small batch = GPUs sitting idle between memory accesses.

## How 4 million tokens get split across GPUs

The global batch doesn't fit on one GPU. It gets divided:

- **Data parallelism** splits the batch across GPU groups. With 8 data-parallel replicas, each processes ~500K tokens.
- **Micro-batch** is what a single GPU actually sees per step — maybe 4 sequences of 8,192 tokens = ~32K tokens.

Each micro-batch computes its own gradients. Those gradients get averaged across all data-parallel replicas during gradient sync (the all-reduce from phase 5, node 2q). The averaged gradient is what the optimizer uses for the update.

## Gradient accumulation

If the micro-batch that fits in GPU memory is still too small relative to the target global batch size, multiple micro-batches run sequentially on each GPU. The gradients accumulate locally before syncing. Instead of one forward+backward pass then sync, you might do 4 forward+backward passes, sum the gradients locally, then sync once. Same math as a larger batch, but fits in memory.

### Performance Profile
- **Compute efficiency:** Larger batch size = better GPU utilization. The matrix multiplication units in the Blackwell GPUs are designed for large, regular workloads — bigger batches keep them saturated.
- **Communication cost:** Gradient sync (all-reduce) happens once per optimizer step, not once per micro-batch. Gradient accumulation amortizes that communication cost across multiple forward+backward passes. More accumulation steps = fewer syncs = less time spent on NVLink/InfiniBand.
- **Memory pressure:** Micro-batch size is bounded by GPU memory. Activations from the forward pass must be held until the backward pass completes. Larger micro-batch = more activations in memory simultaneously. This is the primary constraint on how big each GPU's piece can be.
- **Convergence tradeoff:** Batch size affects learning dynamics. Too small = noisy updates, unstable training. Too large = each step is expensive and the model takes fewer total steps, potentially under-exploring the loss landscape. 4M tokens per step is Meta's empirical sweet spot for 70B at this scale.
- **Step count math:** 15T tokens / 4M tokens per step = ~3.75M optimizer updates. Each update adjusts all 70B parameters. The total compute cost of training is step count x cost per step.
