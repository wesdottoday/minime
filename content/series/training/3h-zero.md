---
title: "ZeRO & memory optimization"
date: 2026-05-19
draft: false
node_id: "3h"
tier: 3
parent: "2r"
children: []
siblings:
  - "3i"
url: "/series/training/train-from-scratch/training-step/optimizer-step/zero/"
summary: "In pure data parallelism, every GPU replica stores a complete copy of the 840GB training state — 430TB of redundant memory across the cluster. ZeRO eliminates that redundancy in three progressive stages, from sharding optimizer states to sharding everything."
series: "training"
mermaid: true
weight: 350
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

The [optimizer step](/series/training/train-from-scratch/training-step/optimizer-step/) (2r) showed that training Llama 3 70B requires ~840GB of state in mixed precision: 140GB weights (BF16) + 140GB gradients (BF16) + 560GB optimizer states (FP32). In pure data parallelism, every GPU replica stores a complete copy. With 512 data-parallel replicas, that's 430TB of total memory — 429.6TB of which is redundant copies. ZeRO eliminates that redundancy.

## ZeRO Stage 1: Shard optimizer states

The optimizer states (Adam's m and v tensors) are the largest single memory consumer: 560GB at FP32. In Stage 1, each of the 512 data-parallel replicas stores only 1/512th of the optimizer states — about 1.1GB per replica instead of 560GB. Each replica still holds the full model weights and gradients (280GB total), but the optimizer states are partitioned.

During the optimizer step, each replica updates the weights for its shard of the optimizer state. Then an all-gather reconstructs the full updated weights across all replicas. The all-gather adds communication but the memory savings are massive.

Per-GPU memory: 140GB (weights) + 140GB (gradients) + ~1.1GB (optimizer shard) = ~281GB. Down from 840GB. Fits on two Blackwell GPUs with room for activations.

## ZeRO Stage 2: Shard gradients too

After the backward pass, each replica has a full 140GB gradient tensor. But the optimizer only needs 1/512th of those gradients (matching its optimizer state shard). Stage 2 shards the gradients the same way: each replica stores only its 1/512th slice. The gradient all-reduce from Phase 5 is replaced with a reduce-scatter — each replica gets the summed gradient for only its shard, not the full gradient tensor.

Per-GPU memory: 140GB (weights) + ~0.27GB (gradient shard) + ~1.1GB (optimizer shard) = ~141GB. The gradient memory practically vanishes. One Blackwell GPU (192GB HBM) can hold weights + sharded gradients + sharded optimizer states with ~50GB left for activations.

## ZeRO Stage 3: Shard everything — including weights

The final step: don't store the full model weights on every replica either. Each replica holds only 1/512th of the weights — about 0.27GB. When a layer needs the full weights for the forward or backward pass, an all-gather reconstructs them just-in-time, the computation runs, and the weights are discarded again.

Per-GPU memory: ~0.27GB (weight shard) + ~0.27GB (gradient shard) + ~1.1GB (optimizer shard) = ~1.6GB for the model state. Essentially nothing — the model "doesn't exist" on any single GPU. The vast majority of per-GPU memory is now available for activations.

## The communication cost of each stage

| Stage | Memory per replica | Additional communication |
|-------|-------------------|------------------------|
| Baseline (no ZeRO) | ~840GB | Gradient all-reduce only |
| Stage 1 | ~281GB | + All-gather for weights after optimizer step |
| Stage 2 | ~141GB | Replace all-reduce with reduce-scatter; + all-gather for weights |
| Stage 3 | ~1.6GB | + All-gather for weights before *every* forward and backward layer |

Stage 1 adds negligible communication — one all-gather per step. Stage 2 is approximately communication-neutral (reduce-scatter is half of all-reduce, but you add a weight all-gather). Stage 3 adds significant communication: an all-gather before every layer in both forward and backward passes. At 80 layers, that's 160 all-gather operations per step, each reconstructing a portion of the model weights.

## FSDP: PyTorch's native ZeRO-3

Fully Sharded Data Parallelism (FSDP) is PyTorch's first-party implementation of ZeRO Stage 3 concepts. Rather than being a separate library (like DeepSpeed), FSDP is built into `torch.distributed`. It wraps model parameters in `ShardedTensor` objects that handle the sharding, all-gather, and discard lifecycle automatically.

Key FSDP behaviors:
- **Forward pass:** Before each layer, all-gather the sharded weights into full weights. Compute the layer. Discard the full weights (keep only the shard).
- **Backward pass:** Same all-gather pattern in reverse. Compute gradients on full weights. Reduce-scatter to shard the gradients. Discard full gradients.
- **Optimizer step:** Each replica updates its shard of weights using its shard of gradients and its shard of optimizer state. No all-gather needed — everything is local.

FSDP supports mixed sharding strategies: you can apply full sharding (Stage 3) across data-parallel replicas while using tensor parallelism within a node. This gives you the memory efficiency of ZeRO-3 where it matters (across many replicas) while keeping the fast NVLink communication for tensor parallelism (within a node).

## When to use which stage

- **Stage 1:** Almost always. The communication overhead is trivial and the memory savings (840GB to 281GB) are decisive. No reason not to use it.
- **Stage 2:** Use when gradient memory matters — specifically when the 140GB gradient tensor competes with activations for HBM. Communication overhead is minimal.
- **Stage 3:** Use when the model is too large to fit even sharded optimizer + full weights on a single GPU. The communication overhead is real (15-25% more step time) and requires careful overlap with compute. For a 70B model on Blackwell GPUs (192GB HBM), Stage 2 is usually sufficient. Stage 3 becomes necessary at 175B+ or when running very large batch sizes that demand more activation memory.

## CPU offloading: the last resort

When even Stage 3 sharding isn't enough — typically for very large models on older hardware with smaller HBM — optimizer states and even weights can be offloaded to CPU memory. On Grace Blackwell, this means spilling to the 512GB LPDDR5X over NVLink-C2C. The bandwidth (~900GB/s) makes this viable, but it adds latency to every optimizer step. CPU offloading is a fallback, not a first choice — it trades memory pressure for slower iteration time.

## Performance Profile

- **Stage 1 memory savings:** 840GB to 281GB per replica. Communication cost: ~negligible (one all-gather per step).
- **Stage 2 memory savings:** 840GB to 141GB per replica. Communication cost: ~neutral (reduce-scatter replaces all-reduce).
- **Stage 3 memory savings:** 840GB to 1.6GB per replica. Communication cost: +15-25% step time (all-gather per layer, 160x per step).
- **FSDP overhead:** 3-8% slower than non-sharded training at Stage 2. 15-25% slower at Stage 3. The overhead comes from all-gather latency and memory copy operations.
- **The decision matrix:** At 70B on 192GB GPUs, Stage 2 is the sweet spot — full model fits with room for activations. Stage 3 is for 175B+, multi-modal models, or when you need very large batch sizes.

### Sources
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
