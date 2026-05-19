---
title: "Phase 1 — Data loading"
date: 2026-05-14
draft: false
node_id: "2m"
tier: 2
parent: "1d"
children: []
siblings:
  - "2m"
  - "2n"
  - "2o"
  - "2p"
  - "2q"
  - "2r"
  - "2s"
url: "/series/training/train-from-scratch/training-step/data-loading/"
summary: "The simplest phase conceptually, but at 16,384 GPUs it becomes a distributed systems problem. Each data-parallel replica needs its micro-batch of token sequences, consuming millions of tokens per optimizer step."
series: "training"
mermaid: true
weight: 230
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

This is the simplest phase conceptually, but at 16,384 GPUs it becomes a distributed systems problem.

**What happens:** Each [data-parallel replica](/series/training/train-from-scratch/training-step/training-units/) needs its micro-batch of token sequences for this training step. With TP=8 and PP=4, it takes 32 GPUs to hold one copy of the model, so 16,384 GPUs yield 512 data-parallel replicas — not 16,384 independent workers. Each replica processes a micro-batch of, say, 4 sequences of 8,192 tokens = ~33K tokens per forward/backward pass. With gradient accumulation and 512 replicas, the global batch reaches roughly **4 million tokens per optimizer step**. At 4 bytes per token (32-bit IDs for a 128K vocabulary), that's ~16MB of raw input data per step — tiny compared to the compute that follows.

**How it works at scale:** The tokenized dataset is sharded across a distributed filesystem — thousands of binary files. A DataLoader process runs on each node's CPU, responsible for:
1. Knowing which shard(s) to read from (coordinated so no two GPUs read the same data in the same epoch)
2. Reading a chunk of sequences from the shard into CPU memory
3. Shuffling and batching the sequences
4. Transferring the batch to GPU memory (CPU → GPU via PCIe or NVLink-C2C on Grace Blackwell)

The DataLoader typically runs several steps ahead — it's prefetching the *next* batch while the GPU is computing the *current* batch. If everything is working, data loading is completely hidden behind compute. The GPU never waits.

**Where it breaks at 227 racks:**

- **Storage bandwidth saturation.** 16MB per step sounds trivial, but training steps happen every few seconds. And it's not just input data — checkpointing (Phase 7) writes terabytes to the same filesystem. If the storage tier can't serve 2,048+ simultaneous readers while also handling checkpoint writes, the DataLoader falls behind and GPUs stall. This is why large training clusters have dedicated storage tiers — often purpose-built with Lustre, GPFS, or AI-native systems like VAST or Weka, with aggregate bandwidth in the hundreds of GB/s.
- **Shard coordination.** Every data-parallel replica must see different data. With 512 replicas consuming ~4 million tokens per step, and a dataset of 15 trillion tokens across ~5 epochs (Meta indicated they ran multiple epochs for some data), the assignment of shards to replicas must be coordinated and deterministic. If two replicas accidentally train on the same data, you've wasted compute and introduced statistical bias. The DataLoader uses a distributed sampler with a random seed — same seed + same replica rank = deterministic, non-overlapping data assignment.
- **Stragglers.** If one node's DataLoader is slow (degraded disk, overloaded filesystem metadata server, network hiccup to storage), it doesn't just slow down that one GPU — it slows down every GPU that needs to synchronize with it in later phases. One slow reader out of 2,048 nodes can bottleneck the entire cluster. This is the "slowest turtle" problem, and it's endemic to distributed training.

### Performance Profile
- **Bottleneck:** Storage I/O throughput and metadata operations, not compute
- **Mitigation:** Aggressive prefetching, local NVMe caching of hot shards, dedicated storage network separated from training network
- **Failure mode:** Silent — a slow DataLoader doesn't crash, it just makes every other GPU wait. Hard to detect, easy to miss in aggregate throughput numbers
- **Scale multiplier:** At 1 rack, storage is trivial. At 227 racks, storage is a first-class infrastructure problem that requires dedicated engineering
