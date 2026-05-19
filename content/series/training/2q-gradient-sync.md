---
title: "Phase 5 — Gradient synchronization"
date: 2026-05-14
draft: false
node_id: "2q"
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
url: "/series/training/train-from-scratch/training-step/gradient-sync/"
summary: "After the backward pass, every GPU has local gradients computed from its slice of the data. Before anyone updates weights, all 512 data-parallel replicas need to agree on a single set of gradients via all-reduce."
series: "training"
mermaid: true
weight: 234
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

After the backward pass, every [data-parallel replica](/series/training/train-from-scratch/training-step/training-units/) has gradients — but they're *local* gradients, computed from that replica's slice of the data. Remember, data parallelism means 512 replicas each saw different training examples. Replica #0's gradients say "based on *my* micro-batches, here's how the weights should change." Replica #256's gradients say something different because it saw different data. Before anyone updates weights, all 512 data-parallel replicas need to agree on a single set of gradients. That agreement is the all-reduce.

**What all-reduce does:** Take 512 copies of the gradient tensor, sum them element-wise, and distribute the result back to all 512 replicas. After the all-reduce, every replica has identical averaged gradients. This guarantees that the optimizer step (Phase 6) produces identical weight updates on every replica, keeping all copies of the model in sync.

**A critical detail about 3D parallelism:** Each replica's 70B parameters are *distributed* across 32 GPUs (TP=8 × PP=4). No single GPU holds a full 140GB gradient tensor. Each GPU holds gradients only for its shard of the model — the parameters in its pipeline stage, further sliced by tensor parallelism. The all-reduce happens in the **data-parallel dimension**: GPU rank 0 in replica 1 syncs with GPU rank 0 in replicas 2 through 512 — the matching ranks that hold the same parameter shard. There are effectively 32 separate all-reduce groups, each with 512 participants, each syncing only its shard of the gradients.

**The scale of the communication:** The total gradient volume across the full model is 70B parameters × 2 bytes (BF16) = ~140GB. Split across 32 GPUs per replica, each GPU syncs roughly 140GB / 32 ≈ **4.4GB** of gradients with its 511 counterparts. A naive approach — send everything to one node, sum it, broadcast back — would require one node to receive 512 × 4.4GB ≈ 2.2TB per shard. Obviously, that doesn't work.

**Ring all-reduce:** The standard algorithm. For each shard's all-reduce group, arrange the 512 participants in a logical ring. Split the ~4.4GB gradient buffer into 512 chunks (~8.6MB each). Each participant sends one chunk to its neighbor while receiving a different chunk from its other neighbor. After 511 steps of simultaneous send-receive, every chunk has been summed across all participants and distributed back. Total data sent per participant: ~2 × 4.4GB ≈ 8.8GB. Across all 32 shards, the aggregate gradient traffic per step is ~280GB (equivalent to 2 × 140GB for the full model). The beauty of ring all-reduce is it runs at full bisection bandwidth regardless of participant count — adding more replicas doesn't slow it down proportionally, because each participant's send/receive volume stays constant.

**Where [NCCL](/series/training/train-from-scratch/training-step/forward-pass/parallelism-stack/) earns its pay:** NCCL doesn't do a single flat ring. It builds a hierarchical communication tree based on the actual topology. Within a NVLink domain (one rack), the all-reduce happens first over NVLink at 1.8TB/s per GPU — near-instant. Then one representative GPU per rack does the cross-rack all-reduce over InfiniBand at ~50-100GB/s per link. Then the result is broadcast back within each rack over NVLink. This hierarchy means the slow link (InfiniBand) only carries a fraction of the data — the intra-rack NVLink handles the heavy aggregation locally.

**The math on timing:** In the hierarchical scheme, the cross-rack all-reduce moves the aggregate gradient data across ~227 racks over InfiniBand. With multiple InfiniBand ports per node and ring/tree algorithms, effective aggregate bandwidth might be 200-400GB/s for this operation. The total volume per GPU's shard is ~4.4GB, but the hierarchical scheme means only a fraction crosses the inter-rack links. Rough estimate: 0.5-1.0 seconds for the full gradient sync at scale. If a training step is 3-5 seconds total, gradient sync is 10-30% of wall clock time. That's the NCCL overhead we mentioned in the orchestration node.

**Overlap with backward pass:** The clever optimization — you don't have to wait until the entire backward pass is done to start syncing. As soon as layer 80's gradients are computed, you can begin all-reducing them while the backward pass continues through layers 79, 78, 77... By the time the backward pass finishes layer 1, layer 80's gradients are already fully synced. This overlap hides a significant portion of the communication cost behind compute. [Megatron-LM](/series/training/train-from-scratch/training-step/forward-pass/parallelism-stack/) does this by default — it hooks gradient all-reduce into the backward pass computation graph so they run concurrently.

**Where it breaks at 227 racks:**

- **The weakest InfiniBand link sets the pace.** Ring all-reduce is only as fast as its slowest link. If one of the 227 racks has a degraded InfiniBand port — flapping switch, marginal cable, congested leaf — every replica waits for it. At this scale, you're near-guaranteed to have at least one suboptimal link at any given time. Network health monitoring and hot-spare capacity are non-optional.
- **Gradient compression and reduced precision.** To cut the communication volume, some training runs all-reduce in lower precision — compute gradients in FP32, but communicate them in BF16 or even FP8, then upcast on the receiving end. This halves or quarters the data volume at the cost of some gradient noise. Whether this hurts quality is empirical — for Llama 3 scale, Meta likely used BF16 gradients. Some researchers have pushed to FP8 gradient communication with mixed results.
- **Straggler amplification.** This is the second place (after data loading) where the slowest-turtle problem bites. All-reduce is a collective — every participant must contribute before any participant gets the result. One replica that's 100ms behind on its backward pass delays the gradient sync for all 512 replicas (and thus all 16,384 GPUs). When you have 16,384 GPUs across 512 replicas, the probability of *at least one* being slow on any given step is essentially 1. The expected slowdown is the tail latency of 16,384 processes — not the average, the maximum. This is why large-scale training throughput is always significantly below theoretical peak.
- **Bit-rot and silent corruption.** If a GPU's memory has a bit flip in its gradient buffer — a real concern at scale, especially without ECC or with marginal ECC — that corrupted gradient gets mixed into the all-reduce and propagated to every replica. Unlike NaN (which is detectable), a single flipped bit might change a gradient from 0.001 to 0.033 — wrong, but not obviously wrong. Over thousands of steps, these silent corruptions can degrade model quality without any clear signal in the loss curve. Some training frameworks run periodic gradient checksums to detect this.

### Performance Profile
- **Communication volume:** ~280GB aggregate per replica per step across all shards (~8.8GB per GPU for its parameter shard)
- **Time cost:** 0.5-1.0 seconds at 227 racks, 10-30% of step time, significantly reduced by overlapping with backward pass
- **Bandwidth utilization:** NVLink (intra-rack) near 100%. InfiniBand (inter-rack) is the bottleneck — real utilization is 50-70% of theoretical due to protocol overhead and congestion
- **Weak points:** Slowest link determines pace. Straggler amplification across 512 replicas (16,384 GPUs). Silent bit corruption. Any rack-level network event (switch reboot, cable reseat) stalls the entire cluster
