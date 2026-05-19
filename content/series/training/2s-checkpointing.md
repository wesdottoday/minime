---
title: "Phase 7 — Checkpointing"
date: 2026-05-14
draft: false
node_id: "2s"
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
url: "/series/training/train-from-scratch/training-step/checkpointing/"
summary: "Hardware fails. At 16,384 GPUs running 24/7 for weeks, it's not if but how often. Checkpointing saves a complete snapshot of training state so you can recover from inevitable failures without starting over."
series: "training"
mermaid: true
weight: 236
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

Hardware fails. At 16,384 GPUs running 24/7 for weeks, it's not a question of if but how often. Meta reported that during the Llama 3 training run, they experienced an average of ~1-2 failures per day requiring intervention. A GPU throws an uncorrectable ECC error. An InfiniBand link flaps. A power supply trips. A node kernel panics. Any of these can corrupt or crash the training run. Without checkpointing, a failure at day 10 of a 14-day run means you start over from random weights.

**What gets saved:** A checkpoint is a complete snapshot of everything needed to resume training exactly where you left off:

- **Model weights:** 70B parameters at BF16 = ~140GB
- **Optimizer states:** Adam's m and v tensors at FP32 = ~560GB
- **Learning rate scheduler state:** Which step you're on, where you are in the cosine schedule
- **Data loader state:** Which shards have been consumed, the random seed state, so you don't re-train on data you've already seen
- **RNG states:** Random number generator states for every GPU, so that dropout and other stochastic operations reproduce exactly

Total checkpoint size: roughly 700GB-1TB, depending on what's included and the precision.

**How often:** Typical checkpoint intervals for runs at this scale are every 500-2,000 training steps. At ~3-5 seconds per step, that's roughly every 25-100 minutes. Too frequent and you waste time writing to storage. Too infrequent and a failure costs you hours of recomputation. The sweet spot depends on failure rate — if you're losing a node once a day, and each step takes 4 seconds, a 1,000-step checkpoint interval means you lose at most ~67 minutes of work per failure. Meta reportedly checkpointed every ~500 steps for Llama 3.

**The I/O storm:** Here's where it gets ugly. 700GB-1TB needs to be written to the distributed filesystem. But it's not one GPU writing 1TB — with [ZeRO](/series/training/train-from-scratch/training-step/optimizer-step/zero/) sharding, the optimizer states are spread across 512 data-parallel replicas. Each replica writes its shard. With pipeline and tensor parallelism, model weights are spread across groups too. So you have potentially thousands of GPUs simultaneously flushing their state to storage.

At 16,384 GPUs, a checkpoint write might look like: 2,048 nodes, each writing 350-500MB of their sharded state, simultaneously. That's ~700GB-1TB hitting the filesystem in a burst. If the storage tier delivers 200GB/s aggregate write bandwidth, that's 3.5-5 seconds of sustained write. Sounds manageable — except the filesystem's metadata server is handling thousands of simultaneous file creates, and the data servers are absorbing a burst that looks nothing like the steady sequential reads of data loading. Checkpoint writes are the bursty, spiky workload that storage architects hate.

**Asynchronous checkpointing:** The standard optimization. Instead of stopping training while the checkpoint writes to disk, you copy the state to CPU memory (fast — NVLink-C2C on Grace Blackwell) and let a background thread write from CPU memory to storage while the GPUs immediately start the next training step. The Grace CPUs with their 512GB of LPDDR5X each are purpose-built for this — they're a large, fast staging buffer between GPU memory and the filesystem. The checkpoint write happens in the background, overlapping with the next few training steps. Training only stalls if the *next* checkpoint arrives before the *previous* one has finished writing — which means your storage needs to sustain checkpoint writes faster than the checkpoint interval.

**Checkpoint management:** You don't keep every checkpoint forever. A typical retention policy:
- Keep the last 3-5 checkpoints (for quick rollback on recent failures)
- Keep one every N checkpoints as a long-term milestone (for recovery from bugs discovered later)
- Delete the rest

Even at 3-5 retained copies, that's 3.5-5TB of checkpoint data on the filesystem at any given time. Over a full training run with milestones, the total checkpoint storage might reach 10-20TB.

**Failure recovery workflow:** When a failure hits:
1. Training halts across all 16,384 GPUs (the collective communication breaks, so everyone notices)
2. The orchestration layer (Slurm + custom health checks) identifies the failed node
3. A replacement node is allocated (or the failed node is rebooted if it's a transient error)
4. The most recent checkpoint is loaded onto all GPUs — weights, optimizer states, scheduler state, RNG states, data loader position
5. Training resumes from the checkpointed step

The gap between the checkpoint and the failure is lost work. At 500-step intervals, that's at most ~2,000 seconds (~33 minutes) of lost compute across 16,384 GPUs. In dollar terms, at a fully loaded cluster cost of $50-100 per GPU-hour (amortizing hardware, power, networking, and facilities across the fleet), 16,384 GPUs for 33 minutes is ~$270K-$540K of wasted compute per failure. With 1-2 failures per day over a 14-day run, that's $4M-$15M in failure-related waste. This is why checkpointing frequency, failure detection speed, and restart efficiency are first-class engineering concerns, not afterthoughts.

**Deterministic replay:** The RNG state and data loader state in the checkpoint enable *exact* resumption — the model sees exactly the same data in exactly the same order it would have if the failure hadn't happened. This matters because training is sensitive to data order (curriculum effects, batch composition), and non-deterministic restarts can introduce subtle distributional shifts that affect final model quality. Getting this right is surprisingly hard — every source of randomness (dropout masks, data shuffling, stochastic rounding) needs its state captured and restored.

**Where it breaks at 227 racks:**

- **Checkpoint write amplification.** With ZeRO sharding and 3D parallelism, the state is fragmented across thousands of GPUs. Reassembling a coherent checkpoint either requires a coordinated gather (expensive communication) or writing thousands of small files (expensive filesystem metadata). Both approaches have scaling limits. Some systems write sharded checkpoints and only reassemble on load.
- **Filesystem metadata pressure.** Thousands of simultaneous file creates/writes hammer the metadata server. Lustre and GPFS handle this differently, but both can become bottlenecks at this scale. Some training setups use node-local NVMe as a first-tier checkpoint destination, then asynchronously replicate to the shared filesystem.
- **Cascade failures.** A failure during a checkpoint write is the worst case — the checkpoint is partially written and corrupted. The training run must fall back to the *previous* checkpoint, doubling the lost work. Atomic writes (write to a temp location, rename on completion) and checkpoint validation (read back and verify checksums) prevent this but add complexity and time.
- **The resume tax.** Loading a checkpoint isn't instant. 1TB of state from storage into 16,384 GPUs takes time — even at 200GB/s aggregate read bandwidth, that's 5 seconds of I/O, plus the time to redistribute sharded state, reinitialize NCCL communicators, verify all GPUs are in sync, and run a test step. Realistic resume time is 2-10 minutes, during which 16,384 GPUs are idle.

### Performance Profile
- **Storage I/O:** 700GB-1TB burst write every 500-2,000 steps. Async checkpoint hides most of this behind compute.
- **Time cost:** Near zero if async checkpointing works and storage keeps up. 5-30 seconds of visible stall if it doesn't.
- **Failure cost:** $270K-$540K per incident in lost compute at Meta's reported scale. The checkpoint interval is a direct tradeoff between failure cost and I/O overhead.
- **Weak points:** Storage burst write capacity, filesystem metadata handling at thousands of simultaneous writers, partial-write corruption risk, resume latency. The entire reliability model of multi-week training rests on checkpointing working perfectly.

### Sources
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — failure rates and checkpoint cadence
