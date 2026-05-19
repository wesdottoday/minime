---
title: "Phase 6 — Optimizer step"
date: 2026-05-14
draft: false
node_id: "2r"
tier: 2
parent: "1d"
children:
  - "3h"
  - "3i"
siblings:
  - "2m"
  - "2n"
  - "2o"
  - "2p"
  - "2q"
  - "2r"
  - "2s"
url: "/series/training/train-from-scratch/training-step/optimizer-step/"
summary: "The gradients are synced. Time to actually update the weights. This is the moment the model learns — and where AdamW's per-parameter adaptive learning rates and 840GB of optimizer state earn their keep."
series: "training"
mermaid: true
weight: 235
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

The gradients are synced. Every data-parallel replica now has identical averaged gradients for all 70 billion parameters. Time to actually update the weights. This is the moment the model learns.

The naive version would be: `new_weight = old_weight - learning_rate × gradient`. That's vanilla SGD (stochastic gradient descent). It works but it's terrible. The loss landscape of a 70B-parameter model is a surface in 70-billion-dimensional space, full of narrow ravines, saddle points, flat plateaus, and sharp cliffs. Vanilla SGD oscillates in the ravines, gets stuck on plateaus, and overshoots cliffs. Modern training uses **Adam** (or its variant **AdamW**), which is dramatically better. Virtually every large language model — Llama 3, GPT-4, Claude, Gemini — is trained with AdamW.

**What Adam does:** Instead of just using the current gradient to update each weight, Adam tracks two running statistics per parameter:

1. **First moment (m):** An exponential moving average of the gradient. This is the *momentum* — it smooths out noisy gradients by averaging over recent history. If a gradient has been consistently pointing left for the last 100 steps, the momentum carries the weight leftward even if this step's gradient is noisy or slightly rightward. Typical decay rate: β₁ = 0.9 (90% of old momentum + 10% of new gradient).

2. **Second moment (v):** An exponential moving average of the *squared* gradient. This tracks how volatile each parameter's gradient has been. If a parameter's gradient is consistently large, v is large. If it's tiny and stable, v is small. Adam uses this to *adapt the learning rate per parameter* — parameters with volatile gradients get smaller updates (caution), parameters with stable gradients get larger updates (confidence). Typical decay rate: β₂ = 0.999.

The update rule: `weight = weight - lr × m / (√v + ε)`. The division by √v is the key — it normalizes the update by the gradient's historical magnitude, giving each of the 70 billion parameters its own effective learning rate.

**AdamW adds weight decay:** A regularization term that nudges weights toward zero each step, preventing them from growing unboundedly. The "W" in AdamW means the weight decay is decoupled from the gradient-based update — a subtle but important distinction from the original Adam that produces better training dynamics.

**The memory cost — this is the killer:** Adam requires storing *two additional tensors* the size of the entire model:
- First moment (m): 70B floats
- Second moment (v): 70B floats

Plus you still have the weights themselves and the gradients. So the optimizer step requires four model-sized tensors simultaneously:

| Tensor | Size at FP32 | Size at mixed precision |
|--------|-------------|------------------------|
| Weights | 70B × 4 bytes = 280GB | 70B × 2 bytes = 140GB (BF16) |
| Gradients | 70B × 4 bytes = 280GB | 70B × 2 bytes = 140GB (BF16) |
| Adam m (first moment) | 70B × 4 bytes = 280GB | 70B × 4 bytes = 280GB (kept FP32) |
| Adam v (second moment) | 70B × 4 bytes = 280GB | 70B × 4 bytes = 280GB (kept FP32) |
| **Total** | **1,120GB** | **~840GB** |

Even in mixed precision, the optimizer states alone (m + v at FP32) are 560GB — *4x the size of the model weights in BF16.* This is why we said earlier that the model "doesn't fit on one GPU" is an understatement. The weights might fit on one 192GB Blackwell GPU. The training state never does.

**Why are the optimizer states kept in FP32?** The m and v tensors are running averages updated with tiny increments every step. At BF16 precision (7 bits of mantissa), small updates get rounded to zero — the running average stops tracking. Over thousands of steps, this rounding error accumulates and the optimizer loses its ability to adapt learning rates per parameter. FP32 (23 bits of mantissa) has enough resolution to track these tiny incremental updates. This is one of the core ideas of mixed-precision training: do the big matrix multiplications (forward/backward) in BF16 for speed, but keep the optimizer state in FP32 for precision.

**[ZeRO optimization](/series/training/train-from-scratch/training-step/optimizer-step/zero/) (Zero Redundancy Optimizer):** In pure data parallelism, every replica stores the full optimizer state — all 840GB. That's 840GB × 512 replicas = 430TB of redundant memory across the cluster. ZeRO, developed by Microsoft (DeepSpeed), partitions the optimizer state across data-parallel replicas. ZeRO Stage 1: shard the optimizer states (m and v). Each replica holds 1/512th of m and v — about 1.1GB each instead of 560GB. When a replica needs to update weights for its shard, it computes the update locally. When it needs the full updated weights, it does an all-gather. This trades a bit of extra communication for massive memory savings, and it's how large-scale training actually fits in memory.

**The learning rate schedule:** The learning rate isn't constant. Llama 3 70B used a cosine learning rate schedule with warmup: start at near-zero, linearly increase to peak over the first ~2,000 steps (warmup), then gradually decay following a cosine curve over the remaining hundreds of thousands of steps. The warmup prevents the randomly-initialized model from making catastrophically large updates on the first few steps when gradients are wild. The cosine decay reduces the learning rate as training progresses, allowing finer-grained adjustments as the model converges.

**Where it breaks at 227 racks:**

- **Memory fragmentation.** The optimizer step allocates and deallocates large temporary buffers. Over thousands of steps, GPU memory can fragment — enough total free memory but no contiguous block large enough for the next allocation. This causes OOM crashes that look random and are notoriously hard to debug. Memory pooling and pre-allocation mitigate this.
- **ZeRO communication overhead.** Sharding optimizer states means an extra all-gather to reconstruct full weights after the update. This is additional communication on top of gradient sync, though it can overlap with the start of the next step's data loading.
- **Learning rate sensitivity.** The learning rate schedule is a hyperparameter that took Meta multiple trial runs to tune. Too high and training diverges (loss explodes). Too low and training converges too slowly, wasting compute. At $millions per run, getting the schedule wrong is expensive. This is typically tuned on smaller-scale runs first and then transferred, but it doesn't always transfer cleanly.

### Performance Profile
- **Compute:** Light compared to forward/backward — just element-wise operations (multiply, divide, sqrt) across 70B parameters. A few seconds at most.
- **Memory:** This is the memory peak of the entire training process. Weights + gradients + two optimizer states = 840GB+ in mixed precision, before ZeRO sharding.
- **Communication:** ZeRO all-gather for weight reconstruction after sharded updates. Overlaps with next step startup.
- **Weak points:** Optimizer state memory dominates total memory footprint (4x model weights at FP32). FP32 requirement is non-negotiable without quality loss. Learning rate schedule is a multi-million-dollar hyperparameter.
