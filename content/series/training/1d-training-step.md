---
title: "The training step"
date: 2026-05-19
draft: false
node_id: "1d"
tier: 1
parent: "0"
children:
  - "2m0"
  - "2m"
  - "2n"
  - "2o"
  - "2p"
  - "2q"
  - "2r"
  - "2s"
siblings:
  - "1a"
  - "1b"
  - "1c"
  - "1e"
  - "1f"
  - "1g"
url: "/series/training/train-from-scratch/training-step/"
summary: "One training step is where the model actually learns. It's the atomic unit of progress — everything else is about how many steps you take and how you configure them."
series: "training"
mermaid: true
weight: 130
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture] --> C[Hardware & Scale] --> D[Training Step]:::hl --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click A "/series/training/train-from-scratch/training-data/"
    click B "/series/training/train-from-scratch/model-architecture/"
    click C "/series/training/train-from-scratch/hardware-and-scale/"
    click D "/series/training/train-from-scratch/training-step/"
    click E "/series/training/train-from-scratch/training-loop/"
    click F "/series/training/train-from-scratch/post-training/"
    click G "/series/training/train-from-scratch/evaluation/"
```

One training step is where the model actually learns. It's the atomic unit of progress — everything else (epochs, batch size, learning rate schedule) is about how many steps you take and how you configure them.

Each step follows seven phases in order:

1. **[Data loading](/series/training/train-from-scratch/training-step/data-loading/)** — Get the next batch of tokens from storage onto the right GPUs
2. **[Forward pass](/series/training/train-from-scratch/training-step/forward-pass/)** — Push tokens through all 80 layers, producing predictions
3. **[Loss calculation](/series/training/train-from-scratch/training-step/loss-calculation/)** — Compare predictions to actual next tokens, compute error
4. **[Backward pass](/series/training/train-from-scratch/training-step/backward-pass/)** — Propagate error back through all 80 layers, computing gradients
5. **[Gradient synchronization](/series/training/train-from-scratch/training-step/gradient-sync/)** — Aggregate gradients across all GPUs that need to agree
6. **[Optimizer step](/series/training/train-from-scratch/training-step/optimizer-step/)** — Use the aggregated gradients to update every weight in the model
7. **[Checkpointing](/series/training/train-from-scratch/training-step/checkpointing/)** — Periodically save the full training state to storage (not every step)

Each phase has different bottlenecks, different failure modes, and different hardware demands. At 227 racks, the weak points aren't theoretical — a GPU memory error, a network switch flap, a slow storage node — any of these can stall or corrupt the entire training step across all 16,384 GPUs.

```mermaid
graph LR
    P1[Data Loading]:::phase --> P2[Forward Pass]:::phase --> P3[Loss]:::phase --> P4[Backward Pass]:::phase --> P5[Gradient Sync]:::phase --> P6[Optimizer]:::phase --> P7[Checkpoint]:::phase
    classDef phase fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
```
