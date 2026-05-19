---
title: "The training loop"
date: 2026-05-19
draft: false
node_id: "1e"
tier: 1
parent: "0"
children:
  - "2t"
  - "2u"
  - "2v"
siblings:
  - "1a"
  - "1b"
  - "1c"
  - "1d"
  - "1f"
  - "1g"
url: "/series/training/train-from-scratch/training-loop/"
summary: "The training step is what happens once. The training loop is what happens 3.75 million times."
series: "training"
mermaid: true
weight: 140
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture] --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop]:::hl --> F[Post-Training] --> G[Evaluation]
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

The training step (1d) is what happens once. The training loop is what happens 3.75 million times. The loop wraps the step in the larger context of how data flows across epochs, how batch size and learning rate evolve, and how compute budgets constrain the total duration.

For Llama 3 70B: 15 trillion tokens, ~4 million tokens per step, ~3.75 million optimizer updates. Each update touches all 70 billion parameters. The decisions that shape the loop — how many epochs, what batch size, what learning rate schedule, when to stop — are not afterthoughts. They're driven by scaling laws that determine how much training compute buys you, and they collectively determine whether the model converges to something useful or plateaus short.
