---
title: "Post-training"
date: 2026-05-19
draft: false
node_id: "1f"
tier: 1
parent: "0"
children:
  - "2w"
  - "2x"
  - "2y"
siblings:
  - "1a"
  - "1b"
  - "1c"
  - "1d"
  - "1e"
  - "1g"
url: "/series/training/train-from-scratch/post-training/"
summary: "Pretraining teaches the model to continue text. Post-training teaches it what kind of continuation is acceptable, useful, formatted, safe, and rewarded."
series: "training"
mermaid: true
weight: 150
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture] --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop] --> F[Post-Training]:::hl --> G[Evaluation]
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

Pretraining teaches the model to continue text. Post-training teaches it what kind of continuation is acceptable, useful, formatted, safe, and rewarded. This is where a base model becomes the assistant people actually interact with.

Much of what feels like "assistant personality" — refusal behavior, formatting discipline, tool-use style, conversational helpfulness — comes from post-training and system scaffolding, not from pretraining. Users experience post-trained assistants, not raw base models.

The post-training pipeline is a sequence of stages, not a single step: continued pretraining, supervised fine-tuning (SFT), preference training (RLHF, DPO), and reinforcement learning with verifiable rewards. Each stage shapes different aspects of model behavior, and critically, they differ in *which tokens count toward the loss* — they are not merely "same training, different data."
