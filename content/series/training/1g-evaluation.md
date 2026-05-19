---
title: "Evaluation"
date: 2026-05-19
draft: false
node_id: "1g"
tier: 1
parent: "0"
children:
  - "2z"
  - "2aa"
siblings:
  - "1a"
  - "1b"
  - "1c"
  - "1d"
  - "1e"
  - "1f"
url: "/series/training/train-from-scratch/evaluation/"
summary: "Training loss tells you whether the model is getting better at predicting held-out tokens. It does not tell you whether the model is getting better at being useful, safe, honest, robust, or non-contaminated."
series: "training"
mermaid: true
weight: 160
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture] --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]:::hl
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

Training loss tells you whether the model is getting better at predicting held-out tokens. It does not tell you whether the model is getting better at being useful, safe, honest, robust, or non-contaminated. Evaluation is where you find out.

Multiple evaluation types exist because they answer different questions. Training metrics track optimization progress. Capability benchmarks test specific skills. Safety evaluations check guardrails. Human preference evaluations compare overall quality. Product telemetry measures real-world performance. No single metric captures readiness — evaluation is the discipline of knowing which signals to trust and when.
