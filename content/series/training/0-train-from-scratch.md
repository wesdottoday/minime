---
title: "What happens when you train a model from scratch?"
date: 2026-05-14
draft: false
node_id: "0"
tier: 0
parent: ""
children:
  - "1a"
  - "1b"
  - "1c"
  - "1d"
  - "1e"
  - "1f"
  - "1g"
siblings: []
url: "/series/training/train-from-scratch/"
summary: "You have a pile of text — trillions of tokens scraped from the internet, books, code repositories, scientific papers. You have a model architecture — a specific arrangement of transformer layers, attention heads, and feed-forward networks."
series: "training"
mermaid: true
weight: 0
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data]:::hl --> B[Architecture]:::hl --> C[Hardware & Scale]:::hl --> D[Training Step]:::hl --> E[Training Loop]:::hl --> F[Post-Training]:::hl --> G[Evaluation]:::hl
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

You have a pile of text — trillions of tokens scraped from the internet, books, code repositories, scientific papers. You have a model architecture — a specific arrangement of transformer layers, attention heads, and feed-forward networks, all defined by a set of hyperparameters somebody chose. And you have a cluster of GPUs — in our case, a GB200 NVL72 rack with 72 Blackwell GPUs connected by NVLink. The model's weights start as random noise. The goal is to adjust those billions of random numbers until the model can predict the next token in a sequence better than chance, and then better than most humans would.

Training is a loop. You feed a batch of text sequences into the model (forward pass), compare the model's predictions against what actually comes next (loss calculation), compute how each weight contributed to the error (backward pass), and nudge every weight slightly in the direction that would have made the prediction better (optimizer step). Then you do it again. And again. For Llama 3 70B, this loop ran for roughly 15 trillion tokens — meaning the model saw and learned from 15 trillion next-token predictions. On a GB200 NVL72 rack, the 70B parameter model doesn't fit on a single GPU (70 billion parameters at 16-bit precision = ~140GB, and that's just the weights before you account for optimizer states, gradients, and activations), so the training is distributed across all 72 GPUs using a combination of parallelism strategies that split the work across data, model layers, and tensor dimensions simultaneously. Every step requires the GPUs to synchronize — sharing gradients, passing activations between pipeline stages, coordinating which chunk of data each GPU is working on. The efficiency of this coordination is what separates a training run that finishes in weeks from one that would take years.

But training the base model is not the end of the story. After pretraining, the model goes through post-training — supervised fine-tuning, preference learning, and reinforcement learning — that transforms a next-token predictor into the assistant people actually interact with. And throughout all of it, evaluation determines whether the training is working, whether the model is safe, and whether it's ready to ship.

### Performance Profile
- **Compute-bound:** Matrix multiplications in forward and backward passes (the actual math of transforming inputs through layers)
- **Memory-bound:** Storing model weights + optimizer states + gradients + activations simultaneously (the "memory wall" that forces distribution across GPUs)
- **Network-bound:** Gradient synchronization across GPUs, activation transfers between pipeline stages, data parallel all-reduce operations (NVLink bandwidth is the ceiling)

### Sources
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — Meta's technical report covering architecture, training data, and scaling decisions referenced throughout this series.
