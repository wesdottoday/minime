---
title: "Epochs"
date: 2026-05-19
draft: false
node_id: "2t"
tier: 2
parent: "1e"
children:
  - "3j"
siblings:
  - "2u"
  - "2v"
url: "/series/training/train-from-scratch/training-loop/epochs/"
summary: "If your dataset has 2 trillion unique tokens and your training run processes 15 trillion tokens total, that's roughly 7-8 epochs — the model sees every piece of training data about 7-8 times."
series: "training"
mermaid: true
weight: 240
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture] --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop]:::hl --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click E "/series/training/train-from-scratch/training-loop/"
```

An epoch is one complete pass through the entire training dataset. If your dataset has 2 trillion unique tokens and your training run processes 15 trillion tokens total, that's roughly 7-8 epochs — the model sees every piece of training data about 7-8 times.

## Model size vs. training data — a critical distinction

The parameter count (70B) is fixed before training starts. It's an architecture decision — 80 layers, d_model 8192, 64 attention heads, FFN hidden dim 28672 — those choices mathematically produce ~70 billion weight values. Training doesn't change the *number* of weights. It changes the *values*. Before training: 70 billion random numbers. After training: the same 70 billion numbers, tuned so the model can predict next tokens well. If you trained the same architecture on 1T tokens, it's still 70B — just worse. On 50T tokens, still 70B — just better (up to a point). The data determines how good the model is at its fixed size.

Each of the 15 trillion tokens is a learning opportunity — the model predicts, gets it wrong, and the optimizer nudges every parameter in the direction that would have made the prediction better. Multiply that by 15T tokens worth of steps and you've got the trained model.

## Chinchilla, over-training, and why Meta did 7-8 epochs

[Chinchilla scaling laws](/series/training/train-from-scratch/training-loop/scaling-laws/) (Hoffmann et al., 2022) say that for a 70B model, compute-optimal training needs ~1.4T tokens — about 20 tokens per parameter. One epoch on a 1.4T dataset and you're done.

Meta deliberately over-trained: 15T tokens on a ~2T unique-token dataset. This is not the same as blindly repeating data. The insight is about the *downstream tradeoff*: a 70B model trained on 15T tokens performs closer to a larger model (100B+) that was trained Chinchilla-optimally. You overspend on training once, but at inference time you're serving a smaller, cheaper model that punches above its weight. Optimized for lifetime cost, not training cost.

## Diminishing returns vs. over-training — not a contradiction

These sound like they conflict but they don't. Diminishing returns (Muennighoff et al., "Scaling Data-Constrained Language Models") are about *repeating the same data*: each additional epoch provides less learning signal. First pass — massive learning. Second — still significant. Third and fourth — diminishing. Beyond ~4 epochs, the model starts memorizing specific sequences rather than generalizing.

Meta accepted this. The marginal gains from later epochs are real but shrinking. They decided those diminishing gains were still *worth it* because the alternative was deploying a bigger, more expensive model. The ~2T unique-token dataset is a constraint, not a target — if they had 10T unique tokens, they'd have used all 10T and done fewer epochs. More unique data is always better than repeated data.

The reason repeated data still helps at all: each pass nudges the weights from a different starting point. Epoch 1, the weights are near-random — gradients are huge, learning is fast. By epoch 7, the weights are already good — gradients are small, adjustments are fine-grained. Early epochs do heavy lifting; later epochs polish.

## Epoch boundaries are invisible

In practice, the data loader shuffles both shard order and within-shard sequence order between epochs. The model doesn't experience a hard restart — no signal says "you've now seen everything once." It's a continuous stream of batches where, statistically, after enough steps every sequence has been visited. The transition from epoch 1 to epoch 2 is invisible to the model. This is deliberate — you don't want the model to learn ordering artifacts.

## Data mixing complicates the count

Meta upsampled code and math relative to their natural proportion. If code gets seen 12 times while web text gets seen 5 times, there's no single "epoch count" — different data categories sit at different effective epoch counts based on the mixing strategy.

## The frontier constraint

The uncomfortable truth: we're running out of high-quality text data. The entire internet, filtered and cleaned, yields maybe 5-10T tokens of genuinely useful training data. Frontier models are already multi-epoch on everything that exists. This is why synthetic data generation, data augmentation, and more efficient learning-per-token are active research — the "just get more data" strategy has a ceiling.

### Performance Profile
- **Epoch count:** Determined by total training tokens / unique dataset size. Llama 3 70B: ~15T / ~2T = ~7-8 epochs.
- **Diminishing returns:** First 1-2 epochs contribute the majority of learning. Beyond ~4 epochs, risk of memorization increases and generalization gains shrink.
- **Over-training tradeoff:** More training tokens = higher training cost but smaller deployable model for equivalent capability. Meta optimized for inference cost, not training cost.
- **Data ceiling:** 5-10T tokens of high-quality text exist globally. Frontier models are already multi-epoch on all of it. Synthetic data and data augmentation are the active research fronts.

### Sources
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Hoffmann et al. (Chinchilla scaling laws)
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) — Muennighoff et al. (diminishing returns from repeated data)
