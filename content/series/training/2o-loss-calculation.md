---
title: "Phase 3 — Loss calculation"
date: 2026-05-14
draft: false
node_id: "2o"
tier: 2
parent: "1d"
children:
  - "3f"
siblings:
  - "2m"
  - "2n"
  - "2o"
  - "2p"
  - "2q"
  - "2r"
  - "2s"
url: "/series/training/train-from-scratch/training-step/loss-calculation/"
summary: "The forward pass produced a prediction for every token position — a probability distribution over 128,000 tokens. The loss calculation is where you find out how wrong those predictions were."
series: "training"
mermaid: true
weight: 232
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

The forward pass produced a prediction — for every token position in every sequence in the batch, the model outputs a probability distribution over the entire 128,000-token vocabulary. "Given everything before this position, here's how likely each possible next token is." The loss calculation is where you find out how wrong those predictions were.

**The function: Cross-entropy loss.** For each token position, you know what actually came next in the training data (the "label"). The model produced a probability for that correct token. Cross-entropy loss is just the negative log of that probability. If the model assigned 90% probability to the correct token, the loss is -log(0.9) = 0.105 — small, good. If it assigned 1% probability, the loss is -log(0.01) = 4.6 — large, bad. The total loss is the average across every token position in the batch.

**The math for one step at scale:** Each micro-batch is 4 sequences × 8,192 tokens = 32,768 token positions per [data-parallel replica](/series/training/train-from-scratch/training-step/training-units/). Each position produces a 128,000-dimensional vector (one logit per vocab entry). That vector goes through softmax to become a probability distribution, then cross-entropy picks out the probability of the correct token. With 512 data-parallel replicas and gradient accumulation, the global batch reaches roughly 4 million token positions per optimizer step — each one producing an individual loss value, all averaged into one number.

**Why this matters more than it sounds:** This single number — the loss — is the *only signal* driving the entire training run. Every gradient, every weight update, every dollar spent on compute traces back to this one scalar. If the loss calculation is wrong, even slightly, the model learns the wrong thing for trillions of tokens. There's a subtlety we touched on in the LLMs series that's worth repeating: the training process treats the training data as *truth*. If the data says "The capital of France is Lyon," the loss function penalizes the model for predicting "Paris." The loss function has no notion of factual accuracy — it only knows what the data says comes next.

**The logit computation:** Before cross-entropy, the model's final layer output (an 8,192-dimensional vector per token) needs to be projected into vocabulary space. This is the "language model head" — a matrix multiplication of `d_model × vocab_size` = 8,192 × 128,000 = ~1 billion parameters. (Often this matrix is the same as the embedding table, transposed — called "weight tying." Llama 3 does not tie weights, so this is a separate 1B-parameter matrix.) This projection is actually one of the larger single operations in the model — producing a 128K-dimensional vector for each of the 32,768 tokens in a micro-batch means materializing a tensor of `32,768 × 128,000 = 4.2 billion` floats. At FP16, that's ~8GB just for the logits of one micro-batch.

**Where it breaks at scale:** Honestly, it mostly doesn't. Loss calculation is embarrassingly parallel — each GPU computes loss on its own micro-batch independently. No communication required. The only coordination is that the losses eventually get averaged across all data-parallel replicas to produce a single loss value for logging and learning rate scheduling, but that's a tiny scalar all-reduce — negligible.

The real risk is numerical. Softmax over 128,000 entries is prone to overflow and underflow — the exponentials can blow up or vanish. Every implementation uses the "log-sum-exp trick" (subtract the max logit before exponentiating) to keep things numerically stable. At FP16/BF16 precision, this is even more critical. A bad softmax implementation at half precision can silently produce garbage probabilities, and the loss looks plausible enough that nobody notices until the model is mysteriously bad at rare tokens 10,000 steps later.

### Performance Profile
- **Compute:** The LM head projection (8,192 × 128,000 matmul) is significant — comparable to one FFN layer. The softmax and cross-entropy after it are cheap by comparison.
- **Memory:** The logit tensor (32,768 × 128,000 × 2 bytes = ~8GB per micro-batch) is a transient spike. It's computed, used for loss, and immediately discarded — but it competes for HBM at the moment it exists.
- **Communication:** Near zero. Each GPU computes its own loss independently. One scalar all-reduce for the global loss average.
- **Weak point:** Numerical precision. FP16 softmax over 128K entries is a minefield. BF16 helps (wider dynamic range), and most implementations use FP32 for the softmax computation even when the rest of training is lower precision.
