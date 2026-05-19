---
title: "Training stability: loss spikes, NaN, and precision"
date: 2026-05-19
draft: false
node_id: "3k"
tier: 3
parent: "2v"
children: []
siblings: []
url: "/series/training/train-from-scratch/training-loop/scaling-laws/training-stability/"
summary: "A training run at scale is a delicate numerical process running for millions of steps across thousands of GPUs. Training stability is the emergent property of many interacting design choices — normalization, precision, clipping, warmup — each contributing a small margin of safety."
series: "training"
mermaid: true
weight: 342
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

A training run at scale is a delicate numerical process running for millions of steps across thousands of GPUs. It can fail in ways that have nothing to do with the model architecture, the data quality, or the learning rate. Training stability is the engineering discipline of keeping the numbers sane.

## Loss spikes

The loss curve for a well-behaved training run decreases smoothly (with noise) over millions of steps. A loss spike is a sudden, dramatic increase — the loss jumps from 2.1 to 15.0 in a single step, then gradually recovers (or doesn't). Causes include:

- **Bad data batches.** A batch containing anomalous data (extremely long sequences, degenerate text, encoding errors) can produce outsized gradients that push weights into unstable territory. The data pipeline's quality filters catch most of this, but at [~4 million tokens per step](/series/training/train-from-scratch/training-step/training-units/) spread across 512 data-parallel replicas, rare anomalies slip through.
- **Learning rate too high.** If the learning rate is slightly too aggressive for the current phase of training, the optimizer overshoots — weights move too far, the loss jumps, and the model must spend many steps recovering. This is especially dangerous near the warmup-to-peak transition.
- **Numerical cascade.** A small precision error in one layer (e.g., a borderline overflow in BF16 attention) produces a slightly wrong activation, which propagates through subsequent layers, amplifying at each step. By the output layer, the error is large enough to produce a bad loss, bad gradients, and bad weight updates.
- **Distributed synchronization glitches.** If one GPU's gradient is corrupted (bit flip, partial NaN) but not detected, the all-reduce mixes the corrupted gradient into the aggregate. All replicas receive a bad update.

**Recovery from loss spikes.** If the spike is transient (bad data batch), the model usually self-corrects within a few hundred steps as subsequent batches provide correct gradient signals. If the spike destabilized the optimizer state (Adam's running averages absorbed the anomaly), recovery is slower. In severe cases, teams roll back to a checkpoint from before the spike and skip the offending data batch.

Meta reported that during Llama 3 training, they encountered loss spikes that required manual intervention — rolling back to a checkpoint and adjusting the data loading to skip problematic data regions. This is operational reality at scale: training runs have human operators watching dashboards.

## NaN propagation — the silent killer

NaN (Not a Number) results from undefined operations: 0/0, infinity minus infinity, square root of a negative number. In training, NaN typically originates from:

- **Softmax overflow.** Logits larger than ~88 (at FP32) or ~11 (at BF16) cause exp() to overflow to infinity. Infinity / infinity = NaN. The log-sum-exp trick prevents this, but a buggy implementation or an extreme input can bypass the trick.
- **Normalization of zero-variance tensors.** RMSNorm divides by the root-mean-square of the input. If the input is all zeros (which shouldn't happen but can in degenerate edge cases), division by zero produces NaN.
- **Gradient explosion.** Through 80 layers of chain rule, a gradient can grow exponentially. If it exceeds the representable range of the format, it becomes infinity. Infinity x 0 (from a dropout mask or a zeroed weight) = NaN.

Once a NaN appears in any tensor, it propagates. NaN + anything = NaN. NaN x anything = NaN. A single NaN in one attention head in one layer on one GPU spreads through the forward pass to every subsequent layer, enters the loss, propagates through the backward pass to every gradient, enters the all-reduce, and contaminates every GPU's gradients. In one step, one bad number infects the entire cluster.

## Detection and recovery

Production training runs include NaN checks:
- **Per-step loss check:** If the loss is NaN or infinity, halt the step, discard the gradients, log the event, and retry with the next batch.
- **Periodic gradient checks:** Scan the gradient tensor for NaN values after the backward pass. If detected, discard and retry.
- **Checkpoint rollback:** If NaN events become frequent (suggesting a corrupted weight state), roll back to the last clean checkpoint.

These checks add overhead (~1-2% of step time) but are non-negotiable at scale. A training run that doesn't check for NaN will eventually produce a silently corrupted model.

## Gradient clipping

The standard defense against gradient explosion. After the backward pass computes all gradients, the global gradient norm is computed (the L2 norm across all 70 billion gradients). If the norm exceeds a threshold (typically 1.0), all gradients are scaled down proportionally: `gradient = gradient x (max_norm / actual_norm)`. This caps the magnitude of any single update without changing the direction.

Clipping doesn't fix the underlying instability — it prevents catastrophic consequences while the model works through whatever caused the large gradients. It's a guardrail, not a cure. But without it, a single unstable step can permanently damage the model's weights.

## Warmup: not ritual, but engineering

At the very start of training, the model weights are random. The loss is at its maximum. Gradients are the largest they'll ever be. If you apply a full-sized learning rate to these enormous gradients, the first weight updates are catastrophically large — the model jumps to an unstable region of the loss landscape and may never recover.

Warmup starts the learning rate near zero and linearly increases it over the first ~2,000 steps. During warmup, the gradients are large but the learning rate is small, so the actual weight updates are controlled. By the time the learning rate reaches its peak, the model has settled into a stable region of the loss landscape where gradients are more reasonable.

The warmup duration is a hyperparameter. Too short and the model doesn't stabilize before the full learning rate hits. Too long and you waste training compute at a suboptimal learning rate. Meta used ~2,000 steps for Llama 3 70B — about 0.05% of total training, but those first 2,000 steps are disproportionately important for the entire run's stability.

## Pre-LN vs. Post-LN: normalization placement matters

The original transformer placed layer normalization *after* each sub-layer (post-norm): attention -> add residual -> LayerNorm -> FFN -> add residual -> LayerNorm. Modern models like Llama 3 use pre-norm: LayerNorm -> attention -> add residual -> LayerNorm -> FFN -> add residual.

The difference is in gradient flow. In post-norm, the normalization sits between the residual connection and the next sub-layer. Gradients flow through the normalization, which can dampen them. Through 80 layers, this dampening accumulates — gradients at the bottom layers can be orders of magnitude smaller than at the top (vanishing gradients).

In pre-norm, the residual connection carries gradients *around* the normalization. The gradient highway from layer 80 to layer 1 passes through addition operations (which preserve gradient magnitude) rather than normalization operations (which can shrink it). This makes deep models (80+ layers) much more stable to train.

GPT-2 discovered this empirically — post-norm training became unstable at 48 layers, pre-norm was stable. Every large-scale model since has used pre-norm. It's not optional at Llama 3's depth.

## The precision-stability interaction

[Precision choices](/series/training/train-from-scratch/training-step/optimizer-step/mixed-precision/) (3i) directly affect stability:

- **BF16 attention scores.** The dot product of query and key vectors can produce values that are fine at FP32 but problematic at BF16 (7 bits mantissa). When these values enter softmax (exponentiation), small precision differences get amplified into noticeably different probability distributions. This is why attention softmax is often computed in FP32 even when the surrounding operations are BF16.
- **FP8 gradient noise.** At 3 mantissa bits, FP8 gradients are inherently noisy. A gradient of 0.037 might be represented as 0.03125 or 0.0625 — either 15% too low or 69% too high. Across 70 billion parameters, these errors average out... mostly. But they add stochastic noise to every update, which can slow convergence or trigger instabilities in sensitive regions of the loss landscape.
- **Accumulation precision.** When summing many small numbers (as in gradient accumulation across micro-batches), BF16 loses information. The sum of 128 BF16 values can have significant rounding error. Accumulation is therefore done in FP32 — collect the BF16 values from each micro-batch, but accumulate the running sum in FP32.

## The meta-point

Training stability is not a single technique. It's the emergent property of many interacting design choices — normalization placement, precision, clipping threshold, warmup schedule, learning rate, batch size, data quality — each one contributing a small margin of safety. Remove any one and the training might still work. Remove two and it probably won't. At scale, stability is the product of dozens of small engineering decisions, each one individually seeming cosmetic but collectively determining whether the run succeeds.

## Performance Profile

- **NaN detection overhead:** ~1-2% of step time for gradient and loss checks. Non-negotiable at scale.
- **Gradient clipping overhead:** Negligible — one norm computation and one scalar multiplication across all gradients. The cost is in the all-reduce to compute the global norm, which is a tiny scalar operation.
- **Warmup cost:** ~2,000 steps at suboptimal learning rate = ~0.05% of total training. Negligible in compute, critical for stability.
- **Loss spike recovery cost:** 500-2,000 steps of progress lost per spike (rollback to last clean checkpoint). At Meta's scale, each recovery costs $130K-$540K in compute time.
- **The stability tax:** The sum of all stability measures (checkpointing, NaN detection, gradient clipping, conservative learning rates, warmup) adds ~5-10% to total training cost. Without them, the training would likely fail entirely — the stability tax is the cost of the run actually completing.
