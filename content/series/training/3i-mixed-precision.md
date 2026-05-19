---
title: "Mixed precision training"
date: 2026-05-19
draft: false
node_id: "3i"
tier: 3
parent: "2r"
children: []
siblings:
  - "3h"
url: "/series/training/train-from-scratch/training-step/optimizer-step/mixed-precision/"
summary: "Every floating-point operation in training has a precision. More bits means more accuracy but more memory and slower compute. Mixed precision uses different precisions for different parts of the computation — BF16 for speed where it's safe, FP32 where it's not."
series: "training"
mermaid: true
weight: 351
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

Every floating-point operation in training has a precision — the number of bits used to represent the number. More bits means more accuracy but more memory and slower compute. Fewer bits means faster and smaller but introduces rounding errors that, over millions of steps, can silently degrade or outright kill a training run. Mixed precision training uses different precisions for different parts of the computation, putting high precision where it matters and low precision where speed matters.

## The precision landscape

| Format | Bits | Exponent | Mantissa | Dynamic range | Use in training |
|--------|------|----------|----------|---------------|----------------|
| FP32 | 32 | 8 | 23 | +/-3.4x10^38 | Optimizer states, loss accumulation |
| FP16 | 16 | 5 | 10 | +/-65,504 | Older training; risky without loss scaling |
| BF16 | 16 | 8 | 7 | +/-3.4x10^38 | Forward/backward pass (modern standard) |
| FP8 (E4M3) | 8 | 4 | 3 | +/-448 | Forward pass compute (emerging) |
| FP8 (E5M2) | 8 | 5 | 2 | +/-57,344 | Backward pass gradients (emerging) |

## Why BF16 won over FP16

FP16 has 5 exponent bits, giving it a dynamic range of +/-65,504. FP16 has limited dynamic range and small gradients can underflow to zero. Loss scaling multiplies the loss by a large factor (say, 1024) before backpropagation so gradients are shifted into representable range, then unscales before the optimizer step. If the scale is too high, gradients overflow and the step is skipped and retried with a lower scale.

BF16 has 8 exponent bits — the same as FP32 — giving it the full +/-3.4x10^38 dynamic range. Gradients that would overflow FP16 are perfectly representable in BF16. No loss scaling needed. The tradeoff: BF16 has only 7 mantissa bits (vs. FP16's 10), so each number has less precision. For matrix multiplications in the forward and backward pass, this lower precision is acceptable — the errors average out across millions of operations. For running averages in the optimizer (Adam's m and v), it's not acceptable — which is why those stay FP32.

BF16 became the default for training on hardware that supports it (all NVIDIA GPUs from A100 onward). It's simpler (no loss scaling), more stable (no overflow risk), and equally fast.

## The mixed-precision recipe

A modern training run (Llama 3 scale) uses:

1. **Forward pass:** BF16. All matrix multiplications, attention, FFN, normalization — everything runs in BF16. Tensor cores on Blackwell GPUs process BF16 matmuls at 2x the speed of FP32.
2. **Backward pass:** BF16. Gradient computations mirror the forward pass in precision. Gradients are computed and stored in BF16.
3. **Loss computation:** FP32. The softmax over 128K vocabulary entries is numerically sensitive. Doing it in BF16 (7 bits mantissa) produces visibly worse probability distributions. The loss scalar itself is accumulated in FP32.
4. **Optimizer states:** FP32. Adam's m and v tensors track running averages with tiny increments. At BF16 precision, updates smaller than ~1% of the current value round to zero — the running average stops updating. FP32's 23 mantissa bits provide enough resolution.
5. **Weight master copy:** FP32. The optimizer computes updates in FP32 on FP32 weight copies. After each step, the FP32 weights are cast to BF16 for the next forward pass. This ensures that small weight updates (which would round to zero in BF16) accumulate over time rather than being silently discarded.

## The FP32 weight master copy is critical

Consider a weight with value 1.0. The optimizer says to subtract 0.00001. In BF16, 1.0 - 0.00001 rounds to 1.0 — the update vanishes. In FP32, 1.0 - 0.00001 = 0.99999. Over 1,000 steps, the BF16 weight hasn't moved. The FP32 weight has moved to 0.99. This is called the **stagnation problem**. Many mixed-precision recipes keep FP32 master weights to avoid small-update stagnation. Some modern BF16 implementations use alternative update strategies, but the core issue remains: tiny updates must accumulate at higher precision somewhere in the system.

## FP8: the frontier

Blackwell GPUs support FP8 tensor core operations at 2x the speed of BF16. FP8 forward passes could theoretically halve training time. But FP8 has severe limitations:

- **E4M3 (4 exponent, 3 mantissa):** Range of +/-448, precision of ~1 part in 8. Used for forward pass computations where the activation values are well-bounded.
- **E5M2 (5 exponent, 2 mantissa):** Range of +/-57,344, precision of ~1 part in 4. Used for backward pass gradients where the dynamic range is more important than precision.

FP8 training requires **per-tensor scaling** — each tensor gets its own scale factor to map its value range into FP8's representable range. This is more complex than BF16 (which needs no scaling) and the tooling is still maturing. Early results from NVIDIA and others show that FP8 training can match BF16 quality for many model architectures, but it's not yet universal. Stability issues at scale (16,384+ GPUs) are still being characterized.

## Per-operation precision decisions

Not all operations tolerate low precision equally:

| Operation | Safe at BF16? | Safe at FP8? | Notes |
|-----------|--------------|-------------|-------|
| FFN matmuls | Yes | Emerging | Large, regular — tolerant of noise |
| Attention QKV projections | Yes | Emerging | Same as FFN — regular matmuls |
| Attention softmax | **No — use FP32** | No | Exponentiation amplifies errors |
| RMSNorm | Yes (element-wise) | Risky | Variance computation sensitive |
| Cross-entropy loss | **No — use FP32** | No | Log of small probabilities underflows |
| Adam m/v updates | **No — use FP32** | No | Running average stagnation |
| Gradient all-reduce | Yes (BF16) | Experimental | Communication volume halves at FP8 |

The pattern: large matmuls are tolerant, reductions (softmax, variance, log) and running averages (optimizer) are not.

## Performance Profile

- **BF16 speedup:** 2x over FP32 on tensor core operations. ~1.5x end-to-end (non-matmul operations don't benefit).
- **FP8 speedup:** 2x over BF16 on tensor core operations. ~1.3x end-to-end (requires per-tensor scaling overhead, and non-matmul ops stay BF16/FP32).
- **Memory savings:** BF16 weights + gradients = 280GB (vs. 560GB at FP32). FP8 weights + gradients = 140GB. Optimizer states stay FP32 regardless.
- **The stagnation risk:** Without FP32 master weights, small parameter updates vanish. This produces a model that appears to train normally (loss decreases) but plateaus earlier than it should — a subtle quality degradation that's hard to detect and expensive to recover from.
- **Precision-stability interaction:** Precision errors compound over millions of steps. A rounding error that's invisible at step 1,000 can produce measurably worse loss at step 100,000. This is why precision choices are validated with long runs, not short ablations.
