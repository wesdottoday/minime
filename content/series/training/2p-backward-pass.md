---
title: "Phase 4 — Backward pass"
date: 2026-05-14
draft: false
node_id: "2p"
tier: 2
parent: "1d"
children:
  - "3g"
siblings:
  - "2m"
  - "2n"
  - "2o"
  - "2p"
  - "2q"
  - "2r"
  - "2s"
url: "/series/training/train-from-scratch/training-step/backward-pass/"
summary: "The backward pass goes top-to-bottom through all 80 layers answering one question for every single parameter: how much did you contribute to the error, and in which direction?"
series: "training"
mermaid: true
weight: 233
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

The forward pass went bottom-to-top through 80 layers producing predictions. The loss told us how wrong those predictions were. Now the backward pass goes top-to-bottom through the same 80 layers answering one question for every single parameter in the model: *how much did you contribute to the error, and in which direction?*

The answer, for each parameter, is its **gradient** — a number that says "if this weight had been slightly larger, the loss would have gone up by this much" (positive gradient) or "down by this much" (negative gradient). For Llama 3 70B, that's 70 billion gradients, one per parameter, computed every single training step.

**How it works — the chain rule at scale:** Backpropagation is just the calculus chain rule applied repeatedly. The loss is a function of the final layer's output. The final layer's output is a function of its weights and its input (which came from layer 79). Layer 79's output is a function of *its* weights and *its* input (from layer 78). And so on, all the way down. The chain rule lets you decompose "how does this weight in layer 12 affect the final loss?" into a chain of local derivatives: layer 80's effect on loss × layer 79's effect on layer 80 × ... × layer 12's effect on layer 13.

In practice, this means the backward pass walks the computation graph in reverse. At each layer, it receives a gradient signal from the layer above ("here's how your output affected the loss") and uses it to compute two things:
1. **Gradients for this layer's weights** — how to update the attention and FFN parameters
2. **Gradients for this layer's input** — passed down to the layer below to continue the chain

**The compute cost: roughly 2x the forward pass.** This surprises people. The backward pass is more expensive than the forward pass because at each layer, you're doing the same matrix multiplications as the forward pass *plus* additional multiplications to compute the weight gradients. The rule of thumb is that backward ≈ 2× forward in FLOPs. For Llama 3 70B, if the forward pass costs X FLOPs, the full forward+backward is ~3X. This is why training is so much more expensive than inference — inference only does the forward pass.

**The activation memory problem:** Here's where the forward pass comes back to haunt you. To compute gradients at layer N, you need the *activations* (intermediate outputs) that were produced at layer N during the forward pass. You need to know what the input to each layer looked like to compute how the weights should change. That means either:

1. **Store all activations during the forward pass.** For Llama 3 70B, the activations for a micro-batch across all 80 layers are enormous. Each layer produces an activation tensor of `batch × seq_len × d_model` = 4 × 8,192 × 8,192 × 2 bytes = ~537MB. Across 80 layers: ~43GB per micro-batch, *per GPU* (for the layers that GPU owns). That's on top of the model weights, gradients, and optimizer states already in HBM. It often doesn't fit.

2. **[Activation checkpointing](/series/training/train-from-scratch/training-step/backward-pass/activation-checkpointing/) (gradient checkpointing).** The standard solution. You store activations at only certain "checkpoint" layers (say, every 4th layer) and throw away the rest. During the backward pass, when you need the activations for a non-checkpointed layer, you re-run the forward pass from the nearest checkpoint to recompute them on the fly. This trades compute for memory — you do extra forward computation during the backward pass, but you cut activation memory by 4x (or whatever your checkpoint interval is). Almost every large-scale training run uses this. The compute overhead is typically 20-35%.

**Pipeline parallelism in reverse:** In the forward pass, activations flowed from pipeline stage 1 → 2 → 3 → 4. In the backward pass, gradients flow in reverse: 4 → 3 → 2 → 1. Stage 4 computes its gradients first and sends the input gradient back to stage 3, which computes its gradients and sends back to stage 2, and so on. The same pipeline bubble problem exists — stages idle while waiting for gradients from the stage above. The 1F1B (one forward, one backward) schedule interleaves forward and backward micro-batches to keep the pipeline fuller, but bubbles remain.

**Tensor parallelism in the backward pass:** Same communication pattern as the forward pass, but in reverse. The partial gradient results need to be combined across the tensor-parallel group via all-reduce, just like the partial forward results did. Same NVLink requirement. Same per-layer communication overhead. The backward pass doesn't introduce new communication patterns — it mirrors the forward pass.

**Gradient accumulation:** At large scale, you often can't fit the full global batch through the model in one shot. Instead, you run multiple micro-batches sequentially, accumulating (summing) their gradients, and only update the weights after all micro-batches are processed. This lets you achieve a large effective batch size without needing the memory for a large batch at once. If your global batch is 4 million tokens and each micro-batch is 32K tokens, you accumulate gradients over 128 micro-batches before doing the optimizer step. The gradients themselves are the same size as the model (70B floats) — they accumulate in place, not growing with each micro-batch.

**Where it breaks at 227 racks:**

- **Memory pressure is at its peak.** During the backward pass, GPU memory holds: model weights + gradients (same size as weights) + optimizer states (2-3x weights for Adam) + whatever activations survived checkpointing + the activation recomputation buffers. This is the moment of maximum memory consumption. If it doesn't fit, the training run OOMs (out-of-memory) and crashes.
- **Numerical stability again.** Gradients can explode or vanish through 80 layers of chain rule. A gradient that's reasonable at layer 80 can become astronomically large by layer 1 (exploding gradients) or effectively zero (vanishing gradients). Gradient clipping (capping the gradient norm to a maximum value, typically 1.0) is standard, but it's a band-aid — it prevents catastrophe without fixing the underlying dynamic. RMSNorm and residual connections help keep gradients in a reasonable range, which is one reason those architectural choices matter.
- **NaN propagation.** If any single gradient computation produces NaN (not-a-number) — due to overflow, division by zero in normalization, or a bad activation checkpoint — that NaN propagates through the chain rule to every downstream gradient. One bad number in one layer on one GPU can corrupt the entire step's gradients across all 512 data-parallel replicas (16,384 GPUs) once gradient sync mixes it into the all-reduce. Detection and recovery from NaN events is a critical part of the training infrastructure.

### Performance Profile
- **Compute:** ~2x the forward pass. The backward pass dominates total training step compute.
- **Memory:** Peak consumption. Weights + gradients + optimizer states + checkpointed activations + recomputation buffers all coexist simultaneously.
- **Communication:** Mirrors the forward pass — tensor-parallel all-reduce per layer over NVLink, pipeline gradient transfers over InfiniBand, same cost as forward.
- **Weak points:** Activation memory forces checkpointing (20-35% compute overhead). Gradient instability through 80 layers. NaN propagation across the entire cluster from a single bad computation.
