---
title: "RMSNorm, RoPE, and SiLU"
date: 2026-05-14
draft: false
node_id: "2j"
tier: 2
parent: "1b"
children: []
siblings: ["2d", "2e", "2f", "2g", "2h", "2i"]
url: "/series/training/train-from-scratch/model-architecture/rmsnorm-rope-silu/"
summary: "These three are the infrastructure choices — less glamorous than attention heads or model dimensions, but each one solves a specific problem that would otherwise break training at scale."
series: "training"
mermaid: true
weight: 216
tags: [llm-training]
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture]:::hl --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click B "/series/training/train-from-scratch/model-architecture/"
```

These three are the "infrastructure" choices — less glamorous than attention heads or model dimensions, but each one solves a specific problem that would otherwise break training at scale.

**RMSNorm (Pre-Norm):** Normalization prevents numbers from exploding to infinity or collapsing to zero as they pass through 80 layers. RMSNorm divides by the root-mean-square of the vector — simpler than LayerNorm (which also subtracts the mean) and ~10-15% faster per operation than LayerNorm, with no quality difference at scale. Applied *before* each sub-layer (pre-norm), not after (post-norm), because pre-norm produces more stable gradients through deep residual stacks. GPT-2 discovered this empirically, and everyone followed.

**Rotary Position Embeddings (RoPE):** Attention is permutation-invariant — without position info, "dog bites man" and "man bites dog" look identical. RoPE encodes position by rotating query and key vectors by an angle proportional to their sequence position. It won over alternatives (absolute, relative, ALiBi) because it generalizes to longer sequences than it trained on, has zero trainable parameters (it's a fixed math operation), and naturally encodes *relative* position (the angle between two positions depends only on their distance). One less thing to learn, one less thing to break.

**SiLU Activation:** The nonlinearity in the FFN. Without an activation function, stacking 80 layers of matrix multiplications would collapse to a single linear transformation — depth would be meaningless. SiLU (x × sigmoid(x)) replaced ReLU because ReLU kills all negative values (zero gradient = dead neuron = stops learning). SiLU is smooth, lets small negative values through, and trains better at scale. The extra sigmoid computation is trivial relative to the surrounding matrix multiplications.

None of these are exciting choices. They're the kind of thing that, if you get right, nobody notices. If you get wrong, training diverges at step 50,000 and you've wasted a week of GPU time.

### Performance Profile
- **RMSNorm:** Element-wise operations only — bandwidth-bound, not compute-bound. ~10-15% faster than LayerNorm (skips mean subtraction). Negligible cost relative to surrounding attention and FFN matmuls, but applied 160 times per token (2× per layer × 80 layers). Tiny cost × high frequency = still worth optimizing.
- **RoPE:** Applied inside attention's Q/K computation — a rotation (sin/cos multiplication) on each head's vectors. Zero trainable parameters (purely mathematical). Compute cost is trivial relative to the QKV projections it sits between. The real value is inference-time: enables context length extension without retraining.
- **SiLU:** One sigmoid + one multiply per element in the FFN hidden dimension (28,672 elements per token per layer). Bandwidth-bound, not compute-bound. ~2x the compute of ReLU but eliminates dead neurons that would otherwise waste capacity in the FFN's 56.5B parameters.
