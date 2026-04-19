---
title: "How do layers transform vectors?"
date: 2026-04-13
node_id: "2f"
tier: 2
parent: "1c"
children:
  - "3b"
siblings:
  - "2b"
  - "2c"
  - "2d"
  - "2e"
  - "2g"
url: "/llms/what-happens/embeddings/layer-transforms/"
summary: "The actual operations are matrix multiplications. Every transformation in a layer — attention, feed-forward — boils down to multiplying a vector by a weight matrix to produce a new vector."
series: "llms"
mermaid: true
weight: 205
tags:
  - llm
---

```mermaid
graph LR
    A[Input Text] --> B[Tokenizer] --> C[Token IDs] --> D[Embedding] --> E["Q,K,V Proj<br/>(matmul)"]:::hl --> F["Attn Scores<br/>(matmul)"]:::hl --> G[Weighted Sum] --> H[Residual]:::hl --> I["FFN W_up<br/>(matmul)"]:::hl --> J["Activation<br/>(nonlinear)"]:::hl --> K["FFN W_down<br/>(matmul)"]:::hl --> L[Residual]:::hl --> M["→ ×80"] --> N[Vocab Proj] --> O[Output]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click A "/llms/what-happens/"
    click B "/llms/what-happens/tokens/"
    click C "/llms/what-happens/tokens/"
    click D "/llms/what-happens/embeddings/"
    click E "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"
    click F "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"
    click G "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"
    click H "/llms/what-happens/embeddings/model-layers/"
    click I "/llms/what-happens/embeddings/model-layers/ffn-deep-dive/"
    click K "/llms/what-happens/embeddings/model-layers/ffn-deep-dive/"
    click L "/llms/what-happens/embeddings/model-layers/"
    click N "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
    click O "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
```
*Highlighted: every matrix multiplication and the nonlinear activation between them*

The actual operations are matrix multiplications. Every transformation in a layer — attention, feed-forward — boils down to multiplying a vector by a weight matrix to produce a new vector.

Here's what matrix multiplication does geometrically to a vector: it **rotates, scales, and projects** it. Imagine your token vector as an arrow pointing in some direction in 8,192-dimensional space. Multiplying it by a weight matrix is like grabbing that arrow and:
- Rotating it to point in a different direction
- Stretching or compressing it along different axes
- Potentially projecting it into a different-dimensional space (though in most of the model, vectors stay at 8,192 dimensions)

So when people say "the model transforms the vectors through its layers," the concrete thing happening is: multiply by a weight matrix, apply a nonlinear function, multiply by another weight matrix, add a residual connection, normalize. Each layer takes an 8,192-dimensional vector in and outputs an 8,192-dimensional vector, but pointing in a different direction — one that encodes more contextual information than before.

The **nonlinear function** (called an activation function) is crucial, and it's easier to understand why if you think about the layer progression from [model layers](/llms/what-happens/embeddings/model-layers/)):

1. After layer 1: "I am 'sat' and my subject is probably nearby"
2. After layer 20: "I am 'sat' with subject 'cat' and location 'mat', past tense"
3. After layer 80: "I am 'sat' in this specific sentence, and the most likely continuation is ___"

Each of those transitions requires the model to combine information in ways that aren't simple scaling. Going from "my subject is probably nearby" to "my subject is 'cat'" requires a *decision* — weigh the evidence, suppress wrong candidates, commit to an interpretation. That's not something a matrix multiplication alone can do — it can only stretch and rotate a vector uniformly. You need a *bend* — a point where the transformation behaves differently depending on the input values.

That's what the activation function does. After a matrix multiplication stretches and rotates the vector, the activation goes through each number and bends it: strongly positive values pass through mostly unchanged (confident signals), values near zero get squished further toward zero (weak/ambiguous signals getting suppressed), and negative values get dampened. The vector comes out shaped differently depending on *what was in it*, not just scaled uniformly.

Without this bend, stacking layers is pointless. Matrix A followed by Matrix B always equals some single Matrix C — you could collapse the entire 80-layer model into one layer and get the same result. The nonlinearity prevents that collapse. It's what makes layer 20's representation genuinely richer than layer 1's, not just a differently-scaled version of the same thing.

**GeLU vs. SiLU — two shapes for the bend:**

The older activation, **ReLU**, was simple: if the number is positive, keep it; if negative, zero it out (`max(0, x)`). Works, but the hard cutoff at zero is harsh — any information encoded as a negative number is permanently destroyed.

**GeLU** (Gaussian Error Linear Unit) smooths out the cutoff. Small negative values get dampened toward zero but not killed outright. The transition is a gentle curve instead of a sharp corner. Used in GPT-2 and BERT.

**SiLU** (Sigmoid Linear Unit, also called Swish): `x × sigmoid(x)`. Very similar effect to GeLU — large positives pass through, negatives get dampened, smooth transition. SiLU actually dips slightly negative before recovering to zero, which preserves a tiny bit of negative signal. Llama 3 uses SiLU.

In practice, GeLU and SiLU produce nearly identical results. The choice between them is a marginal training stability decision, not a fundamental architectural difference. The important thing is that both are smooth (which helps gradient flow during training) and both preserve some negative signal rather than destroying it like ReLU.

**Worked example: one vector through one FFN layer (simplified to 3 dimensions):**

Starting vector for "sat" (post-attention): `[2.0, -0.5, 1.0]`

*Expand* — multiply by W_up (3→4 dimensions):
```
                      W_up
                [ 0.5  0.3 -0.2  0.8]
[2.0 -0.5 1.0] × [-0.1  0.7  0.4  0.1]
                [ 0.6 -0.3  0.9 -0.5]

Element 0: (2.0×0.5)  + (-0.5×-0.1) + (1.0×0.6)  = 1.0 + 0.05 + 0.6  = 1.65
Element 1: (2.0×0.3)  + (-0.5×0.7)  + (1.0×-0.3) = 0.6 - 0.35 - 0.3  = -0.05
Element 2: (2.0×-0.2) + (-0.5×0.4)  + (1.0×0.9)  = -0.4 - 0.2 + 0.9  = 0.30
Element 3: (2.0×0.8)  + (-0.5×0.1)  + (1.0×-0.5) = 1.6 - 0.05 - 0.5  = 1.05
```
Expanded: `[1.65, -0.05, 0.30, 1.05]`

*Activate (SiLU)* — applied to each number independently:
```
SiLU(1.65)  = 1.65 × sigmoid(1.65) = 1.65 × 0.839 = 1.384  ← strong positive, mostly kept
SiLU(-0.05) = -0.05 × sigmoid(-0.05)= -0.05 × 0.488 = -0.024 ← near zero, crushed
SiLU(0.30)  = 0.30 × sigmoid(0.30) = 0.30 × 0.574 = 0.172  ← weak positive, cut in half
SiLU(1.05)  = 1.05 × sigmoid(1.05) = 1.05 × 0.741 = 0.778  ← moderate, dampened
```
After activation: `[1.384, -0.024, 0.172, 0.778]`

The activation treated each value differently based on its magnitude. The confident signal (1.65) survived. The ambiguous near-zero signal (-0.05) got crushed. This is the bend — the model deciding what to keep and what to suppress.

*Compress* — multiply by W_down (4→3 dimensions):
```
Element 0: (1.384×0.4) + (-0.024×0.3) + (0.172×-0.5) + (0.778×0.2) = 0.617
Element 1: (1.384×-0.2)+ (-0.024×0.8) + (0.172×0.1)  + (0.778×0.4) = 0.032
Element 2: (1.384×0.6) + (-0.024×-0.1)+ (0.172×0.7)  + (0.778×0.3) = 1.186
```
Compressed: `[0.617, 0.032, 1.186]`

*Residual add* — add FFN's output back to the original input:
```
  [2.000, -0.500, 1.000]   ← original (before FFN)
+ [0.617,  0.032, 1.186]   ← FFN's modification
= [2.617, -0.468, 2.186]   ← output of this layer
```

The vector shifted: dimension 0 grew, dimension 1 barely moved, dimension 2 roughly doubled. This enters layer 2, where different weights do a different transformation, another activation bends it again, and the residual adds again.

**Why this can't collapse into one step:** without the activation, W_up followed by W_down is just two matrix multiplications — and there always exists a single 3×3 matrix that produces the same result. You could delete the FFN entirely and replace it with one matrix. But because the activation bent the intermediate values — treating 1.65 differently from -0.05 — no single matrix can replicate what happened. The expand-bend-compress is fundamentally more powerful than one step. Multiply that difference across 80 layers and it's the difference between a model that can predict language and one that can't.

The **[residual connection](/llms/what-happens/embeddings/model-layers/)** is also important: instead of replacing the vector with the layer's output, you *add* the layer's output to the original input vector. So each layer computes a *change* to the vector rather than a replacement. This means information can flow straight through the network without being forced through every layer — it prevents early information from getting lost as the vector passes through 80 transformations.

**Performance profile:** The matrix multiplications (W_up, W_down) and the activation function have very different performance characteristics. The **matmuls are compute-bound** during [prefill](/llms/what-happens/prefill-decode/) — they're the big dense operations that GPUs are built for, with high [arithmetic intensity](/llms/what-happens/embeddings/layer-transforms/dimension-tradeoffs/) (many FLOPs per byte loaded from memory). The **activation function is memory-bandwidth bound** — it reads each number, does one cheap operation (multiply by its sigmoid), and writes it back. Almost no math per byte of data. The activation is effectively free relative to the matmuls; it takes negligible time. The **residual addition** is also bandwidth-bound and trivially cheap — it's an element-wise add of two vectors. In total, the FFN's cost is dominated entirely by the two (or three, with SwiGLU) matrix multiplications.
