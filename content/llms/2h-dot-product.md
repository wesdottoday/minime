---
title: "What is a dot product?"
date: 2026-04-13
node_id: "2h"
tier: 2
parent: "1a"
children: []
siblings: []
url: "/llms/what-happens/vectors/dot-product/"
summary: "Take two vectors of the same length. Multiply their elements in pairs, then add up the results. That's the entire operation:"
series: "llms"
mermaid: true
weight: 207
tags:
  - llm
---

```mermaid
graph LR
    A[Input Text] --> B[Tokenizer] --> C[Token IDs] --> D[Embedding] --> E[Q,K,V Proj] --> F["Attn Scores<br/>(Q·K dot prod)"]:::hl --> G[Weighted Sum] --> H[Residual] --> I[FFN] --> J[Residual] --> K["→ ×80"] --> L["Vocab Proj<br/>(dot prod)"]:::hl --> M[Output]
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
    click J "/llms/what-happens/embeddings/model-layers/"
    click L "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
    click M "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
```
*Highlighted: where dot products drive the computation*

Take two vectors of the same length. Multiply their elements in pairs, then add up the results. That's the entire operation:

`[2, 3, 1] · [4, -1, 5] = (2×4) + (3×-1) + (1×5) = 8 + (-3) + 5 = 10`

The result is a single number — a **scalar**, not a vector. That number tells you how much the two vectors point in the same direction:

- **Large positive** → they point in similar directions (similar/related)
- **Zero** → they're perpendicular (unrelated)
- **Negative** → they point in opposite directions (dissimilar/contrasting)

The magnitude matters too — not just the sign. A dot product of 150 means much stronger alignment than 3. The actual value depends on both the direction similarity *and* the length of the vectors, which is why in some contexts (like comparing [embeddings](/llms/what-happens/embeddings/) for similarity) you normalize the vectors to unit length first, giving you **cosine similarity** — a dot product that's purely about direction, scaled between -1 and 1.

In LLMs, dot products show up everywhere:
- **Attention** uses dot products between query and key vectors to compute relevance scores between token pairs
- **The final prediction layer** uses a dot product between the last [hidden state](/llms/what-happens/embeddings/hidden-states/) and each vocabulary token's vector to score how likely each token is to come next
- **Embedding similarity** (for search/retrieval) is measured by dot product or cosine similarity

It's the fundamental "comparison" operation. Anytime the model needs to answer "how similar are these two things?" — it's a dot product.

**Performance profile:** A single dot product is trivial — for an 8,192-d vector, it's 8,192 multiplications and 8,192 additions. **~16K FLOPs**. On a B200 that takes nanoseconds. The cost only matters at scale. In **attention**, you compute T² dot products per head per layer — at T=100K with 64 heads across 80 layers, that's 100K² × 64 × 80 ≈ **51 quadrillion** individual dot products (each 128-d, since they're per-head). That's where the T² compute cost lives. During **[prefill](/llms/what-happens/prefill-decode/)**, these are batched into massive matrix multiplications (the Q × K^T operation) and are **compute-bound** — the GPU does them efficiently as dense matmul. During **decode**, each new token only computes T dot products (one against every cached key), which is far less math but requires loading the entire [KV cache](/llms/what-happens/prefill-decode/kv-cache/) from memory — making it **memory-bandwidth bound**.
