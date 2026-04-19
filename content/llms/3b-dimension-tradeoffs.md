---
title: "Dimension trade-offs: expressiveness vs. cost"
date: 2026-04-13
node_id: "3b"
tier: 3
parent: "2f"
children: []
siblings: []
url: "/llms/what-happens/embeddings/layer-transforms/dimension-tradeoffs/"
summary: "The embedding dimension (`d_model`) is a design choice made before training. Common values:"
series: "llms"
mermaid: true
weight: 301
tags:
  - llm
---

```mermaid
graph LR
    A[Token IDs] --> D["Embedding<br/>128K × D"]:::hl --> E["Q,K,V Proj<br/>D × D"]:::hl --> F["Attn Scores<br/>T² × D"] --> G["FFN W_up<br/>D × 4D"]:::hl --> H["FFN W_down<br/>4D × D"]:::hl --> I["→ ×80"] --> J["Vocab Proj<br/>D × 128K"]:::hl --> K[Output]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click A "/llms/what-happens/tokens/"
    click D "/llms/what-happens/embeddings/"
    click E "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"
    click F "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"
    click G "/llms/what-happens/embeddings/model-layers/ffn-deep-dive/"
    click H "/llms/what-happens/embeddings/model-layers/ffn-deep-dive/"
    click J "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
    click K "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
```
*Highlighted: every operation where D (dimension) determines matrix size and cost*

The embedding dimension (`d_model`) is a design choice made before training. Common values:

| Model | Dimensions | Parameters |
|-------|-----------|------------|
| Llama 3 8B | 4,096 | 8 billion |
| Llama 3 70B | 8,192 | 70 billion |
| Llama 3 405B | 16,384 | 405 billion |
| Claude 3 Opus | ~12,288 | (undisclosed) |

**What more dimensions buy you: capacity, not accuracy.** Each vector has more room to encode nuanced distinctions. With 256 dimensions, the model has to cram everything it knows about a token into 256 numbers — subtle differences between related concepts get lost. With 16,384 dimensions, it can represent much finer-grained relationships. Think of it like color depth: 8-bit color gives you 256 colors, 24-bit gives you 16 million. More dimensions don't make individual representations "more correct" — they give the model more room to distinguish between things.

**What more dimensions cost:** The weight matrices in attention and feed-forward layers are roughly `D × D` in size. So compute scales as **D²** — doubling the dimension roughly quadruples the math per layer. Memory scales similarly: the [weight matrices](/llms/what-happens/embeddings/weights/) themselves are bigger, the activations (intermediate computation values) are bigger, and the [KV cache](/llms/what-happens/prefill-decode/kv-cache/) (see [KV cache](/llms/what-happens/prefill-decode/kv-cache/)) is bigger. Going from 4,096 to 16,384 dimensions is a 4x increase in dimension but approximately a **16x** increase in per-layer compute and weight storage.

This is why scaling model size isn't free — and why there's an architecture design art to choosing the right balance of layers, dimensions, and [attention heads](/llms/what-happens/embeddings/model-layers/attention-deep-dive/multi-head-attention/) for a given parameter budget.
