---
title: "Attention approximations: breaking the T² barrier differently"
date: 2026-04-13
node_id: "3g"
tier: 3
parent: "2i"
children: []
siblings:
  - "3f"
  - "3h"
  - "3i"
  - "3j"
  - "3k"
url: "/llms/what-happens/prefill-decode/kv-cache/attention-approximations/"
summary: "Where sparse attention *skips* certain token pairs, approximation methods try to compute *something like* full attention but with less math. The goal is the same — avoid the T² bottleneck — but the..."
series: "llms"
mermaid: true
weight: 306
tags:
  - llm
---

```mermaid
graph LR
    A["Token vectors"] --> B["Q,K,V Proj"] --> C["Approximate<br/>attention"]:::hl --> D["Output"]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click A "/llms/what-happens/tokens/"
    click B "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"
    click D "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
```

Where sparse attention *skips* certain token pairs, approximation methods try to compute *something like* full attention but with less math. The goal is the same — avoid the T² bottleneck — but the approach is different.

**Linear attention** is the most important class. Standard attention computes `softmax(Q·K^T) × V`, which forces you to materialize the T×T score matrix. Linear attention rearranges the math: instead of computing scores between all token pairs first, it computes `Q × (K^T × V)` — multiplying K and V together first (which produces a D×D matrix, independent of T), then multiplying Q into that. The cost drops from T²×D to T×D² — **linear in sequence length**.

The catch: this rearrangement only works exactly if you remove the softmax. Without softmax, the [attention scores](/llms/what-happens/embeddings/model-layers/attention-deep-dive/) aren't normalized into a [probability distribution](/llms/what-happens/embeddings/model-layers/final-vector-to-token/), which changes the model's behavior. Various methods (Performer, Random Feature Attention, cosine attention) use kernel tricks to approximate softmax while keeping the linear math. Results are decent but not identical to full attention — there's a quality gap, especially for tasks requiring precise long-range recall.

**Other approximation approaches:**
- **Low-rank approximation**: the T×T attention matrix is often approximately low-rank (many rows look similar). You can approximate it by projecting K and V to a smaller number of "landmark" tokens first, reducing the effective T. Linformer does this.
- **Hash-based**: Reformer uses locality-sensitive hashing to bucket tokens by similarity, then only computes attention within buckets. Only similar tokens attend to each other, which is O(T log T).

**Performance profile:** Approximation methods run on the **GPU** and shift the bottleneck from T² compute to D²×T compute (for linear attention) or T log T compute (for hash-based). They're **compute-bound** like full attention, just less compute. The practical challenge is that GPU hardware is heavily optimized for dense matrix multiplication (the T² matmul in standard attention), and the irregular access patterns of some approximation methods don't map well to GPU architectures. [FlashAttention](/llms/what-happens/embeddings/model-layers/attention-deep-dive/multi-head-attention/) (which makes standard full attention much faster by optimizing memory access) has narrowed the gap to the point where approximations are less attractive for moderate sequence lengths — the engineering win from FlashAttention often beats the algorithmic win from approximations until you reach very long sequences (>128K).
