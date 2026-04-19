---
title: "Sparse attention: skipping tokens you don't need"
date: 2026-04-13
node_id: "3f"
tier: 3
parent: "2i"
children: []
siblings:
  - "3g"
  - "3h"
  - "3i"
  - "3j"
  - "3k"
url: "/llms/what-happens/prefill-decode/kv-cache/sparse-attention/"
summary: "Full attention means every token attends to every other token — T² comparisons. Sparse attention says: most of those comparisons are wasted. Token 4,000 probably doesn't need to attend to token 12...."
series: "llms"
mermaid: true
weight: 305
tags:
  - llm
---

```mermaid
graph LR
    A["Token vectors"] --> B["Q,K,V Proj"] --> C["Attention Scores<br/>(T² → sparse)"]:::hl --> D["Softmax"]:::hl --> E["Weighted Sum<br/>(fewer tokens)"]:::hl --> F["Output"]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click A "/llms/what-happens/tokens/"
    click B "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"
    click C "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"
    click D "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"
    click E "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"
    click F "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
```

Full attention means every token attends to every other token — T² comparisons. Sparse attention says: most of those comparisons are wasted. Token 4,000 probably doesn't need to attend to token 12. So instead of computing all T² scores, you define a **pattern** that determines which token pairs are allowed to attend to each other, and skip the rest.

Common sparse attention patterns:

- **Sliding window** (local attention): each token only attends to the nearest W tokens (e.g., 4,096). Token 5,000 can see tokens 4,001–5,000 but not token 1. This captures local context efficiently — most relevant information is nearby. Cost: T × W instead of T². Mistral uses this.

- **Strided / dilated**: attend to every Nth token beyond the local window. Token 5,000 sees its local window plus tokens 1, 1,000, 2,000, 3,000, etc. Gives some long-range reach without full T².

- **Block sparse**: divide the sequence into fixed-size blocks. Full attention within each block, and a selection of cross-block attention for global information flow. BigBird and Longformer use variants of this.

- **Global + local**: designate certain token positions (e.g., the first token, or special delimiter tokens) as "global" tokens that attend to everything and everything attends to them. All other tokens use local attention only. This creates information highways through the sequence.

**The trade-off is always information loss.** If token 5,000 can't attend to token 12, and token 12 contained something critical, the model can't recover that information. Sparse attention assumes that most long-range interactions aren't important — which is often true, but not always. The choice of pattern determines which interactions are preserved and which are sacrificed.

**What this looks like in practice:** With global + local attention, you get a "bathtub" shaped attention distribution: strong attention at the beginning of the context (system prompt, initial instructions — the global anchors), strong attention at the end (recent conversation — the local window), and a shallow middle where tokens only get attended to if they happen to be in a landmark position. You can observe this directly when working with long-context models:

- At 120K tokens, a function definition sitting at token 45,000 is in that shallow middle. The model might have a faint signal that something function-shaped exists there — the concept propagated through the layers during prefill and left a trace in the hidden states — but during decode, it can't attend to the actual implementation details with full fidelity. Those K/V vectors are either receiving near-zero attention weight or (with KV cache eviction) might not be in HBM at all.

- The result: the model knows "there's a function called `processAuth`" but can't recall its exact signature, logic, or output format. When it needs that function, it generates a new one from its trained knowledge of what such a function *probably* looks like — which might differ from what was actually written. This isn't hallucination in the traditional sense — it's attention-limited recall. The information *was* in the context; the model just can't see it with enough resolution.

- This also explains why models anchor to the system prompt and recent messages even in very long contexts — those are the tokens that reliably get full attention (global anchors at the start, local window at the end). Everything in between is best-effort.

**How sparse attention is actually applied during [prefill](/llms/what-happens/prefill-decode/):**

With full attention, every token computes scores against every other token — a 100K × 100K score matrix per head per layer, 10 billion entries, every one computed. With sparse attention, most entries are *never computed*. They're masked out — treated as negative infinity before [softmax](/llms/what-happens/embeddings/model-layers/attention-deep-dive/), which means they become zero probability. The GPU skips the [dot products](/llms/what-happens/vectors/dot-product/) for those positions entirely.

For a sliding window of W=4,096 on a 100K sequence: token 50,000 computes scores against tokens 45,904–50,000 (its local window) — 4,096 dot products. It does NOT compute scores against tokens 1–45,903. Those positions are masked. With global + local, token 50,000 also scores against the designated global tokens (say, the first 128 tokens and any delimiters) — maybe 4,300 total dot products instead of 100,000.

**The computation itself is sparse — not computed then zeroed:**

The naive approach would be to compute the full T×T matrix and zero out the masked entries, but that wastes all the compute. Instead, implementations only compute the non-zero blocks:

- **Block-sparse matmul**: divide the T×T matrix into fixed-size blocks (e.g., 128×128). Only compute blocks that intersect with the attention pattern (local window diagonal, global rows/columns). Skip all other blocks. The GPU kernel only launches work for non-zero blocks.
- **[FlashAttention](/llms/what-happens/embeddings/model-layers/attention-deep-dive/multi-head-attention/) with masks**: FlashAttention computes attention in tiles to avoid materializing the full T×T matrix. It accepts a mask that tells it which tiles to skip. Tiles that are entirely masked get zero compute.

**What about the K/V cache?** During prefill, the model still computes and stores K/V for *all* tokens at *all* layers — prefill is where the cache gets built. The sparsity is in which cached K/V vectors get *read* during each token's attention computation. During decode, the new token only reads K/V from its local window (and global tokens), not the full cache. This is where the memory bandwidth savings come from during generation — you're reading 4K tokens of cached K/V instead of 100K.

**Performance profile:** Sparse attention runs on the **GPU** and is **compute-bound** during prefill (same as full attention, just less of it). The compute savings are proportional to the sparsity — a sliding window of W=4,096 on a T=100K sequence reduces the attention FLOPs from T² to T×W, roughly a **25× reduction**. The [KV cache](/llms/what-happens/prefill-decode/kv-cache/) can also be smaller: if a token only attends to its local window, you only need to cache W tokens worth of K/V per layer, not the full T. Memory savings and compute savings go hand in hand.
