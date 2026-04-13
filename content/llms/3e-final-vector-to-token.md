---
title: "From final vector to predicted token"
date: 2026-04-13
node_id: "3e"
tier: 3
parent: "2e"
children:
  - "4b"
siblings:
  - "3c"
  - "3d"
url: "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
summary: "After 80 layers of attention and FFN, each token's vector has been transformed into a rich contextual representation. But the model needs to produce an actual token — a word (or subword) from its..."
series: "llms"
mermaid: true
weight: 304
tags:
  - llm
---

```mermaid
graph LR
    A["Final hidden<br/>state (8,192-d)"]:::hl --> B["× Vocab matrix<br/>(128K dot products)"]:::hl --> C["Logits<br/>(128K scores)"]:::hl --> D["Softmax"]:::hl --> E["Probabilities"]:::hl --> F["Sample<br/>(temp, top-p)"]:::hl --> G["Output Token"]:::hl
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click C "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"
    click D "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"
    click G "/llms/what-happens/"
```

After 80 layers of attention and FFN, each token's vector has been transformed into a rich contextual representation. But the model needs to produce an actual token — a word (or subword) from its vocabulary. Here's how the final vector becomes a prediction.

**Step 1: Take the last token's final [hidden state](/llms/what-happens/embeddings/hidden-states/).** During generation, only the *last* token's vector matters for prediction — that's the position where the model is predicting "what comes next." (During training, every position predicts, but at inference time, you only need the last one.) This is one 8,192-dimensional vector.

**Step 2: Dot-product against the entire vocabulary.** The model has a projection matrix — often the same [embedding table](/llms/what-happens/embeddings/) from the input, transposed (called **weight tying**). This matrix has 128,000 rows (one per vocabulary token), each row an 8,192-dimensional vector. The final hidden state gets dot-producted against every row, producing 128,000 scores called **logits**.

Each logit is a single number answering: "how well does this vocabulary token match the direction the final hidden state is pointing?" A high [dot product](/llms/what-happens/vectors/dot-product/) means the hidden state ended up pointing in a similar direction to that token's embedding — the model is "predicting" that token.

**Weight tying makes intuitive sense:** if "cat"'s embedding vector defines what "cat-ness" means going into the model, then a final hidden state pointing in the "cat-ness" direction should predict "cat" coming out. The same vector space is used for both input representation and output prediction.

**Step 3: [Softmax](/llms/what-happens/embeddings/model-layers/attention-deep-dive/) to probabilities.** The raw logits get passed through softmax to become a probability distribution — 128,000 probabilities that sum to 1. Token 4821 might have a score of 12.3 as a logit, which becomes 34% probability after softmax.

**Step 4: Sampling.** The model picks a token from that distribution. This is where **temperature** and **top-p** come in:
- **Temperature = 0** (or "greedy"): always pick the highest-probability token. Deterministic, but can feel repetitive.
- **Temperature > 0**: scales the logits before softmax, flattening or sharpening the distribution. Higher temperature = more random, lower = more focused.
- **Top-p (nucleus sampling)**: only consider the smallest set of tokens whose cumulative probability exceeds p (e.g., 0.9). Ignore the long tail of unlikely tokens.

The selected token gets appended to the sequence, and the whole process loops — back to [tokenization](/llms/what-happens/tokens/tokenization/), embedding, and 80 layers of transformation for the next prediction. This is the [decode loop](/llms/what-happens/prefill-decode/).

**Performance profile:** The vocabulary projection runs on the **GPU** and is **memory-bandwidth bound**. The math is small — one 8,192-d vector dot-producted against 128,000 vocabulary vectors = ~2 million FLOPs. Trivial. But the vocabulary projection matrix itself (128,000 × 8,192 × 2 bytes ≈ 2 GB, or shared with the embedding table via weight tying) needs to be loaded from HBM. The softmax over 128,000 logits is also bandwidth-bound and nearly free. The sampling step (choosing a token) runs on the **CPU** and is instantaneous. In total, this step is a rounding error — the 80 layers of attention and FFN before it account for >99.9% of inference time.
