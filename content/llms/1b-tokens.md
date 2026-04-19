---
title: "What is a token?"
date: 2026-04-13
node_id: "1b"
tier: 1
parent: "0"
children:
  - "2a"
siblings:
  - "1a"
  - "1c"
  - "1d"
  - "1e"
  - "1f"
  - "1g"
url: "/llms/what-happens/tokens/"
summary: "A token is the unit of text that an LLM actually works with. It's not a word, not a character — it's somewhere in between. The model has a fixed vocabulary of tokens (typically 30,000–100,000 of..."
series: "llms"
mermaid: true
weight: 101
tags:
  - llm
---

```mermaid
graph LR
    A[Input Text]:::hl --> B[Tokenizer]:::hl --> C[Token IDs]:::hl --> D[Embedding] --> E["Layers ×80"] --> F[Predict] --> G[Output Token]
    G -.->|decode loop| E
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click A "/llms/what-happens/"
    click D "/llms/what-happens/embeddings/"
    click E "/llms/what-happens/embeddings/model-layers/"
    click F "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
    click G "/llms/what-happens/"
```

A token is the unit of text that an LLM actually works with. It's not a word, not a character — it's somewhere in between. The model has a fixed vocabulary of tokens (typically 30,000–100,000 of them), and every piece of text you send gets split into a sequence of tokens from that vocabulary before the model sees it.

Common words like "the" or "hello" are usually a single token. Less common words get broken into pieces: "unbelievable" might become `["un", "believ", "able"]` — three tokens. Very rare words, technical jargon, or typos get split down further, sometimes to individual characters. Spaces, punctuation, and newlines are tokens too.

This splitting is done by a *tokenizer* — a fixed lookup table that was built before training. The tokenizer doesn't understand meaning; it was constructed using a statistical algorithm (usually Byte Pair Encoding or BPE) that looked at a huge corpus of text and figured out the most efficient set of subword chunks to represent it. Frequent character sequences get their own token. Rare sequences get assembled from smaller pieces.

Why does this matter? Because the model doesn't see your words — it sees a sequence of token IDs (integers). "Hello, how are you?" might become `[15496, 11, 703, 527, 499, 30]`. Each of those integers gets looked up in an [embedding table](/llms/what-happens/embeddings/) to become a vector, and *that's* what enters the model. The token is the atom. Everything downstream — vectors, attention, prediction — operates on tokens, not words.
