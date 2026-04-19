---
title: "How tokenization actually works"
date: 2026-04-13
node_id: "2a"
tier: 2
parent: "1b"
children:
  - "3a"
siblings: []
url: "/llms/what-happens/tokens/tokenization/"
summary: "Let's walk through what happens when you type \"I love unbelievable tacos\" and hit send."
series: "llms"
mermaid: true
weight: 200
tags:
  - llm
---

```mermaid
graph LR
    A[Input Text]:::hl --> B[Normalize]:::hl --> C[BPE Split]:::hl --> D[Token IDs]:::hl --> E[Embedding] --> F[Attention] --> G[FFN] --> H["→ ×80"] --> I[Vocab Proj] --> J[Output]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click A "/llms/what-happens/"
    click D "/llms/what-happens/tokens/"
    click E "/llms/what-happens/embeddings/"
    click G "/llms/what-happens/embeddings/model-layers/ffn-deep-dive/"
    click I "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
    click J "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
```

Let's walk through what happens when you type "I love unbelievable tacos" and hit send.

**Step 1: The raw text gets normalized.** Minor cleanup — things like standardizing unicode characters, handling whitespace. Nothing dramatic, just making sure the text is in a consistent format before splitting.

**Step 2: The text gets split into tokens using the vocabulary.** The tokenizer has a fixed vocabulary — let's say 50,000 entries — that was built once before training and never changes. It scans your text and greedily matches the longest sequences it can find in its vocabulary:

- `"I"` → in the vocabulary as-is → token ID 40
- `" love"` → in the vocabulary (note: the space is part of the token) → token ID 3021
- `" un"` → in the vocabulary → token ID 5765
- `"believ"` → in the vocabulary → token ID 18510
- `"able"` → in the vocabulary → token ID 540
- `" tacos"` → in the vocabulary → token ID 83412

So the model receives: `[40, 3021, 5765, 18510, 540, 83412]`

**How was that vocabulary built in the first place?** This is where Byte Pair Encoding (BPE) comes in. Before training the LLM, you take a massive text corpus and run BPE on it:

1. Start with individual characters as your initial vocabulary: `a, b, c, ... z, A, B, ... 0, 1, ... !, ?, ...`
2. Scan the entire corpus and count every pair of adjacent tokens
3. Find the most frequent pair — say `t` + `h` appears 9 million times
4. Merge that pair into a new token `th` and add it to the vocabulary
5. Rescan and repeat

You keep merging until you hit your target vocabulary size (e.g., 50,000 tokens). The result is a vocabulary where common words are single tokens, common subwords are tokens, and rare things get assembled from smaller pieces. The merge order is saved as the tokenizer's rules — that's what gets shipped with the model.

**Why not just split on words?** Three reasons:

- **Unknown words would break the model.** A word-level tokenizer can't handle typos, new slang, code, or any word not in its vocabulary. Subword tokenization can always fall back to characters — nothing is truly "unknown."
- **The vocabulary would be enormous.** English alone has hundreds of thousands of words. Add code, other languages, technical terms — you'd need millions of entries, each requiring its own [embedding vector](/llms/what-happens/embeddings/), which means millions more parameters to store and train.
- **Morphology gets captured for free.** Because "un", "believ", and "able" are separate tokens, the model can learn that "un-" often negates things and "-able" often means "capable of" — patterns that transfer to words it's never seen whole.

**One last detail:** the tokenizer is completely separate from the model. It runs before the model sees anything, and it runs after (to convert [token IDs](/llms/what-happens/tokens/) back into text for the response). It has no neural network, no intelligence — it's a deterministic lookup. The same input always produces the same tokens.
