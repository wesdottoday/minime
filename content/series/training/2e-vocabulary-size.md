---
title: "Vocabulary size: 128,000"
date: 2026-05-14
draft: false
node_id: "2e"
tier: 2
parent: "1b"
children: []
siblings:
  - "2d"
  - "2f"
  - "2g"
  - "2h"
  - "2i"
  - "2j"
url: "/series/training/train-from-scratch/model-architecture/vocabulary-size/"
summary: "The vocabulary is the set of tokens the model can recognize. The vocabulary size determines how many unique tokens exist in that lookup table."
series: "training"
mermaid: true
weight: 211
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture]:::hl --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click B "/series/training/train-from-scratch/model-architecture/"
```

The vocabulary is the set of tokens the model can recognize. We covered tokenization and BPE in the [LLMs series](/series/inference/what-happens/tokens/tokenization/) — the tokenizer learns merge rules from a text corpus, building up from individual bytes to common subwords and full words. The vocabulary size determines how many unique tokens exist in that lookup table.

**Why 128K?** It's a tradeoff between three things:

1. **Compression efficiency.** Larger vocab = fewer tokens per sentence = shorter sequences = faster training and inference. "tokenization" might be one token with a 128K vocab but two tokens ("token" + "ization") with a 32K vocab. Shorter sequences mean less compute per input.
2. **Embedding table size.** The embedding matrix is `vocab_size × d_model`. At 128K × 8,192, that's ~1 billion parameters just for the embedding table — roughly 1.5% of Llama 3's total parameter count. Double the vocab, double that table.
3. **Sparsity.** A huge vocabulary means rare tokens barely get trained. If token #127,998 appears 12 times in 15 trillion tokens, the model never learns a good embedding for it. Wasted capacity.

Llama 2 used 32K. Llama 3 jumped to 128K. The motivation was better multilingual coverage and better handling of code, math notation, and rare subwords. The compute cost of a 4x larger embedding table was acceptable at 70B scale. At smaller scales (7B), the embedding table becomes a proportionally bigger chunk of the model.

**An implementation detail that matters:** Vocabulary size should be divisible by common GPU parallelism factors (64, 128) for efficient tensor operations. 128,000 = 128 × 1,000. Clean.

### Performance Profile
- **Embedding table memory:** 128K × 8,192 × 2 bytes (FP16) = ~2GB. Modest at 70B scale, but at 7B scale this single table is ~15% of total parameters.
- **Compression ratio:** Larger vocab = fewer tokens per input = shorter sequences. Shorter sequences reduce attention's T² cost and shrink [KV cache](/series/inference/what-happens/embeddings/kv-cache/). The jump from 32K to 128K vocab reduced average token count per sentence by roughly 15-20%.
- **Softmax bottleneck:** The final output layer projects from d_model (8,192) to vocab_size (128K). That's a 8,192 × 128K matrix multiplication at every decode step — the single most expensive per-token operation during inference. Larger vocab directly increases this cost.
- **Parallelism alignment:** 128,000 divides cleanly by 64, 128, 256 — critical for efficient tensor parallelism across GPUs. Misaligned vocab sizes cause padding waste on every forward pass.
