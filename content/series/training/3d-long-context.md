---
title: "Long-context training & sequence packing"
date: 2026-05-19
draft: false
node_id: "3d"
tier: 3
parent: "2c"
children: []
siblings: ["3c"]
url: "/series/training/train-from-scratch/training-data/data-mixture/long-context/"
summary: "Llama 3 70B pretrains on 8K-token sequences. Llama 3.1 extended the family to 128K context through additional long-context training. The gap is bridged by sequence packing, RoPE scaling, and a short but carefully tuned long-context fine-tuning phase."
series: "training"
mermaid: true
weight: 303
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data]:::hl --> B[Architecture] --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click A "/series/training/train-from-scratch/training-data/"
```

Llama 3 70B pretrains on 8,192-token sequences. Llama 3.1 extended the family to 128K context through additional long-context training. Training sequence length and inference context length aren't the same thing, and the gap between them is one of the most misunderstood aspects of modern LLMs.

**Training sequence length vs. inference context length.** Pretraining at long context lengths is expensive -- attention scales quadratically with sequence length (T^2 for the attention matrix), and activation memory grows linearly. Training at 128K tokens per sequence would require ~256x the attention compute compared to 8K. That's not feasible for the bulk of pretraining. Instead, the standard approach is staged: pretrain at a short context (8K), then do a relatively brief fine-tuning phase at longer context (32K, 64K, 128K) with a modified positional encoding.

**Sequence packing.** The pretraining dataset is a river of tokenized text from millions of documents. Most documents are much shorter than 8,192 tokens -- a typical web page might be 500-2,000 tokens. Instead of padding every short document to 8,192 and wasting compute on padding tokens, the data pipeline concatenates documents end-to-end with separator tokens (`<|end_of_text|>` or similar) between them. A single 8K training sequence might contain 3-8 documents packed together.

This creates a subtle but important question: **should the model attend across document boundaries?** If Document A ends at position 3,000 and Document B starts at position 3,001, should tokens in Document B be able to attend to tokens in Document A? They're unrelated content that happens to be adjacent in the training sequence.

The answer depends on the implementation:

- **No cross-document attention (masked packing).** An attention mask prevents tokens from attending across document boundaries. Each document is effectively its own independent context. This is cleaner -- the model never learns spurious relationships between unrelated documents. But it requires tracking document boundaries and applying per-document masks, which adds complexity to the data pipeline and the attention implementation.
- **Cross-document attention (naive packing).** No special masking -- the model can attend to all prior tokens in the sequence regardless of document boundaries. This is simpler to implement but means the model occasionally learns to connect unrelated content. In practice, the pretraining loss on tokens near document boundaries is higher (the model can't predict what comes after `<|end_of_text|>` because the next document is random), and some researchers argue this slightly hurts quality. Most large-scale pretraining runs use naive packing because the quality impact is small and the implementation is simpler.

**Long-context curriculum.** Extending to 128K context after pretraining is not simply "fine-tune on longer sequences." The positional encoding must handle positions it never saw during pretraining. For RoPE (Llama 3's positional encoding), this means the rotation angles at position 100,000 are extrapolations -- the model never trained on them.

Several techniques address this:
- **RoPE scaling (linear interpolation).** Divide all position indices by a scaling factor so that position 128K maps to the same rotation angle as position 8K during pretraining. Simple but effective -- the model sees the same angles it learned, just with more positions squeezed into the same range.
- **NTK-aware scaling.** Adjusts the frequency basis of RoPE rather than uniformly scaling positions. This preserves local position resolution (nearby tokens are still distinguishable) while extending global range. Better quality than linear scaling, especially at very long contexts.
- **YaRN (Yet another RoPE extensioN).** Combines NTK-aware scaling with a temperature adjustment to the attention logits. Currently the best-performing open approach for extending context length without full retraining.

The long-context fine-tuning phase is typically short -- thousands of steps, not millions -- and uses a mix of naturally long documents (books, code repositories, concatenated conversations) and synthetically lengthened examples. The goal is to teach the model that position 50,000 is a valid location where useful information might be, not to retrain the model's full capabilities.

**Why long-context training is expensive.** The attention mechanism computes pairwise interactions between all tokens. At sequence length T:
- Attention compute scales as O(T^2): at 128K, that's 256x the attention compute of 8K
- KV cache memory scales as O(T): at 128K, that's 16x the KV cache of 8K
- Activation memory scales as O(T): 16x more activations to store for the backward pass

This is why FlashAttention matters -- it avoids materializing the full TxT attention matrix, reducing the memory scaling. But the compute scaling remains quadratic. Training a full 128K pass costs roughly 16x more per step than an 8K pass (dominated by the attention compute increase), which is why long-context training is done as a short fine-tuning phase, not as the full pretraining run.

**Why long-context benchmarks can be misleading.** A model that scores well on "needle in a haystack" (finding a specific fact embedded in a long document) may still struggle with tasks that require *integrating* information across a long context. Finding a needle tests retrieval; understanding a long argument tests synthesis. Most long-context benchmarks test retrieval, not synthesis. Additionally, if long-context training examples appear in the benchmark evaluation set -- or are very similar to them -- the model may appear more capable than its true long-context reasoning ability warrants.

## Performance Profile
- **Pretraining context:** 8K tokens -- the balance point between context length, compute cost, and training throughput
- **Extension context:** 128K tokens -- achieved through short fine-tuning phase with RoPE scaling
- **Attention compute scaling:** 128K/8K = 16x more sequence length -> ~256x more attention compute per step (quadratic scaling). Offset partially by FlashAttention reducing memory, not compute.
- **Fine-tuning cost:** The long-context extension phase is ~0.1-1% of total pretraining compute -- a rounding error in the total training budget, but requires careful recipe tuning
- **Data scarcity:** Naturally long, high-quality documents (>32K tokens) are rare. Books and long-form code are the main sources. Most long-context training data is synthetically constructed.
