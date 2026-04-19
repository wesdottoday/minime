---
title: "Tokenization performance: where does it run and what's the bottleneck?"
date: 2026-04-13
node_id: "3a"
tier: 3
parent: "2a"
children: []
siblings: []
url: "/llms/what-happens/tokens/tokenization/tokenization-perf/"
summary: "Tokenization runs on the **CPU**. Not the GPU. This surprises people because everything else in the LLM pipeline is GPU-bound, but tokenization is the wrong shape for GPU execution."
series: "llms"
mermaid: true
weight: 300
tags:
  - llm
---

```mermaid
graph LR
    A["Input Text"]:::hl --> B["Normalize<br/>(CPU)"]:::hl --> C["BPE Split<br/>(CPU, memory-bound)"]:::hl --> D["Token IDs"]:::hl --> E[Embedding] --> F["Layers ×80"] --> G[Predict] --> H[Output]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click A "/llms/what-happens/"
    click B "/llms/what-happens/tokens/tokenization/"
    click C "/llms/what-happens/tokens/tokenization/"
    click D "/llms/what-happens/tokens/"
    click E "/llms/what-happens/embeddings/"
    click F "/llms/what-happens/embeddings/model-layers/"
    click G "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
    click H "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
```

Tokenization runs on the **CPU**. Not the GPU. This surprises people because everything else in the LLM pipeline is GPU-bound, but tokenization is the wrong shape for GPU execution.

**Why CPU, not GPU?** GPUs are fast when thousands of threads do the *same operation* on different data simultaneously (SIMD — Single Instruction, Multiple Data). Tokenization is the opposite: it's a sequential string-scanning problem. You read left to right, match the longest token you can, consume those characters, then start the next match from where you left off. Each match depends on where the previous one ended. That's inherently serial *within a single input*. GPUs would waste most of their threads sitting idle.

**Is it multi-threaded?** Yes — across inputs, not within a single input. If you're a serving platform handling 200 requests per second, you tokenize each request independently, and those *can* run in parallel across CPU cores. Libraries like HuggingFace's `tokenizers` (written in Rust) are heavily optimized for this — they'll tokenize a batch of inputs in parallel across threads. But a single input string is still processed sequentially.

**What's the bottleneck?** Tokenization is **memory-bound**, not compute-bound. The actual operations are simple — string matching, hash table lookups, byte comparisons. There's almost no arithmetic. The bottleneck is chasing pointers and looking things up in the vocabulary data structure (usually a trie or hash map), which means the speed depends on how often those lookups hit CPU cache vs. going to main memory. A typical vocabulary trie fits in L2/L3 cache on modern CPUs, so in practice tokenization is *fast* — microseconds to low milliseconds for typical inputs.

**How fast relative to the rest of the pipeline?** Orders of magnitude faster than inference. Tokenizing a 4,000-token input takes maybe 0.1–1ms. Running that through a large model takes hundreds of milliseconds to seconds. Tokenization is almost never the bottleneck in an end-to-end serving pipeline — it's a rounding error next to model execution. The exception would be if you're doing massive batch preprocessing (tokenizing terabytes of training data), where tokenization throughput matters and you'd throw many CPU cores at it in parallel.

**Can you tune it?** Not much at the algorithmic level — BPE tokenization is well-optimized in modern libraries. The levers are:
- **Implementation language**: Rust-based tokenizers (HuggingFace `tokenizers`) are ~10-100x faster than pure Python implementations
- **Batch parallelism**: tokenize many inputs concurrently across CPU threads
- **Vocabulary size trade-off**: larger vocabulary = fewer tokens per input (less work for the model downstream) but larger [embedding](/llms/what-happens/embeddings/) tables and slightly slower lookups. Smaller vocabulary = more tokens per input but smaller memory footprint. Most modern models land at 32K–128K tokens as the sweet spot.
- **Pre-[tokenization](/llms/what-happens/tokens/tokenization/) caching**: some serving systems cache tokenized versions of common prefixes (like system prompts that are the same for every request)

**The real performance story** is that tokenization is "solved" — the interesting optimization problems are all downstream in the model itself, where the GPU is doing trillions of floating-point operations. Tokenization just needs to not be in the way, and with modern implementations, it isn't.
