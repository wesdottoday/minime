---
title: "What does the final training data actually look like?"
date: 2026-05-14
draft: false
node_id: "2a"
tier: 2
parent: "1a"
children: []
siblings:
  - "2b"
  - "2c"
url: "/series/training/train-from-scratch/training-data/final-data-format/"
summary: "After all that filtering, deduplication, and cleaning, the data isn't stored as raw text anymore. It's been tokenized — converted from strings into sequences of integer IDs."
series: "training"
mermaid: true
weight: 200
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

After all that filtering, deduplication, and cleaning, the data isn't stored as raw text anymore. It's been tokenized — converted from strings into sequences of integer IDs from the model's vocabulary. For Llama 3, that vocabulary is 128,000 tokens built using Byte Pair Encoding. The sentence "The cat sat on the mat" becomes something like `[791, 5023, 3831, 389, 279, 5765]` — six integers, each pointing to an entry in the tokenizer's lookup table.

These token sequences get packed into fixed-length chunks. Llama 3's context window is 8,192 tokens during pretraining (they extended it to 128K in Llama 3.1 with a separate long-context fine-tuning stage). So the pipeline takes that continuous river of tokenized text and chops it into 8,192-token sequences. Documents are concatenated end-to-end with special separator tokens between them — you don't waste a whole sequence on a short document, and you don't pad short documents with empty tokens. You just pack them tight.

The storage format is typically binary — arrays of 32-bit integers (a 128K vocabulary exceeds the 65,536 limit of 16-bit integers) saved in memory-mappable files. Meta and others commonly use formats like NumPy `.npy` files, Apache Arrow, or custom binary formats. Memory-mapping is the key design choice here: the dataset is too large to fit in CPU RAM all at once (tens of terabytes), so the data loader maps the file into virtual memory and reads only the chunks it needs for the current batch. The OS handles paging data in and out from storage transparently.

In practice, the dataset is sharded — split into thousands of files spread across a distributed filesystem (think Lustre, GPFS, or cloud object storage like S3). Each training worker (each GPU or group of GPUs) reads from different shards simultaneously. The data loader shuffles which shards each worker reads from, and shuffles sequences within shards, to ensure the model doesn't see the same data in the same order each epoch. This shuffle-at-two-levels approach (shard order + within-shard order) gives you near-random access patterns without needing to truly randomize a multi-terabyte file.

A single training batch for Llama 3 70B across a GB200 NVL72 rack might look like: 72 GPUs, each processing a micro-batch of, say, 4 sequences of 8,192 tokens. That's `72 × 4 × 8,192 = 2,359,296` tokens per global batch step. Each token is a 4-byte integer. So each step, you're pulling ~9MB of raw token data from storage — tiny compared to the compute that follows, but it has to arrive on time, every time, or you stall.

### Performance Profile
- **Storage format:** Binary integer arrays, memory-mapped, sharded across distributed filesystem
- **I/O pattern:** Sequential reads within shards, random shard selection — optimized for throughput, not latency
- **Bottleneck risk:** Not the size of each read (small), but the *consistency* — one slow read from a degraded storage node stalls the entire training step across all 72 GPUs
