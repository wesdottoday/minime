---
title: "What are weights?"
date: 2026-04-13
node_id: "2b"
tier: 2
parent: "1c"
children: []
siblings:
  - "2c"
  - "2d"
  - "2e"
  - "2f"
  - "2g"
url: "/llms/what-happens/embeddings/weights/"
summary: "A weight is a single number that the model can adjust to change its behavior. That's it. When people say a model has \"405 billion parameters,\" they mean it has 405 billion individual numbers —..."
series: "llms"
mermaid: true
weight: 201
tags:
  - llm
---

```mermaid
graph LR
    A[Input Text] --> B[Tokenizer] --> C[Token IDs] --> D[Embedding]:::hl --> E[Q,K,V Proj]:::hl --> F[Attn Scores] --> G[Weighted Sum] --> H["FFN: W_up"]:::hl --> I[Activation] --> J["FFN: W_down"]:::hl --> K["→ ×80"] --> L[Vocab Proj]:::hl --> M[Output]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click A "/llms/what-happens/"
    click B "/llms/what-happens/tokens/"
    click C "/llms/what-happens/tokens/"
    click D "/llms/what-happens/embeddings/"
    click E "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"
    click F "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"
    click G "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"
    click H "/llms/what-happens/embeddings/model-layers/ffn-deep-dive/"
    click I "/llms/what-happens/embeddings/layer-transforms/"
    click J "/llms/what-happens/embeddings/model-layers/ffn-deep-dive/"
    click L "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
    click M "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
```
*Highlighted: every step that uses learned weight matrices*

A weight is a single number that the model can adjust to change its behavior. That's it. When people say a model has "405 billion parameters," they mean it has 405 billion individual numbers — weights — that collectively define everything the model does.

Weights live in matrices (grids of numbers). The [embedding table](/llms/what-happens/embeddings/) you just learned about is one matrix of weights — 128,000 rows × 8,192 columns = ~1 billion weights, just for embeddings. But that's a tiny fraction. The vast majority of weights live in the model's layers (more on those in [model layers](/llms/what-happens/embeddings/model-layers/)), where they define the transformations that happen to your token vectors as they flow through the network.

Here's the important mental model: **the weights *are* the model.** The code that runs an LLM is relatively simple — it's matrix multiplications and a few other operations in a loop. What makes GPT-4 different from a random noise generator is the *values* of the weights. Two models with identical code but different weights will produce completely different outputs. Training is the process of finding weight values that produce useful behavior. The architecture (code) defines what operations happen; the weights define what those operations *do*.

Every weight started as a small random number. Through training, each one got nudged — millions of times — toward a value that makes the model slightly better at predicting the next token. The final trained weights encode everything the model "knows."
