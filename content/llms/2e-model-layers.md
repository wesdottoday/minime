---
title: "What are model layers?"
date: 2026-04-13
node_id: "2e"
tier: 2
parent: "1c"
children:
  - "3c"
  - "3d"
  - "3e"
siblings:
  - "2b"
  - "2c"
  - "2d"
  - "2f"
  - "2g"
url: "/llms/what-happens/embeddings/model-layers/"
summary: "A layer is one complete round of transformation that every token vector passes through. If the model has 80 layers (typical for a large LLM), then each token vector gets transformed 80 times in..."
series: "llms"
mermaid: true
weight: 204
tags:
  - llm
---

```mermaid
graph LR
    A[Input Text] --> B[Tokenizer] --> C[Token IDs] --> D[Embedding] --> E[Q,K,V Proj]:::hl --> F[Attn Scores]:::hl --> G[Weighted Sum]:::hl --> H[Residual]:::hl --> I["FFN"]:::hl --> J[Residual]:::hl --> K["→ ×80"]:::hl --> L[Vocab Proj] --> M[Output]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click A "/llms/what-happens/"
    click B "/llms/what-happens/tokens/"
    click C "/llms/what-happens/tokens/"
    click D "/llms/what-happens/embeddings/"
    click E "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"
    click F "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"
    click G "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"
    click I "/llms/what-happens/embeddings/model-layers/ffn-deep-dive/"
    click L "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
    click M "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
```

A layer is one complete round of transformation that every token vector passes through. If the model has 80 layers (typical for a large LLM), then each token vector gets transformed 80 times in sequence before the model produces its output. During [prefill](/llms/what-happens/prefill-decode/) (see [prefill vs decode](/llms/what-happens/prefill-decode/)), all tokens in the input — your message, prior conversation history, system prompt, everything — enter each layer simultaneously and are processed in parallel.

Here's what happens inside a single layer, step by step. Let's say your input is 500 tokens — that's 500 vectors, each 8,192 dimensions, all entering the layer at the same time:

1. **Attention** — this is the only place in a layer where tokens interact with each other. Each token's vector gets multiplied by three different [weight matrices](/llms/what-happens/embeddings/weights/) to produce three new vectors: a **query** (Q), a **key** (K), and a **value** (V). Then every token's query is compared against every other token's key using a [dot product](/llms/what-happens/vectors/dot-product/), producing a relevance score — "how much should token i pay attention to token j?" Those scores get normalized into weights that determine how much of each token's value vector gets mixed into each other token's representation. The result: each of the 500 tokens now has a new vector that's a weighted blend of information from all the tokens it attended to. This is where "the cat sat on the mat" lets "sat" pull in information from "cat" and "mat" — because the [attention scores](/llms/what-happens/embeddings/model-layers/attention-deep-dive/) between those tokens will be high.

2. **[Feed-forward network](/llms/what-happens/embeddings/model-layers/ffn-deep-dive/) (FFN)** — after attention has mixed information *between* tokens, each token's blended vector passes *independently* through a small neural network (two matrix multiplications with a nonlinear activation in between). Every token goes through the same feed-forward weights, but independently — no cross-token interaction here. This is per-token transformation, not mixing.

3. **Residual connection** — the output of each step gets *added back* to the input that entered the step, rather than replacing it. Information from before the layer survives through. This is why deep models don't lose early information — there's a direct path through all 80 layers via these residual additions.

The result: 500 vectors go in, 500 vectors come out, still 8,192 dimensions each, but now pointing in slightly different directions — enriched with contextual information from the other tokens. Here's what that progression actually looks like for a single token, say "sat" in the sentence "The cat sat on the mat":

1. **Raw [embedding](/llms/what-happens/embeddings/)** (before any layers): "I am the token 'sat'" — no context, just the static lookup
2. **After layer 1**: "I am 'sat' and my subject is probably nearby"
3. **After layer 20**: "I am 'sat' with subject 'cat' and location 'mat', past tense"
4. **After layer 80**: "I am 'sat' in this specific sentence, and given everything I've absorbed, the most likely continuation is ___"

Each layer's attention and FFN nudge the vector's direction in 8,192-dimensional space. By layer 80, the vector has been nudged so many times that it's pointing in a direction that *is* the prediction. The final [hidden state](/llms/what-happens/embeddings/hidden-states/) then gets dot-producted against every token in the vocabulary (128,000 vectors) — whichever vocabulary token's vector has the highest dot product with the final hidden state is the predicted next token. The expensive part was getting the vector to the right place in space across 80 layers. The actual prediction is just 128,000 dot products.

Each layer has its *own* set of weights — its own attention weight matrices (Q, K, V projections) and its own feed-forward weights. Layer 1's weights are completely independent from layer 40's weights. This is where the bulk of the model's parameters live: 80 layers × (attention weights + FFN weights) = hundreds of billions of parameters.

**Why many layers instead of one big layer?** Depth lets the model build increasingly abstract representations. Early layers tend to capture surface-level patterns (syntax, word associations). Middle layers capture more complex relationships (coreference, semantic roles). Late layers capture high-level reasoning and prediction-relevant features. This hierarchy wasn't designed — it emerged from training — but it's been consistently observed when researchers probe what each layer does.

**Performance profile:** Each layer runs on the **GPU** and layers execute **sequentially** — layer 2 can't start until layer 1 finishes, because it needs layer 1's output as input. (Within a layer, the token-level operations run in parallel across the GPU.) For Llama 3 70B, each layer's weights are roughly 70B / 80 layers ≈ **875 million parameters** ≈ **1.75 GB** at FP16. During **prefill**, each layer is **compute-bound** — large matrix multiplications across all input tokens in parallel. During **decode**, each layer is **memory-bandwidth bound** — you still load the full 1.75 GB of layer weights from HBM, but you're only processing one token, so the math is tiny relative to the data movement. This is the fundamental reason decode is slow: every single token generated requires reading *all* model weights (140 GB for Llama 3 70B) from memory, 80 layers × 1.75 GB, just to produce one token.
