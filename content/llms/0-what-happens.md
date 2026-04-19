---
title: "What happens when you send a message to an LLM"
date: 2026-04-13
node_id: "0"
tier: 0
parent: ""
children:
  - "1a"
  - "1b"
  - "1c"
  - "1d"
  - "1e"
  - "1f"
  - "1g"
siblings: []
url: "/llms/what-happens/"
summary: "When you send a message to an LLM like ChatGPT or Claude, here's what happens at a high level:"
series: "llms"
mermaid: true
weight: 0
tags:
  - llm
---

```mermaid
graph LR
    A[Input Text]:::hl --> B[Tokenizer]:::hl --> C[Token IDs]:::hl --> D[Embedding]:::hl --> E["Layers ×80"]:::hl --> F[Predict]:::hl --> G[Output Token]:::hl
    G -.->|decode loop| E
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click B "/llms/what-happens/tokens/"
    click C "/llms/what-happens/tokens/"
    click D "/llms/what-happens/embeddings/"
    click E "/llms/what-happens/embeddings/model-layers/"
    click F "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
```

When you send a message to an LLM like ChatGPT or Claude, here's what happens at a high level:

1. Your message, along with any previous messages in the conversation (the "context"), gets converted from human-readable text into numbers — specifically, a sequence of numerical vectors. This step is called *[tokenization](/llms/what-happens/tokens/tokenization/)* and *[embedding](/llms/what-happens/embeddings/)*.
2. Those numbers flow through the model — a stack of mathematical layers that transform them, over and over, each layer refining the model's internal representation of what you said and what should come next.
3. The final layer outputs a [probability distribution](/llms/what-happens/embeddings/model-layers/final-vector-to-token/): a ranked list of every possible next word (technically "token") the model could produce, with a score for how likely each one is.
4. A token is selected from that distribution, appended to the sequence, and the whole process repeats — the model now takes everything so far (your messages + its own partial response) and predicts the next token again. This loop continues until the model produces a [stop signal](/llms/what-happens/embeddings/model-layers/final-vector-to-token/stopping/).

That's it. Every response you've ever gotten from an LLM was generated one token at a time, left to right, by a system that only knows how to do one thing: predict what comes next.
