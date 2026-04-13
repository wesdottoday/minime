---
title: "How does memory work?"
date: 2026-04-13
node_id: "1g"
tier: 1
parent: "0"
children: []
siblings:
  - "1a"
  - "1b"
  - "1c"
  - "1d"
  - "1e"
  - "1f"
url: "/llms/what-happens/memory/"
summary: "LLMs have no persistent memory. Every conversation starts from zero — the model has its weights (fixed, trained knowledge) and whatever tokens are in the current context window. When the context is..."
series: "llms"
mermaid: true
weight: 106
tags:
  - llm
---

```mermaid
graph LR
    A["User message"] --> B["Harness checks<br/>memory store"]:::hl
    B --> C["Retrieve relevant<br/>memories"]:::hl
    C --> D["Inject into<br/>context"]:::hl
    D --> E["Model sees memories<br/>as tokens"] --> F["Layers ×80"] --> G["Response"]
    G --> H{"Save new<br/>memories?"}:::hl
    H -->|"yes"| I["Write to<br/>memory store"]:::hl
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click A "/llms/what-happens/"
    click D "/llms/what-happens/"
    click E "/llms/what-happens/tokens/"
    click F "/llms/what-happens/embeddings/model-layers/"
```

LLMs have no persistent memory. Every conversation starts from zero — the model has its weights (fixed, trained knowledge) and whatever tokens are in the current context window. When the context is gone, everything discussed is gone. "Memory" systems are external software that works around this limitation.

**The common pattern — retrieval-augmented memory:**

1. **Storage**: memories are text files, database entries, or vector [embeddings](/llms/what-happens/embeddings/) stored outside the model. They persist between conversations on disk or in a database.

2. **Retrieval at conversation start**: when a new conversation begins, the harness (the software wrapping the model) queries the memory store for entries relevant to the current context — the user, the project, the topic. Retrieved memories get injected into the system prompt or early in the conversation as tokens the model can attend to.

3. **Retrieval during conversation**: as the conversation progresses, the harness may query the memory store for additional relevant memories based on what's being discussed.

4. **Saving**: when the model encounters something worth remembering, it can generate a structured output (similar to a tool call) that the harness intercepts and writes to the memory store. The model doesn't "decide" to remember — it generates tokens that match a save pattern, and external software does the actual persistence.

**How does the model know what to save?** It doesn't, fundamentally. Through training (including [RLHF](/llms/what-happens/thinking/) and instruction tuning), the model learned patterns like "when the user explicitly says 'remember this'" or "when information seems like it would be useful in future conversations." But this is pattern matching on the text in context, not genuine metacognition. The model generates save-shaped output when the current context statistically resembles the training examples where saving was appropriate.

**How does retrieval decide what's relevant?** This is where embeddings and dot products come back. The most common approach:

1. Each stored memory gets embedded — run through an embedding model to produce a vector that captures its semantic meaning.
2. The current query (user message, or a summary of the current conversation) also gets embedded.
3. **[Dot product](/llms/what-happens/vectors/dot-product/) (or cosine similarity)** between the query embedding and every stored memory embedding. Highest scores = most relevant memories.
4. Top-K results get retrieved and injected into context.

This is exactly the same "similarity via dot product" mechanism from [dot products](/llms/what-happens/vectors/dot-product/)), just applied to memory retrieval instead of attention. Memories that are semantically close to the current conversation get high similarity scores and get pulled in.

**How does the model prioritize memories over other context?** It doesn't — not through any special mechanism. Memories are tokens in the context, and the [attention mechanism](/llms/what-happens/embeddings/model-layers/attention-deep-dive/) treats them like any other tokens. However, there are a few factors that give memories influence:

- **Positioning**: memories are typically injected in the system prompt or early in the context. Some models pay more attention to early and recent tokens (a consequence of positional encoding patterns and how attention distributions tend to shape during training).
- **Framing**: the harness usually frames memories with explicit instructions like "The following are your saved memories about this user. Use these to personalize your response." The model learned during training to follow such instructions.
- **Training**: models that support memory features are often specifically fine-tuned on conversations that include memory context, teaching the model to attend to and use memory tokens when they're present.

There is no special "memory [attention head](/llms/what-happens/embeddings/model-layers/attention-deep-dive/multi-head-attention/)" or weighted priority system. It's all regular attention — the model attends to memory tokens the same way it attends to any other token, and the training just makes it likely that relevant memory tokens will get high attention scores.

**Different approaches across providers:**

- **File-based** (like CLAUDE.md / memory files): memories are literal text files loaded into context at conversation start. Simple, transparent, user-editable. Retrieval is basic (load everything, or keyword matching).
- **Vector database-backed**: memories are embedded and stored in a vector DB (Pinecone, Chroma, etc.). Retrieval uses embedding similarity. Scales to many memories because you only inject the most relevant ones, not all of them.
- **Conversation summarization**: instead of saving specific memories, periodically summarize the conversation and store the summary. Less granular but captures overall patterns.

**Performance profile:** Memory systems add cost at two points. **Retrieval** (before the model runs) involves embedding the query and computing similarity against the memory store — this is a lightweight operation, typically running on CPU or a small embedding model, adding milliseconds. **Context injection** is where the real cost is: every memory added to context increases the input token count, increasing [prefill](/llms/what-happens/prefill-decode/) compute (more tokens × all 80 layers) and [KV cache](/llms/what-happens/prefill-decode/kv-cache/) usage. If you inject 2,000 tokens of memories into every conversation, that's 2,000 additional tokens processed through the full model on every request — non-trivial at scale, but usually worth the quality improvement.
