---
title: "\"Directions in the space encode relationships\""
date: 2026-04-13
node_id: "2d"
tier: 2
parent: "1c"
children: []
siblings:
  - "2b"
  - "2c"
  - "2e"
  - "2f"
  - "2g"
url: "/series/inference/what-happens/embeddings/directions/"
aliases:
  - "/llms/what-happens/embeddings/directions/"
summary: "This is one of the most surprising things about embeddings. Take the classic example: king - man + woman ≈ queen. What's actually happening in the vector math?"
series: "inference"
mermaid: true
weight: 203
tags:
  - llm
---

```mermaid
graph LR
    A[Input Text] --> B[Tokenizer] --> C[Token IDs] --> D[Embedding]:::hl --> E[Attention] --> F[FFN] --> G["→ ×80"] --> H[Vocab Proj] --> I[Output]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click A "/series/inference/what-happens/"
    click B "/series/inference/what-happens/tokens/"
    click C "/series/inference/what-happens/tokens/"
    click D "/series/inference/what-happens/embeddings/"
    click F "/series/inference/what-happens/embeddings/model-layers/ffn-deep-dive/"
    click H "/series/inference/what-happens/embeddings/model-layers/final-vector-to-token/"
    click I "/series/inference/what-happens/embeddings/model-layers/final-vector-to-token/"
```
*Highlighted: embedding space where geometric relationships emerge*

This is one of the most surprising things about embeddings. Take the classic example: king - man + woman ≈ queen. What's actually happening in the vector math?

Each token's embedding is a point in 8,192-dimensional space. The *direction* from one point to another is itself a vector — you get it by subtraction. So `king - man` gives you a vector that represents the *difference* between "king" and "man." That difference turns out to point in roughly the same direction as `queen - woman`. The direction encodes the concept "royalty" or "gendered royal title" — and it's consistent across pairs.

**Nobody programmed these directions.** They emerged because the model saw, across billions of sentences, that "king" and "queen" appear in similar contexts with a consistent gender-based offset, the same way "man" and "woman" do. The training process discovered that organizing the embedding space this way — where consistent relationships map to consistent directions — was useful for prediction.

**The relationships aren't stored as an explicit list.** There's no lookup table that says "king relates to queen." The relationships are implicit in the *geometry* of the embedding space — the relative positions of the vectors. The [embedding table](/series/inference/what-happens/embeddings/) gets reorganized during training (through [gradient updates](/series/inference/what-happens/embeddings/gradients/) to its weights) so that these geometric regularities emerge naturally.

Other directions encode other relationships:
- Country → capital (France - Paris ≈ Germany - Berlin)
- Verb tense (walk - walked ≈ swim - swam)
- Comparative forms (big - bigger ≈ small - smaller)

These are all approximate — it's not perfect arithmetic. But the fact that it works at all tells you something profound: the model learned a *structured* representation of meaning, not just a bag of associations.
