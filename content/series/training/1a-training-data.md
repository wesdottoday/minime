---
title: "What is the training data?"
date: 2026-05-14
draft: false
node_id: "1a"
tier: 1
parent: "0"
children:
  - "2a"
  - "2b"
  - "2c"
siblings:
  - "1b"
  - "1c"
  - "1d"
  - "1e"
  - "1f"
  - "1g"
url: "/series/training/train-from-scratch/training-data/"
summary: "Training data for a model like Llama 3 70B is a massive, curated corpus — Meta reported using over 15 trillion tokens from a mix of publicly available sources."
series: "training"
mermaid: true
weight: 100
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data]:::hl --> B[Architecture] --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click A "/series/training/train-from-scratch/training-data/"
    click B "/series/training/train-from-scratch/model-architecture/"
    click C "/series/training/train-from-scratch/hardware-and-scale/"
    click D "/series/training/train-from-scratch/training-step/"
    click E "/series/training/train-from-scratch/training-loop/"
    click F "/series/training/train-from-scratch/post-training/"
    click G "/series/training/train-from-scratch/evaluation/"
```

Training data for a model like Llama 3 70B is a massive, curated corpus — Meta reported using over 15 trillion tokens from a mix of publicly available sources. The raw inputs are web crawls (Common Crawl is the backbone — petabytes of raw HTML scraped from the open internet), books, Wikipedia, GitHub repositories, scientific papers (arXiv), StackExchange, and various other public text sources. The "wavy hands" part is real — companies are deliberately vague about exact sources because of ongoing copyright lawsuits, licensing gray areas, and competitive advantage. Meta says "publicly available" and leaves it there.

The raw data is garbage without heavy processing. A typical pipeline looks like this: First, you deduplicate — the internet is full of repeated content, boilerplate, copied articles, and scraped mirrors. MinHash or exact-match hashing removes near-duplicates at scale. Then you filter for quality — language detection throws out non-target languages, heuristic classifiers score text for "quality" (coherence, information density, not being SEO spam or porn), and perplexity filters use a smaller pre-trained model to flag text that's statistically weird. Then you clean — strip HTML, normalize Unicode, remove excessive whitespace, handle encoding issues. Some pipelines use classifier-based filtering where a model trained on "high-quality" reference text (Wikipedia, published books) scores every document on a quality spectrum.

The result is a curated dataset that's maybe 5-10% the size of the raw crawl. For Llama 3, Meta also deliberately upsampled high-quality sources — code and math got boosted relative to their natural proportion in the crawl because those domains disproportionately improve reasoning capabilities. The data then gets tokenized (chunked into the model's vocabulary) and packed into fixed-length sequences for efficient batching during training.

### Performance Profile
- **Compute-bound:** Quality classifier inference over billions of documents, MinHash computation for deduplication
- **Storage-bound:** Raw Common Crawl is petabytes; even after filtering, the tokenized dataset for Llama 3 is tens of terabytes that need to be streamed to GPUs continuously
- **I/O-bound:** Data loading must keep up with GPU training speed — if the data pipeline stalls, 72 GPUs sit idle burning money
