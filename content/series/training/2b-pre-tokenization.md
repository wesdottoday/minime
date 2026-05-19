---
title: "Why can't training data be pre-tokenized at the source?"
date: 2026-05-14
draft: false
node_id: "2b"
tier: 2
parent: "1a"
children: []
siblings:
  - "2a"
  - "2c"
url: "/series/training/train-from-scratch/training-data/pre-tokenization/"
summary: "If Common Crawl is the backbone for everyone, why doesn't Common Crawl just ship pre-tokenized binary files? Several reasons, and they compound."
series: "training"
mermaid: true
weight: 201
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

Reasonable question — if Common Crawl is the backbone for everyone, why doesn't Common Crawl just ship pre-tokenized binary files and save every lab on the planet those compute cycles? Several reasons, and they compound.

**Every model has its own vocabulary.** Llama 3 uses a 128K-token BPE vocabulary. GPT-4 uses a different one (~100K tokens, different merge rules). Mistral uses yet another. The tokenizer isn't a standard — it's a design decision baked into the model architecture. "The" might be token 791 in Llama 3's vocabulary and token 464 in GPT-4's. Pre-tokenized data from one vocabulary is useless to a model built on another. You'd have to detokenize it back to text and retokenize with your own vocab, which is worse than just starting from text.

**Quality filtering is opinionated and proprietary.** What Meta considers "high quality" text is not what Google considers high quality. Each lab builds custom classifiers, uses different perplexity thresholds, applies different deduplication radii. One lab might aggressively filter out all content below a quality score of 0.7; another might keep more borderline content but apply curriculum-style weighting later. The filtering pipeline is where a lot of the secret sauce lives — it's one of the few remaining levers that meaningfully differentiates training runs when everyone is using roughly the same raw sources.

**Data mixing ratios are a core hyperparameter.** Llama 3 upsampled code and math. Anthropic presumably has a different mix optimized for instruction-following and safety. The ratio of web text to books to code to scientific papers isn't standardized — it's tuned per model, sometimes adjusted mid-training based on loss curves. A pre-packaged dataset locks you into someone else's mix decisions.

**Sequence packing depends on context length.** Llama 3 pretrains at 8,192 tokens per sequence. Other models might use 2,048, 4,096, or 32,768. The way documents get concatenated, where separator tokens go, how padding is handled — all of this is context-length-dependent. A dataset packed for 8K sequences can't be directly reused for a 4K or 32K training run without re-packing.

**Reproducibility and auditability.** When your training run produces unexpected behavior — the model is weirdly good at Turkish, or weirdly bad at chemistry — you need to trace back to the data. If you started from someone else's pre-processed binary blob, you can't inspect what's actually in there without reverse-engineering their pipeline. Starting from raw text with your own pipeline gives you a complete audit trail from source document to token sequence.

**Legal exposure.** If Common Crawl shipped tokenized data and it contained copyrighted material, every downstream user inherits that liability with less visibility into what they're training on. Keeping the processing in-house means each lab makes its own decisions about what to include, exclude, or license — and can demonstrate due diligence if challenged.

The net result: everyone starts from roughly the same raw text, but the path from raw text to GPU-ready binary is where most of the differentiation happens. It's not wasted compute — it's the curation step that determines what the model actually learns.
