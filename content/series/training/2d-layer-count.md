---
title: "How do you decide the number of layers?"
date: 2026-05-14
draft: false
node_id: "2d"
tier: 2
parent: "1b"
children: []
siblings:
  - "2e"
  - "2f"
  - "2g"
  - "2h"
  - "2i"
  - "2j"
url: "/series/training/train-from-scratch/model-architecture/layer-count/"
summary: "Short answer: you don't calculate it from first principles. You inherit it from a body of empirical research, scaling laws, and ablation studies."
series: "training"
mermaid: true
weight: 210
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture]:::hl --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click B "/series/training/train-from-scratch/model-architecture/"
```

Short answer: you don't calculate it from first principles. You inherit it from a body of empirical research, scaling laws, and ablation studies, then validate with smaller experiments.

The foundational work here is the Chinchilla scaling laws (Hoffmann et al., 2022 from DeepMind). They showed that for a given compute budget, there's an optimal ratio of model parameters to training tokens. But within a fixed parameter count, how you distribute those parameters — wide and shallow vs. narrow and deep — is a separate design axis. 70 billion parameters with 80 layers and d_model=8,192 is one arrangement. You could also get ~70B with 60 layers and d_model=9,500, or 120 layers and d_model=6,800. They'd have roughly the same number of trainable weights but very different behavior.

**Depth (more layers) buys you compositional reasoning.** Each layer is a processing step — information gets transformed, recombined, and refined. Deeper models can build more complex representations because they have more sequential steps to compose features. A shallow model might learn "this word follows that word." A deep model learns "this concept, built from those subfeatures, in this syntactic structure, implies that conclusion." There's research showing that certain reasoning tasks (multi-hop logic, complex code generation) specifically benefit from depth more than width.

**Width (larger d_model) buys you capacity per layer.** A wider model can represent more features simultaneously within each layer. If your model is too narrow, each layer is a bottleneck — it can't hold enough distinct concepts in a single vector. If it's too wide relative to its depth, it has plenty of room to store information but not enough sequential steps to do anything sophisticated with it.

**In practice, the ratios converge.** Look across Llama 3 (70B: 80 layers, d=8192), GPT-3 (175B: 96 layers, d=12288), PaLM (540B: 118 layers, d=18432). The depth-to-width ratio clusters in a surprisingly narrow band. The community has converged on rough ratios through years of trial and error: layer count typically scales as roughly the cube root of parameter count, while d_model scales faster. This isn't a law of physics — it's an empirical regularity that people have found works well.

**The practical process looks like this:** You pick a target parameter budget (70B, because that's what your compute and data budget can support per Chinchilla). You look at what worked for similar-scale models. You run small-scale ablations — maybe train a 1B-parameter model with different depth/width ratios for a few thousand steps and see which one's loss curve looks best. Then you scale up, keeping the ratios roughly constant. If you're Meta, you also have internal results from Llama 1 and Llama 2 that tell you what worked and what didn't.

Nobody is solving an equation that outputs "80 layers." It's more like: the research says deeper is better for reasoning, the empirics say this ratio works at this scale, the hardware constraints say this fits across our GPUs with acceptable pipeline-parallel granularity, and previous versions of this model used a similar ratio. Ship it.

### Performance Profile
- **Pipeline parallelism granularity:** Layer count determines how the model splits across GPUs in pipeline parallelism. 80 layers across 8 pipeline stages = 10 layers per stage. Uneven division (e.g., 80 layers across 6 stages) creates load imbalance where the largest stage becomes the bottleneck.
- **Depth vs. latency:** Each layer is a sequential dependency — token processing must pass through all 80 layers in order. More layers = higher per-token latency during inference. This is the fundamental tension between model quality (deeper is better) and serving speed.
- **Memory per stage:** In pipeline parallelism, each stage holds only its slice of layers. 80 layers at ~880M params/layer = ~11B params per 10-layer stage = ~22GB at FP16. Fits comfortably in a single GPU's 192GB HBM3e, leaving room for activations and KV cache.
