---
title: "Why 16,384 GPUs?"
date: 2026-05-19
draft: false
node_id: "2l"
tier: 2
parent: "1c"
children: []
siblings: ["2k"]
url: "/series/training/train-from-scratch/hardware-and-scale/why-scale/"
summary: "A 70B model trained on 15 trillion tokens literally cannot be done at small scale in a reasonable timeframe. Scale is not a luxury. It's the only way the math works."
series: "training"
mermaid: true
weight: 221
tags: [llm-training]
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture] --> C[Hardware & Scale]:::hl --> D[Training Step] --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click C "/series/training/train-from-scratch/hardware-and-scale/"
```

Meta trained Llama 3 70B on 16,384 H100 GPUs. That's roughly 2,048 nodes of 8 GPUs each (DGX H100 form factor), or in our GB200 NVL72 terms, ~227 racks. This isn't a nice-to-have. A 70B model trained on 15 trillion tokens literally cannot be done at small scale in a reasonable timeframe.

**Why this can't run on a single rack:** The math is unforgiving. Training Llama 3 70B requires roughly 6 × 70B × 15T = 6.3 × 10²⁴ FLOPs (the standard approximation is 6 × parameters × tokens for total training compute). A single GB200 NVL72 rack delivers ~180 petaFLOPS peak at FP8. Realistic MFU (Model FLOP Utilization) for training is 30-50%, giving ~54-90 petaFLOPS effective throughput. At the optimistic end: 6.3 × 10²⁴ / 9 × 10¹⁶ ≈ 70 million seconds ≈ 2.2 years. At the conservative end: ~3.7 years. Nobody is going to babysit a training run for years and hope nothing breaks.

At 16,384 GPUs (~227 racks), you divide that compute across all of them: the run finishes in roughly 1-2 weeks. That's consistent with what Meta reported for Llama 3 70B on H100s. Scale is not a luxury. It's the only way the math works.

### Performance Profile
- **Total training compute:** ~6.3 × 10²⁴ FLOPs (6 × 70B params × 15T tokens). This is the fundamental cost — every other number derives from dividing this across GPUs and time.
- **Single-rack time:** ~2-4 years at realistic utilization (30-50% MFU). Impractical — too long, too many failure opportunities.
- **At 16,384 GPUs (~227 racks):** 1-2 weeks. The only scale where the math works for a production training run.
- **Per-step data consumption:** ~537M tokens = ~2GB raw input per step. Tiny relative to compute, but must arrive on time across 2,048+ nodes every few seconds.
- **Failure cost at scale:** Any single GPU failure, network flap, or storage stall affects all 16,384 GPUs. Mean time between failures across the cluster is hours, not days — operational reliability is the dominant engineering challenge.
