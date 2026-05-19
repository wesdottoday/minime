---
title: "Scaling laws & compute economics"
date: 2026-05-19
draft: false
node_id: "2v"
tier: 2
parent: "1e"
children:
  - "3k"
siblings:
  - "2t"
  - "2u"
url: "/series/training/train-from-scratch/training-loop/scaling-laws/"
summary: "Scaling laws determine how much capability you get for your compute budget, and they explain why model training is a resource allocation problem, not just an ML procedure."
series: "training"
mermaid: true
weight: 242
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture] --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop]:::hl --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click E "/series/training/train-from-scratch/training-loop/"
```

Every training run is an economic decision. Before a single GPU warms up, someone has decided: how many parameters, how many tokens, how many GPU-hours. Scaling laws are the mathematical relationships that govern these tradeoffs — they predict how much capability you get for your compute budget, and they explain why model training is a resource allocation problem, not just an ML procedure.

## The Core Relationship: Parameters, Tokens, Compute

Training a language model involves three interacting quantities:

- **N** — the number of model parameters (70 billion for Llama 3 70B)
- **D** — the number of training tokens (15 trillion for Llama 3)
- **C** — the total compute budget in FLOPs (approximately `6 × N × D` for a dense transformer)

For Llama 3 70B: `C ≈ 6 × 70B × 15T = 6.3 × 10²⁴ FLOPs.` At the GB200's theoretical ~2,500 TFLOPS (BF16), that's roughly 2.5 million GPU-hours of perfectly efficient computation. Real utilization is 40-50%, so the actual GPU-hours are roughly 2x that — call it ~5 million GPU-hours. At cloud rates, that's tens of millions of dollars.

The fundamental question scaling laws answer is: **given a fixed compute budget C, how should you split it between model size N and training tokens D to minimize loss?**

## Chinchilla: The Compute-Optimal Insight

In 2022, DeepMind's Chinchilla paper (Hoffmann et al.) established the most influential scaling law to date. They trained over 400 language models ranging from 70M to 16B parameters with varying amounts of data, and fit the resulting loss curves to find the optimal allocation.

The key result: **for a fixed compute budget, the optimal model size and training tokens scale roughly equally.** If you double your compute, you should both increase parameters and increase tokens by roughly the same factor. Specifically, Chinchilla found that the compute-optimal ratio is approximately 20 training tokens per parameter.

This was a bombshell because it implied that most existing large models were **undertrained.** GPT-3 (175B parameters, 300B tokens) had a ratio of ~1.7 tokens per parameter — an order of magnitude below compute-optimal. The original Chinchilla model (70B parameters, 1.4T tokens, ratio ~20:1) outperformed the much larger Gopher (280B parameters, 300B tokens, ratio ~1:1) despite using similar compute. The larger model was spectacularly undertrained.

| Model | Parameters | Training Tokens | Tokens/Param Ratio | Relative Performance |
|-------|-----------|----------------|-------------------|---------------------|
| GPT-3 | 175B | 300B | 1.7 | Baseline |
| Gopher | 280B | 300B | 1.1 | Slightly better than GPT-3 |
| Chinchilla | 70B | 1.4T | 20 | Better than Gopher with same compute |
| Llama 3 70B | 70B | 15T | 214 | Far beyond compute-optimal |

The Chinchilla insight reframed the entire field: you don't need to keep making models bigger if you haven't finished training the current size.

## Compute-Optimal vs. Product-Optimal: Why Meta Over-Trained

Here's where it gets interesting. Llama 3 70B was trained on 15 trillion tokens — a tokens-per-parameter ratio of ~214, more than 10x the Chinchilla-optimal ratio of 20. By Chinchilla's framework, this is massively over-trained. Meta should have either used a much larger model with the same compute, or stopped training much earlier.

Meta did this on purpose.

The distinction is between **compute-optimal** and **product-optimal.** Chinchilla optimizes for *training efficiency* — minimizing loss per FLOP spent during training. But training is a one-time cost. Inference is an ongoing cost. Every request to the model costs compute proportional to the model's parameter count. A 70B model is roughly 10x cheaper to serve per query than a 700B model.

Meta's calculation: if we spend the compute budget to train a smaller model for much longer than Chinchilla-optimal, we get a 70B model that approaches the quality of a much larger compute-optimal model — but at 1/10th the inference cost per query. The extra training compute is a one-time investment that pays dividends on every future inference call.

This is the **inference-aware scaling** paradigm. The optimal training strategy depends not just on the training compute budget but on the expected inference volume. For a model that will serve billions of queries, over-training a smaller model is economically rational even if it's training-compute-suboptimal.

**The math works out roughly like this:** Suppose a Chinchilla-optimal allocation produces a 300B model. Serving that model to 1 billion queries per day costs ~4x more in GPU-hours than serving a 70B model. If the 70B model, trained 10x longer than Chinchilla-optimal, achieves 95% of the 300B model's quality, the inference savings over the model's deployment lifetime vastly exceed the extra training cost. Meta reportedly spent weeks of extra training time to save months of inference costs.

## The Cost Curve: Doubling Compute

Scaling laws follow **power laws** — smooth curves on a log-log plot. The relationship between compute and loss looks roughly like:

`Loss ∝ C^(-α)`

where α is a small exponent (approximately 0.05-0.1 depending on the study). This means:

- **Doubling compute reduces loss by ~5-7%.** That sounds small, but on benchmarks it can translate to 5-15 percentage points of improvement depending on the task.
- **10x compute reduces loss by ~15-20%.** This is why each generation of frontier models requires roughly an order of magnitude more compute than the last.
- **Diminishing returns are baked in.** The power law means each successive doubling of compute buys less improvement than the last. Going from 10²² to 10²³ FLOPs buys more than going from 10²⁴ to 10²⁵.

At current frontier scales (~10²⁵ FLOPs for the largest training runs), each meaningful quality improvement requires roughly a 3-10x increase in compute. This is why training budgets are growing exponentially — from millions to tens of millions to hundreds of millions of dollars per run — while model quality improves incrementally.

## Why This Matters for NVIDIA

Every training run is a purchase order for GPU-hours. Scaling laws directly determine how many GPUs a lab needs and for how long. When DeepMind showed that Chinchilla-optimal training requires 20 tokens per parameter, they effectively told the industry: you need more training time, not just more parameters. More training time means more GPU-hours. More GPU-hours means more GPU purchases.

The inference-aware scaling paradigm is equally significant. Over-training smaller models doesn't reduce total compute — it shifts it from inference (which can use smaller GPUs and quantized models) to training (which demands the highest-end hardware). A lab that over-trains a 70B model by 10x is buying 10x more training GPU-hours, and that training must happen on the latest hardware where memory bandwidth and interconnect matter most.

The hardware implications cascade:

- **Larger training runs** require more GPUs running longer, driving demand for training-optimized hardware (GB200, NVL72 racks)
- **Over-training strategies** increase total training compute even for smaller models
- **Multi-epoch training** on the full corpus means the data must be served faster (storage bandwidth) and processed more times (GPU utilization)
- **Scaling law research itself** requires running hundreds of experimental models at various sizes — each one consuming GPU-hours

## Where Scaling Laws Break Down

Scaling laws are empirical fits, not physical laws. They describe smooth trends in aggregate loss, but several phenomena break the clean picture:

**Emergent capabilities** don't follow smooth scaling. Certain abilities — multi-step reasoning, chain-of-thought, in-context learning with many examples — appear to emerge abruptly at specific scales rather than improving gradually. A model with 10B parameters might score 0% on a reasoning benchmark, while a 70B model scores 60%. The scaling law predicted a smooth curve, but the capability jumped. This makes it hard to predict which capabilities a given compute budget will unlock.

**Data constraints** break the tokens-per-parameter relationship. Chinchilla assumes you have unlimited unique training data. In practice, the high-quality text corpus tops out at ~5-10 trillion unique tokens. Beyond that, you're training multi-epoch on repeated data, and the marginal value of each additional token seen drops faster than the scaling law predicts. This is one reason the Chinchilla ratio has been stretched so far — labs are spending compute on additional epochs because they can't find enough new data.

**Post-training effects** are not captured by pretraining scaling laws. A base model's loss follows clean power laws. But the model users actually interact with has been through SFT, RLHF/DPO, and potentially further fine-tuning. The relationship between pretraining loss and final assistant quality is noisy and non-monotonic — a model with slightly higher pretraining loss can be a better assistant after post-training if its representations are more amenable to alignment.

**Architecture matters more than scaling laws suggest.** The original scaling laws were fit on vanilla transformer architectures. Mixture-of-experts models, which activate only a fraction of their parameters per token, break the simple N-D-C relationship. A 400B MoE model that activates 50B parameters per token doesn't follow the same scaling curve as a 400B dense model. This is relevant because MoE architectures (used by GPT-4, Mixtral, DeepSeek-V3) are increasingly common at the frontier.

## The Current Frontier

As of 2025, the frontier training runs are estimated at 10²⁵-10²⁶ FLOPs. The largest publicly disclosed runs (Llama 3 405B, GPT-4, Gemini Ultra) consumed compute budgets in the hundreds of millions of dollars. The next generation will likely cost billions.

The scaling laws say this will produce measurably better models. But the returns are diminishing, the data is running out, and the costs are growing exponentially. The field is responding with inference-time compute scaling (letting models "think longer" on hard problems), synthetic data generation, and architectural innovations that get more capability per FLOP. Scaling laws got us here, but pure parameter-and-token scaling is approaching its limits.

### Performance Profile
- **Economic framing:** Training compute for Llama 3 70B: ~5M GPU-hours. At a fully loaded cluster cost of $50-100/GPU-hour (amortizing hardware, power, networking, and facilities — cloud on-demand rates can be lower for raw GPU-hours, but don't capture total cost of ownership), that's tens of millions of dollars in training compute. The scaling law says 10x more compute would reduce loss by another ~15-20%.
- **Chinchilla ratio:** 20 tokens per parameter is compute-optimal for training. Llama 3 70B used 214 tokens per parameter — deliberately over-trained to optimize inference economics.
- **Diminishing returns:** Each doubling of compute buys ~5-7% loss reduction. Each generation of frontier models requires ~10x more compute for meaningful improvement.
- **Inference-aware scaling:** Over-training a smaller model is economically rational when inference volume is high. The extra training cost is a one-time investment; the inference savings are continuous.
- **Weak points:** Scaling laws don't predict emergent capabilities, don't account for data exhaustion, don't capture post-training effects, and don't generalize cleanly to MoE architectures. They're useful approximations, not guarantees.

### Sources
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla scaling laws
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
