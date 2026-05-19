---
title: "Synthetic data & distillation"
date: 2026-05-19
draft: false
node_id: "3c"
tier: 3
parent: "2c"
children: []
siblings:
  - "3d"
url: "/series/training/train-from-scratch/training-data/data-mixture/synthetic-data/"
summary: "Synthetic data is not free data. It is a way of converting model capability, filtering, and compute into more training signal. For modern training at the frontier, it's no longer optional."
series: "training"
mermaid: true
weight: 302
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

Synthetic data is not free data. It is a way of converting model capability, filtering, and compute into more training signal. For modern training at the frontier, it's no longer optional — high-quality natural data is running out, and synthetic data is how labs extend the runway.

**Why stronger models generate training data for weaker models.** The core idea is simple: you have a powerful model (the "teacher") that can produce high-quality outputs — reasoning traces, code solutions, formatted responses, tool-use demonstrations. You use the teacher to generate thousands or millions of examples, then train a smaller model (the "student") on those examples. The student learns to mimic the teacher's outputs at a fraction of the teacher's inference cost. This is **distillation** — compressing the teacher's capability into a smaller, cheaper model.

The mechanism works because the teacher's outputs contain more information than raw next-token labels. When a teacher model writes out a step-by-step solution to a math problem, the student doesn't just learn the answer — it learns the reasoning structure, the formatting conventions, and the style of decomposition. Those intermediate tokens are "free" training signal that doesn't exist in the original pretraining data.

**Self-instruct and rejection sampling.** Self-instruct generates instruction-following training data without human annotators. A model generates instructions, then generates responses to its own instructions, then a filter (human or automated) keeps the good ones. Rejection sampling is the quality lever: generate N responses to each prompt, score them (with a reward model, verifier, or execution feedback), and keep only the top-k. For math and code, you can verify correctness automatically — does the code pass tests? Does the math reach the right answer? This creates high-confidence training data at scale.

**Distilling specific capabilities.** Different capabilities require different distillation strategies:
- **Reasoning traces:** The teacher produces chain-of-thought explanations. The student learns not just the answer but the thinking pattern. This is how smaller reasoning models are bootstrapped from larger ones.
- **Tool use:** The teacher demonstrates how to call APIs, format tool requests, and interpret tool responses. The student learns tool-use conventions from demonstrations rather than from sparse reward signals.
- **Code generation:** The teacher writes solutions, the verifier runs tests, and only passing solutions enter the training set. This is one of the highest-value distillation targets because correctness is automatically checkable.
- **Formatting and style:** The teacher demonstrates consistent output formatting — markdown, citations, structured data. The student absorbs these patterns from examples rather than from explicit instructions.

**The risks are real and compound.**

- **Monoculture.** If every lab distills from the same frontier model (or a narrow set of frontier models), the resulting ecosystem of models converges on the same style, the same biases, the same failure modes. Diversity in model behavior decreases even as the number of models increases.
- **Error amplification.** The teacher's mistakes become the student's training data. If the teacher confidently generates a subtly wrong math proof, the student learns that wrong pattern as ground truth. Unlike human-generated data where errors are random, teacher errors are systematic — the same mistake propagates to every student.
- **Style collapse.** Models trained heavily on synthetic data converge on the teacher's voice. The resulting model sounds like the teacher, reasons like the teacher, and fails like the teacher. This is especially visible in conversational AI where synthetic data creates a homogeneous "AI assistant voice" across otherwise different models.
- **Reward-model bias amplification.** If synthetic data is filtered by a reward model, the surviving examples are the ones the reward model likes — which may correlate with verbosity, confident tone, or specific formatting rather than actual quality. The student inherits the reward model's biases, compounded.

**The economic framing.** Synthetic data converts inference compute into training signal. Running a frontier model to generate 10 million high-quality examples costs compute, but that cost is orders of magnitude lower than the cost of training the frontier model in the first place. It's a form of amortization — the frontier model's capabilities, developed at enormous expense, are compressed and redistributed through its outputs.

The uncomfortable implication: if synthetic data from proprietary models enters public training sets (through crawled AI-generated content, for example), every model trained on web data inherits traces of the proprietary model's training. The data ecosystem is no longer separable.

## Performance Profile

- **Generation cost:** Producing 10M synthetic examples at ~1K tokens each = ~10B tokens of inference. At frontier model serving costs, this is $10K-$100K — trivial compared to pretraining compute.
- **Quality-compute tradeoff:** Rejection sampling with N=16 (generate 16, keep 1) costs 16x inference but dramatically improves data quality. For verifiable tasks (math, code), this is almost always worth it.
- **Filtering bottleneck:** The reward model or verifier becomes the quality ceiling. Synthetic data is only as good as your ability to evaluate it.
- **Diminishing returns:** Synthetic data from the same teacher shows diminishing returns faster than diverse natural data. The student saturates on the teacher's capability distribution.
