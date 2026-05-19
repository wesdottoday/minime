---
title: "Validation loss & benchmarks"
date: 2026-05-19
draft: false
node_id: "2z"
tier: 2
parent: "1g"
children: []
siblings:
  - "2aa"
url: "/series/training/train-from-scratch/evaluation/validation-benchmarks/"
summary: "The quantitative signals teams track during and after training — from validation loss curves to downstream benchmarks to human preference evaluations."
series: "training"
mermaid: true
weight: 260
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture] --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]:::hl
    classDef hl fill:#b45309,stroke:#92400e,color:#fff
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click G "/series/training/train-from-scratch/evaluation/"
```

Training a model is not the same as evaluating it. The training loss tells you whether the optimizer is making progress on next-token prediction — whether the weights are getting better at the specific objective they're being trained on. But "better at predicting tokens" and "better at being useful" are different things, and the gap between them is where evaluation lives.

Evaluation during and after training involves a stack of increasingly expensive, increasingly informative signals: validation loss, downstream benchmarks, human preference evaluations, and finally real-world telemetry. Each answers a different question. None is sufficient alone.

## Training loss vs. validation loss

Training loss is computed on the data the model is currently learning from. Training loss generally trends downward over a stable run, though it can be noisy and can rise temporarily during learning-rate changes, data-mixture shifts, or instability events. This is expected — the model is fitting the training data.

Validation loss is computed on a held-out dataset that the model never trains on. It answers the question: is the model learning generalizable patterns, or is it memorizing the training data? When training loss keeps dropping but validation loss plateaus or begins rising, the model is overfitting — it's fitting the specific training examples rather than the underlying distribution.

For a Llama 3 70B model trained on 15T tokens, classic overfitting in the pretraining sense is rare. The dataset is enormous relative to the model's capacity, and most data is seen only once (or very few times). But overfitting can still appear in subtler forms: the model becomes increasingly good at predicting the *style* of its training data without improving its general capability, or it fits distributional quirks of the training corpus that don't generalize.

Validation loss is typically computed every few hundred to few thousand training steps — often enough to track trends, infrequent enough to not waste compute. A single validation evaluation on a meaningful held-out set might process millions of tokens, taking several minutes of cluster time. At a fully loaded cluster cost of $50-100 per GPU-hour (amortizing hardware, power, networking, and facilities across the fleet) and 16,384 GPUs, even a brief validation pause is expensive.

## Per-domain validation loss

Aggregate validation loss hides important structure. A model that improves its overall validation loss might be doing so by getting dramatically better at English web text (the largest data source) while getting slightly worse at code or multilingual text. If you only track the aggregate, you miss this.

Per-domain validation loss tracks performance separately on held-out sets from each major data domain: general web text, books, code (Python, C++, Java, etc.), mathematics, scientific papers, multilingual text (by language or language family), and conversational data. This reveals the model's capability profile at each checkpoint.

During Llama 3 training, Meta tracked per-domain validation loss to ensure that the data mixture was producing balanced improvement. When code loss plateaued while general text loss continued improving, that was a signal to adjust the data mixture — increase the code fraction, or upweight higher-quality code data, or extend training on code-heavy batches.

The per-domain view also reveals *cannibalization*: improving one domain at the expense of another. A data mixture change that upsamples math data might improve math loss but degrade multilingual loss. The total loss might look fine because math gains outweigh multilingual losses in the aggregate. But the model has become worse at something it was previously good at. Per-domain tracking catches this.

## Downstream benchmarks

Validation loss measures how well the model predicts held-out tokens. Downstream benchmarks measure whether the model can actually *do things* — answer questions, write code, solve math problems, reason through multi-step problems.

The standard evaluation suite for large language models includes:

- **MMLU (Massive Multitask Language Understanding):** 57 subjects, multiple-choice questions ranging from elementary math to professional law and medicine. Tests breadth of knowledge. Llama 3 70B scores roughly 79-82% depending on the evaluation configuration.
- **HumanEval / MBPP:** Code generation benchmarks. The model is given a function signature and docstring and must produce a working implementation. Tests functional code generation, not just code-like text.
- **GSM8K:** Grade-school math word problems requiring multi-step arithmetic reasoning. Tests whether the model can chain together arithmetic operations, not just pattern-match on math-like text.
- **MATH:** Competition-level math problems. Much harder than GSM8K — requires genuine mathematical reasoning, not just arithmetic.
- **ARC (AI2 Reasoning Challenge):** Science questions requiring common-sense and scientific reasoning.
- **HellaSwag:** Commonsense reasoning about physical and social situations.
- **TruthfulQA:** Tests whether the model produces truthful answers rather than plausible-sounding falsehoods.
- **WinoGrande:** Commonsense pronoun resolution — tests whether the model understands who or what a pronoun refers to in context.

These benchmarks are run periodically during training — typically every few thousand steps, or at every checkpoint. They're more expensive than validation loss evaluation because they require generation (not just likelihood computation) and sometimes multi-step inference.

**A critical caveat:** Benchmark results are not directly comparable across papers and leaderboards without controlling for evaluation configuration. The same model can score significantly differently depending on: prompt template, few-shot vs zero-shot setup, whether chain-of-thought is allowed, pass@1 vs pass@k scoring, sampling temperature, exact-match vs model-judged scoring, and whether the base model or instruct-tuned variant was tested. When the series cites benchmark numbers, they reflect the evaluation conditions reported by the original authors — which may not match conditions used by other teams reporting on different models.

## The gap between loss and capability

Here's where evaluation gets interesting: validation loss and benchmark performance don't always agree.

A model can have excellent validation loss — it predicts held-out tokens very accurately — but score poorly on reasoning benchmarks. This happens because token prediction and task completion are different skills. A model might predict the *most likely* next token with high accuracy while being unable to chain together the *correct* sequence of tokens to solve a multi-step problem. The most likely token isn't always the right token for reasoning.

Conversely, a model can show modest validation loss improvement between two checkpoints but significant benchmark improvement. A small change in the model's ability to maintain coherence over long reasoning chains might barely move the average loss over millions of tokens but substantially improve its ability to solve GSM8K problems.

This disconnect is why teams track both metrics. Loss is cheap, continuous, and reliable as a trend indicator. Benchmarks are expensive, periodic, and informative about specific capabilities. Neither tells the whole story.

## Checkpoint selection

The final checkpoint in a training run is not necessarily the best checkpoint. Training runs can exhibit late-stage instability, data order effects, or subtle capability regressions in the final stages.

Teams typically evaluate the last several dozen checkpoints across the full benchmark suite and select the one with the best overall profile. "Best overall profile" is itself a judgment call — one checkpoint might score highest on MMLU while another scores highest on HumanEval. The selected checkpoint reflects the team's priorities for the intended use case.

Meta's Llama 3 report described evaluating checkpoints across multiple benchmark categories and selecting based on a composite score. In some cases, the shipped checkpoint was from several thousand steps before the final one, because the final checkpoint showed a regression on coding benchmarks that had been introduced by the last batch of training data.

Checkpoint averaging — taking the element-wise mean of weights across multiple checkpoints — is another technique. It smooths out the noise of individual checkpoints and can produce a model that scores slightly better than any individual checkpoint on average, though it rarely beats the best individual checkpoint on any single benchmark.

## Capability regressions

Improving one capability can degrade another. This is not a theoretical concern — it happens routinely during post-training.

Safety tuning is the clearest example. A model trained to refuse harmful requests may start refusing harmless ones too (over-refusal). A model trained to be more helpful may become less cautious about harmful content. Improving coding ability through code-heavy fine-tuning data can degrade conversational ability. Improving multilingual performance can slightly reduce English performance.

These regressions happen because the model's parameters are shared across all capabilities. Adjusting weights to improve one behavior inevitably changes the model's behavior elsewhere. The effect is usually small — a 2-3% regression on one benchmark while gaining 5-10% on the target benchmark — but it compounds across multiple rounds of optimization.

The practical consequence is that evaluation must track a wide capability surface, not just the metric being optimized. Teams maintain evaluation dashboards with dozens of metrics, and any training run that improves the target metric but regresses significantly on other metrics is flagged for review.

## Human preference evaluations

Benchmarks are automated, reproducible, and cheap relative to human evaluation. But they're also limited — they test specific, narrow capabilities using artificial formats (multiple choice, function completion, word problems). They don't capture the holistic quality of a model's responses in open-ended conversation.

Human preference evaluation is the gold standard. Human annotators are presented with the same prompt and two different model responses (from different checkpoints, different models, or different training configurations) and asked which response they prefer. Over thousands of comparisons, a reliable preference ranking emerges.

This is expensive. A single evaluation campaign comparing two models might require 2,000-10,000 pairwise comparisons. At $1-5 per comparison (depending on task complexity and annotator expertise), that's $2K-$50K per evaluation round. For a training pipeline that produces dozens of candidate checkpoints across multiple post-training configurations, the total evaluation cost can reach hundreds of thousands of dollars.

Human evaluation is also slow — days to weeks for a full evaluation round, versus minutes for automated benchmarks. And it's subject to annotator bias: annotators tend to prefer longer responses (verbosity bias), more confident responses (style bias), and better-formatted responses. These biases mirror the reward model biases discussed in the previous article, because the reward model was trained on the same kind of human judgments.

Despite these limitations, human preference evaluation catches things benchmarks miss. A model that scores identically to another on all automated benchmarks might produce noticeably worse conversational responses — awkward phrasing, unhelpful hedging, unnecessary verbosity, or subtle factual errors that benchmarks don't test. Human annotators notice these differences.

## Product telemetry

After deployment, the most informative evaluation signal comes from real users. Product telemetry — thumbs up/down ratings, conversation length, user retention, task completion rates, retry rates (how often users rephrase and try again), and session duration — reveals how the model performs in its actual use case.

This is the ultimate evaluation, but it's also the noisiest and hardest to attribute. If user retention drops after a model update, was it the model quality, a product UI change, seasonality, or something else entirely? If thumbs-down ratings increase, is it because the model is worse or because the user population shifted (e.g., more novice users joining)?

Telemetry also has a survivorship bias: it only measures users who stayed. It doesn't capture the users who tried the model once, found it unhelpful, and never returned. And it's subject to the model's own influence — a more verbose model might show longer conversations without those conversations being more productive.

Despite these challenges, product telemetry is the signal that ultimately matters. A model that scores well on benchmarks and human preference evaluations but poorly on product telemetry has a real problem. The reverse — strong product telemetry with modest benchmark scores — is a model that's doing its job.

## The evaluation stack

These signals form a hierarchy of increasing cost and informativeness:

1. **Training loss** — continuous, cheap, measures optimization progress
2. **Validation loss** — periodic, cheap, detects overfitting
3. **Per-domain validation loss** — periodic, moderate cost, reveals capability balance
4. **Automated benchmarks** — periodic, moderate cost, tests specific capabilities
5. **Human preference evaluation** — batch, expensive, captures holistic quality
6. **Product telemetry** — continuous, free (in marginal cost), measures real-world performance

Each layer catches problems the previous layer missed. A model that passes all six is genuinely good. A model that passes the first four but fails human evaluation has benchmark-friendly behavior that doesn't translate to quality. A model that passes human evaluation but fails product telemetry has evaluator-friendly behavior that doesn't translate to user value.

No single metric suffices. Evaluation is the practice of triangulating between many imperfect signals to estimate a quantity — model quality — that can't be directly measured.

### Performance Profile
- **Validation loss evaluation cost:** Millions of tokens processed every few thousand training steps. At scale, each evaluation pass takes 2-10 minutes of cluster time. Typically 0.5-1% of total training compute.
- **Benchmark evaluation cost:** Full benchmark suite (MMLU, HumanEval, GSM8K, etc.) requires model generation on thousands of problems. Running the full suite on a 70B model takes 1-4 hours depending on the harness and hardware. Run at every checkpoint during critical training phases.
- **Human evaluation cost:** $2K-$50K per evaluation round. 2,000-10,000 pairwise comparisons per round. Turnaround time: days to weeks. The expense limits human evaluation to a handful of key checkpoints and training configurations.
- **Checkpoint selection window:** Teams typically evaluate the final 20-50 checkpoints across the full evaluation stack. The selected checkpoint may be thousands of steps before the final one. This selection process itself can take days.
- **Regression detection latency:** Automated benchmarks catch regressions within hours (at the next evaluation pass). Human evaluation catches regressions within days to weeks. Product telemetry catches regressions within days to months. The slower the signal, the more expensive the regression — a capability loss detected by product telemetry may affect millions of users before it's identified.
