---
title: "The training objective: shifted tokens & loss masking"
date: 2026-05-19
draft: false
node_id: "3f"
tier: 3
parent: "2o"
children: []
siblings: []
url: "/series/training/train-from-scratch/training-step/loss-calculation/training-objective/"
summary: "Most explanations say the model learns to predict the next word. That's correct but papers over details that matter enormously — shifted token prediction, SFT loss masking, and why the three training stages shape different behaviors."
series: "training"
mermaid: true
weight: 320
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture] --> C[Hardware & Scale] --> D[Training Step]:::hl --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click D "/series/training/train-from-scratch/training-step/"
```

Most explanations say "the model reads text and learns to predict the next word." That's correct, but it papers over details that matter enormously. The training objective differs meaningfully across pretraining, SFT, and preference tuning — not in the architecture or the optimizer, but in *which tokens count toward the loss.*

**The shifted token prediction problem.** A training example is a sequence of token IDs: `[791, 5023, 3831, 389, 279, 5765]`. The model receives tokens at positions 0 through 4: `[791, 5023, 3831, 389, 279]`. The labels — what the model should predict — are the tokens at positions 1 through 5: `[5023, 3831, 389, 279, 5765]`. Every position produces a prediction, and every prediction has a corresponding label. The loss is the average cross-entropy across all positions.

This is the **shifted** part: inputs are tokens 0...n-1, labels are tokens 1...n. The model learns to predict the next token at every position simultaneously, not just at the end. A sequence of 8,192 tokens gives 8,191 training signals in a single forward pass. This parallelism is why pretraining is efficient — you get thousands of gradient signals per sequence.

**A concrete worked example.** Take a 6-token sequence: "The cat sat on the mat" → tokens `[791, 5023, 3831, 389, 279, 5765]`.

| Position | Input token | Model predicts distribution over 128K vocab | True next token | Loss |
|----------|------------|---------------------------------------------|----------------|------|
| 0 | 791 ("The") | P(next) = {5023: 0.02, ...} | 5023 ("cat") | -log(0.02) = 3.91 |
| 1 | 5023 ("cat") | P(next) = {3831: 0.15, ...} | 3831 ("sat") | -log(0.15) = 1.90 |
| 2 | 3831 ("sat") | P(next) = {389: 0.40, ...} | 389 ("on") | -log(0.40) = 0.92 |
| 3 | 389 ("on") | P(next) = {279: 0.60, ...} | 279 ("the") | -log(0.60) = 0.51 |
| 4 | 279 ("the") | P(next) = {5765: 0.05, ...} | 5765 ("mat") | -log(0.05) = 3.00 |

Average loss: (3.91 + 1.90 + 0.92 + 0.51 + 3.00) / 5 = 2.05. The model is bad at predicting which noun comes first ("The ___") and what "the" refers to at the end ("the ___"), but decent at predicting prepositions and articles. Gradients from this example push the model to assign higher probability to "cat" after "The" and "mat" after "on the."

**Loss masking in SFT changes everything.** During pretraining, every token contributes to the loss. During [supervised fine-tuning](/series/training/train-from-scratch/post-training/sft/), you have prompt-response pairs:

```
[System: You are a helpful assistant.]
[User: What is the capital of France?]
[Assistant: The capital of France is Paris.]
```

In SFT, the loss is computed *only on the assistant's response tokens*. The system prompt and user message are masked — the model processes them (they provide context during the forward pass) but receives no gradient signal from them. The model is not penalized for failing to predict what the user will say. It's only penalized for failing to predict what the assistant should say.

This masking has profound consequences:

- **The model learns what to generate, not what to expect.** Pretraining trains the model on the entire distribution of text — including predicting user queries, error messages, and random web content. SFT focuses the learning exclusively on the assistant's output distribution.
- **Prompt tokens are "free" context.** The system prompt and user message enter the forward pass and shape the hidden states, but they don't generate gradients. This means SFT can use long, detailed system prompts without wasting gradient signal on predicting them.
- **Capability formation is targeted.** The model's behavior changes in the direction of producing good assistant responses — formatting, helpfulness, instruction following — without also being trained to produce good user questions or system prompts.

**Why this matters: pretraining ≠ SFT ≠ preference tuning.** The three major training stages use the same model, the same optimizer, and often the same hardware — but the loss function differs:

| Stage | What tokens get loss | What the model learns |
|-------|---------------------|----------------------|
| Pretraining | All tokens | How text continues in general |
| SFT | Assistant tokens only | How to respond like a good assistant |
| Preference tuning (RLHF/DPO) | Neither directly — loss is over response pairs | Which responses are preferred over others |

In preference tuning, the loss isn't token-level at all. The model generates two complete responses (or is shown a chosen/rejected pair), and the loss pushes the model toward the preferred response and away from the rejected one. This is a fundamentally different optimization signal — it operates at the response level, not the token level.

**The implication for understanding model behavior.** When a model refuses a harmless request, that behavior came from the loss signal during post-training — specifically, from preference pairs where refusal was marked as "preferred" for certain categories of requests. When a model formats code in a specific way, that came from SFT demonstrations that showed that formatting. When a model knows facts about the world, that came from pretraining loss on the tokens that contained those facts. Different training stages shape different aspects of behavior because they apply loss to different parts of the output.

## Performance Profile

- **Pretraining efficiency:** 8,191 training signals per 8,192-token sequence. Every token contributes. Maximum gradient signal per forward pass.
- **SFT efficiency:** Only assistant tokens contribute loss — typically 30-70% of the sequence. System prompts and user messages are "free" forward-pass context. Effective training signals per sequence drop proportionally.
- **Masking implementation:** A binary mask tensor the same shape as the sequence. Multiply the per-token loss by the mask before averaging. Compute cost: negligible. Implementation complexity: moderate (must correctly track which tokens belong to which role).
- **Gradient concentration:** In SFT, gradients are concentrated on assistant tokens. This means the layers and weights most responsible for generating output text receive stronger, more targeted updates. Weights that mostly process input context receive weaker signals.
