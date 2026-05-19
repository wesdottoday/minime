---
title: "Supervised fine-tuning (SFT)"
date: 2026-05-19
draft: false
node_id: "2w"
tier: 2
parent: "1f"
children: []
siblings:
  - "2x"
  - "2y"
url: "/series/training/train-from-scratch/post-training/sft/"
summary: "Demonstrations of desired behavior. The first step after pretraining that begins shaping the model from a text completer into an assistant."
series: "training"
mermaid: true
weight: 250
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture] --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop] --> F[Post-Training]:::hl --> G[Evaluation]
    classDef hl fill:#b45309,stroke:#92400e,color:#fff
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click F "/series/training/train-from-scratch/post-training/"
```

A pretrained model is a text completer. You give it the beginning of a document and it continues it plausibly. Ask it a question and it might answer — or it might generate ten more questions, or continue in the style of a FAQ page, or produce something that looks like a Wikipedia article about the question. It has learned the statistical structure of language across 15 trillion tokens, but nobody ever told it what a helpful response looks like.

Supervised fine-tuning is how you tell it.

## What SFT Is

SFT trains the model on **demonstrations of desired behavior** — thousands of examples of prompt-response pairs where the response is exactly what you want the model to produce. The prompt might be "Explain quantum entanglement to a 10-year-old" and the response is a clear, age-appropriate explanation. The prompt might be "Write a Python function to merge two sorted lists" and the response is clean, well-commented code.

These demonstrations are not scraped from the internet. They're **curated.** Human annotators — often teams of dozens to hundreds of people with specific writing guidelines — craft responses that embody the desired assistant behavior: helpful, structured, appropriately detailed, honest about uncertainty, and formatted consistently. Some labs supplement human-written demonstrations with responses distilled from stronger models (e.g., using GPT-4 outputs to train a smaller model), but the principle is the same: show the model what good looks like.

The SFT dataset is typically small relative to pretraining — **10,000 to 100,000 examples** compared to trillions of pretraining tokens. Meta's Llama 3 SFT dataset was in this range. The dataset is small because quality matters more than quantity here, and because a little bit of high-quality signal goes a remarkably long way when applied to a pretrained model that already understands language.

## Loss Masking: The Fundamental Difference from Pretraining

This is the most important technical distinction between pretraining and SFT, and it's where most casual explanations get sloppy.

In pretraining, the model computes loss on **every token** in the sequence. Given a document of 8,192 tokens, each position contributes to the cross-entropy loss — the model is trained to predict token 2 from token 1, token 3 from tokens 1-2, and so on across all 8,191 prediction positions. Every token is both a training signal and a label.

In SFT, the training sequence looks like a conversation:

```
[System] You are a helpful assistant.
[User] What is the capital of France?
[Assistant] The capital of France is Paris.
```

But the model **only computes loss on the assistant tokens.** The system prompt and user message are fed through the model — they provide context for the forward pass — but the gradients only flow from the assistant's response. The loss is masked to zero on all non-assistant tokens.

Why does this matter? Because it fundamentally changes what the model learns. In pretraining, the model learns to predict all text, including user prompts, system instructions, HTML boilerplate, and metadata. In SFT, the model learns *only* to generate good assistant responses given a prompt as context. The user's question provides the conditioning signal, but the model is never trained to *generate* user questions or system prompts. This is what makes SFT directional — it trains the model toward a specific role.

Node 3f covers the training objective mechanics in depth — the shifted token prediction, the cross-entropy calculation, and exactly how the loss mask interacts with the autoregressive structure. The key point here is: **SFT is not just pretraining on different data.** The loss function is structurally different because of which tokens count.

## What SFT Teaches (And What It Cannot)

SFT is remarkably effective at teaching surface-level behavior patterns:

**Formatting conventions.** A pretrained model doesn't know it should use markdown headers, bullet points, or code blocks. SFT demonstrations teach it that code goes in fenced code blocks, lists use consistent bullet styles, and mathematical expressions use LaTeX notation. After a few thousand examples, the model reliably produces well-formatted responses.

**Instruction following.** The pretrained model treats everything as text continuation. SFT teaches it the conversational contract: when someone asks a question, you answer it. When someone says "write this in Python," you produce Python code, not a discussion about Python. When someone says "be brief," the response should be short. This is not semantic understanding of instructions — it's pattern matching on instruction-response pairs. But it's effective pattern matching.

**Tone and register.** SFT demonstrations establish what the assistant "sounds like" — professional but approachable, precise but not pedantic, willing to say "I don't know." The model absorbs these patterns from even a small number of consistent examples.

**Response structure.** Demonstrations teach the model to open with a direct answer, provide supporting detail, use examples, and conclude cleanly. Before SFT, the model might bury the answer in paragraph three or trail off into an unrelated tangent. After SFT, it structures responses like the demonstrations.

**Tool-use patterns.** If the SFT data includes examples of the model generating function calls, API requests, or structured outputs, it learns when and how to produce them. This is how models learn to use tools — not from understanding APIs, but from demonstrations of tool invocation and result handling.

But SFT has hard limits:

**It cannot teach subjective quality.** Given two grammatically correct, factually accurate, well-formatted responses — one concise and one verbose — SFT has no mechanism to teach the model which is *better.* Both are valid demonstrations. SFT can only show the model examples of good behavior; it cannot teach it to distinguish good from great. That's what preference training handles.

**It cannot teach safety boundaries reliably.** You can include demonstrations where the model refuses harmful requests, and the model will learn to refuse *those specific types of requests.* But SFT doesn't generalize well to novel harmful requests that don't match the refusal patterns in the demonstrations. A model SFT'd on 50 refusal examples might refuse "how to make a bomb" but comply with a slightly rephrased version that doesn't pattern-match. Robust safety usually requires more than SFT alone — preference data, adversarial examples, safety-specific evaluation, and often runtime policy layers.

**It cannot teach nuanced judgment.** When the correct response depends on context, audience, or subtle priorities — when there's no single "right" answer — SFT struggles because it can only learn from individual demonstrations, not from comparative feedback. Is it better to be thorough or concise? It depends on the question. SFT teaches the model to produce *a* reasonable response, but preference training teaches it to produce the *best* response.

## Data Quality vs. Quantity: The SFT Paradox

One of the most counterintuitive findings in post-training is that **a small number of high-quality demonstrations dramatically outperforms a large number of mediocre ones.**

The LIMA paper (Less Is More for Alignment) showed that fine-tuning a 65B-parameter model on just 1,000 carefully curated examples produced a model that competed with models trained on 50x more SFT data. Other studies have confirmed: the quality curve for SFT is steep. Going from 0 to 10,000 high-quality examples transforms the model from a text completer to a usable assistant. Going from 10,000 to 100,000 examples improves it further. Going from 100,000 to 1,000,000 examples of lower quality can actually make it *worse* — the model starts mimicking the artifacts and inconsistencies in the lower-quality demonstrations.

This is because SFT is fitting to demonstrations, not optimizing a capability. If the demonstrations are inconsistent — some verbose, some terse, some using markdown, some not — the model learns a blurred average of all styles, which is worse than any individual style. High-quality demonstrations are consistent in formatting, tone, accuracy, and structure. The model can learn a clean target from clean data.

The practical consequence: SFT data curation is a quality control problem, not a scale problem. Labs spend enormous effort on annotator guidelines, response auditing, and inter-annotator agreement. A single bad demonstration — one that's factually wrong, poorly formatted, or stylistically inconsistent — contributes noise that takes many good demonstrations to overcome.

## The SFT Training Process

Mechanically, SFT looks like continued pretraining with two key differences: the loss mask and the data.

The model starts from the pretrained checkpoint — all 70 billion parameters initialized to their pretrained values. The learning rate is much lower than pretraining (typically 1-2 orders of magnitude smaller) because you're refining an already-capable model, not training from scratch. A few epochs over the SFT dataset is usually sufficient — 2-5 passes for most configurations.

The total compute cost is tiny compared to pretraining. SFT on 100,000 examples of ~2,000 tokens each is 200 million tokens — roughly 0.001% of the pretraining token count. On a single node with 8 GPUs, SFT takes hours, not weeks. On the full training cluster, it would take minutes. The constraint is data quality, not compute.

After SFT, the model is a functional assistant. It follows instructions, formats responses cleanly, and behaves conversationally. But it has one major gap: it produces *acceptable* responses, not *optimal* ones. It doesn't know which of several acceptable responses a human would prefer. Closing that gap is the job of preference training.

### Performance Profile
- **Data scale:** 10K-100K curated demonstrations, ~200M tokens. Roughly 0.001% of pretraining data volume.
- **Compute cost:** Hours on a single node, minutes on a full cluster. Negligible compared to pretraining.
- **Behavioral impact:** Transforms the model from a text completer to a functional assistant. Formatting, instruction following, tone, and response structure all improve dramatically.
- **Quality sensitivity:** High-quality demonstrations matter far more than quantity. 1K excellent examples can outperform 50K mediocre ones. Data curation is the binding constraint.
- **Weak points:** Cannot teach subjective preferences, robust safety, or nuanced judgment. The model learns to produce acceptable responses but has no mechanism for choosing between two acceptable options. [Loss masking](/series/training/train-from-scratch/training-step/loss-calculation/training-objective/) means the model never learns to generate user messages, which can cause issues with role confusion in edge cases.
