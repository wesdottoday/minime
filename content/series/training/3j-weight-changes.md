---
title: "What actually changes in the weights"
date: 2026-05-19
draft: false
node_id: "3j"
tier: 3
parent: "2t"
children: []
siblings: []
url: "/series/training/train-from-scratch/training-loop/epochs/weight-changes/"
summary: "Training doesn't insert records into a lookup table. The model's knowledge lives in the geometry of its transformation space — distributed across millions of weights, each contributing a tiny piece to many different capabilities simultaneously."
series: "training"
mermaid: true
weight: 340
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

After 3.75 million optimizer steps, Llama 3 70B's 70 billion parameters have been adjusted from random noise to a state that predicts next tokens well. But what *is* that state? What does it mean for a weight to be "trained"?

**Training doesn't insert records into a lookup table.** The most common lay explanation — "the model stores facts in its weights" — is misleading in an important way. There is no weight that stores "Paris is the capital of France." There is no row in a matrix that you can edit to change "France -> Paris" to "France -> Lyon." The knowledge is **distributed** across millions of weights, each contributing a tiny piece to many different capabilities simultaneously.

## What a single weight "does"

Each of the 70 billion weights is a number — say, 0.0347 — that participates in a matrix multiplication. When token representations flow through the model, each weight scales one dimension of one vector by that amount. A single weight might contribute to the model's knowledge of French geography, its ability to parse prepositional phrases, its tendency to use formal register, and its understanding of capitalization — all at once. Changing that single weight would slightly alter all of those capabilities.

This is because the model's knowledge lives in the **geometry of its transformation space.** Each layer transforms 8,192-dimensional vectors. The directions in that space encode concepts. "Paris" occupies a region of the space. "Capital" occupies a nearby direction. "France" occupies another direction such that the vector arithmetic "France + capital -> region near Paris" works. Training adjusts the weights so that these geometric relationships — the directions, the distances, the clusters — align with the patterns in the training data.

## What changes over 3.75 million steps

In early training (first few thousand steps), the weights are near-random. The model predicts nearly uniform distributions — every next token is equally unlikely. Gradients are enormous because the loss is high. The model is learning the most basic statistical regularities: common words exist, sentences have structure, certain tokens follow other tokens. The weight changes are large and affect the entire geometry.

By mid-training (hundreds of thousands of steps), the model has learned grammar, common knowledge, and basic reasoning patterns. Gradients are smaller. The weight changes are more targeted — refining relationships rather than establishing them. The geometry is largely stable, with incremental adjustments in specific subspaces.

By late training (millions of steps), the model is polishing. The loss curve has flattened. Gradients are tiny. Each step adjusts weights by amounts that are small relative to the weights themselves. The model is learning rare patterns, edge cases, and subtle distributional facts. This is where diminishing returns from repeated data (node 2t) become visible — the weight changes are so small that noisy data repetitions contribute more noise than signal.

## Catastrophic forgetting

Because knowledge is distributed across shared parameters, training on new data can overwrite old capabilities. If you fine-tune Llama 3 70B exclusively on French text for thousands of steps, the model gets better at French but worse at English — not because "English weights" were deleted, but because the shared geometric relationships that supported English were nudged in the direction of French. The same weights serve both capabilities, and optimizing for one perturbs the other.

This is why fine-tuning uses small learning rates (to limit the magnitude of weight changes) and [KL penalties](/series/training/train-from-scratch/post-training/preference-training/) (to penalize divergence from the original model's behavior). The goal is to adjust the geometry slightly — enough to add new behavior without destroying what was already there.

## Interference between capabilities

Related to forgetting, but subtler. Training to improve code generation might slightly degrade multilingual performance — not because code and language use "different weights," but because they share geometric structure. Improving the model's representation of Python syntax might shift directions that also participate in encoding morphological patterns in Turkish. At 70 billion parameters, these interferences are individually tiny, but they accumulate.

This is one reason post-training is done carefully with mixed datasets (blending new capabilities with reminders of old ones) and why preference training uses KL constraints. The model is a single geometric object, and every change to it ripples across all capabilities to some degree.

## Why model editing is hard

Researchers have tried to update individual facts in trained models — change "CEO of Twitter is Jack Dorsey" to "CEO of Twitter is Elon Musk" — by identifying and modifying the specific weights that encode that fact. This works for simple factual associations (localized in specific FFN layers, per the [knowledge localization research](/series/training/train-from-scratch/model-architecture/ffn-hidden-dimension/ffn-parameters/) in 3a), but it's fragile. Changing the weights that encode one fact can subtly corrupt related facts ("Who founded Twitter?" might lose accuracy). The geometry is tangled — facts aren't isolated records, they're directions in a space where nearby directions encode related knowledge.

The fundamental insight: training a neural network is not like filling a database. It's more like sculpting a high-dimensional surface where every groove affects every nearby groove. This is why training from scratch is so expensive, why fine-tuning requires care, and why you can't just "update" a model by editing a few weights.

## Performance Profile

- **Early training dynamics:** Gradients are ~100-1000x larger than late training. Weight change per step is large. The model's internal geometry is being established from scratch.
- **Late training dynamics:** Gradients are tiny. Weight changes per step are <0.01% of weight magnitude. The model is refining, not restructuring.
- **Forgetting rate:** Fine-tuning with a learning rate of 1e-5 (typical for SFT) changes each weight by ~0.001% per step. After 10,000 SFT steps, cumulative changes are ~10% of the pretraining weight magnitude — enough to add new behavior, enough to subtly degrade old behavior.
- **Why this matters for deployment:** A model is a snapshot of its geometry at checkpoint time. It cannot be incrementally updated — new knowledge requires retraining or fine-tuning, both of which risk disrupting existing capabilities. This is the fundamental limitation that retrieval-augmented generation (RAG) works around.
