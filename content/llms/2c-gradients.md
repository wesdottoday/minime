---
title: "Gradients and gradient updates (how weights get their values)"
date: 2026-04-13
node_id: "2c"
tier: 2
parent: "1c"
children: []
siblings:
  - "2b"
  - "2d"
  - "2e"
  - "2f"
  - "2g"
url: "/llms/what-happens/embeddings/gradients/"
summary: "Training works by repeated trial and error, automated at massive scale. Here's the loop:"
series: "llms"
mermaid: true
weight: 202
tags:
  - llm
---

```mermaid
graph LR
    A[Training Text]:::hl --> B[Forward Pass]:::hl --> C[Prediction]:::hl --> D[Loss Calc]:::hl --> E[Backward Pass]:::hl --> F[Gradients]:::hl --> G[Weight Update]:::hl --> H["Repeat ×trillions"]:::hl
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click A "/llms/what-happens/"
    click C "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"
```
*Training loop — separate from inference pipeline*

Training works by repeated trial and error, automated at massive scale. Here's the loop:

1. **Forward pass**: Feed the model a chunk of text. It predicts the next token at each position.
2. **Loss calculation**: Compare the model's predictions to what the actual next tokens were. This is a critical point: **the training data is treated as ground truth.** The model doesn't "know" what's correct — it just knows what the training text actually said. If the next token in the training data was "Paris," then "Paris" is the right answer, period. The *loss* is a single number measuring how far the model's [probability distribution](/llms/what-happens/embeddings/model-layers/final-vector-to-token/) was from that known-correct answer — higher loss means worse predictions. This is why training data quality matters so much: the model will learn to reproduce whatever patterns exist in the data, accurate or not.
3. **Backward pass (backpropagation)**: This is where gradients come in. The model works backward from the loss through every layer, calculating for each weight: "if I nudged this weight up slightly, would the loss go up or down, and by how much?" That "how much and in what direction" is the **gradient** for that weight. It's a direction sign (up or down) and a magnitude (how sensitive the loss is to this weight).
4. **Weight update**: Nudge every weight in the direction that reduces the loss, by an amount proportional to its gradient. A weight with a large gradient gets a bigger nudge. A weight with a tiny gradient barely moves. The size of these nudges is controlled by the **learning rate** — a hyperparameter that determines how aggressive each update is. These updated weights are written back to the same [weight matrices](/llms/what-happens/embeddings/weights/) they came from — the [embedding table](/llms/what-happens/embeddings/), the attention matrices, the [feed-forward](/llms/what-happens/embeddings/model-layers/ffn-deep-dive/) matrices in every layer. The weights live in GPU memory (HBM) during training, and the updates happen in-place. When training is done, the final weight matrices are saved to disk as the model's checkpoint — that file *is* the trained model.
5. **Repeat** — trillions of times, across terabytes of text.

The gradient is the compass. It doesn't tell you the right value for a weight — it tells you which direction to step. Each update takes one small step. Over billions of steps, the weights converge toward values that make the model good at prediction.

**Why "gradient"?** It's a calculus term — the gradient of a function tells you the direction of steepest increase. Training walks *downhill* on the loss landscape (called gradient *descent*), trying to find a valley where the loss is low. The loss landscape has one dimension per weight — so for a model like Llama 3 405B with 405 billion parameters, that's a 405-billion-dimensional landscape. You can't visualize it, but the math works the same as walking downhill on a real hill — just with incomprehensibly more directions to choose from.

**Performance profile:** Training runs on **GPUs** and hits all three bounds at different stages. The forward pass (step 1) is **compute-bound** — it's the same matmuls as inference, across many tokens in parallel. The backward pass (step 3) is also **compute-bound** and costs roughly **2× the forward pass** (it has to compute gradients for every weight through every layer). But training is also heavily **memory-bound**: the backward pass needs the intermediate activations from the forward pass to compute gradients, so all of those must be stored in HBM. For Llama 3 70B, the model weights alone are ~140 GB in FP16, but the optimizer states (which track momentum and variance per weight, e.g., Adam stores 2 extra copies) add another ~280 GB, and the stored activations add more on top. This is why training requires far more GPU memory than inference — and why large models are trained across hundreds or thousands of GPUs. At multi-GPU scale, training also becomes **network-bound**: after each backward pass, every GPU must synchronize its gradients with every other GPU before updating weights. This gradient synchronization (via all-reduce operations over NVLink and InfiniBand) becomes the dominant bottleneck at large cluster sizes.
