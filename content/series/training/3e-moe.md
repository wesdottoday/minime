---
title: "Mixture of Experts (MoE)"
date: 2026-05-19
draft: false
node_id: "3e"
tier: 3
parent: "2i"
children: []
siblings: ["3a"]
url: "/series/training/train-from-scratch/model-architecture/ffn-hidden-dimension/moe/"
summary: "MoE decouples total parameter count from per-token compute by routing each token through a subset of expert sub-networks. The architecture behind Mixtral, DeepSeek-V3, and reportedly GPT-4 -- with deep tradeoffs in memory, communication, and training stability."
series: "training"
mermaid: true
weight: 301
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

Llama 3 70B is a dense model -- every token passes through every parameter. Mixture of Experts is an alternative architecture where each token only uses a fraction of the model's total parameters, routed through a subset of "expert" sub-networks. This decouples total parameter count from per-token compute cost, and it's the architecture behind Mixtral 8x7B, DeepSeek-V3, and reportedly GPT-4.

**The core idea.** In a dense model, the FFN in each layer is a single network: 8,192 -> 28,672 -> 8,192, with three matrices totaling ~707M parameters per layer. In a MoE model, that single FFN is replaced by N separate FFN "experts" -- say, 8 experts, each identical in structure but with independent weights. A small router network examines each token and decides which experts to send it to.

**Top-k routing.** The router produces a score for each expert, and only the top-k experts (typically k=1 or k=2) are activated for each token. The router's output scores also serve as mixing weights -- the outputs of the selected experts are weighted-summed based on the router's confidence.

**The parameter math.** The key to MoE economics is that each expert is *smaller* than what a single dense FFN would be at the same model scale. Mixtral 8x7B is the clearest example:
- 8 expert FFNs per layer, each roughly 1/8 the size of what a single large dense FFN would be at that total parameter count
- Top-2 routing: each token is routed through 2 of the 8 experts
- Total parameters: ~47B (all 8 experts loaded across all layers)
- Active parameters per token: ~13B (only the 2 selected experts fire, plus attention and embeddings)

A 47B total / 13B active MoE model has the *knowledge capacity* approaching a ~47B dense model but the per-token *inference compute* of a ~13B dense model. You get much broader knowledge coverage without paying the full compute cost. This is the fundamental value proposition.

**Load balancing -- the hard part.** Left to its own devices, the router learns to send most tokens to 1-2 "favorite" experts and ignore the rest. This is expert collapse -- the majority of the model's parameters go unused. To prevent it, training adds an auxiliary loss that penalizes uneven expert utilization. The auxiliary loss pushes the router toward distributing tokens roughly equally across all experts.

Getting the balance right is delicate. Too much auxiliary loss and the router can't specialize experts (every expert does the same thing, defeating the purpose). Too little and experts collapse. DeepSeek-V3 introduced a "bias term" approach that achieves better balance without distorting the router's learned preferences.

**Memory implications -- the painful tradeoff.** Even though each token only activates k experts, **all expert weights must reside in GPU memory.** A 47B MoE model at BF16 requires ~94GB just for weights -- far more than a 13B dense model with equivalent per-token compute. The compute savings don't translate to proportional memory savings. This is the primary constraint on MoE serving: you need enough GPUs to hold all expert weights even though most weights are idle for any given token.

**Expert parallelism and all-to-all communication.** In a dense model, tensor parallelism splits each matrix across GPUs and every GPU processes every token. In MoE, a natural strategy is expert parallelism: assign different experts to different GPUs. Each GPU holds one or two experts. But tokens are routed to different experts -- so token representations need to move between GPUs based on the routing decision.

This creates an **all-to-all communication pattern**: every GPU potentially needs to send tokens to every other GPU and receive tokens from every other GPU. This is fundamentally different from the all-reduce in data parallelism (which sums tensors) or the point-to-point in pipeline parallelism (which passes activations to the next stage). All-to-all is the most communication-intensive collective, and it happens *every layer* in a MoE model.

For DeepSeek-V3 (671B total, 37B active, 256 experts per layer, top-8 routing), the all-to-all communication was the dominant engineering challenge. They designed custom communication kernels that overlap expert computation with token routing, and they exploited the NVLink topology to minimize cross-rack all-to-all traffic.

**Training challenges unique to MoE:**
- **Router instability.** Early in training, the router's scores are near-random. Small perturbations in gradient noise can cause sudden routing shifts -- one step, expert 3 gets 30% of tokens; next step, it gets 2%. This makes training less stable than dense models.
- **Expert specialization.** The ideal outcome is that each expert learns a different "skill" -- one expert handles code, another handles multilingual, another handles math. In practice, specialization is partial and emergent. Researchers observe some domain specialization but it's not as clean as the theory suggests.
- **Evaluation difficulty.** Standard benchmarks don't capture MoE's advantages well. A large MoE model and a dense model with similar active parameter counts may score similarly on many benchmarks, but the MoE model has better coverage of rare knowledge and rare languages -- capabilities that matter in deployment but don't dominate benchmarks.

## Performance Profile
- **Compute efficiency:** k/N ratio determines per-token FFN savings. Top-2 out of 8 experts (Mixtral) = 2/8 of FFN compute per token. Top-8 out of 256 = ~3% of FFN compute (DeepSeek-V3). Attention is unchanged.
- **Memory cost:** All expert weights must be loaded. Mixtral's ~47B total at BF16 = ~94GB. DeepSeek-V3's 671B total requires multi-node serving regardless of per-token compute cost.
- **Communication:** All-to-all every layer for expert parallelism. This replaces all-reduce (gradient sync) as the dominant communication pattern during MoE training.
- **Training instability:** Router learning and expert collapse are additional failure modes that don't exist in dense models. Auxiliary loss tuning is a MoE-specific hyperparameter.
- **Serving tradeoff:** Lower per-token compute but higher memory footprint relative to active parameter count. MoE models shine when the bottleneck is compute (high-throughput batched serving) rather than memory (low-batch serving where weight loading dominates).

### Sources
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
