---
title: "The parallelism orchestration stack"
date: 2026-05-19
draft: false
node_id: "3b"
tier: 3
parent: "2n"
children: []
siblings: []
url: "/series/training/train-from-scratch/training-step/forward-pass/parallelism-stack/"
summary: "Slurm launches the job and allocates hardware. But Slurm doesn't know what tensor parallelism is. The actual orchestration — which GPU holds which slice of which layer — is a software stack inside the training job."
series: "training"
mermaid: true
weight: 310
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

Slurm launches the job and allocates hardware. But Slurm doesn't know what tensor parallelism is. The actual orchestration of how 16,384 GPUs coordinate — which GPU holds which slice of which layer, how they communicate, when they synchronize — that's a software stack that sits *inside* the training job.

## The framework: PyTorch (with FSDP or custom parallelism)

Almost all frontier model training runs on PyTorch. Not TensorFlow, not JAX (except at Google), not custom frameworks. PyTorch's `torch.distributed` module provides the communication primitives — all-reduce, all-gather, broadcast, point-to-point send/recv. These map down to the actual GPU communication libraries.

But vanilla PyTorch doesn't know how to set up a 3D parallelism strategy across 16,384 GPUs. That's where the higher-level libraries come in.

## Megatron-LM (NVIDIA)

This is the de facto standard for large-scale transformer training. Built by NVIDIA's applied deep learning team, Megatron-LM provides:

- **Tensor parallelism implementation.** It knows how to split attention and FFN matrices across GPUs within a group. It handles the column-parallel and row-parallel splits, inserts the all-reduce communications at the right places, and fuses operations to minimize communication rounds. You tell it "tensor parallel degree = 8" and it handles the matrix slicing.
- **Pipeline parallelism with schedules.** Megatron implements multiple pipeline schedules — 1F1B (one forward, one backward), interleaved schedules that reduce bubble size, and virtual pipeline stages. You specify "pipeline parallel degree = 4" and it assigns layers to stages, manages micro-batch routing, and handles the activation transfers.
- **Data parallelism.** The outermost ring. Each data-parallel replica is a full tensor-parallel x pipeline-parallel group. Megatron coordinates gradient sync (all-reduce) across data-parallel replicas after the backward pass.
- **3D parallelism configuration.** You specify three numbers: TP (tensor parallel), PP (pipeline parallel), DP (data parallel). The product must equal total GPUs: `TP x PP x DP = 16,384`. For Llama 3 70B on H100s, a realistic config might be TP=8 (within one node's 8 GPUs), PP=4 (4 pipeline stages across 4 nodes), DP=512 (512 data-parallel replicas). That's 8 x 4 x 512 = 16,384 GPUs.

## NCCL (NVIDIA Collective Communications Library)

Underneath Megatron and PyTorch, the actual GPU-to-GPU communication uses NCCL. It implements collective operations (all-reduce, all-gather, reduce-scatter, broadcast) optimized for NVIDIA GPU topologies. NCCL knows about NVLink, NVSwitch, PCIe, InfiniBand, and RoCE. When Megatron says "all-reduce these gradients across 512 data-parallel replicas," NCCL figures out the optimal communication pattern — which data goes over NVLink, which goes over InfiniBand, how to pipeline the transfers to overlap with compute.

NCCL is topology-aware at runtime. It detects the physical interconnect hierarchy and builds communication rings or trees that minimize the number of hops over slow links. This is why the same training script can run on different cluster configurations without code changes — NCCL adapts.

## The process group abstraction

PyTorch's distributed module organizes GPUs into "process groups" — subsets of GPUs that communicate together. Megatron creates separate process groups for each parallelism dimension:

- A **tensor parallel group** of 8 GPUs that share a layer's computation (communicate every layer)
- A **pipeline parallel group** of 4 GPU groups that form the pipeline stages (communicate at stage boundaries)
- A **data parallel group** of 512 GPU groups that sync gradients (communicate once per step)

Each GPU belongs to exactly one group in each dimension. GPU #0 might be in tensor group 0, pipeline stage 0, data replica 0. GPU #8 might be in tensor group 1, pipeline stage 0, data replica 0. And so on. The mapping of GPU ranks to process groups is determined at launch by the 3D parallelism configuration and the cluster topology.

## What Meta actually used for Llama 3

Meta built their own training infrastructure on top of PyTorch, using code that evolved from Megatron-LM concepts but was heavily customized. Their internal system handles the parallelism configuration, custom communication kernels, fault detection, and automatic restart. It's not open source. The public-facing version is `torchtitan`, a reference implementation that demonstrates similar concepts at smaller scale.

The key point: this is not a single orchestration tool. It's a stack. Slurm at the bottom (hardware allocation), NCCL in the middle (communication), Megatron-LM / custom framework at the top (parallelism strategy). Each layer is independently complex, and they all have to work together perfectly across 16,384 GPUs for weeks without interruption.

## Performance Profile

- **Configuration complexity:** The 3D parallelism config (TP x PP x DP) is one of the most consequential decisions in the entire training run. Wrong ratios waste compute, saturate interconnects, or blow out memory.
- **NCCL's role:** Typically consumes 20-40% of total step time at scale. The rest is compute. Reducing NCCL overhead by even 5% translates to days saved on a multi-week run.
- **Failure blast radius:** A bug in the parallelism setup doesn't produce wrong answers — it produces NaN gradients, hung processes, or silently incorrect weight updates that take thousands of steps to notice in the loss curve. These are among the hardest bugs in all of ML engineering.

### Sources
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
