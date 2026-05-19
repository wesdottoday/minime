---
title: "Hardware & scale"
date: 2026-05-14
draft: false
node_id: "1c"
tier: 1
parent: "0"
children:
  - "2k"
  - "2l"
siblings:
  - "1a"
  - "1b"
  - "1d"
  - "1e"
  - "1f"
  - "1g"
url: "/series/training/train-from-scratch/hardware-and-scale/"
summary: "Let's ground this in the actual iron. We're using a GB200 NVL72 rack as our reference, so here's what's physically sitting in the datacenter."
series: "training"
mermaid: true
weight: 120
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture] --> C[Hardware & Scale]:::hl --> D[Training Step] --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click A "/series/training/train-from-scratch/training-data/"
    click B "/series/training/train-from-scratch/model-architecture/"
    click C "/series/training/train-from-scratch/hardware-and-scale/"
    click D "/series/training/train-from-scratch/training-step/"
    click E "/series/training/train-from-scratch/training-loop/"
    click F "/series/training/train-from-scratch/post-training/"
    click G "/series/training/train-from-scratch/evaluation/"
```

Let's ground this in the actual iron. We're using a GB200 NVL72 rack as our reference, so here's what's physically sitting in the datacenter.

**The rack:** One GB200 NVL72 is a single rack containing 36 Grace Blackwell Superchips. Each Superchip is two Blackwell GPUs bonded to one Grace CPU via NVLink-C2C (chip-to-chip). So: 36 Superchips = 72 Blackwell GPUs + 36 Grace CPUs.

**GPU memory:** Each Blackwell GPU has 192GB of HBM3e. Across the rack: 72 × 192GB = 13.8TB of GPU memory. This is where the model weights, gradients, optimizer states, and activations live during training. For context, Llama 3 70B's weights alone are ~140GB at FP16 — that's less than one GPU's memory. But during training, you need 4-20x the weight memory for gradients and optimizer states (we'll get into why), so the full training state for 70B can easily consume 2-3TB+.

**The interconnect — NVLink:** This is the critical piece. All 72 GPUs in the rack are connected via 5th-generation NVLink in a fully connected (all-to-all) topology through an NVLink Switch. Each GPU has 1.8TB/s of NVLink bandwidth. This is not going over PCIe, not going over the network — it's a direct GPU-to-GPU fabric. When GPUs need to exchange gradients or pass activations between pipeline stages, they do it over NVLink at speeds that dwarf anything Ethernet or InfiniBand can offer within a single rack.

**Why this matters for training:** Distributed training is fundamentally a communication problem. Every training step, GPUs need to synchronize — share gradient updates, pass intermediate results between pipeline stages, reduce partial results across data-parallel replicas. The speed of that synchronization directly determines training throughput. NVLink at 1.8TB/s per GPU means a 70B model's full gradient set (~140GB at FP16) can be all-reduced across all 72 GPUs in well under a second. On a slower interconnect, that same operation might take 10-30 seconds, and your expensive GPUs sit idle waiting.

**The Grace CPUs:** Each Grace CPU provides 512GB of LPDDR5X memory. That's 36 × 512GB = 18.4TB of CPU memory across the rack. This serves as a staging area — the data loading pipeline runs on the CPUs, reading tokenized training data from storage and feeding it to the GPUs. CPU memory also enables offloading: optimizer states or rarely-accessed tensors can spill from GPU HBM to CPU memory over NVLink-C2C, which is slower than HBM but faster than going to storage.

**Storage:** The rack connects to a high-performance distributed filesystem — typically Lustre, GPFS, or a purpose-built AI storage tier (like VAST, Weka, or DDN). The tokenized training data (tens of TB) lives here. The data loading pipeline streams it to CPU memory, which stages it for GPU consumption. Storage also handles checkpointing — periodically saving the full training state (weights + optimizer + scheduler state) so you can resume if something fails. A full checkpoint for Llama 3 70B is 2-3TB. You're writing that every few thousand steps.

**Networking beyond the rack:** For single-rack training (our reference setup), NVLink handles everything. But Meta trained Llama 3 70B on 16,384 GPUs — that's ~227 racks. Between racks, the interconnect drops from NVLink to InfiniBand (400Gb/s per port, multiple ports per GPU) or RoCE (RDMA over Converged Ethernet). This is an order of magnitude slower than NVLink, which is why parallelism strategies are designed to keep the heaviest communication within the NVLink domain and minimize cross-rack traffic.

### Performance Profile
- **Compute:** 72 Blackwell GPUs, each capable of ~2.5 petaFLOPS at FP8, ~1.25 petaFLOPS at FP16. Rack total: ~180 petaFLOPS FP8
- **Memory bandwidth:** HBM3e at ~8TB/s per GPU for feeding data to the compute units
- **Interconnect hierarchy:** NVLink (1.8TB/s, intra-rack) >> InfiniBand (50-100GB/s, inter-rack) >> Storage (10-50GB/s depending on setup)
- **The bottleneck cascade:** Compute waits on memory bandwidth, memory bandwidth waits on interconnect, interconnect waits on nothing as long as you stay within NVLink — the moment you go inter-rack, everything slows down
