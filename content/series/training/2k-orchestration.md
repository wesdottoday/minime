---
title: "Orchestration — Slurm vs Kubernetes"
date: 2026-05-14
draft: false
node_id: "2k"
tier: 2
parent: "1c"
children: []
siblings: ["2l"]
url: "/series/training/train-from-scratch/hardware-and-scale/orchestration/"
summary: "Somebody has to actually launch the training job, allocate GPUs, handle failures, and manage the queue of researchers waiting for compute time."
series: "training"
mermaid: true
weight: 220
tags: [llm-training]
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture] --> C[Hardware & Scale]:::hl --> D[Training Step] --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click C "/series/training/train-from-scratch/hardware-and-scale/"
```

Somebody has to actually launch the training job, allocate GPUs, handle failures, and manage the queue of researchers waiting for compute time. That's the job scheduler.

**Slurm** is the dominant choice for large-scale model training. It's been the standard in HPC (high-performance computing) for decades — supercomputers, national labs, physics simulations. It thinks in terms of nodes, cores, and GPUs as first-class resources. You submit a job script that says "I need 72 GPUs on nodes with NVLink connectivity for 14 days," and Slurm finds the right nodes, allocates them exclusively to your job, launches your training script across all of them, and holds the reservation until the job completes or you release it.

**Why Slurm wins for training:**

- **Exclusive allocation.** Training needs all 72 GPUs working in lockstep. If another job steals one GPU, the entire training run stalls because the distributed communication pattern breaks. Slurm gives you exclusive, bare-metal access to your allocation. No sharing, no noisy neighbors.
- **Gang scheduling.** All nodes for a distributed job start simultaneously. This is non-negotiable for training — you can't have 70 of 72 GPUs running while 2 are still being allocated. Every GPU must start the same step at the same time.
- **Topology awareness.** Slurm can be configured to understand the physical network topology — which nodes share an NVLink switch, which are on the same InfiniBand leaf switch. It allocates nodes that are physically close to minimize communication latency.
- **Long-running job support.** Training runs last days to weeks. Slurm is built for this — it handles node reservations, job priorities, fair-share scheduling across teams, and preemption policies. It doesn't assume jobs are short-lived.
- **Checkpoint/restart integration.** When a GPU fails mid-training (and it will — at scale, hardware failures are not exceptional, they're expected), Slurm can be configured to detect the failure, save a checkpoint, reallocate healthy nodes, and restart from the last checkpoint automatically.

**Kubernetes** dominates cloud-native and inference workloads but historically struggled with training. Kubernetes thinks in pods and containers — it's designed for stateless, short-lived, horizontally-scalable microservices. Training is the opposite: stateful, long-lived, tightly-coupled, and intolerant of disruption.

**Where Kubernetes falls short for training:**

- **No native gang scheduling.** Kubernetes schedules pods independently. Getting 72 pods to start simultaneously requires add-ons (like Volcano or custom schedulers).
- **No topology awareness out of the box.** Kubernetes doesn't know about NVLink domains or InfiniBand fabrics. Without custom topology plugins, it might scatter your pods across racks.
- **Container overhead.** Minor, but real — the container runtime, network namespace, and overlay networking add latency that bare-metal Slurm jobs don't have.
- **Preemption model.** Kubernetes can preempt pods to make room for higher-priority workloads. Having a 3-day training run killed is catastrophic without robust checkpointing.

**The convergence:** The line is blurring. NVIDIA's Base Command Manager and Run:ai layer training-aware scheduling on top of Kubernetes. Volcano and Kubeflow add gang scheduling. But the default path for a team that says "I need to train a 70B model" is still overwhelmingly Slurm.

### Performance Profile
- **Slurm overhead:** Near zero — it's a job launcher, not a runtime. Once the job starts, Slurm is out of the way
- **Kubernetes overhead:** Container runtime + overlay networking adds 1-5% latency to inter-node communication, plus scheduling complexity
- **The real cost of orchestration failures:** A mis-scheduled training run that puts GPUs on different InfiniBand leaf switches instead of the same NVLink domain can cut training throughput by 30-50%. The scheduler's topology decisions are worth more than most hyperparameter tuning
