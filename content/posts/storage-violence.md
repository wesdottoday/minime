---
title: "AI Commits Storage Violence"
date: 2026-05-22
draft: false
hero: /img/tyson-vs-atc.png
url: /storage-violence
---

Imagine you're working air traffic control. Every transmission is critical. None can be missed. None can be delayed. If you mess up, hundreds of people die. And every 30 minutes Mike Tyson walks in and punches you square in the face, because you owe him money. And he's coming to remind you every 30 minutes until you pay up. You can't go down. The planes don't stop. You're the only one on duty.

Your brain after years of training immediately returns to your job directing flights. Keeping people alive. It all but ignores that Mike Tyson just rearranged your face. Not because it didn't hurt, but because you're so tuned in to each and every transmission that your brain says that the most important thing in the room is the next plane, it doesn't matter that Uncle Mike will be back in 30 minutes, your day to day are the planes.

If you're reading this, there's a {{< tooltip trigger="1 in 843,438,210,000" >}}<span class="tt-factor"><span class="tt-label">P(you're a certified ATC) = <span class="tt-math">1 in 18,789</span></span>14,264 FAA controllers ÷ 268M US adults<br><span class="tt-note">Source: FAA Controller Workforce Plan, 2025</span></span><span class="tt-factor"><span class="tt-label">P(you owe Mike Tyson money) = <span class="tt-math">1 in 670,000</span></span>~400 people in his financial inner circle ÷ 268M<br><span class="tt-note">He paid a man named Crocodile $300K to shout "guerrilla warfare" at press conferences. The bar was not high.</span></span><span class="tt-factor"><span class="tt-label">P(he punches you about it) = <span class="tt-math">1 in 67</span></span>6 documented non-ring assaults ÷ ~400 relationships<br><span class="tt-note">Parking lot attendant '87, Mitch Green at 4AM '88, two motorists '98, photographer at LAX '09, JetBlue passenger '22. Measurable recidivism coefficient: 1.5%.</span></span><span class="tt-factor"><span class="tt-label">P(on a 30-minute schedule) = <span class="tt-math">1 in 1</span></span>2,500 sit-ups/day. 500 push-ups. 500 dips. In sets of 50, spread across the day. 4AM runs, six days a week, for a decade.<br><span class="tt-note">We do not doubt his discipline. P = 1.</span></span><span class="tt-result"><span class="tt-math">18,789 × 670,000 × 67 × 1 = 843,438,210,000</span><br>≈ 120× the world population</span>{{< /tooltip >}} chance this is your lived reality. For the storage architects and AI infrastructure engineers reading this, there's a more reasonable {{< tooltip trigger="1 in 12,500" >}}<span class="tt-factor"><span class="tt-label">Senior storage architects globally: <span class="tt-math">~35,000</span></span>~200K storage professionals × 15-20% at architect/principal level<br><span class="tt-note">Source: derived from Uptime Institute data center staffing estimates</span></span><span class="tt-factor"><span class="tt-label">Senior AI infrastructure engineers: <span class="tt-math">~30,000</span></span>People who build training clusters, not people who pip install torch.<br><span class="tt-note">This job title barely existed 5 years ago.</span></span><span class="tt-factor"><span class="tt-label">Overlap: <span class="tt-math">~5,000</span></span><span class="tt-note">Rare enough to be interesting at parties.</span></span><span class="tt-factor"><span class="tt-label">English-speaking working-age adults: <span class="tt-math">~750,000,000</span></span></span><span class="tt-result"><span class="tt-math">60,000 ÷ 750,000,000 = 1 in 12,500</span><br>If you're still reading, you're probably one of them.</span>{{< /tooltip >}} chance the rest of this post is about your Tuesday.

## The Violence

Artifical Intelligence is breaking the designs and architectures of every datacenter architecture we've ever had. While GPUs get most of the headlines, something unlike anything we've ever experienced is happening in the storage world.

I've been deeply attached to the storage portion of the tech industry for a long time. I've spent countless hours designing storage platforms to ensure they'll meet the workload requirements and constraints of some of the worlds most complex and demanding mission critical workloads. For the most part though, I've never had that difficult of a time sizing something properly to meet those needs, just as I imagine most of you have managed fine all this time too. While we've been through technology transitions in this space before, many times, architecture changes in this space, many times, nothing has shaken the industry like AI.

Let me explain.

### AI meets Storage

I split my thinking of artificial intelligence, for the most part, into two buckets: [Training](/series/training/) and [Inference](/series/inference/). For the sake of this blog post, I'll be referencing numbers based on Llama 3 70B and then we will extrapolate some numbers for a 15T parameter MoE frontier model (_Note: This extrapolation is just math, I have no insight into Frontier Models_). Both Training and Inference have unique storage patterns that are worth exploring a bit deeper.

### Training Storage Patterns

When you think of training a large language model, you probably think about the training data set(s) that are used to train it. But just like everything else on the planet, there's much more to it than that. If you want to go deep on this, make sure you dive in the [LLM Training](/series/training) blog series I wrote. For the sake of this post, we'll keep it fairly simple.

Training does require huge data sets, for example, Llama 3 70B was trained on 15T [tokens](/series/inference/what-happens/tokens/), which would consume about 60TB (uncompressed) of storage at 4 bits per token. Storing this data is childsplay. Even petabyte+ sized data sets are easy to store these days. 

Accessing it is a bit more complex, but we'll get there.

There are also two other steps of the training proces that hit storage: [Optimizer](/series/training/train-from-scratch/training-step/optimizer-step/) and then [Checkpointing](/series/training/train-from-scratch/training-step/checkpointing/).

While the Optimizer step can run without hitting the storage, solely living in GPU HBM throughout the cluster (using ZeRO-2, for the sake of Llama 3), once models get large enough you have to put storage in the path to offload HBM by using ZeRO-Infinity. This means that for every single training step (3-5 seconds), each GPU must do a read-modify-write pattern. This is the constant load, the load the air traffic controller deals with all day long, hundreds of planes, thousands of souls. With 16k GPUs, that means 16k clients over GPUDirect all reading and writing their shards of data to the storage. The data footprint is pretty small for this step at about 30MB per GPU, it's the metadata load that really kicks storage's butt.

> ZeRO-Infinity isn't the normal optimizer path today, but as models get larger and larger, it becomes harder to fit everything in HBM.

The final step in the training loop is Checkpointing. You can think of this like a game save point, a place to respawn from when things go sideways. And things absolutely go sideways when you have 16k GPUs cranking hard for months on end. A checkpoint is made up of:

- Model Weights: 140GB (70B parameters at BF16)
- Optimizer States: 560GB (Adam's m and v at FP32)
- Gradients: 140GB (70B parameters at BF16)

**840GB per Checkpoint**

If a failure hits and the last checkpoint is corrupt or missing, you rewind to the one before it and redo all the training steps in between. At the scale of a 16k GPU cluster, that lost work has a price tag. Meta has reported cluster operating costs that put the per-failure cost somewhere between $270,000 and $540,000 depending on how far back you rewind. That's where the $300K number comes from throughout this post, it's a rough midpoint. Every checkpoint is a $300K insurance policy.

For a model like Llama 3 70B there can be 40-60 checkpoints over the full run, so you can see how this becomes a large amount of data to store. But honestly, it's not the data at rest that is the issue, it's the timeframe to write that checkpoint that is the problem. 

With 16k GPUs writing around 51MB each, approximately every 30 minutes, that's a manageable storm for an adequately sized storage platform. The total burst is under a terabyte. Your metadata service takes a beating from 16k simultaneous file creates, but the raw bandwidth is nothing special. Over a full training run of 40-60 checkpoints, you're looking at 34-50TB of total checkpoint writes. Not trivial, but not terrifying either.

Now scale it up.

A 15T parameter MoE model is the direction frontier labs are heading. More experts, more parameters, more knowledge capacity. The active parameters per token stay manageable (maybe 200-400B), but the checkpoint has to save everything. All 15 trillion parameters. All the optimizer state for all of them.

| Component | Llama 3 70B | 15T MoE |
|-----------|-------------|---------|
| Weights (BF16) | 140GB | 30TB |
| Adam m (FP32) | 280GB | 60TB |
| Adam v (FP32) | 280GB | 60TB |
| Gradients (BF16) | 140GB | 30TB |
| **Total** | **~840GB** | **~180TB** |

**Read that again.** 180 terabytes. Per checkpoint. Written all at once, from every GPU in the cluster, every 10-30 minutes, for months. And while you're writing that checkpoint, you still need to serve up the next batch of training data to all those clients.

The cluster scales too. You're not running a 15T model on 16k GPUs. You need 32k or more. Each GPU is now writing a 5.5GB shard instead of 51MB. That's 100x the per-GPU volume.

| Metric | Llama 3 70B | 15T MoE |
|--------|-------------|---------|
| GPU count | 16,384 | ~32,768 |
| Per-GPU checkpoint shard | ~51MB | ~5.5GB |
| Checkpoint burst | ~840GB | ~180TB |
| Checkpoints per run | 40-60 | 50-100 |
| Total checkpoint writes | ~34-50TB | **9-18PB** |

At 70B, the violence is primarily a metadata storm. Small shards, millions of files, the metadata service is the bottleneck. The raw bandwidth is solvable. At 15T, you still have the metadata storm, but now you also have a bandwidth crisis. 180TB needs to land on storage within the checkpoint window. If you need it done in 5 minutes so you don't stall the next training step, that's 600 GB/s of sustained write throughput from the storage tier. Simultaneously. From 32,000 clients.

And over the full training run? 9 to 18 petabytes of checkpoint writes. That's not just a capacity conversation anymore, it's a write endurance conversation for the NVMe drives underneath. Every one of those writes is dense FP32 data that doesn't compress, doesn't deduplicate, and doesn't benefit from any of the efficiency features your storage platform was built around.

That's the punch. And Mike Tyson is getting stronger.

### Inference Storage Patterns

Inference is a different animal. If training is violence, inference is a crowd. A very impatient crowd, all talking at once, and every one of them expects a response in milliseconds.

The first thing that happens when you spin up an inference instance is model loading. You have to get the weights from storage into GPU HBM before you can serve a single request. For Llama 3 70B, that's 140GB. For a 15T MoE, it's 30TB. This isn't a periodic event like checkpointing, it's a startup cost, but it matters when you're autoscaling in response to demand. Every new instance that comes online needs to pull those weights from somewhere, fast. If your storage can't serve 140GB in a few seconds, your autoscaler is useless.

Once the model is loaded, the ongoing storage interaction is the [KV cache](/series/inference/what-happens/kv-cache/). Every request generates key-value pairs that represent the model's attention state for that conversation. The cache grows with context length. A single request at 128K context on a 70B model can generate several gigabytes of KV cache data. Multiply that by hundreds or thousands of concurrent requests, and you're looking at terabytes of hot, transient state that needs to live somewhere.

At 128K context that's already a lot of state to manage. Now consider something like Claude Opus running at a 1 million token context window. The KV cache for a single conversation at that length can balloon to tens of gigabytes, and that's one user. Run a few hundred of those concurrently and you're managing terabytes of transient attention state across the cluster just to keep conversations alive.

When HBM fills up, and it will, that KV cache has to spill. First to CPU memory, then to NVMe, and potentially to network storage. This is a tiered memory hierarchy where the hottest cache entries stay in HBM, warm entries drop to CPU, and cold entries land on flash. Your storage tier just became part of the inference latency path. Every cache miss that hits NVMe instead of HBM adds microseconds to the response. At scale, those microseconds stack.

The access pattern here is nothing like training. Training reads are sequential, predictable, and prefetchable. KV cache reads are random, request-dependent, and latency-critical. Training writes are periodic bursts. KV cache writes are constant, small, and scattered. It looks a lot more like a traditional database workload, just at a scale and latency requirement that most database storage was never designed for.

| Metric | Training | Inference |
|--------|----------|-----------|
| Primary storage interaction | Checkpoint writes (burst) | KV cache tiering (constant) |
| Access pattern | Sequential reads, burst writes | Random reads, scattered writes |
| Latency sensitivity | Step-level (3-5 sec budget) | Request-level (milliseconds) |
| Data lifecycle | Checkpoints retained for days/weeks | KV cache is ephemeral, seconds to minutes |
| Client count | 16-32K GPUs, synchronized | Hundreds to thousands of instances*, independent |
| Dominant bottleneck | Metadata throughput | Read latency |

_*An inference instance is a group of 1-8 GPUs working together via tensor parallelism to serve requests. The GPUs within an instance are coordinated, but each instance operates independently from the storage tier's perspective. The storage client count scales with instance count, not raw GPU count._

In practice, training and inference almost never share the same storage. They're separate clusters, separate networks, separate storage tiers. Nobody is running a 90-day training run and serving production inference off the same platform. These aren't two workloads competing for the same resources, they're two entirely different storage design problems that happen to both live under the umbrella of "AI." You might be the same architect designing both, but the designs share almost nothing.

The saving grace with inference is that the data is ephemeral. KV cache entries live for the duration of a request or session, then they're gone. You don't need replication. You don't need durability. If a node dies, the request retries, the cache rebuilds. Nobody lost $300K of compute because a KV cache entry disappeared.

Training storage is the punch. Inference storage is the crowd noise. Both demand your full attention, but for completely different reasons.

## Your Storage is Blind to it

Every storage platform you've ever worked with makes decisions about your data based on how often it's accessed. Hot data gets fast storage. Cold data gets archived. Frequently read data gets cached, replicated, protected. This is the foundational heuristic that every vendor has built their platform around for the last 20 years, and it works. It works for databases, for virtual desktops, for file shares, for object stores. It works for basically everything.

Except checkpoints.

A checkpoint is written once and, if everything goes according to plan, never read again. It sits there, untouched, until it's superseded by the next checkpoint and eventually deleted. From your storage platform's perspective, this looks like the least important data in the cluster. Cold. Stale. A candidate for archival, tiering, or reduced protection.

In reality, it's a $300,000 insurance policy. The only time you read it is when everything has gone wrong, and at that moment, it's the most important data in your entire infrastructure. Your storage platform has no way to know this.

I spent some time surveying how major storage platforms handle IO profiling and data classification. The pattern is consistent across the industry, regardless of vendor.

The platforms that actively profile your IO and make automatic decisions about data placement and protection are the ones that get this the most wrong. If your platform tracks access recency, checkpoint data gets flagged as cold within minutes. Write-once-never-read is the definition of cold. So the system demotes it to slower storage, reduces the replication factor, applies erasure coding to save space. It's doing exactly what it was designed to do. It's just that what it was designed to do is wrong for this workload.

Platforms with adaptive IO engines that profile write patterns and auto-select protection schemes see a large sequential write and optimize for space efficiency. Erasure coding instead of replication. Less protection, more capacity. On the data that most needs maximum protection.

Platforms that tier based on access temperature push checkpoint data out first. It's the coldest thing on the flash tier. First to get evicted to object storage. If you need to recover from a failure, your checkpoint is now sitting behind a network round-trip instead of on local flash. Recovery time just went from minutes to potentially hours.

Platforms that don't tier at all are inadvertently safer. They don't punish write-once data. But they don't elevate it either. Your $300K insurance policy gets the exact same protection as your log files and temp data.

And here's the part that really keeps me up at night. You know what the ideal storage profile for checkpoint data actually looks like? Write once, sequential, never read unless disaster, durability above all else, doesn't benefit from flash-optimized features. That's tape. The ideal at-rest profile for the most critical data in a hundred-million-dollar AI training cluster is, absurdly, tape. Of course you can't actually use tape because when you need that checkpoint you need it in seconds, not the 30 minutes it takes a robot arm to find your cartridge. But the fact that tape is a closer philosophical match than your all-flash array should tell you something about how broken the assumptions are.

**This is not an indictment of the storage industry.** Every one of these heuristics is correct for every workload that existed before this one. The assumption that access frequency correlates with data importance has held true for decades. Checkpoints break that assumption completely. The most important data in the cluster is the data you hope you never have to read.

No storage platform today has a concept of write-path importance. There's no mechanism for the training framework to tell the storage system "this data is irreplaceable, protect it accordingly" at write time. The framework knows which checkpoints matter, epoch boundaries versus intermediate saves, best-so-far versus routine. But that knowledge stays in the application layer. The storage layer never sees it.

The fix is probably simpler than you'd think. A metadata flag. A hint at write time that says "this object is critical" or "this object is transient." The training framework already knows. It just needs a way to tell the storage. The first AI-focused storage vendor that ships this as a first-class feature will own the conversation. Until then, your storage platform is making decisions about your most valuable data using heuristics that were designed for a world where this workload didn't exist.

## The Cost

Let's put a dollar figure on this.

When a training run fails, you rewind to the last good checkpoint and redo everything since. The gap between that checkpoint and the failure is lost work. At Meta's reported cluster costs for Llama 3, that's somewhere between $270,000 and $540,000 per incident, depending on how far back you have to rewind.

Hardware failures aren't rare at this scale. With 16,000+ GPUs running 24/7 for weeks or months, you're going to lose nodes. Memory errors, NVMe failures, network flaps, power events. The industry reports 1-2 failures per day on large training clusters. Over a 14-day run, that's $4M to $15M in lost compute from failures alone.

Checkpoint frequency is the only lever you have. More frequent checkpoints mean less lost work per failure. But more frequent checkpoints mean more storage violence. Every additional checkpoint is another 840GB burst (or 180TB at 15T scale) that your storage tier has to absorb without flinching.

This is a direct dollars-per-failure tradeoff, and storage is the thing that determines where you land on that curve. If your storage can absorb a checkpoint in 2 minutes, you can checkpoint every 10 minutes and cap your exposure at ~$90K per failure. If your storage takes 15 minutes to land a checkpoint, you're spacing them further apart and your exposure per failure goes up proportionally.

Now think about what happens when the heuristics from the last section kick in. Your storage platform demotes the checkpoint data, reduces its protection, tiers it to slower storage. A failure hits. You reach for the last checkpoint and it's sitting on an HDD tier behind an erasure coding rebuild. Or it's been tiered to object storage and needs to be recalled over the network. Recovery time goes from minutes to hours. The $300K failure just became a $600K failure because the recovery took long enough to lose another checkpoint interval worth of work.

The storage platform didn't lose the data. It just made it slower to get back, at the exact moment when speed is the only thing that matters.

## What Can We Do?

If you're designing or managing the storage for a training cluster, here's what I'd be thinking about.

**RF2 minimum for all checkpoint data.** Don't get clever with replication factors on checkpoints. Yes, RF2 means doubling your checkpoint storage footprint. That's real capacity. But think about what an unrecoverable checkpoint actually means. You don't just lose one checkpoint's worth of work. You have to fall back to an earlier checkpoint, which means even more retraining time. If multiple checkpoints are compromised, or the failure is severe enough, you could be looking at starting the entire run over. On a 90-day training run with 1-2 hardware failures per day, that's a lot of opportunities for things to go wrong. RF2 is cheap insurance. However you slice the storage cost, it's a rounding error on a project of this scale.

**Disable deduplication and compression on checkpoint volumes.** FP32 optimizer states are dense floating point values that change slightly every checkpoint. They don't deduplicate. They don't compress. You're paying the full computational overhead of both features, hash lookups, compression attempts, index maintenance, for a return of essentially zero. Turn them off for checkpoint paths and reclaim those CPU cycles.

**Think about erasure coding carefully.** EC defers its work to quiet periods, and that's smart design for most workloads. But a training run doesn't have quiet periods. The 25 minutes between checkpoint bursts is when the EC engine, the scrubber, the rebalancer, and the replication catch-up process all need to run. Stack those background queues up and the next burst arrives before the last one's housekeeping is done. If you're using EC on checkpoint data, make sure the math works out on your background processing budget, not just your capacity math.

**Design for burst metadata capacity, not just data throughput.** The chokepoint at current scale isn't GB/s, it's the metadata service handling millions of concurrent file creates. Every checkpoint is thousands of sharded files landing simultaneously. Your metadata tier needs to absorb this burst without queuing, because any delay in checkpoint writes is a delay in resuming training, which is a delay in getting to the next checkpoint, which is increased exposure to failure.

**Validate your checkpoints.** A partial write that looks complete is worse than a failed write. If a checkpoint is corrupt and you don't know it until you try to recover, you've lost not just the work since the last checkpoint, but the work since the last *good* checkpoint, which could be two or three intervals back. Read-back and checksum validation after every checkpoint write is cheap insurance.

**Talk to the training team.** The training framework knows things your storage platform doesn't. It knows which checkpoints are critical and which are intermediate. It knows the checkpoint schedule. It knows the expected shard sizes and file counts before they happen. None of this information makes it to the storage layer today. If you can get even basic coordination, a heads-up before a burst, a flag for checkpoint priority, you'll be ahead of every other storage deployment in the industry.

This is new territory. The storage industry has spent decades optimizing for workloads that look nothing like this. The heuristics are wrong. The efficiency features are counterproductive. The assumptions about data importance are inverted. None of that is anyone's fault. This workload didn't exist five years ago.

But it exists now. And it's getting bigger.

---

You're still in the tower. The planes haven't stopped.

The door opens. Right on schedule. Mike walks in, fist cocked.

But this time you've got a mouthguard in. Your stance is set. You've been training for this moment since the last time he walked through that door. He still hits you. It still hurts. But you don't go down. The planes never stopped.

You can't stop Mike from showing up. But you can stop getting knocked out.