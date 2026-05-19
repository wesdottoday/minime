# LLM Training Series Review Feedback

This is a detailed review of `./content/series/training` focused on technical inaccuracies, internal inconsistencies, and major content gaps. The series is conceptually strong and readable, but several numerical examples mix abstraction levels: per GPU, per tensor-parallel rank, per pipeline stage, per data-parallel replica, per optimizer step, and per global batch are sometimes treated as interchangeable. That creates most of the serious issues below.

## Highest Priority Corrections

### 1. Global Batch Math Conflicts With Step-Level Token Counts

**Severity:** High

**Core issue:** The training loop says Llama 3 70B uses roughly 4 million tokens per optimizer step, which gives about 3.75 million optimizer updates across 15 trillion tokens. That is internally coherent:

- `content/series/training/1e-training-loop.md:43-45`
- `content/series/training/2u-batch-size.md:30`
- `content/series/training/2u-batch-size.md:57-58`

But several step-level pages say every one of 16,384 GPUs receives a micro-batch of 4 sequences x 8,192 tokens, giving:

`16,384 x 4 x 8,192 = 536,870,912 tokens per step`

That is about 537M tokens per step, not 4M. If that were true, 15T tokens would be only about 27,940 optimizer steps, not 3.75M.

**Affected files:**

- `content/series/training/2m-data-loading.md:37`
- `content/series/training/2m-data-loading.md:50`
- `content/series/training/2o-loss-calculation.md:40`
- `content/series/training/2l-why-scale.md:37`
- `content/series/training/2n-forward-pass.md:38`
- `content/series/training/3k-training-stability.md:34`
- `content/series/training/2p-backward-pass.md:58`

**Suggested fix:** Add a short definitions section before the phase pages that distinguishes:

- Global batch size: total tokens per optimizer update.
- Micro-batch: tokens processed per forward/backward pass per model-parallel group.
- Gradient accumulation: number of micro-batches accumulated before optimizer step.
- Data-parallel replica: one full logical model replica, often composed of multiple GPUs due to tensor and pipeline parallelism.
- Tensor-parallel rank: one shard of a layer, not an independent full batch worker.
- Pipeline stage: owns a subset of layers and passes activations forward/backward.

Then revise every “tokens per step” calculation to use the same global-batch assumption. If keeping 4M tokens/step, do not multiply 16,384 GPUs by 4 sequences each as though every GPU has an independent full model replica. The relevant unit is the model-parallel replica or data-parallel group, not raw GPU count.

### 2. Training Compute Exponent Is Wrong In One Page

**Severity:** High

**Core issue:** `6 x 70B x 15T` equals `6.3 x 10^24` FLOPs, not `6.3 x 10^23`.

**Affected files:**

- Incorrect: `content/series/training/2l-why-scale.md:29`
- Incorrect: `content/series/training/2l-why-scale.md:34`
- Correct elsewhere: `content/series/training/2v-scaling-laws.md:41`

**Why it matters:** This is a 10x error. It affects the single-rack runtime estimate, the comparison between one rack and 227 racks, and the credibility of the cost/timing narrative.

**Suggested fix:** Change `6.3 x 10^23` to `6.3 x 10^24` in `2l-why-scale.md`, then recompute the timing section. Also check whether the effective rack throughput assumption is consistently “effective throughput” or “theoretical throughput adjusted by MFU.” Right now that section may be double-counting or undercounting utilization depending on how the 90 PFLOP figure was derived.

### 3. MoE Active-Parameter Math Is Incorrect

**Severity:** High

**Core issue:** The MoE example says: take Llama 3's dense FFN, replace it with 8 experts of the same size, use top-2 routing, and therefore active FFN parameters are:

`2 x 56.4B / 8`

That is not correct for the architecture described. If each expert is a full dense FFN, top-2 routing activates two full FFN experts per token. Active FFN compute is about `2 x dense FFN`, not `25% of dense FFN`.

**Affected files:**

- `content/series/training/3e-moe.md:30-40`
- `content/series/training/3e-moe.md:59-64`

**Why it matters:** The page currently says a hypothetical 464B-total MoE has about 27.2B active parameters. Under the architecture as described, the active parameter count is closer to:

- Attention + embeddings/output: about 13B
- Active FFN: about `2 x 56.4B = 112.8B`
- Total active per token: about 126B

So the serving-cost claim is off by roughly 4-5x.

**Suggested fix:** Choose one of these approaches:

1. Keep the “8 experts, top-2, each expert same size as dense FFN” example, but revise active parameters upward and explain that it buys capacity, not lower-than-dense compute.
2. Change the example so each expert is smaller, matching common MoE designs where total active compute is deliberately kept near a dense baseline.
3. Use a real published MoE configuration, such as Mixtral or DeepSeek-V3, and explain total vs active parameters from that architecture instead of deriving from Llama 3's dense FFN dimensions.

### 4. Llama 3 vs Llama 3.1 Context Length Is Conflated

**Severity:** High

**Core issue:** The long-context page says “Llama 3 70B pretrains on 8,192-token sequences but supports 128K tokens at inference.” Official model cards list Llama 3 as 8k context and Llama 3.1 as 128k context.

**Affected files:**

- `content/series/training/3d-long-context.md:11`
- `content/series/training/3d-long-context.md:28`
- `content/series/training/3d-long-context.md:60-63`
- `content/series/training/2a-final-data-format.md:32`

**Sources:**

- Meta Llama 3 70B model card: https://huggingface.co/meta-llama/Meta-Llama-3-70B
- Meta Llama 3.1 70B model card: https://huggingface.co/meta-llama/Llama-3.1-70B

**Suggested fix:** Rephrase as either:

> Llama 3 70B was released with an 8K context window. Llama 3.1 extended the Llama 3 family to 128K using additional long-context training and RoPE/context-extension techniques.

Or make the article explicitly about “how Llama 3.1-style context extension works after an 8K pretraining phase.”

### 5. Token Storage Size Is Undercounted For A 128K Vocabulary

**Severity:** Medium-High

**Core issue:** A 128,000-token vocabulary cannot be represented in 16 bits, because 16-bit unsigned integers only cover 0-65,535. The text says datasets may be stored as 16-bit or 32-bit integers, then uses 2 bytes/token for raw input size.

**Affected files:**

- `content/series/training/2a-final-data-format.md:30`
- `content/series/training/2a-final-data-format.md:34`
- `content/series/training/2m-data-loading.md:37`
- `content/series/training/2l-why-scale.md:37`

**Suggested fix:** Use 32-bit token IDs for Llama 3's 128K vocab unless explicitly discussing compressed or packed formats. If keeping “2 bytes per token,” explain that it only works for vocabularies up to 65,536 or for a custom packed representation.

### 6. ZeRO And Gradient Sync Examples Ignore Tensor/Pipeline Parallel Sharding

**Severity:** Medium-High

**Core issue:** The parallelism page correctly introduces a plausible `TP=8, PP=4, DP=512` setup:

- `content/series/training/3b-parallelism-stack.md:43`

But gradient sync and ZeRO pages often describe each data-parallel replica as if it holds a complete 70B model and complete 140GB gradient tensor on a single GPU:

- `content/series/training/2q-gradient-sync.md:37-45`
- `content/series/training/3h-zero.md:29-49`
- `content/series/training/2r-optimizer-step.md:69`

In a 3D-parallel setup, each data-parallel replica is not one GPU. It is a group of GPUs collectively holding a full logical model. Gradient sync occurs across matching parameter shards/ranks across data-parallel replicas. The 140GB full-model gradient exists conceptually, but not as one full tensor on each GPU.

**Why it matters:** The current wording inflates per-GPU memory and communication numbers and makes ZeRO stages look different than they would in a tensor/pipeline-parallel implementation.

**Suggested fix:** Revise language from “each GPU/replica holds the full 140GB gradient tensor” to “each data-parallel replica collectively holds the full model, while each rank owns the shard corresponding to its tensor/pipeline-parallel position.” Then describe all-reduce or reduce-scatter across equivalent ranks in the DP dimension.

### 7. Cost Assumptions Conflict Across Pages

**Severity:** Medium

**Core issue:** Some pages use `$50-100 per GPU-hour`; another uses `$2-3 per GPU-hour`.

**Affected files:**

- `content/series/training/2z-validation-benchmarks.md:41`
- `content/series/training/2s-checkpointing.md:69`
- `content/series/training/2v-scaling-laws.md:122`

**Suggested fix:** Pick one cost basis, or explicitly separate:

- Cloud on-demand H100/B200 price.
- Reserved/committed cloud price.
- Internal amortized fleet cost.
- Fully loaded cluster cost including power, networking, storage, labor, and idle time.

The current mix makes dollar estimates hard to trust.

## Additional Accuracy Notes

### 8. “Training Loss Always Decreases” Is Too Strong

**Severity:** Medium

**Affected file:** `content/series/training/2z-validation-benchmarks.md:35`

**Issue:** The statement “Training loss always decreases (with noise) as long as the learning rate is nonzero and the model has capacity” is too absolute. Training loss can plateau, spike, rise during curriculum changes, react to learning-rate schedule changes, or worsen due to instability or distribution shifts.

**Suggested fix:** Replace with something like:

> Training loss is expected to trend downward over a stable training run, though it can be noisy and can rise temporarily during learning-rate changes, data-mixture changes, instability events, or curriculum shifts.

### 9. DPO Loss Formula Is Too Compressed And Slightly Ambiguous

**Severity:** Medium

**Affected file:** `content/series/training/2x-preference-training.md:99-105`

**Issue:** The DPO formula uses `log π(chosen)/π_ref(chosen)`, which can be read as `log π(chosen)` divided by `π_ref(chosen)`. The intended DPO term is the log probability ratio:

`log π(y_w | x) - log π_ref(y_w | x) - log π(y_l | x) + log π_ref(y_l | x)`

Also, the explanation says DPO applies stronger signal when the reference model was ambivalent. That is directionally okay, but incomplete: the signal depends on the policy-reference log-ratio margin for chosen vs rejected responses, not only the reference model's preference.

**Suggested fix:** Write the formula as log-probability differences, not a slash ratio.

### 10. SFT Safety Limit Is Overstated

**Severity:** Medium

**Affected file:** `content/series/training/2w-sft.md:80`

**Issue:** The statement that robust safety “requires preference training with negative examples” is directionally common, but too categorical. Safety behavior can come from SFT, preference optimization, classifiers, constitutional/RLAIF data, system policies, refusal datasets, red-teaming, tool gating, and runtime enforcement.

**Suggested fix:** Rephrase to:

> Robust safety usually requires more than SFT alone: preference data, adversarial examples, safety-specific evaluation, and often runtime policy layers.

### 11. FP16 Loss Scaling Explanation Is Backwards In One Spot

**Severity:** Medium

**Affected file:** `content/series/training/3i-mixed-precision.md:43`

**Issue:** It says loss scaling multiplies the loss so gradients are larger, helping keep gradient values in FP16's representable range. The major reason is to prevent underflow of small gradients, not overflow of large gradients. Dynamic loss scaling then backs off if overflow occurs.

**Suggested fix:** Explain:

> FP16 has limited dynamic range and small gradients can underflow to zero. Loss scaling multiplies the loss before backprop so gradients are shifted into representable range, then unscales before the optimizer step. If the scale is too high, gradients overflow and the step is skipped/retried with a lower scale.

### 12. FP32 Master Weights May Be Too Absolute For Modern BF16 Training

**Severity:** Medium

**Affected files:**

- `content/series/training/3i-mixed-precision.md:57-61`
- `content/series/training/3i-mixed-precision.md:93`
- `content/series/training/2r-optimizer-step.md:67`

**Issue:** The page says FP32 master weights are non-negotiable. This is a classic mixed-precision recipe, but modern BF16 training implementations vary. Some keep FP32 master parameters; others use BF16 parameters with FP32 optimizer states and carefully designed update paths, depending on framework and optimizer.

**Suggested fix:** Make it less absolute:

> Many mixed-precision recipes keep FP32 master weights to avoid small-update stagnation. Some modern BF16 stacks use alternative update strategies, but the general issue remains: tiny updates must accumulate at higher precision somewhere.

### 13. “RMSNorm Is 10-15% Faster” Needs Qualification

**Severity:** Low-Medium

**Affected file:** `content/series/training/2j-rmsnorm-rope-silu.md:29`

**Issue:** RMSNorm can be faster than LayerNorm, but the exact speedup depends on kernel fusion, hardware, tensor shape, and memory bandwidth. “10-15% faster” may sound like end-to-end model speedup when it is more likely operation-level.

**Suggested fix:** Clarify that the speedup is for the normalization operation, not total training throughput.

### 14. FFN “Stores Factual Knowledge” Is Too Strong

**Severity:** Low-Medium

**Affected file:** `content/series/training/3a-ffn-parameters.md:51`

**Issue:** The FFN/MLP layers are often implicated in factual associations and memory-like behavior, but saying factual knowledge “lives in FFN weights” is too clean. Knowledge is distributed across embeddings, attention, MLPs, layer interactions, and representations.

**Suggested fix:** Rephrase to:

> Much factual association appears to be mediated by FFN/MLP weights, but knowledge is distributed across the network rather than stored in a single component.

### 15. Vocabulary Parallelism Alignment Claim Is Overconfident

**Severity:** Low-Medium

**Affected file:** `content/series/training/2e-vocabulary-size.md:44`

**Issue:** `128,000 = 128 x 1,000`, and it divides by 64, 128, and 256, but “vocab size should be divisible by common GPU parallelism factors” is more of an implementation convenience than a primary design driver. Many systems pad vocabulary/logit dimensions internally.

**Suggested fix:** Keep it as a minor systems note, not “the constraint nobody talks about.”

## Major Content Gaps

### A. Missing Distributed Training Units Page

This is the biggest gap. Add an article before the training-step phase pages explaining the units used throughout the rest of the series:

- GPU
- Node
- Rack
- Tensor-parallel rank
- Tensor-parallel group
- Pipeline stage
- Data-parallel replica
- Global batch
- Micro-batch
- Gradient accumulation step
- Optimizer step
- Token count per optimizer step

This page should include one consistent worked example and all later math should reference it.

### B. Missing Context Parallelism / Sequence Parallelism

The series covers tensor, pipeline, and data parallelism, but not context parallelism or sequence parallelism. For long context and large-batch training, these matter. The long-context page especially would benefit from a section on splitting long sequences across devices.

Suggested locations:

- Add as child of `2n-forward-pass.md`
- Cross-link from `3d-long-context.md`

### C. Data Governance Is Too Thin

The data pages mention copyright and legal ambiguity, but a production training data pipeline also needs:

- PII detection/removal
- license/provenance tracking
- opt-out handling
- malware/code safety filtering
- adult/violent/toxic content policy
- benchmark and eval-set quarantine
- source-level audit trails
- privacy-sensitive data handling

Suggested locations:

- Expand `1a-training-data.md`
- Add a child page under `2c-data-mixture.md`
- Cross-link with `2aa-contamination.md`

### D. Evaluation Could Use A “Configuration Matters” Warning

Benchmarks are listed, but the series should explicitly call out that results vary by:

- prompt template
- few-shot vs zero-shot
- chain-of-thought allowed or not
- pass@1 vs pass@k
- sampling temperature
- exact match vs model-judged scoring
- contamination/decontamination settings
- base model vs instruct model

Suggested location:

- `content/series/training/2z-validation-benchmarks.md`

### E. Optimizer Coverage Is AdamW-Centric

AdamW is a good default explanation, but frontier training discussions often mention:

- AdamW variants
- Adafactor-style memory savings
- Lion and other alternatives
- fused optimizers
- optimizer-state sharding
- learning-rate transfer from smaller-scale experiments

Suggested location:

- Add a short “AdamW is the default mental model, but not the whole space” section in `2r-optimizer-step.md`.

### F. Missing Sources / Citations For Specific Public Claims

The series makes several specific claims about Meta/Llama 3: training tokens, GPU count, failures per day, annealing, DPO variant, GQA quality preservation, checkpoint cadence, data mixture adjustments. Those should be linked to the Llama 3 paper/model cards or softened to “public reports suggest.”

Suggested approach:

- Add source links at the bottom of each page or inline footnotes.
- Use official Meta/Hugging Face pages where possible.
- Use “reportedly” only when not directly sourced.

## Suggested Rewrite Order

1. Add the distributed-training-units explainer.
2. Fix global batch/token-count math across `2m`, `2n`, `2o`, `2p`, `2q`, `2u`, `2l`, and `3k`.
3. Fix the FLOP exponent and recompute timing/cost language.
4. Fix the MoE example or replace it with a real architecture.
5. Split Llama 3 vs Llama 3.1 long-context claims.
6. Normalize cost assumptions.
7. Add citation links for Llama-specific claims.
8. Patch the medium-priority wording issues.

## Summary

The series works well as an explanatory arc. The main problem is not conceptual structure; it is numerical consistency. Once the distributed-training units are defined and reused consistently, most of the high-severity issues become straightforward edits. The second major repair is source precision: distinguish Llama 3 from Llama 3.1, distinguish public fact from plausible industry practice, and cite the official model cards or papers for concrete claims.
