---
title: "Preference training (RLHF, DPO)"
date: 2026-05-19
draft: false
node_id: "2x"
tier: 2
parent: "1f"
children: []
siblings:
  - "2w"
  - "2y"
url: "/series/training/train-from-scratch/post-training/preference-training/"
summary: "SFT says 'here's what a good response looks like.' Preference training says 'this response is better than that one — learn why.'"
series: "training"
mermaid: true
weight: 251
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

SFT teaches the model what a good response looks like. But "good" is a low bar. Given a complex question, the model might produce five different responses that are all grammatically correct, factually accurate, and properly formatted — yet a human reading all five would have a clear preference. One is more concise. One addresses the actual intent behind the question. One avoids an unnecessary caveat that makes the others feel evasive. SFT has no mechanism to learn these distinctions because it trains on individual demonstrations, not comparisons.

Preference training closes this gap. Instead of showing the model good responses and saying "be like this," you show it *pairs* of responses and say "this one is better than that one — learn why."

## The Gap SFT Leaves

After SFT, a model is a functional assistant. But functional is not the same as good. Consider two responses to "Explain why the sky is blue":

**Response A:** "The sky is blue due to Rayleigh scattering. Shorter wavelengths of light (blue and violet) are scattered more by atmospheric molecules than longer wavelengths (red, orange). Since our eyes are more sensitive to blue light and violet is absorbed higher in the atmosphere, we perceive the sky as blue."

**Response B:** "Great question! The sky appears blue because of a phenomenon known as Rayleigh scattering, which was first described by Lord Rayleigh in the 19th century. Light from the sun is made up of many different wavelengths, and when these wavelengths interact with the molecules in Earth's atmosphere (primarily nitrogen and oxygen), the shorter wavelengths — specifically blue and violet — are scattered in all directions much more than the longer wavelengths like red and orange. Now, you might wonder why the sky doesn't appear violet, since violet light has an even shorter wavelength. There are actually two reasons for this..."

Both are factually correct. Both demonstrate the instruction-following behavior SFT taught. But most humans prefer Response A — it's direct, complete, and doesn't pad with rhetorical filler. SFT can't teach this preference because both responses are valid demonstrations of "answer questions about physics." Preference training can, because it learns from the comparison itself.

## Reward Modeling: Teaching a Model to Judge

The first approach to preference training uses a **reward model** — a separate neural network trained to predict which of two responses a human would prefer.

The training data is a dataset of **preference pairs:** for a given prompt, two model-generated responses are shown to a human annotator who selects the better one. The annotator doesn't write anything — they just choose. Datasets typically contain 50,000-500,000 preference pairs, collected from teams of trained annotators following detailed rubrics.

| Prompt | Chosen Response | Rejected Response |
|--------|----------------|-------------------|
| "Explain recursion" | [Concise, clear explanation with a code example] | [Lengthy, meandering explanation without examples] |
| "Write a haiku about rain" | [Follows 5-7-5 syllable structure] | [Interesting imagery but wrong syllable count] |
| "Should I invest in crypto?" | [Balanced discussion of risks and potential] | [Enthusiastic recommendation without caveats] |

The reward model is typically initialized from the SFT model itself (same architecture, same pretrained representations) and then trained with a different head — instead of predicting the next token, it outputs a scalar reward score for a complete response. The training objective is: for each preference pair, the reward model should assign a higher score to the chosen response than the rejected one. The loss function is binary cross-entropy on the comparison:

`Loss = -log(σ(r(chosen) - r(rejected)))`

where `r(x)` is the reward model's score for response `x` and `σ` is the sigmoid function. This loss pushes the reward model to assign higher scores to responses humans preferred.

Once trained, the reward model encodes a compressed representation of human preferences. It can score any new response without needing a human to evaluate it. This is the key — it turns expensive, slow human judgment into fast, cheap model inference.

## RLHF / PPO: The Original Recipe

Reinforcement Learning from Human Feedback (RLHF), popularized by OpenAI's InstructGPT paper, uses the reward model as a signal to optimize the language model (the "policy") via reinforcement learning.

The setup has four components:

1. **The policy model** — the language model being optimized (starts from the SFT checkpoint)
2. **The reward model** — the trained judge that scores responses
3. **The reference model** — a frozen copy of the SFT checkpoint, used as an anchor
4. **The PPO optimizer** — the reinforcement learning algorithm that updates the policy

The training loop:
1. Sample a batch of prompts from the training set
2. The policy model generates responses to each prompt
3. The reward model scores each response
4. PPO updates the policy to increase the probability of generating high-reward responses

This is reinforcement learning in the classic sense: the model takes actions (generates tokens), receives a reward signal (the reward model's score), and updates its policy to maximize expected reward. PPO (Proximal Policy Optimization) is the specific RL algorithm used — it's relatively stable and well-understood, though applying it to language models required significant engineering effort.

**The KL penalty is the critical safety mechanism.** Without it, the policy model will find and exploit whatever patterns the reward model scores highly — even if those patterns are degenerate. The reward model is imperfect. It might give high scores to verbose responses (because annotators sometimes confuse length with quality), or to responses with confident-sounding hedging phrases, or to responses that repeat the question back before answering. Given unlimited optimization pressure, the policy model will discover these exploits and produce responses that score well but are actually terrible. This is **reward hacking.**

The KL penalty constrains the policy model's distance from the reference model. For each response, the optimizer computes the KL divergence between the policy's token distribution and the reference model's token distribution. This divergence is subtracted from the reward:

`Adjusted reward = Reward model score - β × KL(policy || reference)`

where β controls the strength of the constraint. The effect: the model can only improve its reward score to the extent that it stays "close" to the SFT baseline. Small, targeted improvements in response quality are rewarded. Radical departure from the reference model's behavior is penalized. This prevents the worst reward hacking while still allowing meaningful optimization.

The choice of β is a tuning problem. Too small and the model drifts too far, gaming the reward model. Too large and the model barely changes from the SFT baseline, wasting the entire preference training effort. In practice, β is set so that the KL divergence stays in a range of roughly 5-15 nats — enough to meaningfully change behavior without destabilizing it.

## DPO: Cutting Out the Middle Model

Direct Preference Optimization (DPO), introduced in 2023, asks a simple question: do we actually need the reward model?

The insight: the reward modeling objective and the RL objective can be collapsed into a single loss function that operates directly on preference pairs. Instead of training a reward model to score responses, then using RL to optimize against those scores, DPO optimizes the policy model directly from the preference data.

The DPO loss:

`Loss = -log(σ(β × ((log π(y_w | x) - log π_ref(y_w | x)) - (log π(y_l | x) - log π_ref(y_l | x)))))`

where `π` is the policy model, `π_ref` is the reference model, `y_w` is the chosen (winning) response, `y_l` is the rejected (losing) response, `x` is the prompt, and β controls the strength of the implicit KL constraint. The DPO loss compares the log-probability margin between chosen and rejected responses under the current policy versus the reference model. The intuition is clean: **increase the relative probability of chosen responses over rejected responses, but only in proportion to how much the reference model already distinguished them.**

If the reference model already strongly preferred the chosen response, DPO applies less gradient signal (the model already has it right). If the reference model was ambivalent between chosen and rejected, DPO applies stronger signal (this is where the model most needs to learn).

DPO's advantages are practical:

- **No reward model to train.** One fewer model to build, validate, and host. This eliminates an entire stage of the pipeline and its associated compute and hyperparameter tuning.
- **No RL loop.** PPO is notoriously unstable and sensitive to hyperparameters. DPO is standard supervised learning — compute a loss, compute gradients, update weights. The training infrastructure for DPO is the same as for SFT.
- **Fewer moving parts.** RLHF requires coordinating four models (policy, reward, reference, value function) during training. DPO requires two (policy and reference).

DPO has become increasingly popular since its introduction. Meta used a DPO variant for Llama 3 post-training. Many open-source models use DPO exclusively for preference training. The quality is competitive with RLHF on most benchmarks, though some researchers argue that RLHF with a well-trained reward model still holds an edge on the hardest alignment tasks.

## Constitutional AI / RLAIF: Scaling Preferences with AI

Human preference annotation is expensive and slow. A team of 50 annotators might produce 1,000 labeled preference pairs per day. For a large-scale preference training run requiring 500,000 pairs, that's months of annotation work.

**Constitutional AI (CAI)**, developed by Anthropic, and the broader category of **RLAIF** (Reinforcement Learning from AI Feedback) replace human annotators with AI-generated judgments. The approach:

1. Write a set of principles ("constitution") that define desired behavior: be helpful, be honest, don't assist with harmful tasks, acknowledge uncertainty, etc.
2. For each prompt, generate two candidate responses from the model
3. Ask a strong AI model (often the same model or a stronger one) to evaluate both responses against the principles and select the better one
4. Use these AI-generated preferences in place of human-generated ones for reward modeling or DPO

The advantage is scale. An AI critic can produce tens of thousands of preference judgments per hour at minimal cost. This allows much larger preference datasets and more iterations of preference training.

The disadvantage is that the AI critic inherits the biases and blind spots of whatever model generates the judgments. If the critic model has a verbosity bias, it will select verbose responses as "better," and the policy model will learn to be verbose. If the critic model's safety judgments are miscalibrated, those miscalibrations propagate into the trained model. Constitutional AI scales preference data generation but doesn't solve the fundamental problem of defining what "better" means — it just moves the problem from annotation guidelines to constitutional principles.

In practice, most frontier labs use a hybrid approach: AI-generated preferences for the bulk of the training data, with human-annotated preferences for the highest-stakes categories (safety-critical scenarios, edge cases, subjective quality judgments where AI critics are least reliable).

## RL with Verifiable Rewards: Clean Signal for Math and Code

For math and code, there's a shortcut that avoids both human annotators and AI critics: **check whether the answer is correct.**

Given a math problem with a known answer, you can automatically verify whether the model's response produces the right result. Given a coding problem with test cases, you can run the model's code and check whether the tests pass. These verifiable rewards provide a clean, unambiguous training signal — no annotation bias, no reward model approximation, no constitutional principles required.

DeepSeek-R1 and similar reasoning-focused models have used RL with verifiable rewards extensively. The training loop:
1. Present the model with a math problem or coding task
2. The model generates a solution (possibly with intermediate reasoning steps)
3. Check the final answer against the ground truth (or run the code against test cases)
4. Correct solutions get reward +1, incorrect solutions get reward -1 (or 0)
5. Use RL (PPO or similar) to optimize toward correct solutions

This approach is powerful for two reasons. First, the reward signal is *exact.* There's no gap between the reward and the actual objective — a correct answer is correct, period. Second, it naturally encourages the model to develop whatever intermediate reasoning is useful for reaching correct answers. The model learns chain-of-thought reasoning not because it was demonstrated (as in SFT) but because it's the strategy that maximizes reward.

The limitation is scope: verifiable rewards only work for tasks where correctness can be automatically checked. For open-ended writing, conversational quality, or nuanced reasoning about ambiguous situations, you can't write a unit test. These domains still require human or AI preference judgments.

## What Happens Without the KL Penalty

The KL penalty deserves emphasis because it's the difference between a well-aligned model and a degenerate one.

Without the KL constraint, optimization against a reward model (or any fixed reward signal) produces **mode collapse.** The model discovers a narrow set of response patterns that score well and produces them regardless of the prompt. This might look like: every response starts with "That's a great question!" followed by a numbered list, followed by a disclaimer. These responses score well because the reward model associates these patterns with the annotator-preferred responses it was trained on. But they're formulaic and unhelpful.

**Reward hacking** is the more insidious failure mode. The model finds specific tokens, phrases, or structures that exploit imperfections in the reward model. If the reward model slightly overweights confident language, the model becomes maximally confident about everything — including wrong answers. If the reward model slightly prefers longer responses, the model pads every response with tangential elaboration. These exploits are adversarial in the game-theoretic sense: the policy model is effectively adversarially attacking the reward model.

The KL penalty prevents both by saying: you can improve on the SFT baseline, but you can't become a completely different model. The improvement must be targeted, gradual, and grounded in the reference model's original behavior distribution. In practice, this means the preference-trained model produces responses that look like slightly better versions of what the SFT model would have said — not radically different responses optimized for a proxy metric.

### Performance Profile
- **Data scale:** 50K-500K preference pairs for reward modeling or DPO. Significantly more than SFT but still tiny relative to pretraining.
- **Compute cost:** RLHF requires maintaining 4 models simultaneously (policy, reward, reference, value function), making it 3-4x more memory-intensive than SFT. DPO requires 2 models and has compute costs comparable to SFT.
- **Behavioral impact:** Moves the model from "acceptable responses" to "preferred responses." Reduces verbosity, improves directness, teaches safety refusals, and aligns subtle quality preferences that demonstrations alone cannot capture.
- **KL constraint:** Typical KL divergence budget of 5-15 nats from the reference model. Too tight and the model barely changes. Too loose and it drifts into reward hacking.
- **Weak points:** Reward model quality is the ceiling — the policy can only be as well-aligned as the reward signal allows. Human annotator disagreement introduces noise in preference data. DPO is simpler but may be less effective on the hardest alignment problems. Constitutional AI scales preference generation but propagates critic model biases. All preference methods are sensitive to the distribution of prompts in the training set.

### Sources
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT / RLHF
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
