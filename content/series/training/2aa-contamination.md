---
title: "Contamination & evaluation integrity"
date: 2026-05-19
draft: false
node_id: "2aa"
tier: 2
parent: "1g"
children: []
siblings:
  - "2z"
url: "/series/training/train-from-scratch/evaluation/contamination/"
summary: "If benchmark examples are in the training data, the model appears more capable than it really is. This is not a theoretical concern — it's an active problem."
series: "training"
mermaid: true
weight: 261
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data] --> B[Architecture] --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]:::hl
    classDef hl fill:#b45309,stroke:#92400e,color:#fff
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click G "/series/training/train-from-scratch/evaluation/"
```

If benchmark test questions appear in the training data, the model can recall answers rather than reason its way to them. It scores higher than its actual capability warrants. This is contamination, and it's not a theoretical concern — it's an active, persistent problem that affects every large-scale training run and every benchmark score you've ever seen reported.

The fundamental issue is simple: you cannot fairly test someone on questions they've already seen. But when your training corpus is 15 trillion tokens scraped from the entire internet, and your test set is a few thousand questions that have been publicly available for years, keeping the two apart is harder than it sounds.

## What contamination looks like

Contamination exists on a spectrum. At the obvious end: a document in the training corpus contains the exact text of a benchmark question and its correct answer. The model memorizes the pair. When evaluated on that benchmark, it produces the correct answer not because it reasoned through the problem but because it saw this specific question during training.

At the subtler end: a document discusses the topic of a benchmark question, walks through similar reasoning, or presents a paraphrase of the problem with a worked solution. The model hasn't memorized the exact question, but it's been exposed to the specific reasoning chain the benchmark tests. Its advantage is less dramatic but still real — it has rehearsed this type of problem in a way that wouldn't have happened without the contaminated data.

Concrete examples of how benchmark data ends up in training corpora:

- **Direct inclusion.** Many benchmark datasets are publicly available on GitHub, HuggingFace, and academic websites. Common Crawl indexes these sites. The benchmark questions and answers appear verbatim in the crawled text.
- **Academic papers.** Papers introducing or analyzing benchmarks frequently include example questions with solutions. Papers discussing MMLU might reproduce 10-20 questions to illustrate the benchmark's difficulty. These papers are crawled and included in the training data.
- **Forums and Q&A sites.** Stack Overflow, Reddit, Quora, and other forums contain discussions where users post benchmark questions and community members provide solutions. These conversations are high-quality text that passes quality filters easily.
- **Study guides and tutoring materials.** Websites designed to help students prepare for standardized tests often include problems that overlap with or are derived from evaluation benchmarks. GSM8K-style math problems appear in countless tutoring resources.
- **Other models' outputs.** As LLMs become more prevalent, web text increasingly contains LLM-generated content, including responses to benchmark questions. Training on this text creates a feedback loop where benchmark knowledge propagates across model generations.

## Why decontamination is hard at scale

A Llama 3-class training run processes roughly 15 trillion tokens across billions of documents. Decontamination means scanning this entire corpus against every benchmark the model will be evaluated on and removing matches. The computational cost alone is substantial — but the real difficulty is defining what constitutes a "match."

**Exact-match decontamination** is the simplest approach: for every document in the training corpus, check whether it contains a verbatim copy of any benchmark question or answer. This catches the most obvious contamination but misses everything else. A question reworded from "What is the capital of France?" to "Which city serves as France's capital?" passes exact-match decontamination completely.

**N-gram overlap detection** is more aggressive. For each benchmark item, extract all n-grams (subsequences of n words) and flag training documents that share a high fraction of n-grams with any benchmark item. With n=8 or n=13, this catches many paraphrases. But it also produces false positives: common phrases, standard mathematical notation, and frequently used code patterns all generate high n-gram overlap without representing actual contamination. At 15T tokens, even a low false-positive rate means flagging millions of documents for manual review — which is impractical.

**Semantic similarity detection** uses embeddings to find documents that are "close" to benchmark items in meaning, even if the surface text is different. This catches more true contamination but requires embedding every training document and every benchmark item, computing similarity scores, and setting a threshold. The threshold is the hard part: too aggressive and you remove documents that merely discuss the same topic as a benchmark question (removing all discussion of French geography because MMLU has a question about Paris). Too lenient and paraphrased contamination slips through.

No decontamination method is complete. Each catches a different subset of contaminated documents, and none catches all of them. Practical decontamination pipelines combine multiple methods — exact match, n-gram overlap, and sometimes semantic similarity — and accept that some contamination will survive. The goal is to reduce contamination to the level where it doesn't meaningfully inflate benchmark scores, not to eliminate it entirely.

## The contamination spectrum

Not all overlap between training data and evaluation data is equally problematic. It helps to think about contamination as a spectrum:

**Exact reproduction** — The benchmark question and answer appear verbatim in the training data. This is clearly contamination and easy to detect. The model's correct answer on this question tells you nothing about its reasoning ability.

**Close paraphrase** — The same question is stated differently, or the same solution is presented with different variable names or slightly reworded steps. This is probably contamination and hard to detect reliably. The model hasn't memorized the exact text but has been exposed to the specific problem.

**Topic overlap** — The training data contains extended discussion of the topic a benchmark question tests, including similar (but not identical) problems and reasoning patterns. This is hard to detect and may or may not be contamination. A model that's read thousands of documents about French geography "should" know the capital of France — is that contamination or just learning?

**General domain knowledge** — The training data contains broad coverage of a subject area that a benchmark tests. This is not contamination — it's the entire point of training. A model that's read millions of pages of code and can solve HumanEval problems is demonstrating genuine capability, not recall.

The difficulty is that the boundaries between these categories are blurry, and the same benchmark score could reflect any mix of them. A model scoring 85% on MMLU might be 80% genuine knowledge and 5% contamination-driven recall, or it might be 70% genuine and 15% contamination. From the outside, the scores look the same.

## Benchmark saturation

When models consistently score 90%+ on a benchmark, the benchmark stops being informative. It can no longer discriminate between models of different capability — a model scoring 92% and a model scoring 94% might differ by noise, evaluation configuration, or contamination rather than genuine capability.

This drives a cycle: researchers create a challenging benchmark. Models improve and scores rise. The benchmark saturates. Researchers create a harder benchmark. The new benchmark's test data is published (or leaked, or discussed). Models train on data that overlaps with it. Scores rise. The benchmark saturates. Repeat.

MMLU was challenging when introduced. By 2025, frontier models score above 85%, and the benchmark primarily discriminates between models in a narrow band. MMLU-Pro, GPQA, and other harder benchmarks have been introduced, but they're on the same trajectory. The benchmark lifecycle — from informative to saturated — is measured in months to a few years.

The arms race has no stable equilibrium. As long as benchmarks are public and training data is drawn from the public internet, contamination pressure is continuous. Even benchmarks designed to be contamination-resistant (held-out test sets, dynamically generated problems, private evaluation sets) eventually face pressure as their formats, topics, and difficulty distributions become known.

## Evaluation set security

Keeping test data out of training pipelines requires operational discipline, not just algorithmic decontamination.

**Data handling separation.** The evaluation team and the training data team should have separate data pipelines. Evaluation sets should not be stored in the same repositories, filesystems, or databases as training data. Access controls should prevent evaluation data from being accidentally included in a data processing pipeline.

**Audit trails.** Every document in the training corpus should be traceable to its source. If a contamination concern arises, the team needs to be able to determine whether and when a specific document was included in training data. This requires logging and metadata that many training pipelines don't maintain at the document level.

**Temporal separation.** For benchmarks created after the training data cutoff, contamination is less likely (though not impossible — benchmark data can be retroactively added to web crawls if the crawl index is updated). For benchmarks that predate the training data, contamination is almost certain to exist to some degree.

**Canary strings.** Some evaluation sets include unique strings that should never appear in training data. If a model can reproduce these canary strings, it's strong evidence of contamination. This is a detection mechanism, not a prevention mechanism — by the time you find the canary, the model is already contaminated.

**Private evaluation sets.** The most robust approach is to evaluate models on test data that has never been public. Several organizations maintain private evaluation sets that are administered under controlled conditions. This eliminates web-crawl contamination but introduces other problems: private benchmarks can't be independently reproduced, verified, or critiqued by the community.

## Goodhart's Law and the measurement trap

The contamination problem is a specific instance of a deeper issue: when a measure becomes a target, it ceases to be a good measure. Benchmarks were designed to measure model capability. When benchmark scores became the primary metric for comparing models, building investor confidence, and marketing releases, they became optimization targets. Labs don't just evaluate on benchmarks — they select data, tune training mixtures, choose checkpoints, and adjust post-training specifically to improve benchmark numbers.

This doesn't require intentional contamination. A lab that notices its model scores poorly on math benchmarks might upweight mathematical text in the training mixture. The resulting model is genuinely better at math — but it's also been specifically shaped to perform well on the kind of problems math benchmarks test. The benchmark score reflects both genuine capability improvement and targeted optimization. Separating the two is difficult.

The result is a benchmarking ecosystem where scores are informative but not trustworthy. A model scoring 85% on a benchmark is probably better than one scoring 60%, but the difference between 85% and 88% might reflect contamination, benchmark-specific optimization, or genuine capability — and you can't tell which from the score alone.

## What contamination means for the reader

If you're evaluating models using public benchmark scores, know that those scores are upper bounds on the model's true capability. The real capability is somewhere below the reported number, but by how much is unknown and varies by benchmark, by model, and by training methodology.

The most reliable evaluation signals combine multiple sources: automated benchmarks (many of them, not just one), human preference evaluation (expensive but harder to game), held-out evaluation sets (private is better), and real-world task performance (the ultimate test). Any single benchmark, no matter how well-designed, is an incomplete and gameable measure.

### Performance Profile
- **Decontamination compute cost:** N-gram overlap detection across 15T tokens against thousands of benchmark items requires significant compute — typically a few hundred GPU-hours for a full scan. This is run once per training corpus version, not per training run.
- **False positive rate:** At n=8 overlap threshold, 0.1-1% of training documents may be flagged as potentially contaminated. Manual review of even 0.1% of billions of documents is impractical, so most pipelines use conservative thresholds that accept some false negatives.
- **Contamination impact on scores:** Studies have estimated 1-5% score inflation on heavily contaminated benchmarks. The effect varies widely by benchmark format — multiple-choice benchmarks (MMLU) are more affected than open-generation benchmarks (HumanEval) because the answer space is constrained.
- **Benchmark lifecycle:** A new benchmark is typically informative for 1-3 years before saturation and contamination pressure reduce its discriminative value. The community creates ~5-10 new major benchmarks per year, roughly matching the rate of saturation.
- **Private evaluation overhead:** Maintaining private evaluation sets requires dedicated infrastructure, access controls, and operational discipline. The marginal cost is low, but the organizational overhead is significant — especially when evaluation teams span multiple organizations or are subject to competitive pressure to publish results.
