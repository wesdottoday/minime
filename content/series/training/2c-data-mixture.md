---
title: "Data mixture & curation"
date: 2026-05-19
draft: false
node_id: "2c"
tier: 2
parent: "1a"
children:
  - "3c"
  - "3d"
siblings:
  - "2a"
  - "2b"
url: "/series/training/train-from-scratch/training-data/data-mixture/"
summary: "The optimizer updates the weights, but the data mixture decides what the gradients are usually about."
series: "training"
mermaid: true
weight: 202
tags:
  - llm-training
---

```mermaid
graph LR
    A[Training Data]:::hl --> B[Architecture] --> C[Hardware & Scale] --> D[Training Step] --> E[Training Loop] --> F[Post-Training] --> G[Evaluation]
    classDef hl fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
    classDef default fill:#1a1a2e,stroke:#16213e,color:#e0e0e0

    click A "/series/training/train-from-scratch/training-data/"
```

The optimizer updates the weights, but the data mixture decides what the gradients are usually about. This is probably the deepest non-obvious topic in all of training. Two models with identical architectures, identical optimizers, identical hardware — trained on different data mixtures — will have dramatically different capabilities. One will write solid code. The other will struggle with FizzBuzz. The difference isn't in the weights at initialization or the learning rate schedule. It's in what the model spent its trillion-token budget looking at.

Llama 3 was trained on approximately 15 trillion tokens. That's not 15 trillion tokens of "the internet." It's a carefully constructed blend of domains, filtered for quality, deduplicated at massive scale, with specific domains deliberately upsampled to shape the model's capabilities. The data mixture is where capability formation actually happens.

## Domain Mixture: The Ratio Matters More Than the Volume

A pretraining corpus isn't just "a lot of text." It's a blend of distinct domains, each contributing different capabilities to the final model:

| Domain | Typical Share | What It Teaches |
|--------|--------------|-----------------|
| Web crawl (CommonCrawl, etc.) | 50-70% | Broad world knowledge, general language patterns, factual associations |
| Code (GitHub, StackOverflow) | 10-20% | Logical reasoning, structured output, variable tracking, algorithmic thinking |
| Books | 5-10% | Long-range coherence, narrative structure, deep topical coverage |
| Scientific/academic text | 3-8% | Technical vocabulary, mathematical reasoning, citation patterns |
| Math (textbooks, problem sets) | 2-5% | Numerical reasoning, proof structure, symbolic manipulation |
| Multilingual text | 5-15% | Cross-lingual transfer, non-English capabilities |
| Conversational/forum data | 2-5% | Dialog patterns, Q&A structure, informal language |

These percentages are approximate and vary across labs, but the key insight is: **the ratio matters more than the volume.** If you train on 15 trillion tokens that are 95% web crawl and 1% code, you'll get a model that's fluent but can't code. If you shift to 80% web and 15% code, the model becomes dramatically better at programming — not because it saw more data overall, but because a larger fraction of its gradient updates came from code.

Meta's Llama 3 paper explicitly noted that they increased the proportion of code and math data relative to Llama 2. The result: significantly better performance on coding and mathematical benchmarks without a proportional loss in general language capability. The domain mixture is a capability allocation mechanism — every percentage point you give to one domain is a percentage point taken from another.

## Quality Filtering: Defining What "Good" Means

Raw web crawl is, to put it technically, garbage. A typical CommonCrawl dump contains spam, SEO-stuffed pages, auto-generated text, cookie banners, navigation menus, boilerplate legal text, porn, and machine-translated nonsense. Training on all of it wastes compute on gradients that teach the model to generate junk.

Quality filtering is how you separate the signal from the noise, and it operates at multiple levels:

**Heuristic filters** are the blunt instruments. They remove documents that are too short, have too many special characters, have abnormally high or low perplexity, contain excessive repetition, or match known spam patterns. These catch the obvious trash — a page that's 90% HTML tags or a document that repeats "buy cheap viagra" 400 times.

**Classifier-based filtering** is the sharp instrument. You train a text classifier — often a small fastText or BERT model — on examples of "high quality" text (Wikipedia, published books, curated reference material) versus "low quality" text (random web pages). Then you score every document in the corpus and keep only those above a threshold. This is how projects like Dolma and RedPajama filter their web crawl: a classifier trained to distinguish "Wikipedia-like" from "random web-like" text.

The problem is that "quality" is opinionated. Wikipedia is well-structured and factual, but it's also formal, encyclopedic, and stylistically narrow. Filtering too aggressively toward Wikipedia-like text can strip out valid conversational language, dialectal variation, creative writing, and informal technical discussion. The quality threshold is a design decision that shapes the model's voice — there's no objectively correct setting.

**Perplexity filtering** uses a pre-trained language model to score documents. High perplexity means the text is surprising to the scorer — which might mean it's garbled, in a foreign language the scorer doesn't know, or genuinely novel. Low perplexity means it's predictable — which might mean it's well-written or it might mean it's boilerplate. Perplexity filters are useful but crude; they can accidentally discard domain-specific technical writing that looks "surprising" to a general-domain scorer.

## Deduplication: The Internet Is Mostly Copies of Itself

This is not an exaggeration. Studies of CommonCrawl have found that after deduplication, the unique content is a fraction of the raw volume. The same news article gets syndicated across hundreds of sites. The same boilerplate privacy policy appears on millions of pages. The same Stack Overflow answer gets scraped into dozens of tutorial sites. Without deduplication, the model trains on the same text multiple times per epoch — wasting compute on repeated gradients and, worse, memorizing specific sequences rather than learning general patterns.

Deduplication happens at three levels:

**Exact-match deduplication** removes documents with identical content. Simple, fast, catches the obvious copies. This is typically done by hashing each document and removing duplicates. It catches copy-paste plagiarism and content syndication but misses paraphrases.

**Near-duplicate detection** catches documents that are 90%+ identical — the same article with a different byline, the same tutorial with minor formatting changes. The standard tool is **MinHash with Locality-Sensitive Hashing (LSH).** MinHash approximates the Jaccard similarity between documents by comparing sets of n-gram hashes. LSH makes this tractable at scale — you can compare billions of document pairs without doing all-pairs comparison. At 15 trillion tokens, even the approximate methods are computationally expensive: Llama 3's deduplication pipeline was itself a significant engineering effort.

**Substring/paragraph-level deduplication** removes repeated chunks within documents or across the corpus. A paragraph that appears in 10,000 documents is essentially boilerplate and contributes little training signal after the first few encounters. Some pipelines (like the one used for Dolma) detect and remove these repeated substrings even when the surrounding documents are otherwise unique.

The impact of deduplication on model quality is well-documented. Training on deduplicated data produces lower validation loss per token seen, reduces memorization of specific sequences, and improves generalization. It also reduces the effective dataset size — which matters when you're trying to decide how many epochs to run.

## Upsampling: Boosting Scarce but Valuable Domains

Not all domains have equal volume. The web produces petabytes of text annually. High-quality math textbooks produce megabytes. If you train proportional to available volume, the model sees math so rarely that it never develops strong mathematical reasoning.

**Upsampling** means showing the model certain domains more often than their natural proportion — effectively training multiple epochs on scarce, high-value data while training fewer epochs on abundant web data. Llama 3 upsampled code and math data significantly. The result: disproportionate capability gains in those domains.

This works because some domains have unusually high information density per token. A single page of well-written mathematical proof teaches the model more about formal reasoning than a hundred pages of web prose. By upsampling, you ensure these high-value tokens get proportionally more influence on the weights.

The risk of upsampling is overfitting. If you show the model the same 50GB of math textbooks 20 times, it may memorize specific problems rather than learning general mathematical reasoning. The practical solution is to combine upsampling with data augmentation (rephrasing, reformatting, synthetic extensions) and to monitor validation loss on held-out examples from the upsampled domain. When the validation loss plateaus or increases while training loss continues to drop, you've overfit.

## Curriculum Changes: Adjusting Mid-Flight

The data mixture doesn't have to be static. Some training runs adjust the mix during training — a strategy called **curriculum learning** or **annealing.**

A common pattern: train on the broad web-heavy mixture for the majority of training, then shift toward higher-quality, more domain-specific data in the final phase. Llama 3 used an annealing phase in the last portion of training where they increased the proportion of high-quality data. The intuition is that early training builds broad representations, and late training refines them — so the model benefits from seeing cleaner, more targeted data when its representations are mature enough to absorb the signal.

Curriculum changes can also respond to loss curves. If the training loss on code stops improving while the loss on web text is still dropping, you might increase the code proportion to give that domain more gradient signal. This is manual and expensive to tune — you can only observe the effect over thousands of steps, and you're making irreversible decisions about a multi-million-dollar run. But the gains can be significant: Meta reported that data mixture adjustments during Llama 3 training contributed measurably to final benchmark performance.

## Data Quantity vs. Quality: More Tokens Are Not Automatically Better

There's a tempting assumption that more data is always better. It's wrong in important ways.

Training on low-quality data doesn't just waste compute — it actively degrades the model. Noisy, contradictory, or duplicated data introduces conflicting gradient signals that slow convergence and reduce final model quality. A model trained on 10 trillion tokens of heavily filtered, deduplicated, high-quality text will outperform a model trained on 20 trillion tokens of unfiltered web crawl, even though it saw half as much data.

This is why data curation teams exist at every major AI lab, and why the data pipeline — crawling, filtering, deduplicating, classifying, mixing — is often as complex as the training infrastructure itself. The data pipeline for Llama 3 involved multiple stages of filtering, each with its own classifiers, thresholds, and validation. The curation effort was not an afterthought; it was a core part of the training investment.

## The Frontier Data Constraint

Here's the constraint that few people discuss publicly but that every frontier lab is hitting: **there are only about 5-10 trillion tokens of high-quality, unique, English-language text on the internet.** Once you've filtered, deduplicated, and quality-scored the global crawl, that's roughly what you're left with. Add multilingual data and you can stretch to maybe 15-20 trillion unique tokens across all languages.

Frontier models are already multi-epoch on this data. Llama 3's 15 trillion training tokens represent multiple passes over the available high-quality corpus. When you've seen all the good data twice, seeing it a third time gives diminishing returns. The gradient signal from the third epoch is weaker — the model has already absorbed most of what those tokens can teach.

This constraint is driving several responses across the industry: aggressive investment in [synthetic data](/series/training/train-from-scratch/training-data/data-mixture/synthetic-data/) generation (using strong models to create training data for future models), exploration of non-text modalities (images, video, audio) as sources of language-relevant training signal, and partnerships for access to proprietary text corpora (books, scientific journals, private databases) that aren't in the public crawl.

The data wall is real, and it's one of the binding constraints on continued scaling of language models.

### Performance Profile
- **Capability impact:** The data mixture is the single largest determinant of model capabilities after raw scale. Shifting 5% of tokens from web to code can move coding benchmarks by double-digit percentages.
- **Compute cost of curation:** The filtering, deduplication, and classification pipeline for a 15T-token corpus is itself a significant compute job — days of GPU time for classifier inference, weeks of CPU time for MinHash deduplication.
- **Quality vs. quantity tradeoff:** 10T high-quality tokens consistently outperform 20T unfiltered tokens. The curation effort pays for itself in faster convergence and better final quality.
- **Frontier constraint:** ~5-10T unique high-quality English tokens exist globally. Frontier models are multi-epoch on all of it. Synthetic data and multimodal training are the current responses to this ceiling.
- **Weak points:** Quality filtering is subjective and can introduce bias. Upsampling risks overfitting on scarce domains. Curriculum changes are expensive to tune and irreversible mid-run. Deduplication at scale is computationally expensive and imperfect.
