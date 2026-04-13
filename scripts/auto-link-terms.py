#!/usr/bin/env python3
"""Auto-link first occurrence of key terms in LLMs content to their node pages."""

import os
import re
import glob

CONTENT_DIR = os.path.expanduser("~/Projects/minime/content/llms")

# Term -> (URL, case_sensitive)
# Order matters: longer/more specific phrases first to avoid partial matches.
# Each term is linked only on its FIRST occurrence per page, and never on its own page.
TERM_MAP = [
    # Multi-word phrases first (most specific)
    ("sparse attention", "/llms/what-happens/prefill-decode/kv-cache/sparse-attention/"),
    ("paged attention", "/llms/what-happens/prefill-decode/kv-cache/paged-attention/"),
    ("KV cache offloading", "/llms/what-happens/prefill-decode/kv-cache/kv-cache-offloading/"),
    ("KV cache", "/llms/what-happens/prefill-decode/kv-cache/"),
    ("multi-head attention", "/llms/what-happens/embeddings/model-layers/attention-deep-dive/multi-head-attention/"),
    ("multi-head", "/llms/what-happens/embeddings/model-layers/attention-deep-dive/multi-head-attention/"),
    ("attention heads", "/llms/what-happens/embeddings/model-layers/attention-deep-dive/multi-head-attention/"),
    ("attention head", "/llms/what-happens/embeddings/model-layers/attention-deep-dive/multi-head-attention/"),
    ("grouped-query attention", "/llms/what-happens/prefill-decode/kv-cache/mqa-gqa/"),
    ("multi-query attention", "/llms/what-happens/prefill-decode/kv-cache/mqa-gqa/"),
    ("dot product", "/llms/what-happens/vectors/dot-product/"),
    ("dot products", "/llms/what-happens/vectors/dot-product/"),
    ("dot-product", "/llms/what-happens/vectors/dot-product/"),
    ("dot-producted", "/llms/what-happens/vectors/dot-product/"),
    ("hidden states", "/llms/what-happens/embeddings/hidden-states/"),
    ("hidden state", "/llms/what-happens/embeddings/hidden-states/"),
    ("feed-forward network", "/llms/what-happens/embeddings/model-layers/ffn-deep-dive/"),
    ("feed-forward", "/llms/what-happens/embeddings/model-layers/ffn-deep-dive/"),
    ("arithmetic intensity", "/llms/what-happens/embeddings/layer-transforms/dimension-tradeoffs/"),
    ("gradient descent", "/llms/what-happens/embeddings/gradients/"),
    ("gradient updates", "/llms/what-happens/embeddings/gradients/"),
    ("Byte Pair Encoding", "/llms/what-happens/tokens/tokenization/"),
    ("token IDs", "/llms/what-happens/tokens/"),
    ("token ID", "/llms/what-happens/tokens/"),
    ("cosine similarity", "/llms/what-happens/vectors/dot-product/"),
    ("residual connection", "/llms/what-happens/embeddings/model-layers/"),
    ("residual connections", "/llms/what-happens/embeddings/model-layers/"),
    ("activation function", "/llms/what-happens/embeddings/layer-transforms/"),
    ("activation functions", "/llms/what-happens/embeddings/layer-transforms/"),
    ("weight matrices", "/llms/what-happens/embeddings/weights/"),
    ("weight matrix", "/llms/what-happens/embeddings/weights/"),
    ("attention mechanism", "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"),
    ("attention scores", "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"),
    ("attention score", "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"),
    ("attention patterns", "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"),
    ("context window", "/llms/what-happens/prefill-decode/kv-cache/"),
    ("embedding table", "/llms/what-happens/embeddings/"),
    ("embedding space", "/llms/what-happens/embeddings/"),
    ("embedding vector", "/llms/what-happens/embeddings/"),
    ("embedding vectors", "/llms/what-happens/embeddings/"),
    ("probability distribution", "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"),
    ("next-token prediction", "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"),
    ("stop signal", "/llms/what-happens/embeddings/model-layers/final-vector-to-token/stopping/"),
    ("EOS token", "/llms/what-happens/embeddings/model-layers/final-vector-to-token/stopping/"),

    # Single-word / short terms (less specific, matched after phrases)
    ("tokenization", "/llms/what-happens/tokens/tokenization/"),
    ("tokenizer", "/llms/what-happens/tokens/tokenization/"),
    ("tokenized", "/llms/what-happens/tokens/tokenization/"),
    ("embeddings", "/llms/what-happens/embeddings/"),
    ("embedding", "/llms/what-happens/embeddings/"),
    ("quantization", "/llms/what-happens/prefill-decode/kv-cache/quantization/"),
    ("quantized", "/llms/what-happens/prefill-decode/kv-cache/quantization/"),
    ("backpropagation", "/llms/what-happens/embeddings/gradients/"),
    ("gradients", "/llms/what-happens/embeddings/gradients/"),
    ("gradient", "/llms/what-happens/embeddings/gradients/"),
    ("softmax", "/llms/what-happens/embeddings/model-layers/attention-deep-dive/"),
    ("prefill", "/llms/what-happens/prefill-decode/"),
    ("decode", "/llms/what-happens/prefill-decode/"),
    ("logits", "/llms/what-happens/embeddings/model-layers/final-vector-to-token/"),
    ("GQA", "/llms/what-happens/prefill-decode/kv-cache/mqa-gqa/"),
    ("MQA", "/llms/what-happens/prefill-decode/kv-cache/mqa-gqa/"),
    ("SiLU", "/llms/what-happens/embeddings/layer-transforms/"),
    ("GeLU", "/llms/what-happens/embeddings/layer-transforms/"),
    ("ReLU", "/llms/what-happens/embeddings/layer-transforms/"),
    ("FlashAttention", "/llms/what-happens/embeddings/model-layers/attention-deep-dive/multi-head-attention/"),
    ("BPE", "/llms/what-happens/tokens/tokenization/"),
    ("RLHF", "/llms/what-happens/thinking/"),
]


def get_page_url(filepath):
    """Extract the url from front matter."""
    with open(filepath) as f:
        content = f.read()
    match = re.search(r'^url:\s*"([^"]+)"', content, re.MULTILINE)
    return match.group(1) if match else None


def split_content(text):
    """Split content into front matter and body."""
    if text.startswith("---"):
        end = text.index("---", 3)
        return text[:end + 3], text[end + 3:]
    return "", text


def protect_zones(body):
    """Identify zones that should not be modified: code blocks, mermaid, links, bold, headings."""
    protected = []

    # Mermaid blocks
    for m in re.finditer(r'```mermaid.*?```', body, re.DOTALL):
        protected.append((m.start(), m.end()))

    # Code blocks (fenced)
    for m in re.finditer(r'```.*?```', body, re.DOTALL):
        protected.append((m.start(), m.end()))

    # Inline code
    for m in re.finditer(r'`[^`]+`', body):
        protected.append((m.start(), m.end()))

    # Existing markdown links [text](url)
    for m in re.finditer(r'\[[^\]]+\]\([^)]+\)', body):
        protected.append((m.start(), m.end()))

    # Note: bold (**text**) and italic (*text*) are NOT protected —
    # links work fine inside them and we want terms like *embedding* to be linkable.

    # Headings
    for m in re.finditer(r'^#+\s.*$', body, re.MULTILINE):
        protected.append((m.start(), m.end()))

    # Table header rows with |
    for m in re.finditer(r'^\|.*\|$', body, re.MULTILINE):
        protected.append((m.start(), m.end()))

    return protected


def is_protected(pos, length, zones):
    """Check if a position falls within any protected zone."""
    end = pos + length
    for zs, ze in zones:
        if pos >= zs and pos < ze:
            return True
        if end > zs and end <= ze:
            return True
    return False


def auto_link_body(body, page_url):
    """Replace first occurrence of each term with a markdown link."""
    protected = protect_zones(body)
    linked_urls = set()  # Track which URLs we've already linked to
    linked_urls.add(page_url)  # Never link to self

    for term, url in TERM_MAP:
        if url in linked_urls:
            continue

        # Build regex: word-boundary match, case-insensitive for multi-word,
        # exact case for acronyms (all caps or specific casing)
        is_acronym = term.isupper() and len(term) <= 5
        is_camel = term[0].isupper() and not term.isupper() and not ' ' in term

        if is_acronym:
            # Exact case match for acronyms
            pattern = re.compile(r'(?<!\w)' + re.escape(term) + r'(?!\w)')
        elif is_camel:
            # Case-sensitive for CamelCase terms like FlashAttention, SiLU
            pattern = re.compile(r'(?<!\w)' + re.escape(term) + r'(?!\w)')
        else:
            # Case-insensitive for regular terms
            pattern = re.compile(r'(?<!\w)' + re.escape(term) + r'(?!\w)', re.IGNORECASE)

        for match in pattern.finditer(body):
            pos = match.start()
            matched_text = match.group(0)

            if is_protected(pos, len(matched_text), protected):
                continue

            # Don't link if it's immediately preceded by [ (already part of a link)
            if pos > 0 and body[pos - 1] == '[':
                continue

            # Replace this occurrence
            link = f"[{matched_text}]({url})"
            body = body[:pos] + link + body[pos + len(matched_text):]

            # Update protected zones (shift everything after the insertion)
            diff = len(link) - len(matched_text)
            protected = [(s + diff if s > pos else s, e + diff if e > pos else e)
                         for s, e in protected]
            # Protect the new link itself
            protected.append((pos, pos + len(link)))

            linked_urls.add(url)
            break  # Only first occurrence

    return body


def process_file(filepath):
    """Process a single file."""
    with open(filepath) as f:
        content = f.read()

    page_url = get_page_url(filepath)
    if not page_url:
        return False

    frontmatter, body = split_content(content)
    new_body = auto_link_body(body, page_url)

    if new_body == body:
        return False

    with open(filepath, 'w') as f:
        f.write(frontmatter + new_body)

    return True


def main():
    files = glob.glob(os.path.join(CONTENT_DIR, "[0-4]*.md"))
    modified = 0
    for f in sorted(files):
        name = os.path.basename(f)
        if process_file(f):
            print(f"  + {name}")
            modified += 1
        else:
            print(f"    {name} (no changes)")
    print(f"\nModified {modified} files")


if __name__ == "__main__":
    main()
