#!/usr/bin/env python3
"""Replace node ID link text like [1d](...) with descriptive text."""

import os
import re
import glob

CONTENT_DIR = os.path.expanduser("~/Projects/minime/content/llms")

# Map node IDs to short descriptive labels for inline references
NODE_LABELS = {
    "0": "the root overview",
    "1a": "vectors",
    "1b": "tokens",
    "1c": "embeddings",
    "1d": "prefill vs decode",
    "1e": "thinking",
    "1f": "tool calls",
    "1g": "memory",
    "2a": "tokenization",
    "2b": "weights",
    "2c": "gradients",
    "2d": "direction relationships",
    "2e": "model layers",
    "2f": "layer transforms",
    "2g": "hidden states",
    "2h": "dot products",
    "2i": "KV cache",
    "2j": "skills",
    "2k": "planning",
    "3a": "tokenization performance",
    "3b": "dimension trade-offs",
    "3c": "attention",
    "3d": "feed-forward networks",
    "3e": "token prediction",
    "3f": "sparse attention",
    "3g": "attention approximations",
    "3h": "quantization",
    "3i": "paged attention",
    "3j": "KV cache offloading",
    "3k": "MQA and GQA",
    "4a": "multi-head attention",
    "4b": "stopping",
}


def fix_file(filepath):
    with open(filepath) as f:
        content = f.read()

    original = content

    # Pattern: [node_id](url) where node_id is like 1d, 2e, etc.
    def replace_ref(match):
        node_id = match.group(1)
        url = match.group(2)
        label = NODE_LABELS.get(node_id, node_id)
        return f"[{label}]({url})"

    content = re.sub(
        r'\[([0-4][a-k])\](\(/llms/[^)]+\))',
        replace_ref,
        content
    )

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False


def main():
    files = glob.glob(os.path.join(CONTENT_DIR, "[0-4]*.md"))
    modified = 0
    for f in sorted(files):
        name = os.path.basename(f)
        if fix_file(f):
            print(f"  + {name}")
            modified += 1
    print(f"\nFixed {modified} files")


if __name__ == "__main__":
    main()
