#!/usr/bin/env python3
"""Add click handlers to the pipeline Mermaid diagrams in LLMs content pages."""

import os
import re
import glob

CONTENT_DIR = os.path.expanduser("~/Projects/minime/content/llms")

# Map Mermaid node IDs in the standard pipeline diagram to page URLs.
# The pipeline uses single-letter IDs (A, B, C, D, E, F, G, H, etc.)
# with labels like "Input Text", "Tokenizer", "Embedding", etc.
# We match by label content since the same letter can mean different things.

LABEL_TO_URL = {
    "Input Text": "/llms/what-happens/",
    "Tokenizer": "/llms/what-happens/tokens/",
    "Token IDs": "/llms/what-happens/tokens/",
    "Embedding": "/llms/what-happens/embeddings/",
    "Layers": "/llms/what-happens/embeddings/model-layers/",
    "Predict": "/llms/what-happens/embeddings/model-layers/final-vector-to-token/",
    "Output Token": "/llms/what-happens/",
    "Output": "/llms/what-happens/embeddings/model-layers/final-vector-to-token/",
    "Attn Scores": "/llms/what-happens/embeddings/model-layers/attention-deep-dive/",
    "Weighted Sum": "/llms/what-happens/embeddings/model-layers/attention-deep-dive/",
    "FFN": "/llms/what-happens/embeddings/model-layers/ffn-deep-dive/",
    "Residual": "/llms/what-happens/embeddings/model-layers/",
    "Q,K,V Proj": "/llms/what-happens/embeddings/model-layers/attention-deep-dive/",
    "Vocab Proj": "/llms/what-happens/embeddings/model-layers/final-vector-to-token/",
    "Activation": "/llms/what-happens/embeddings/layer-transforms/",
    "Softmax": "/llms/what-happens/embeddings/model-layers/attention-deep-dive/",
    "Normalize": "/llms/what-happens/embeddings/model-layers/",
    "BPE Split": "/llms/what-happens/tokens/tokenization/",
    "Normalize": "/llms/what-happens/tokens/tokenization/",
    # Prefill/decode diagram
    "Tokenize + Embed": "/llms/what-happens/embeddings/",
    "User message": "/llms/what-happens/",
    # KV cache diagram
    "KV Cache": "/llms/what-happens/prefill-decode/kv-cache/",
}


def add_clicks_to_mermaid(mermaid_block, current_url):
    """Add click directives to a mermaid block based on node labels."""
    lines = mermaid_block.split("\n")

    # Find all node definitions: ID["label"] or ID[label]
    node_pattern = re.compile(r'^\s+(\w+)\[(?:"([^"]+)"|([^\]]+))\]')

    # Also match nodes in edge definitions: A["label"] --> B["label"]
    edge_node_pattern = re.compile(r'(\w+)\[(?:"([^"]+)"|([^\]]+))\]')

    # Collect all node_id -> label mappings
    nodes = {}
    for line in lines:
        for match in edge_node_pattern.finditer(line):
            node_id = match.group(1)
            label = match.group(2) or match.group(3)
            # Strip HTML tags and class defs
            clean_label = re.sub(r':::?\w+', '', label).strip()
            clean_label = re.sub(r'<[^>]+>', '', clean_label).strip()
            nodes[node_id] = clean_label

    # Generate click directives
    clicks = []
    for node_id, label in nodes.items():
        # Skip special nodes like classDef, style directives
        if node_id in ('classDef', 'style', 'graph', 'subgraph'):
            continue

        # Try to match label to a URL
        matched_url = None
        for keyword, url in LABEL_TO_URL.items():
            if keyword.lower() in label.lower():
                matched_url = url
                break

        # Also match partial labels for nodes like "FFN: W_up" or "Layers ×80"
        if not matched_url:
            for keyword, url in LABEL_TO_URL.items():
                # Check if any word in the keyword appears in the label
                if any(w.lower() in label.lower() for w in keyword.split() if len(w) > 2):
                    matched_url = url
                    break

        if matched_url and matched_url != current_url:
            clicks.append(f"    click {node_id} \"{matched_url}\"")

    if not clicks:
        return mermaid_block

    # Insert clicks before the closing ```
    # Find the last classDef or style line, insert after it
    insert_idx = len(lines) - 1  # before closing ```
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith("classDef") or stripped.startswith("style"):
            insert_idx = i + 1
            break

    # Add blank line then clicks
    click_block = [""] + clicks
    lines = lines[:insert_idx] + click_block + lines[insert_idx:]

    return "\n".join(lines)


def process_file(filepath):
    """Process a single markdown file, adding clicks to its mermaid blocks."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Get the current page's URL from front matter
    url_match = re.search(r'^url:\s*"([^"]+)"', content, re.MULTILINE)
    if not url_match:
        return False
    current_url = url_match.group(1)

    # Find and process mermaid blocks
    # Only process the FIRST mermaid block (the pipeline overview)
    mermaid_pattern = re.compile(r'(```mermaid\n)(.*?)(```)', re.DOTALL)

    match = mermaid_pattern.search(content)
    if not match:
        return False

    original = match.group(2)

    # Skip specialized diagrams that don't have the pipeline pattern
    # (e.g., subgraph-based diagrams like MHA/GQA/MQA comparison)
    if 'subgraph' in original:
        return False

    modified = add_clicks_to_mermaid(original, current_url)

    if modified == original:
        return False

    content = content[:match.start(2)] + modified + content[match.end(2):]

    with open(filepath, 'w') as f:
        f.write(content)

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
