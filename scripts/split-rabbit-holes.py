#!/usr/bin/env python3
"""Split RABBIT-HOLES.md into individual Hugo content pages for the LLMs collection."""

import re
import os
import textwrap

SOURCE = os.path.expanduser(
    "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Minime/"
    "Blog/LLMs - All the rabbit holes/RABBIT-HOLES.md"
)
OUTPUT_DIR = os.path.expanduser("~/Projects/minime/content/llms")

# --- Parse the Mermaid graph for edges ---

EDGES = {
    # parent_id -> [child_ids]
    "0": ["1a", "1b", "1c", "1d", "1e", "1f", "1g"],
    "1a": ["2h"],
    "1b": ["2a"],
    "1c": ["2b", "2c", "2d", "2e", "2f", "2g"],
    "1d": ["2i"],
    "1e": ["2k"],
    "1f": ["2j"],
    "2a": ["3a"],
    "2e": ["3c", "3d", "3e"],
    "2f": ["3b"],
    "2i": ["3f", "3g", "3h", "3i", "3j", "3k"],
    "3c": ["4a"],
    "3e": ["4b"],
}

# Build reverse map: child -> parent
PARENTS = {}
for parent, children in EDGES.items():
    for child in children:
        PARENTS[child] = parent

# Node ID -> slug mapping
SLUGS = {
    "0": "what-happens",
    "1a": "vectors", "1b": "tokens", "1c": "embeddings", "1d": "prefill-decode",
    "1e": "thinking", "1f": "tool-calls", "1g": "memory",
    "2a": "tokenization", "2b": "weights", "2c": "gradients",
    "2d": "directions", "2e": "model-layers", "2f": "layer-transforms",
    "2g": "hidden-states", "2h": "dot-product", "2i": "kv-cache",
    "2j": "skills", "2k": "planning",
    "3a": "tokenization-perf", "3b": "dimension-tradeoffs",
    "3c": "attention-deep-dive", "3d": "ffn-deep-dive",
    "3e": "final-vector-to-token", "3f": "sparse-attention",
    "3g": "attention-approximations", "3h": "quantization",
    "3i": "paged-attention", "3j": "kv-cache-offloading", "3k": "mqa-gqa",
    "4a": "multi-head-attention", "4b": "stopping",
}


def get_url_path(node_id):
    """Build the full nested URL path for a node based on its ancestry."""
    chain = []
    current = node_id
    while current:
        chain.append(SLUGS[current])
        current = PARENTS.get(current, "")
    chain.reverse()
    return "/llms/" + "/".join(chain) + "/"


# Pre-compute all URL paths
URL_PATHS = {nid: get_url_path(nid) for nid in SLUGS}


def get_siblings(node_id):
    """Get sibling node IDs (same parent, excluding self)."""
    parent = PARENTS.get(node_id, "")
    if not parent:
        return []
    return [c for c in EDGES.get(parent, []) if c != node_id]


# Weight for ordering (tier * 100 + letter index)
def compute_weight(node_id):
    if node_id == "0":
        return 0
    tier = int(node_id[0])
    letter = node_id[1:]
    val = 0
    for ch in letter:
        val = val * 26 + (ord(ch) - ord('a'))
    return tier * 100 + val


def extract_summary(body):
    """Get first non-empty, non-mermaid line as summary."""
    in_mermaid = False
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```mermaid"):
            in_mermaid = True
            continue
        if in_mermaid:
            if stripped.startswith("```"):
                in_mermaid = False
            continue
        if stripped.startswith("*") and stripped.endswith("*"):
            continue  # skip italic annotations
        if stripped.startswith("`") and stripped.endswith("`"):
            continue  # skip code-only lines like legend lines
        if stripped and not stripped.startswith("#") and not stripped.startswith("---"):
            # Truncate to ~150 chars
            summary = stripped[:200]
            if len(stripped) > 200:
                summary = summary[:summary.rfind(" ")] + "..."
            return summary
    return ""


def convert_cross_references(body):
    """Convert node ID references like (see 2c) or (from 1d) to Hugo links."""
    def replace_ref(match):
        prefix = match.group(1)
        node_id = match.group(2)
        if node_id in URL_PATHS:
            return f'{prefix}[{node_id}]({URL_PATHS[node_id]})'
        return match.group(0)

    body = re.sub(
        r'((?:see |from |in |node |\())'
        r'([0-4][a-k](?:\b))',
        replace_ref,
        body
    )
    return body


def parse_nodes(content):
    """Parse the markdown content into individual nodes."""
    nodes = {}

    # Split by ### headings
    # Pattern: ### {id} — {title} (or ### 0 — {title} for root)
    pattern = re.compile(
        r'^### (\S+)\s*(?:—|---)\s*(.+)$',
        re.MULTILINE
    )

    matches = list(pattern.finditer(content))

    for i, match in enumerate(matches):
        raw_id = match.group(1).strip()
        title = match.group(2).strip()

        # Normalize node ID
        node_id = raw_id
        if node_id.startswith("0"):
            node_id = "0"

        # Extract body: from end of heading to start of next heading (or end of file)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].strip()

        # Remove tier separator lines
        body = re.sub(r'^---\s*$', '', body, flags=re.MULTILINE).strip()
        # Remove tier headers like "## Tier 1"
        body = re.sub(r'^## Tier \d+\s*$', '', body, flags=re.MULTILINE).strip()

        nodes[node_id] = {
            "title": title,
            "body": body,
        }

    return nodes


def write_node(node_id, title, body):
    """Write a single node as a Hugo content page."""
    slug = SLUGS.get(node_id, node_id)
    filename = f"{node_id}-{slug}.md"
    filepath = os.path.join(OUTPUT_DIR, filename)

    tier = 0 if node_id == "0" else int(node_id[0])
    parent = PARENTS.get(node_id, "")
    children = EDGES.get(node_id, [])
    siblings = get_siblings(node_id)
    weight = compute_weight(node_id)
    summary = extract_summary(body)
    url_path = URL_PATHS[node_id]

    # Convert cross-references
    body = convert_cross_references(body)

    # Build front matter
    children_yaml = "\n".join(f'  - "{c}"' for c in children)
    children_block = f"children:\n{children_yaml}" if children else "children: []"

    siblings_yaml = "\n".join(f'  - "{s}"' for s in siblings)
    siblings_block = f"siblings:\n{siblings_yaml}" if siblings else "siblings: []"

    # Escape any quotes in title/summary
    safe_title = title.replace('"', '\\"')
    safe_summary = summary.replace('"', '\\"')

    lines = [
        "---",
        f'title: "{safe_title}"',
        "date: 2026-04-13",
        f'node_id: "{node_id}"',
        f"tier: {tier}",
        f'parent: "{parent}"',
        children_block,
        siblings_block,
        f'url: "{url_path}"',
        f'summary: "{safe_summary}"',
        'series: "llms"',
        "mermaid: true",
        f"weight: {weight}",
        "tags:",
        "  - llm",
        "---",
    ]
    frontmatter = "\n".join(lines) + "\n"

    with open(filepath, "w") as f:
        f.write(frontmatter)
        f.write("\n")
        f.write(body)
        f.write("\n")

    print(f"  wrote {filename} ({len(body)} chars)")


def write_index():
    """Write the _index.md landing page with the interactive Mermaid map."""
    filepath = os.path.join(OUTPUT_DIR, "_index.md")

    # Build click directives for the Mermaid map
    click_lines = []
    for node_id in sorted(SLUGS.keys()):
        mermaid_id = "R0" if node_id == "0" else node_id
        click_lines.append(f'    click {mermaid_id} "{URL_PATHS[node_id]}"')
    clicks = "\n".join(click_lines)

    content = textwrap.dedent(f"""\
        ---
        title: "LLMs: All the Rabbit Holes"
        description: "What actually happens when you send a message to an LLM — from tokens to attention to output, one rabbit hole at a time."
        layout: "llms-home"
        mermaid: true
        ---

        ```mermaid
        graph TD
            R0["<b>0</b><br/>What happens when you send a message"]:::reviewed
            R0 --> 1a["<b>1a</b><br/>What are vectors?"]:::reviewed
            R0 --> 1b["<b>1b</b><br/>What is a token?"]:::reviewed
            1b --> 2a["<b>2a</b><br/>How tokenization works"]:::reviewed
            2a --> 3a["<b>3a</b><br/>Tokenization performance"]:::reviewed
            R0 --> 1c["<b>1c</b><br/>What are embeddings?"]:::reviewed
            1c --> 2b["<b>2b</b><br/>What are weights?"]:::reviewed
            1c --> 2c["<b>2c</b><br/>Gradients and gradient updates"]:::reviewed
            1c --> 2d["<b>2d</b><br/>Directions encode relationships"]:::reviewed
            1c --> 2e["<b>2e</b><br/>What are model layers?"]:::reviewed
            1c --> 2f["<b>2f</b><br/>How layers transform vectors"]:::reviewed
            1c --> 2g["<b>2g</b><br/>What are hidden states?"]:::reviewed
            R0 --> 1d["<b>1d</b><br/>Prefill vs decode"]:::reviewed
            1a --> 2h["<b>2h</b><br/>What is a dot product?"]:::reviewed
            2f --> 3b["<b>3b</b><br/>Dimension trade-offs"]:::reviewed
            1d --> 2i["<b>2i</b><br/>KV cache and context memory costs"]:::reviewed
            2e --> 3c["<b>3c</b><br/>Attention deep dive"]:::reviewed
            3c --> 4a["<b>4a</b><br/>Multi-head attention and its cost"]:::reviewed
            2e --> 3d["<b>3d</b><br/>FFN deep dive"]:::reviewed
            2e --> 3e["<b>3e</b><br/>From final vector to predicted token"]:::reviewed
            2i --> 3f["<b>3f</b><br/>Sparse attention"]:::reviewed
            2i --> 3g["<b>3g</b><br/>Attention approximations"]:::unreviewed
            2i --> 3h["<b>3h</b><br/>What is quantization?"]:::reviewed
            2i --> 3i["<b>3i</b><br/>Paged attention"]:::unreviewed
            2i --> 3j["<b>3j</b><br/>KV cache offloading"]:::unreviewed
            2i --> 3k["<b>3k</b><br/>MQA and GQA"]:::reviewed
            3e --> 4b["<b>4b</b><br/>How does the model know when to stop?"]:::unreviewed
            R0 --> 1e["<b>1e</b><br/>How does thinking work?"]:::unreviewed
            R0 --> 1f["<b>1f</b><br/>How do tool calls work?"]:::unreviewed
            1f --> 2j["<b>2j</b><br/>Skills"]:::unreviewed
            1e --> 2k["<b>2k</b><br/>Planning and multi-step execution"]:::unreviewed
            R0 --> 1g["<b>1g</b><br/>How does memory work?"]:::unreviewed

            classDef reviewed fill:#2d6a4f,stroke:#1b4332,color:#d8f3dc
            classDef unreviewed fill:#e76f51,stroke:#9c3a1a,color:#fff

        {clicks}
        ```
    """)

    with open(filepath, "w") as f:
        f.write(content)
    print(f"  wrote _index.md")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(SOURCE, "r") as f:
        content = f.read()

    print("Parsing nodes...")
    nodes = parse_nodes(content)
    print(f"Found {len(nodes)} nodes")

    print("\nWriting individual pages...")
    for node_id, data in sorted(nodes.items()):
        write_node(node_id, data["title"], data["body"])

    print("\nWriting _index.md...")
    write_index()

    print(f"\nDone! {len(nodes)} node pages + _index.md written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
