"""
data/download.py
Downloads and saves all datasets used in this project.
Run: python data/download.py
"""

import os
import random
import numpy as np
from datasets import load_dataset

# Fix random seeds
random.seed(42)
np.random.seed(42)

# Output directories
os.makedirs("data/raw/pubmedqa", exist_ok=True)
os.makedirs("data/raw/meddec", exist_ok=True)

# ── Dataset 1: PubMedQA (RAG evaluation) ──────────────────────────────────
print("Downloading PubMedQA...")
try:
    pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
except TypeError as exc:
    # Work around stale/incompatible HF cache metadata on some setups.
    print(f"Default HuggingFace cache failed ({exc}). Retrying with project cache...")
    cache_dir = os.path.join("data", "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    pubmedqa = load_dataset(
        "qiaojin/PubMedQA",
        "pqa_labeled",
        cache_dir=cache_dir
    )
pubmedqa.save_to_disk("data/raw/pubmedqa")
print(f"  Train: {len(pubmedqa['train'])} samples")
print(f"  Columns: {pubmedqa['train'].column_names}")
print("  Saved to data/raw/pubmedqa")

# ── Dataset 2: MedDec (if cloned via git, just verify it exists) ───────────
meddec_path = "data/raw/meddec"
if os.path.exists(meddec_path):
    files = os.listdir(meddec_path)
    print(f"\nMedDec found at {meddec_path}")
    print(f"  Files: {files}")
else:
    print(f"\nMedDec not found. Run:")
    print(f"  git clone https://github.com/CLU-UML/MedDec data/raw/meddec")

print("\nAll downloads complete.")