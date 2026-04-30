# pyright: reportMissingImports=false

"""
src/build_index.py
Chunks processed text, embeds it, builds FAISS index.
Run: python src/build_index.py
"""

import os
import json
import pickle
import random
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Fix random seeds
random.seed(42)
np.random.seed(42)

# Paths
PROCESSED_DIR = "data/processed"
INDEX_DIR     = "data/index"
os.makedirs(INDEX_DIR, exist_ok=True)

# Settings
MODEL_NAME    = "all-MiniLM-L6-v2"   # small fast model, works on any PC
CHUNK_SIZE    = 150                    # words per chunk
CHUNK_OVERLAP = 30                     # overlap between chunks


def chunk_text(text):
    """Split text into overlapping word chunks."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += (CHUNK_SIZE - CHUNK_OVERLAP)
    return chunks


def load_passages():
    """Load all processed JSON files and chunk the context text."""
    passages = []

    for split in ["train", "val", "test"]:
        path = os.path.join(PROCESSED_DIR, f"{split}.json")

        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Loaded {split}.json — {len(data)} samples")

        for sample in data:
            context = sample.get("context", "")
            if not context:
                continue

            for idx, chunk in enumerate(chunk_text(context)):
                passages.append({
                    "text":      chunk,
                    "question":  sample.get("question", ""),
                    "answer":    sample.get("answer", ""),
                    "label":     sample.get("label", ""),
                    "source_id": sample.get("id", ""),
                    "chunk_idx": idx,
                    "split":     split
                })

    print(f"\nTotal chunks : {len(passages)}")
    return passages


def main():

    # ── Step 1: Load passages ─────────────────────────────────────────────
    print("Loading processed data and chunking...\n")
    passages = load_passages()

    if len(passages) == 0:
        print("ERROR: No passages found. Run load_data.py first.")
        return

    # ── Step 2: Load embedding model ──────────────────────────────────────
    print(f"\nLoading embedding model: {MODEL_NAME}")
    print("(Downloads ~90MB on first run, cached after that)\n")
    model = SentenceTransformer(MODEL_NAME)

    # ── Step 3: Embed all chunks ──────────────────────────────────────────
    texts = [p["text"] for p in passages]
    print(f"Embedding {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print(f"\nEmbedding matrix shape : {embeddings.shape}")

    # ── Step 4: Build FAISS index ─────────────────────────────────────────
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built — {index.ntotal} vectors, dim={dim}")

    # ── Step 5: Quick sanity test ─────────────────────────────────────────
    query      = "effect of aspirin on heart disease"
    q_emb      = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(q_emb, 3)

    print(f"\nSanity check — query: '{query}'")
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        p = passages[idx]
        print(f"  Rank {rank+1} | score={dist:.4f} | label={p['label']}")
        print(f"  Text: {p['text'][:120]}...")

    # ── Step 6: Save everything ───────────────────────────────────────────
    faiss_path    = os.path.join(INDEX_DIR, "pubmedqa.index")
    passages_path = os.path.join(INDEX_DIR, "passages.pkl")
    meta_path     = os.path.join(INDEX_DIR, "index_meta.json")

    faiss.write_index(index, faiss_path)
    print(f"\nSaved FAISS index   → {faiss_path}")

    with open(passages_path, "wb") as f:
        pickle.dump(passages, f)
    print(f"Saved passage store → {passages_path}")

    with open(meta_path, "w") as f:
        json.dump({
            "model":          MODEL_NAME,
            "chunk_size":     CHUNK_SIZE,
            "chunk_overlap":  CHUNK_OVERLAP,
            "total_passages": len(passages),
            "embedding_dim":  dim,
            "random_seed":    42
        }, f, indent=2)
    print(f"Saved metadata      → {meta_path}")

    print("\nbuild_index.py complete ✓")


if __name__ == "__main__":
    main()