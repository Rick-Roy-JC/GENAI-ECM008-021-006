# pyright: reportMissingImports=false

"""
src/compare_embedders.py

Compares retrieval quality between general embedder (MiniLM)
and biomedical embedder (PubMedBERT) on clinical queries.

Run from project root: python src/compare_embedders.py
"""

import os
import json
import pickle
import random
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

random.seed(42)
np.random.seed(42)

INDEX_DIR = "data/index"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

PUBMED_INDEX = "pubmedqa_pubmedbert.index"
PUBMED_PASSAGES = "passages_pubmedbert.pkl"
MINILM_INDEX = "pubmedqa_minilm.index"

# Clinical test queries — chosen to test biomedical understanding
TEST_QUERIES = [
    "Does aspirin reduce cardiovascular mortality?",
    "Is metformin effective for type 2 diabetes treatment?",
    "What is the effect of statins on LDL cholesterol?",
    "Does surgery improve outcomes in lower back pain?",
    "Is vitamin D supplementation effective for bone health?",
    "Does beta blocker therapy reduce heart failure mortality?",
    "Is chemotherapy effective for early stage breast cancer?",
    "Does ACE inhibitor therapy prevent diabetic nephropathy?",
]


def load_index_and_passages(index_file, passages_file):
    index = faiss.read_index(os.path.join(INDEX_DIR, index_file))
    with open(os.path.join(INDEX_DIR, passages_file), "rb") as f:
        passages = pickle.load(f)
    return index, passages


def retrieve(query, index, passages, model, top_k=3):
    q_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    distances, indices = index.search(q_emb, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            results.append(
                {
                    "text": passages[idx]["text"][:150],
                    "score": round(float(dist), 4),
                    "label": passages[idx].get("label", "?"),
                }
            )
    return results


def ensure_minilm_index(passages, minilm_model):
    """
    MiniLM vectors are not compatible with the PubMedBERT FAISS index (different dim).
    Build or load a separate IndexFlatIP on the same passages, in the same order as
    build_index.py used for PubMedBERT.
    """
    path = os.path.join(INDEX_DIR, MINILM_INDEX)
    if os.path.exists(path):
        print(f"Loading existing MiniLM index → {path}")
        return faiss.read_index(path)

    print("Building MiniLM FAISS index (first run only; same chunks as PubMedBERT)...")
    texts = [p["text"] for p in passages]
    embeddings = minilm_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, path)
    print(f"Saved MiniLM index → {path}")
    return index


def main():
    pubmed_index_path = os.path.join(INDEX_DIR, PUBMED_INDEX)
    pubmed_passages_path = os.path.join(INDEX_DIR, PUBMED_PASSAGES)
    if not os.path.exists(pubmed_index_path) or not os.path.exists(pubmed_passages_path):
        raise FileNotFoundError(
            "PubMedBERT index or passages missing. Run: python src/build_index.py"
        )

    print("Loading MiniLM (general)...")
    minilm_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading PubMedBERT (biomedical)...")
    pubmed_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

    print("\nLoading PubMedBERT index and passages...")
    pubmed_index, passages = load_index_and_passages(PUBMED_INDEX, PUBMED_PASSAGES)

    print("\nMiniLM index...")
    minilm_index = ensure_minilm_index(passages, minilm_model)

    print("\n── Retrieval Comparison ─────────────────────────────\n")

    comparison_results = []
    pubmed_avg_scores = []
    minilm_avg_scores = []

    for query in TEST_QUERIES:
        pubmed_results = retrieve(query, pubmed_index, passages, pubmed_model)
        minilm_results = retrieve(query, minilm_index, passages, minilm_model)

        pubmed_top_score = pubmed_results[0]["score"] if pubmed_results else 0.0
        minilm_top_score = minilm_results[0]["score"] if minilm_results else 0.0

        pubmed_avg_scores.append(pubmed_top_score)
        minilm_avg_scores.append(minilm_top_score)

        pubmed_snip = (
            (pubmed_results[0]["text"][:80] + "...")
            if pubmed_results
            else "(no hits)"
        )
        minilm_snip = (
            (minilm_results[0]["text"][:80] + "...")
            if minilm_results
            else "(no hits)"
        )

        print(f"Query: {query}")
        print(f"  PubMedBERT top score : {pubmed_top_score:.4f} | {pubmed_snip}")
        print(f"  MiniLM     top score : {minilm_top_score:.4f} | {minilm_snip}")
        print()

        comparison_results.append(
            {
                "query": query,
                "pubmedbert_score": pubmed_top_score,
                "minilm_score": minilm_top_score,
                "pubmedbert_top_text": pubmed_results[0]["text"]
                if pubmed_results
                else "",
                "minilm_top_text": minilm_results[0]["text"]
                if minilm_results
                else "",
            }
        )

    print("── Summary ──────────────────────────────────────────")
    if pubmed_avg_scores and minilm_avg_scores:
        p_mean = float(np.mean(pubmed_avg_scores))
        m_mean = float(np.mean(minilm_avg_scores))
        print(f"  PubMedBERT avg top-1 score : {p_mean:.4f}")
        print(f"  MiniLM     avg top-1 score : {m_mean:.4f}")
        improvement = p_mean - m_mean
        print(f"  Improvement (PubMed − MiniLM): {improvement:+.4f}")
    else:
        improvement = 0.0
        print("  No scores collected.")

    summary = {
        "pubmedbert_avg_score": round(float(np.mean(pubmed_avg_scores)), 4)
        if pubmed_avg_scores
        else 0.0,
        "minilm_avg_score": round(float(np.mean(minilm_avg_scores)), 4)
        if minilm_avg_scores
        else 0.0,
        "improvement": round(float(improvement), 4),
        "queries": comparison_results,
    }

    out_path = os.path.join(RESULTS_DIR, "embedder_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_path}")
    print("\ncompare_embedders.py complete ✓")


if __name__ == "__main__":
    main()
