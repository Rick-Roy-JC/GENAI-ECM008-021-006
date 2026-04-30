# pyright: reportMissingImports=false

"""
src/rag_pipeline.py

Full RAG pipeline with continuous knowledge updating.
- Loads FAISS index and passage store
- Retrieves relevant passages for a query
- Generates answer using a lightweight LLM (Flan-T5)
- Supports adding new documents without rebuilding the index

Run: python src/rag_pipeline.py
"""

import os
import json
import pickle
import random
import datetime
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Fix random seeds
random.seed(42)
np.random.seed(42)

# Paths
INDEX_DIR     = "data/index"
RESULTS_DIR   = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Settings — must match build_index.py exactly
MODEL_NAME    = "all-MiniLM-L6-v2"
LLM_NAME      = "google/flan-t5-base"
TOP_K         = 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — Loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_index_and_passages():
    """Load FAISS index and passage store from disk."""

    faiss_path    = os.path.join(INDEX_DIR, "pubmedqa.index")
    passages_path = os.path.join(INDEX_DIR, "passages.pkl")

    if not os.path.exists(faiss_path):
        raise FileNotFoundError(
            "FAISS index not found. Run src/build_index.py first."
        )

    print("Loading FAISS index...")
    index = faiss.read_index(faiss_path)
    print(f"  Vectors in index : {index.ntotal}")

    print("Loading passage store...")
    with open(passages_path, "rb") as f:
        passages = pickle.load(f)
    print(f"  Total passages   : {len(passages)}")

    return index, passages


def load_models():
    """Load embedding model and LLM."""

    print(f"\nLoading embedding model : {MODEL_NAME}")
    embedder = SentenceTransformer(MODEL_NAME)

    print(f"Loading LLM             : {LLM_NAME}")
    print("(Downloads ~1GB on first run, cached after)\n")
    tokenizer = T5Tokenizer.from_pretrained(LLM_NAME)
    llm       = T5ForConditionalGeneration.from_pretrained(LLM_NAME)

    return embedder, tokenizer, llm


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — Retrieval
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def retrieve(query, index, passages, embedder, top_k=TOP_K):
    """
    Given a query string, find the top_k most relevant passages
    from the FAISS index.
    Returns list of (passage_dict, score) tuples.
    """

    # Embed the query
    q_emb = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Search FAISS
    distances, indices = index.search(q_emb, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:      # FAISS returns -1 if not enough results
            continue
        results.append((passages[idx], float(dist)))

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 — Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_prompt(query, retrieved_passages):
    """
    Combines retrieved passages with the query into a prompt
    that Flan-T5 can understand.
    """

    context_parts = []
    for i, (passage, score) in enumerate(retrieved_passages):
        context_parts.append(f"[Passage {i+1}] {passage['text']}")

    context = " ".join(context_parts)

    # Flan-T5 works best with instruction-style prompts
    prompt = (
        f"You are a clinical assistant. "
        f"Answer the medical question based on the context below.\n\n"
        f"Context: {context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    # Truncate if too long — Flan-T5 has 512 token limit
    words = prompt.split()
    if len(words) > 450:
        prompt = " ".join(words[:450])

    return prompt


def generate_answer(prompt, tokenizer, llm):
    """
    Generate an answer using Flan-T5.
    Returns the generated answer string.
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    outputs = llm.generate(
        inputs["input_ids"],
        max_new_tokens=100,
        num_beams=4,             # beam search for better quality
        early_stopping=True,
        no_repeat_ngram_size=3   # avoid repeating phrases
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4 — Continuous Knowledge Updating (CORE CONTRIBUTION)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def chunk_text(text, chunk_size=150, overlap=30):
    """Split text into overlapping chunks — same as build_index.py."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks


def add_new_document(new_doc, index, passages, embedder):
    """
    THE CORE FUNCTION — adds a new clinical document to the
    knowledge base WITHOUT rebuilding the entire index.

    This is what makes the system continuously updatable.

    new_doc is a dict with keys:
        text   — the document content
        source — where it came from (e.g. 'NEJM_2025', 'WHO_guideline')
        date   — when it was added

    Steps:
    1. Chunk the new document text
    2. Embed each chunk
    3. Add embeddings directly to existing FAISS index
    4. Add chunk metadata to passage store
    5. Log the update
    """

    print("\n-- Adding new document to knowledge base ------------")
    print(f"  Source : {new_doc.get('source', 'unknown')}")
    print(f"  Date   : {new_doc.get('date', 'unknown')}")
    print(f"  Length : {len(new_doc['text'].split())} words")

    # Step 1 — Chunk the document
    chunks = chunk_text(new_doc["text"])
    print(f"  Chunks : {len(chunks)}")

    if not chunks:
        print("  WARNING: No chunks extracted, skipping.")
        return index, passages

    # Step 2 — Embed the new chunks
    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Step 3 — Add to FAISS index directly
    # This is the key operation — FAISS IndexFlatIP supports
    # incremental addition without full rebuild
    index.add(embeddings)

    # Step 4 — Add metadata to passage store
    timestamp = datetime.datetime.now().isoformat()
    for i, chunk in enumerate(chunks):
        passages.append({
            "text":      chunk,
            "source":    new_doc.get("source", "unknown"),
            "date":      new_doc.get("date", timestamp),
            "chunk_idx": i,
            "label":     "updated",      # mark as new knowledge
            "split":     "live_update"   # distinguish from original data
        })

    # Step 5 — Log the update to results/
    log_path = os.path.join(RESULTS_DIR, "update_log.jsonl")
    log_entry = {
        "timestamp":     timestamp,
        "source":        new_doc.get("source", "unknown"),
        "date":          new_doc.get("date", "unknown"),
        "chunks_added":  len(chunks),
        "index_size_after": index.ntotal
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"  Added {len(chunks)} chunks to index")
    print(f"  Index size now : {index.ntotal} vectors")
    print(f"  Logged to      : {log_path}")
    print("  Update complete.")

    return index, passages


def save_updated_index(index, passages):
    """
    Save the updated index and passage store back to disk
    so updates persist across runs.
    """
    faiss.write_index(index, os.path.join(INDEX_DIR, "pubmedqa.index"))
    with open(os.path.join(INDEX_DIR, "passages.pkl"), "wb") as f:
        pickle.dump(passages, f)
    print("\nUpdated index saved to disk.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5 — Full Pipeline + Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_pipeline(query, index, passages, embedder, tokenizer, llm):
    """
    Full RAG pipeline for one query.
    Returns a result dict with question, retrieved passages, and answer.
    """

    # Step 1 — Retrieve
    retrieved = retrieve(query, index, passages, embedder)

    # Step 2 — Build prompt
    prompt = build_prompt(query, retrieved)

    # Step 3 — Generate
    answer = generate_answer(prompt, tokenizer, llm)

    return {
        "question":          query,
        "answer":            answer,
        "retrieved_passages": [
            {
                "text":   p["text"][:200],
                "score":  round(score, 4),
                "source": p.get("source", p.get("source_id", "unknown")),
                "label":  p.get("label", "")
            }
            for p, score in retrieved
        ]
    }


def evaluate_on_test_set(index, passages, embedder, tokenizer, llm, n_samples=20):
    """
    Run the pipeline on n_samples from the test set.
    Saves results to results/preliminary_metrics.txt
    This is your Milestone 1 preliminary results.
    """

    import json

    test_path = "data/processed/test.json"
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Use first n_samples only — full eval is for Final submission
    test_data = test_data[:n_samples]

    print(f"\nRunning evaluation on {n_samples} test samples...")

    results       = []
    correct_label = 0

    for sample in tqdm(test_data, desc="  Evaluating"):
        result = run_pipeline(
            sample["question"],
            index, passages,
            embedder, tokenizer, llm
        )
        result["ground_truth_label"]  = sample["label"]
        result["ground_truth_answer"] = sample["answer"]
        results.append(result)

        # Simple accuracy — does generated answer contain correct label word?
        pred   = result["answer"].lower()
        label  = sample["label"].lower()
        if label in pred:
            correct_label += 1

    accuracy = correct_label / len(results) * 100

    # Save full results
    results_path = os.path.join(RESULTS_DIR, "preliminary_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save metrics summary
    metrics_path = os.path.join(RESULTS_DIR, "preliminary_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("CS5202 — RAG Clinical NLP — Milestone 1 Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model (embedder) : {MODEL_NAME}\n")
        f.write(f"Model (LLM)      : {LLM_NAME}\n")
        f.write(f"Top-K retrieved  : {TOP_K}\n")
        f.write(f"Samples evaluated: {n_samples}\n\n")
        f.write(f"Label match accuracy : {accuracy:.1f}%\n\n")
        f.write("Sample Predictions (first 3):\n")
        f.write("-" * 40 + "\n")
        for r in results[:3]:
            f.write(f"Q  : {r['question'][:100]}\n")
            f.write(f"GT : {r['ground_truth_label']}\n")
            f.write(f"A  : {r['answer'][:150]}\n\n")

    print(f"\nLabel match accuracy : {accuracy:.1f}%")
    print(f"Full results saved   -> {results_path}")
    print(f"Metrics saved        -> {metrics_path}")

    return accuracy, results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():

    # ── Load index, passages, models ──────────────────────────────────────
    index, passages = load_index_and_passages()
    embedder, tokenizer, llm = load_models()

    # ── Test basic pipeline on one query ─────────────────────────────────
    print("\n-- Single Query Test --------------------------------")
    test_query = "Does aspirin reduce the risk of cardiovascular disease?"
    result     = run_pipeline(test_query, index, passages, embedder, tokenizer, llm)

    print(f"Question : {result['question']}")
    print(f"Answer   : {result['answer']}")
    print(f"\nTop retrieved passages:")
    for i, p in enumerate(result["retrieved_passages"]):
        print(f"  [{i+1}] score={p['score']} | {p['text'][:100]}...")

    # ── Demonstrate continuous knowledge update ───────────────────────────
    print("\n-- Continuous Knowledge Update Demo -----------------")

    # Simulate a new clinical document arriving
    # In a real deployment this could be a new WHO guideline,
    # a new drug approval notice, or a new research paper
    new_document = {
        "source": "WHO_guideline_2025",
        "date":   "2025-11-01",
        "text": (
            "Updated WHO guidelines 2025 recommend that low-dose aspirin "
            "should no longer be used for primary prevention of cardiovascular "
            "disease in adults over 60 due to increased bleeding risk. "
            "This represents a significant change from previous recommendations. "
            "Patients already on aspirin therapy should consult their physician "
            "before making any changes to their medication regimen. "
            "Secondary prevention use of aspirin remains unchanged. "
            "New antiplatelet agents show better safety profiles in elderly patients."
        )
    }

    # Add it to the live index — no rebuild needed
    index, passages = add_new_document(
        new_document, index, passages, embedder
    )

    # Query again — system now knows about the new guideline
    print("\n-- Query After Knowledge Update ---------------------")
    result_after = run_pipeline(
        test_query, index, passages, embedder, tokenizer, llm
    )
    print(f"Question : {result_after['question']}")
    print(f"Answer   : {result_after['answer']}")

    # ── Save updated index to disk ────────────────────────────────────────
    save_updated_index(index, passages)

    # ── Run evaluation on test set ────────────────────────────────────────
    print("\n-- Milestone 1 Evaluation ---------------------------")
    accuracy, results = evaluate_on_test_set(
        index, passages, embedder, tokenizer, llm,
        n_samples=20
    )

    print("\nrag_pipeline.py complete.")
    print(f"\nFiles saved to results/:")
    print(f"  preliminary_results.json")
    print(f"  preliminary_metrics.txt")
    print(f"  update_log.jsonl")


if __name__ == "__main__":
    main()