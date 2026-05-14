# pyright: reportMissingImports=false

"""
src/evaluate.py

Full evaluation suite for the Clinical RAG system.
Computes:
  - Exact match accuracy
  - Per-class F1, Precision, Recall
  - ROUGE-L score
  - Retrieval quality (avg similarity score)
  - Confusion matrix

Run: python src/evaluate.py
"""

import os
import re
import json
import pickle
import random
import numpy as np
import torch
import faiss
from tqdm import tqdm
from rouge_score import rouge_scorer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Fix seeds
random.seed(42)
np.random.seed(42)

# Paths
INDEX_DIR   = "data/index"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Settings — must match rag_pipeline.py exactly
EMBEDDER_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
LLM_NAME      = "google/flan-t5-base"
TOP_K         = 3
N_EVAL        = 100    # evaluate on 100 samples for final — more than M1's 20


# ── Loading ───────────────────────────────────────────────────────────────

def load_everything():
    print("Loading index and passages...")
    faiss_path = os.path.join(INDEX_DIR, "pubmedqa_pubmedbert.index")
    passages_path = os.path.join(INDEX_DIR, "passages_pubmedbert.pkl")
    if not os.path.isfile(faiss_path):
        raise FileNotFoundError(
            f"Missing {faiss_path}. Run src/build_index.py first."
        )
    if not os.path.isfile(passages_path):
        raise FileNotFoundError(
            f"Missing {passages_path}. Run src/build_index.py first."
        )

    index = faiss.read_index(faiss_path)
    with open(passages_path, "rb") as f:
        passages = pickle.load(f)
    print(f"  Index size : {index.ntotal} vectors")
    print(f"  Passages   : {len(passages)}")

    print(f"\nLoading embedder : {EMBEDDER_NAME}")
    embedder = SentenceTransformer(EMBEDDER_NAME)

    print(f"Loading LLM      : {LLM_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(LLM_NAME)
    llm       = T5ForConditionalGeneration.from_pretrained(LLM_NAME)

    print("\nLoading test data...")
    test_path = os.path.join("data", "processed", "test.json")
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Missing {test_path}. Run preprocessing first.")
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"  Test samples : {len(test_data)}")

    return index, passages, embedder, tokenizer, llm, test_data


# ── Retrieval ─────────────────────────────────────────────────────────────

def retrieve(query, index, passages, embedder, top_k=TOP_K):
    q_emb = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    distances, indices = index.search(q_emb, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            results.append((passages[idx], float(dist)))
    return results


# ── Generation ────────────────────────────────────────────────────────────

def build_prompt(query, retrieved):
    """Same truncation policy as rag_pipeline.build_prompt (512-token budget)."""
    context_parts = [
        f"Context {i+1}: {p['text']}"
        for i, (p, _) in enumerate(retrieved)
    ]
    context = " ".join(context_parts)

    prompt = (
        f"Based on the medical context provided, answer the question "
        f"with only one word: yes, no, or maybe.\n\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer with only yes, no, or maybe:"
    )

    words = prompt.split()
    if len(words) > 450:
        context_words = context.split()
        allowed = 450 - 40
        context = " ".join(context_words[:allowed])
        prompt = (
            f"Based on the medical context provided, answer the question "
            f"with only one word: yes, no, or maybe.\n\n"
            f"{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer with only yes, no, or maybe:"
        )
    return prompt


def parse_yes_no_maybe(answer_raw: str) -> str:
    """
    Map free-form model output to exactly one of yes / no / maybe.
    Avoids substring bugs (e.g. 'unknown' matching 'no').
    """
    t = answer_raw.lower().strip()
    for token in re.split(r"[\s,;.]+", t):
        if token in ("yes", "no", "maybe"):
            return token
    if "maybe" in t:
        return "maybe"
    if re.search(r"\byes\b", t):
        return "yes"
    if re.search(r"\bno\b", t):
        return "no"
    return "maybe"


def generate_answer(prompt, tokenizer, llm):
    device = next(llm.parameters()).device
    llm.eval()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=5,
            num_beams=4,
            no_repeat_ngram_size=2,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return parse_yes_no_maybe(answer)


# ── ROUGE-L ───────────────────────────────────────────────────────────────

def compute_rouge_l(predictions, references):
    """
    Computes average ROUGE-L score between predicted and
    reference answers.
    ROUGE-L measures longest common subsequence overlap.
    """
    scorer  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores  = []

    for pred, ref in zip(predictions, references):
        if not ref.strip():
            continue
        score = scorer.score(ref, pred)
        scores.append(score["rougeL"].fmeasure)

    return round(float(np.mean(scores)), 4) if scores else 0.0


# ── Retrieval Quality ─────────────────────────────────────────────────────

def compute_retrieval_quality(all_retrieved_scores):
    """
    Average top-1 similarity score across all queries.
    Higher = retrieval is more confident and relevant.
    """
    nonempty = [s for s in all_retrieved_scores if s]
    if not nonempty:
        return {
            "mean_top1_score": 0.0,
            "mean_top3_score": 0.0,
            "min_top1_score": 0.0,
            "max_top1_score": 0.0,
        }
    top1_scores = [scores[0] for scores in nonempty]
    top3_means = [float(np.mean(scores)) for scores in nonempty]
    return {
        "mean_top1_score": round(float(np.mean(top1_scores)), 4),
        "mean_top3_score": round(float(np.mean(top3_means)), 4),
        "min_top1_score": round(float(np.min(top1_scores)), 4),
        "max_top1_score": round(float(np.max(top1_scores)), 4),
    }


# ── Main Evaluation ───────────────────────────────────────────────────────

def main():

    # Load everything
    index, passages, embedder, tokenizer, llm, test_data = load_everything()

    # Use up to N_EVAL samples (fewer if test split is smaller)
    eval_data = test_data[:N_EVAL]
    n_run = len(eval_data)
    print(f"\nRunning evaluation on {n_run} samples...")
    print("(This takes ~5 minutes with Flan-T5-Base)\n")

    # Storage
    predictions      = []
    references       = []
    long_predictions = []
    long_references  = []
    all_scores       = []

    for sample in tqdm(eval_data, desc="  Evaluating"):

        query    = sample["question"]
        gt_label = parse_yes_no_maybe(sample.get("label", "maybe"))
        gt_answer = sample.get("answer", "").strip()

        # Retrieve
        retrieved = retrieve(query, index, passages, embedder)
        scores    = [score for _, score in retrieved]
        all_scores.append(scores)

        # Generate
        prompt = build_prompt(query, retrieved)
        pred   = generate_answer(prompt, tokenizer, llm)

        predictions.append(pred)
        references.append(gt_label)

        # For ROUGE-L use long answer vs generated answer
        long_predictions.append(pred)
        long_references.append(gt_answer if gt_answer else gt_label)

    # ── Compute all metrics ───────────────────────────────────────────────
    print("\nComputing metrics...\n")

    # 1. Exact match accuracy
    accuracy = accuracy_score(references, predictions)

    # 2. Per-class classification report
    labels = ["yes", "no", "maybe"]
    report = classification_report(
        references, predictions,
        labels=labels,
        zero_division=0,
        output_dict=True
    )
    report_str = classification_report(
        references, predictions,
        labels=labels,
        zero_division=0
    )

    # 3. ROUGE-L
    rouge_l = compute_rouge_l(long_predictions, long_references)

    # 4. Retrieval quality
    retrieval_quality = compute_retrieval_quality(all_scores)

    # 5. Confusion matrix
    cm = confusion_matrix(references, predictions, labels=labels)

    # ── Print results ─────────────────────────────────────────────────────
    print("=" * 55)
    print("  FINAL EVALUATION RESULTS")
    print("  CS5202 — Clinical RAG with Knowledge Updating")
    print("=" * 55)

    print(f"\nExact Match Accuracy : {accuracy*100:.1f}%")
    print(f"ROUGE-L Score        : {rouge_l:.4f}")
    print(f"\nRetrieval Quality:")
    print(f"  Mean top-1 score   : {retrieval_quality['mean_top1_score']}")
    print(f"  Mean top-3 score   : {retrieval_quality['mean_top3_score']}")

    print(f"\nPer-Class Results:")
    print(report_str)

    print(f"Confusion Matrix (rows=actual, cols=predicted):")
    print(f"           yes    no   maybe")
    for i, label in enumerate(labels):
        print(f"  {label:<6}  {cm[i]}")

    # ── Save all results ──────────────────────────────────────────────────
    full_results = {
        "model_embedder":     EMBEDDER_NAME,
        "model_llm":          LLM_NAME,
        "top_k":              TOP_K,
        "n_evaluated":        n_run,
        "exact_match_accuracy": round(accuracy * 100, 1),
        "rouge_l":            rouge_l,
        "retrieval_quality":  retrieval_quality,
        "per_class": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall":    round(report[label]["recall"],    4),
                "f1":        round(report[label]["f1-score"],  4),
                "support":   int(report[label]["support"])
            }
            for label in labels
        },
        "confusion_matrix": cm.tolist(),
        "predictions": predictions,
        "references":  references,
    }

    json_path = os.path.join(RESULTS_DIR, "final_evaluation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nFull results saved → {json_path}")

    # Save human readable metrics summary
    summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("CS5202 Spring 2026 — Clinical RAG Evaluation Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Embedder  : {EMBEDDER_NAME}\n")
        f.write(f"LLM       : {LLM_NAME}\n")
        f.write(f"Top-K     : {TOP_K}\n")
        f.write(f"Samples   : {n_run}\n\n")
        f.write(f"Exact Match Accuracy : {accuracy*100:.1f}%\n")
        f.write(f"ROUGE-L Score        : {rouge_l:.4f}\n\n")
        f.write(f"Retrieval Quality:\n")
        f.write(f"  Mean top-1 score   : {retrieval_quality['mean_top1_score']}\n")
        f.write(f"  Mean top-3 score   : {retrieval_quality['mean_top3_score']}\n\n")
        f.write("Per-Class Results:\n")
        f.write(report_str)
        f.write("\nConfusion Matrix (rows=actual, cols=predicted):\n")
        f.write(f"         yes    no   maybe\n")
        for i, label in enumerate(labels):
            f.write(f"  {label:<6}  {cm[i]}\n")

    print(f"Summary saved      → {summary_path}")
    print("\nevaluate.py complete ✓")


if __name__ == "__main__":
    main()