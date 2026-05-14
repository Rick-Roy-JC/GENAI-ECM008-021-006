# pyright: reportMissingImports=false

"""
src/ablation.py

Ablation study comparing three system configurations:
  1. No RAG    — LLM answers from memory, no retrieval
  2. RAG Static  — Standard RAG, fixed index, no updates
  3. RAG + Update — Full system with continuous knowledge updating

Run: python src/ablation.py
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
from sklearn.metrics import accuracy_score, classification_report
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

random.seed(42)
np.random.seed(42)

INDEX_DIR   = "data/index"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

EMBEDDER_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
LLM_NAME      = "google/flan-t5-base"
TOP_K         = 3
N_EVAL        = 100


# ── Helpers ───────────────────────────────────────────────────────────────

def load_models():
    print(f"Loading embedder : {EMBEDDER_NAME}")
    embedder  = SentenceTransformer(EMBEDDER_NAME)
    print(f"Loading LLM      : {LLM_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(LLM_NAME)
    llm       = T5ForConditionalGeneration.from_pretrained(LLM_NAME)
    return embedder, tokenizer, llm


def load_index():
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
    print(f"Index loaded — {index.ntotal} vectors")
    return index, passages


def load_test_data(n=N_EVAL):
    path = os.path.join("data", "processed", "test.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}. Run preprocessing first.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[: min(n, len(data))]


def retrieve(query, index, passages, embedder, top_k=TOP_K):
    q_emb = embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    distances, indices = index.search(q_emb, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            results.append((passages[idx], float(dist)))
    return results


def parse_yes_no_maybe(answer_raw: str) -> str:
    """
    Map free-form text to exactly one of yes / no / maybe.
    Avoids substring false positives (e.g. 'unknown' matching 'no').
    """
    t = str(answer_raw).lower().strip()
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


def build_rag_prompt(query, retrieved):
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


def generate(prompt, tokenizer, llm):
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


def compute_rouge_l(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for p, r in zip(preds, refs):
        r_str = str(r).strip()
        if not r_str:
            continue
        scores.append(scorer.score(r_str, str(p))["rougeL"].fmeasure)
    return round(float(np.mean(scores)), 4) if scores else 0.0


def compute_metrics(predictions, references):
    predictions = [parse_yes_no_maybe(p) for p in predictions]
    references = [parse_yes_no_maybe(r) for r in references]
    accuracy = accuracy_score(references, predictions)
    labels   = ["yes", "no", "maybe"]
    report   = classification_report(
        references, predictions,
        labels=labels, zero_division=0, output_dict=True
    )
    report_str = classification_report(
        references, predictions,
        labels=labels, zero_division=0
    )
    rouge_l = compute_rouge_l(predictions, references)
    return {
        "accuracy":  round(accuracy * 100, 1),
        "rouge_l":   rouge_l,
        "report":    report,
        "report_str": report_str,
        "yes_f1":    round(report["yes"]["f1-score"], 4),
        "no_f1":     round(report["no"]["f1-score"],  4),
        "maybe_f1":  round(report["maybe"]["f1-score"], 4),
        "macro_f1":  round(report["macro avg"]["f1-score"], 4),
    }


# ══════════════════════════════════════════════════════════════════════════
# CONDITION 1 — No RAG
# LLM answers purely from its own weights, no retrieved context
# ══════════════════════════════════════════════════════════════════════════

def run_no_rag(test_data, tokenizer, llm):
    print("\n── Condition 1: No RAG ──────────────────────────────")
    print("LLM answers from memory only, no retrieval\n")

    predictions = []
    references  = []

    for sample in tqdm(test_data, desc="  No RAG"):
        query = sample["question"]
        label = parse_yes_no_maybe(sample.get("label", ""))

        # No context — just the question
        prompt = (
            f"Answer this medical question with only one word: "
            f"yes, no, or maybe.\n\n"
            f"Question: {query}\n\n"
            f"Answer with only yes, no, or maybe:"
        )

        pred = generate(prompt, tokenizer, llm)
        predictions.append(pred)
        references.append(label)

    metrics = compute_metrics(predictions, references)
    print(f"  Accuracy : {metrics['accuracy']}%")
    print(f"  Macro F1 : {metrics['macro_f1']}")
    return metrics, predictions


# ══════════════════════════════════════════════════════════════════════════
# CONDITION 2 — RAG Static
# Standard RAG with fixed index — no knowledge updates
# ══════════════════════════════════════════════════════════════════════════

def run_rag_static(test_data, index, passages, embedder, tokenizer, llm):
    print("\n── Condition 2: RAG Static ──────────────────────────")
    print("Standard RAG, fixed index, no knowledge updating\n")

    predictions = []
    references  = []

    for sample in tqdm(test_data, desc="  RAG Static"):
        query = sample["question"]
        label = parse_yes_no_maybe(sample.get("label", ""))

        retrieved = retrieve(query, index, passages, embedder)
        prompt = build_rag_prompt(query, retrieved)

        pred = generate(prompt, tokenizer, llm)
        predictions.append(pred)
        references.append(label)

    metrics = compute_metrics(predictions, references)
    print(f"  Accuracy : {metrics['accuracy']}%")
    print(f"  Macro F1 : {metrics['macro_f1']}")
    return metrics, predictions


# ══════════════════════════════════════════════════════════════════════════
# CONDITION 3 — RAG + Continuous Update
# Full system: RAG + new documents added to live index
# ══════════════════════════════════════════════════════════════════════════

def run_rag_with_update(test_data, index, passages, embedder, tokenizer, llm):
    print("\n── Condition 3: RAG + Continuous Update ─────────────")
    print("Full system: RAG with live knowledge updating\n")

    # Simulate adding 3 new real medical knowledge updates
    # These represent documents NOT in the original training set
    new_documents = [
        {
            "source": "WHO_guideline_update_2025",
            "text": (
                "WHO 2025 updated guidelines state that low-dose aspirin is "
                "no longer recommended for primary prevention of cardiovascular "
                "disease in adults over 60 years due to increased bleeding risk. "
                "Secondary prevention use remains unchanged for patients with "
                "established cardiovascular disease. Clinicians should reassess "
                "patients currently on aspirin therapy for primary prevention."
            )
        },
        {
            "source": "FDA_drug_safety_update_2025",
            "text": (
                "FDA 2025 safety communication: Metformin extended-release "
                "tablets have shown improved glycemic control with reduced "
                "gastrointestinal side effects compared to immediate-release "
                "formulations. New evidence supports metformin as first-line "
                "therapy for type 2 diabetes with cardiovascular comorbidities. "
                "Dose adjustment required for patients with eGFR below 45."
            )
        },
        {
            "source": "NEJM_clinical_trial_2025",
            "text": (
                "A randomised controlled trial published in NEJM 2025 found that "
                "intensive blood pressure lowering to below 120 mmHg systolic "
                "significantly reduced major cardiovascular events compared to "
                "standard treatment targeting below 140 mmHg. Benefits were "
                "observed across age groups and diabetic status. Adverse events "
                "including hypotension were more frequent in the intensive group."
            )
        }
    ]

    # Add all new documents to the live index
    print("  Adding new knowledge documents to live index...")
    original_size = index.ntotal

    for doc in new_documents:
        # Chunk
        words  = doc["text"].split()
        chunks = []
        start  = 0
        while start < len(words):
            end   = min(start + 200, len(words))
            chunks.append(" ".join(words[start:end]))
            start += 160

        # Embed and add
        embeddings = embedder.encode(
            chunks, convert_to_numpy=True, normalize_embeddings=True
        )
        index.add(embeddings)

        for i, chunk in enumerate(chunks):
            passages.append({
                "text":   chunk,
                "source": doc["source"],
                "label":  "updated",
                "split":  "live_update"
            })

        print(f"    Added '{doc['source']}' — {len(chunks)} chunks")

    print(f"  Index size: {original_size} → {index.ntotal} vectors")

    # Now run RAG with the updated index
    predictions = []
    references  = []

    for sample in tqdm(test_data, desc="  RAG+Update"):
        query = sample["question"]
        label = parse_yes_no_maybe(sample.get("label", ""))

        retrieved = retrieve(query, index, passages, embedder)
        prompt = build_rag_prompt(query, retrieved)

        pred = generate(prompt, tokenizer, llm)
        predictions.append(pred)
        references.append(label)

    metrics = compute_metrics(predictions, references)
    print(f"  Accuracy : {metrics['accuracy']}%")
    print(f"  Macro F1 : {metrics['macro_f1']}")
    return metrics, predictions


# ══════════════════════════════════════════════════════════════════════════
# MAIN — Run all three, print comparison table
# ══════════════════════════════════════════════════════════════════════════

def main():

    # Load everything once — shared across all conditions
    print("Loading models and data...")
    embedder, tokenizer, llm = load_models()
    index, passages          = load_index()
    test_data                = load_test_data()
    n_samples                = len(test_data)

    print(f"\nEvaluating {n_samples} samples across 3 conditions...")
    print("Each condition takes ~5 minutes\n")

    # Run all three conditions
    metrics1, preds1 = run_no_rag(
        test_data, tokenizer, llm
    )
    metrics2, preds2 = run_rag_static(
        test_data, index, passages, embedder, tokenizer, llm
    )
    # Clone index + passage list so live updates do not mutate the static baseline objects.
    index_live = faiss.deserialize_index(faiss.serialize_index(index))
    passages_live = list(passages)
    metrics3, preds3 = run_rag_with_update(
        test_data, index_live, passages_live, embedder, tokenizer, llm
    )

    # ── Print comparison table ────────────────────────────────────────────
    print("\n")
    print("=" * 60)
    print("  ABLATION STUDY RESULTS")
    print("  CS5202 — Clinical RAG with Knowledge Updating")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'No RAG':>10} {'RAG Static':>12} {'RAG+Update':>12}")
    print("-" * 60)
    print(f"{'Accuracy (%)':<25} {metrics1['accuracy']:>10} {metrics2['accuracy']:>12} {metrics3['accuracy']:>12}")
    print(f"{'Macro F1':<25} {metrics1['macro_f1']:>10} {metrics2['macro_f1']:>12} {metrics3['macro_f1']:>12}")
    print(f"{'Yes F1':<25} {metrics1['yes_f1']:>10} {metrics2['yes_f1']:>12} {metrics3['yes_f1']:>12}")
    print(f"{'No F1':<25} {metrics1['no_f1']:>10} {metrics2['no_f1']:>12} {metrics3['no_f1']:>12}")
    print(f"{'Maybe F1':<25} {metrics1['maybe_f1']:>10} {metrics2['maybe_f1']:>12} {metrics3['maybe_f1']:>12}")
    print(f"{'ROUGE-L':<25} {metrics1['rouge_l']:>10} {metrics2['rouge_l']:>12} {metrics3['rouge_l']:>12}")
    print("-" * 60)

    # Improvements
    acc_improvement_rag    = round(metrics2['accuracy'] - metrics1['accuracy'], 1)
    acc_improvement_update = round(metrics3['accuracy'] - metrics2['accuracy'], 1)
    f1_improvement_rag     = round(metrics2['macro_f1'] - metrics1['macro_f1'],  4)
    f1_improvement_update  = round(metrics3['macro_f1'] - metrics2['macro_f1'],  4)

    print(f"\nRAG vs No RAG improvement     : {acc_improvement_rag:+.1f}% accuracy, {f1_improvement_rag:+.4f} macro F1")
    print(f"Update vs Static RAG improvement: {acc_improvement_update:+.1f}% accuracy, {f1_improvement_update:+.4f} macro F1")

    # ── Save results ──────────────────────────────────────────────────────
    ablation_results = {
        "conditions": {
            "no_rag":     metrics1,
            "rag_static": metrics2,
            "rag_update": metrics3,
        },
        "improvements": {
            "rag_vs_no_rag_accuracy":      acc_improvement_rag,
            "update_vs_static_accuracy":   acc_improvement_update,
            "rag_vs_no_rag_macro_f1":      f1_improvement_rag,
            "update_vs_static_macro_f1":   f1_improvement_update,
        }
    }

    # Remove non-serialisable report objects
    for cond in ablation_results["conditions"].values():
        cond.pop("report",     None)
        cond.pop("report_str", None)

    out_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\nAblation results saved → {out_path}")

    # Save readable summary
    summary_path = os.path.join(RESULTS_DIR, "ablation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("CS5202 Spring 2026 — Ablation Study\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Metric':<25} {'No RAG':>10} {'RAG Static':>12} {'RAG+Update':>12}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Accuracy (%)':<25} {metrics1['accuracy']:>10} {metrics2['accuracy']:>12} {metrics3['accuracy']:>12}\n")
        f.write(f"{'Macro F1':<25} {metrics1['macro_f1']:>10} {metrics2['macro_f1']:>12} {metrics3['macro_f1']:>12}\n")
        f.write(f"{'Yes F1':<25} {metrics1['yes_f1']:>10} {metrics2['yes_f1']:>12} {metrics3['yes_f1']:>12}\n")
        f.write(f"{'No F1':<25} {metrics1['no_f1']:>10} {metrics2['no_f1']:>12} {metrics3['no_f1']:>12}\n")
        f.write(f"{'Maybe F1':<25} {metrics1['maybe_f1']:>10} {metrics2['maybe_f1']:>12} {metrics3['maybe_f1']:>12}\n")
        f.write(f"{'ROUGE-L':<25} {metrics1['rouge_l']:>10} {metrics2['rouge_l']:>12} {metrics3['rouge_l']:>12}\n")
        f.write("-" * 60 + "\n\n")
        f.write(f"RAG vs No RAG          : {acc_improvement_rag:+.1f}% accuracy\n")
        f.write(f"Update vs Static RAG   : {acc_improvement_update:+.1f}% accuracy\n")

    print(f"Ablation summary saved → {summary_path}")
    print("\nablation.py complete ✓")


if __name__ == "__main__":
    main()