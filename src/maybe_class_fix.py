# pyright: reportMissingImports=false

"""
src/maybe_class_fix.py

Diagnoses and fixes the maybe-class F1 = 0.00 problem from evaluate.py.

Root cause (confirmed from results/final_evaluation.json's confusion
matrix): the "maybe" column is all zeros — across all 100 test samples,
beam search under max_new_tokens=5 never once generates "maybe", regardless
of the true label. Beam search picks the single highest-probability
sequence; "yes"/"no" are far more common training targets for yes/no
questions in general, so they dominate the beams even when "maybe" is the
correct answer. This is a decoding bias, not necessarily a hard model-size
ceiling — so it's tested here before concluding a bigger model is needed.

Fix tested: instead of free-form generation + string parsing, directly
score each of the 3 candidate answers via teacher-forced log-likelihood
under the model, and pick the argmax. This guarantees "maybe" is scored as
a first-class candidate on every single example, rather than needing to
win an open-ended beam search. Verified that "yes", "no", and "maybe" each
tokenize to exactly 1 content token + EOS under Flan-T5's tokenizer, so raw
summed log-likelihood is directly comparable across candidates (no length
normalization needed).

Follow-up finding: directly inspecting per-candidate log-likelihoods shows
"maybe" scores far below "yes"/"no" even when forced to compete on equal
footing — e.g. with NO input context at all, flan-t5-base's likelihood for
"maybe" is already ~-5.8 vs ~-0.2 for "yes". This means the bias isn't
purely a beam-search artifact; it's a prior baked into the model's weights.
This motivated adding contextual calibration (Zhao et al. 2021,
"Calibrate Before Use"): subtract each candidate's content-free baseline
log-likelihood before taking the argmax, isolating the actual evidence
contributed by the retrieved context from the model's intrinsic class bias.

Three conditions compared on the same 100 test samples (a planned 4th —
flan-t5-large, to test whether a bigger model helps on top of the fix — was
dropped; see the note in main() for why):
  1. flan-t5-base + beam search (original)             — reproduces the bug
  2. flan-t5-base + likelihood scoring + few-shot       — tests decoding-bias fix
  3. flan-t5-base + calibrated likelihood + few-shot    — tests prior-bias fix

Run: python src/maybe_class_fix.py
"""

import os
import sys
import json
import pickle
import random

import numpy as np
import torch
import faiss
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

INDEX_DIR     = "data/index"
RESULTS_DIR   = "results"
EMBEDDER_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
TOP_K         = 3
N_EVAL        = 100
CANDIDATES    = ["yes", "no", "maybe"]

# One worked example showing "maybe" is a legitimate, expected answer —
# the original prompt never demonstrated this, only described it.
FEWSHOT_EXAMPLE = (
    "Example Context: A small pilot study suggested a possible benefit, but "
    "the effect size was inconsistent across subgroups and the sample size "
    "was too small to draw firm conclusions.\n"
    "Example Question: Does the treatment improve outcomes?\n"
    "Example Answer: maybe\n\n"
)


# ── Loading ───────────────────────────────────────────────────────────────

def load_index_and_data():
    index = faiss.read_index(os.path.join(INDEX_DIR, "pubmedqa_pubmedbert.index"))
    with open(os.path.join(INDEX_DIR, "passages_pubmedbert.pkl"), "rb") as f:
        passages = pickle.load(f)
    with open(os.path.join("data", "processed", "test.json"), "r", encoding="utf-8") as f:
        test_data = json.load(f)
    return index, passages, test_data[:N_EVAL]


def retrieve(query, index, passages, embedder, top_k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(q_emb, top_k)
    return [passages[idx] for idx in indices[0] if idx != -1]


# ── Prompting ─────────────────────────────────────────────────────────────

def build_prompt(query, retrieved, few_shot):
    context = " ".join(f"Context {i+1}: {p['text']}" for i, p in enumerate(retrieved))
    prefix = FEWSHOT_EXAMPLE if few_shot else ""
    prompt = (
        f"{prefix}"
        f"Based on the medical context provided, answer the question "
        f"with only one word: yes, no, or maybe.\n\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer with only yes, no, or maybe:"
    )
    words = prompt.split()
    if len(words) > 450:
        context_words = context.split()
        allowed = 450 - 40 - (len(prefix.split()) if few_shot else 0)
        context = " ".join(context_words[:allowed])
        prompt = (
            f"{prefix}"
            f"Based on the medical context provided, answer the question "
            f"with only one word: yes, no, or maybe.\n\n"
            f"{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer with only yes, no, or maybe:"
        )
    return prompt


# ── Decoding strategies ───────────────────────────────────────────────────

def generate_beam(prompt, tokenizer, llm, device):
    """Original strategy: open-ended beam search + substring parsing."""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.inference_mode():
        outputs = llm.generate(
            **inputs, max_new_tokens=5, num_beams=4, no_repeat_ngram_size=2,
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).lower().strip()
    if "yes" in answer:
        return "yes"
    elif "no" in answer:
        return "no"
    elif "maybe" in answer:
        return "maybe"
    return answer


def score_candidates(prompt, tokenizer, llm, device, candidate_ids):
    """Returns {candidate: summed teacher-forced log-likelihood}."""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    scores = {}
    with torch.inference_mode():
        for candidate, target_ids in candidate_ids.items():
            labels = target_ids.unsqueeze(0)  # (1, seq_len)
            out = llm(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
            scores[candidate] = -out.loss.item() * labels.shape[1]
    return scores


NEUTRAL_PROMPT = (
    "Based on the medical context provided, answer the question "
    "with only one word: yes, no, or maybe.\n\n"
    "Context 1: N/A\n\n"
    "Question: N/A\n\n"
    "Answer with only yes, no, or maybe:"
)


def generate_likelihood(prompt, tokenizer, llm, device, candidate_ids):
    """Fix #1: score each candidate's raw log-likelihood and argmax."""
    scores = score_candidates(prompt, tokenizer, llm, device, candidate_ids)
    return max(scores, key=scores.get)


def generate_likelihood_calibrated(prompt, tokenizer, llm, device, candidate_ids, baseline_scores, alpha=1.0):
    """
    Fix #2: subtract alpha * each candidate's content-free baseline
    log-likelihood before taking the argmax (contextual calibration,
    Zhao et al. 2021). alpha=1.0 is full calibration; alpha=0.0 is
    equivalent to generate_likelihood (no correction at all).
    """
    scores = score_candidates(prompt, tokenizer, llm, device, candidate_ids)
    adjusted = {c: scores[c] - alpha * baseline_scores[c] for c in scores}
    return max(adjusted, key=adjusted.get)


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(predictions, references):
    """
    Reports the full per-class precision/recall/F1, not just F1 — a fix
    that raises maybe-F1 by sacrificing yes-recall is a real trade-off, and
    hiding that behind a single macro-F1 number would be misleading.
    """
    accuracy = accuracy_score(references, predictions)
    report = classification_report(
        references, predictions, labels=CANDIDATES, zero_division=0, output_dict=True
    )
    metrics = {"accuracy": round(accuracy * 100, 1), "macro_f1": round(report["macro avg"]["f1-score"], 4)}
    for label in CANDIDATES:
        metrics[f"{label}_precision"] = round(report[label]["precision"], 4)
        metrics[f"{label}_recall"] = round(report[label]["recall"], 4)
        metrics[f"{label}_f1"] = round(report[label]["f1-score"], 4)
        # Support (n) matters here: maybe support is only 17/100, so an F1
        # jump from 0.00 to 0.32 is a handful of examples flipping, not a
        # large-sample trend — keep the denominator visible, not just the F1.
        metrics[f"{label}_support"] = int(report[label]["support"])
    metrics["confusion_matrix"] = classification_report(
        references, predictions, labels=CANDIDATES, zero_division=0
    )
    return metrics


# ── Conditions ────────────────────────────────────────────────────────────

def run_condition(name, test_data, index, passages, embedder, tokenizer, llm,
                   device, decode="beam", few_shot=False, candidate_ids=None,
                   baseline_scores=None, alpha=1.0):
    print(f"\n-- {name} --")

    predictions, references = [], []
    for sample in tqdm(test_data, desc=f"  {name}"):
        retrieved = retrieve(sample["question"], index, passages, embedder)
        prompt = build_prompt(sample["question"], retrieved, few_shot)
        if decode == "beam":
            pred = generate_beam(prompt, tokenizer, llm, device)
        elif decode == "likelihood":
            pred = generate_likelihood(prompt, tokenizer, llm, device, candidate_ids)
        else:
            pred = generate_likelihood_calibrated(
                prompt, tokenizer, llm, device, candidate_ids, baseline_scores, alpha=alpha
            )
        predictions.append(pred)
        references.append(sample["label"].lower().strip())

    metrics = compute_metrics(predictions, references)
    print(f"  Accuracy={metrics['accuracy']}%  Macro F1={metrics['macro_f1']}  "
          f"Yes F1={metrics['yes_f1']}  No F1={metrics['no_f1']}  Maybe F1={metrics['maybe_f1']}")
    return metrics, predictions


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading embedder: {EMBEDDER_NAME}")
    embedder = SentenceTransformer(EMBEDDER_NAME, device=str(device))

    index, passages, test_data = load_index_and_data()
    references_gt = [s["label"].lower().strip() for s in test_data]
    print(f"Test samples: {len(test_data)}")

    results = {}

    # Condition 1 — Flan-T5-base, beam search (reproduce the original bug)
    # NOTE: uses few_shot=False (the original prompt), while every condition
    # below uses few_shot=True. This is a real confound: the ~3pt accuracy
    # drop from condition 1 to condition 2 is NOT purely "beam vs likelihood
    # decoding" — some of it is the few-shot example itself shifting the
    # model's answer distribution (raw likelihood's recall shifts hard
    # toward "no": 0.41->0.62, while "yes" recall drops 0.48->0.24). A clean
    # ablation would add a beam+few-shot cell to isolate the two variables;
    # not done here, but flagged so the comparison isn't read as apples-to-apples.
    print("\nLoading google/flan-t5-base...")
    tok_base = T5Tokenizer.from_pretrained("google/flan-t5-base")
    llm_base = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)
    llm_base.eval()

    metrics, preds = run_condition(
        "Flan-T5-base + beam search (baseline)",
        test_data, index, passages, embedder, tok_base, llm_base, device,
        decode="beam", few_shot=False,
    )
    results["base_beam_baseline"] = {"metrics": metrics, "predictions": preds}

    # Shared setup for likelihood-based conditions — compute once, reuse
    # across the raw-likelihood condition and the whole alpha sweep below.
    candidate_ids = {
        c: tok_base(c, return_tensors="pt").input_ids[0].to(device)
        for c in CANDIDATES
    }
    baseline_scores = score_candidates(NEUTRAL_PROMPT, tok_base, llm_base, device, candidate_ids)
    print(f"\nContent-free baseline log-likelihoods: {baseline_scores}")

    # Condition 2 — Flan-T5-base, raw likelihood scoring + few-shot
    # (equivalent to alpha=0.0 in the sweep below — no calibration at all)
    metrics, preds = run_condition(
        "Flan-T5-base + likelihood scoring + few-shot",
        test_data, index, passages, embedder, tok_base, llm_base, device,
        decode="likelihood", few_shot=True, candidate_ids=candidate_ids,
    )
    results["base_likelihood_fewshot"] = {"metrics": metrics, "predictions": preds}

    # Conditions 3a-3d — calibration strength sweep. alpha=1.0 (full
    # calibration) is the one extreme operating point; sweeping shows the
    # actual trade-off curve between maybe-F1 and yes-recall instead of
    # reporting only the harshest correction.
    for alpha in (0.25, 0.5, 0.75, 1.0):
        metrics, preds = run_condition(
            f"Flan-T5-base + calibrated likelihood (alpha={alpha}) + few-shot",
            test_data, index, passages, embedder, tok_base, llm_base, device,
            decode="likelihood_calibrated", few_shot=True,
            candidate_ids=candidate_ids, baseline_scores=baseline_scores, alpha=alpha,
        )
        results[f"base_calibrated_alpha_{alpha}"] = {"metrics": metrics, "predictions": preds}

    del llm_base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Flan-T5-large was planned as a 4th condition (does a bigger model help
    # on top of the calibration fix?) but is dropped here: Hugging Face's CDN
    # was unreliable in this environment — 5 download attempts (10 min each,
    # resuming from scratch each time rather than from the partial bytes)
    # all timed out without completing the ~3GB download. The 3 conditions
    # below on Flan-T5-base already show the full diagnose-and-fix arc.

    # ── Save ──────────────────────────────────────────────────────────────
    out = {
        "ground_truth": references_gt,
        "conditions": {
            k: {"metrics": v["metrics"], "predictions": v["predictions"]}
            for k, v in results.items()
        },
    }
    out_path = os.path.join(RESULTS_DIR, "maybe_class_fix_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    header = f"{'Condition':<48}{'Acc%':>7}{'MacroF1':>9}{'YesF1':>8}{'NoF1':>8}{'MaybeF1':>9}{'YesRec':>8}"

    def row(name, m):
        return (f"{name:<48}{m['accuracy']:>7}{m['macro_f1']:>9}{m['yes_f1']:>8}"
                f"{m['no_f1']:>8}{m['maybe_f1']:>9}{m['yes_recall']:>8}")

    summary_path = os.path.join(RESULTS_DIR, "maybe_class_fix_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Maybe-Class F1 Fix — Calibration Strength Sweep\n")
        f.write("=" * 100 + "\n\n")
        f.write(header + "\n")
        f.write("-" * 100 + "\n")
        for name, v in results.items():
            f.write(row(name, v["metrics"]) + "\n")
        f.write(
            "\nYesRec = yes-class recall, included alongside MaybeF1 to make the "
            "trade-off visible: full calibration (alpha=1.0) maximizes maybe-F1 "
            "but at the cost of yes-recall. Pick the alpha that best matches the "
            "deployment's tolerance for false negatives on \"yes\" vs missing "
            "\"maybe\" entirely.\n"
        )
        for name, v in results.items():
            f.write(f"\n{name} — per-class report:\n{v['metrics']['confusion_matrix']}\n")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(header)
    for name, v in results.items():
        print(row(name, v["metrics"]))
    print(f"\nSaved -> {out_path}")
    print(f"Saved -> {summary_path}")


if __name__ == "__main__":
    main()
