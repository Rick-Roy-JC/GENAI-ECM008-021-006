# pyright: reportMissingImports=false

"""
src/llm_judge.py

RAGAS-style LLM-judge metrics for the Clinical RAG system, using
Gemini 2.5 Flash as the judge (free tier — no OpenAI key needed).

Computes, per sample, on a 1-5 scale:
  - Faithfulness     : is the generated answer supported by the retrieved
                        context, or does it contradict / go beyond it?
  - Answer Relevance  : does the generated answer actually address the
                        question asked?

These mirror RAGAS's `faithfulness` and `answer_relevancy` metrics, but are
computed with a direct Gemini call + structured JSON output instead of
RAGAS's LangChain wrapper, since RAGAS's default judges are OpenAI-shaped.

Alignment note: this script re-derives retrieved context for each sample
itself (using the same FAISS index + embedder + retrieve() logic as
evaluate.py) and reuses the "predictions" list already saved in
results/final_evaluation.json. Both scripts slice test_data the same way
(test_data[:N_EVAL], no shuffling), so index i in final_evaluation.json's
predictions corresponds to eval_data[i] here.

Run: python src/llm_judge.py
"""

import os
import sys
import json
import time
import random
import pickle
import winreg

import numpy as np
import faiss
from tqdm import tqdm
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError

random.seed(42)
np.random.seed(42)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

INDEX_DIR    = "data/index"
RESULTS_DIR  = "results"
EMBEDDER_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
TOP_K        = 3
JUDGE_MODEL  = "gemini-2.5-flash"
# Gemini free tier for gemini-2.5-flash is 5 requests/minute (observed from
# 429 responses) and a limited daily quota — 25 samples keeps a full run
# (~6 minutes incl. pacing) safely within that without risking a mid-run
# quota exhaustion that would waste partial progress.
N_JUDGE      = 25


# ── API key resolution ───────────────────────────────────────────────────

def get_gemini_api_key():
    """
    Prefer the env var. Fall back to the Windows User-level registry value,
    since `setx` writes there but doesn't propagate to already-running
    parent processes until they restart.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        reg = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment")
        value, _ = winreg.QueryValueEx(reg, "GEMINI_API_KEY")
        return value
    except OSError:
        return None


# ── Structured judge output ──────────────────────────────────────────────

class JudgeScore(BaseModel):
    faithfulness: int      # 1-5: is the answer grounded in the given context?
    relevance: int         # 1-5: does the answer address the question?
    reasoning: str         # one-sentence justification


JUDGE_PROMPT = """You are evaluating a biomedical question-answering system.

Question: {question}

Retrieved context the system had access to:
{context}

System's answer: {answer}

Rate the system's answer on two dimensions, each from 1 (worst) to 5 (best):

1. Faithfulness: Is the answer consistent with and supported by the retrieved
   context? An answer that contradicts the context, or makes claims the
   context doesn't support, should score low. An answer that is clearly
   grounded in the context should score high.
2. Relevance: Does the answer actually address the question asked? A
   one-word yes/no/maybe answer can still be fully relevant if it correctly
   answers a yes/no/maybe question.

Respond with the structured fields only."""


def _extract_retry_delay_seconds(error):
    """Gemini's 429 responses include an explicit retryDelay — honor it
    instead of guessing, since the free tier's RPM limit is tight (5/min)."""
    try:
        for detail in error.details.get("error", {}).get("details", []):
            if detail.get("@type", "").endswith("RetryInfo"):
                delay_str = detail["retryDelay"]   # e.g. "39s"
                return float(delay_str.rstrip("s"))
    except (AttributeError, KeyError, ValueError):
        pass
    return None


def judge_one(client, question, context, answer, max_retries=6):
    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=JudgeScore,
                    temperature=0.0,
                ),
            )
            return resp.parsed
        except (ServerError, ClientError) as e:
            wait = _extract_retry_delay_seconds(e)
            if wait is None:
                wait = min(15 * (attempt + 1), 60)   # 503s have no retryDelay
            print(f"    Gemini error ({e.__class__.__name__}), retrying in {wait:.0f}s...")
            time.sleep(wait)
    raise RuntimeError(f"Gemini judge failed after {max_retries} retries")


# ── Retrieval (mirrors evaluate.py) ──────────────────────────────────────

def retrieve(query, index, passages, embedder, top_k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(q_emb, top_k)
    return [
        passages[idx] for idx in indices[0] if idx != -1
    ]


def load_eval_inputs():
    faiss_path = os.path.join(INDEX_DIR, "pubmedqa_pubmedbert.index")
    passages_path = os.path.join(INDEX_DIR, "passages_pubmedbert.pkl")
    index = faiss.read_index(faiss_path)
    with open(passages_path, "rb") as f:
        passages = pickle.load(f)

    print(f"Loading embedder : {EMBEDDER_NAME}")
    embedder = SentenceTransformer(EMBEDDER_NAME)

    with open(os.path.join("data", "processed", "test.json"), "r", encoding="utf-8") as f:
        test_data = json.load(f)

    eval_results_path = os.path.join(RESULTS_DIR, "final_evaluation.json")
    if not os.path.isfile(eval_results_path):
        raise FileNotFoundError(
            f"Missing {eval_results_path}. Run src/evaluate.py first — "
            "the judge reuses its predictions to stay consistent."
        )
    with open(eval_results_path, "r", encoding="utf-8") as f:
        eval_results = json.load(f)

    n_eval = eval_results["n_evaluated"]
    eval_data = test_data[:n_eval]
    predictions = eval_results["predictions"]

    if len(predictions) != len(eval_data):
        raise ValueError(
            f"predictions ({len(predictions)}) and eval_data ({len(eval_data)}) "
            "length mismatch — final_evaluation.json may be stale, re-run evaluate.py."
        )

    return index, passages, embedder, eval_data, predictions


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found in environment or User registry. "
            "Set it with: setx GEMINI_API_KEY \"your-key\" and restart the shell."
        )
    client = genai.Client(api_key=api_key)

    index, passages, embedder, eval_data, predictions = load_eval_inputs()

    n = min(N_JUDGE, len(eval_data))
    # Deterministic subsample, not just the first N, so it isn't biased
    # toward whatever ordering test.json happens to have.
    sample_indices = sorted(random.sample(range(len(eval_data)), n))

    # Resume support: Gemini's free tier is rate/quota limited tightly
    # enough that one run can get throttled mid-way. Re-running this script
    # picks up from the last checkpoint instead of re-spending API calls
    # on samples already judged.
    checkpoint_path = os.path.join(RESULTS_DIR, "llm_judge_results.json")
    per_sample = []
    if os.path.isfile(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            prior = json.load(f)
        if prior.get("judge_model") == JUDGE_MODEL and prior.get("n_planned") == n:
            per_sample = prior["per_sample"]
            print(f"Resuming from checkpoint — {len(per_sample)}/{n} already judged.")

    already_judged = {s["index"] for s in per_sample}
    remaining_indices = [i for i in sample_indices if i not in already_judged]

    print(f"\nJudging {len(remaining_indices)} remaining of {n} planned samples with {JUDGE_MODEL}...")

    for i in tqdm(remaining_indices, desc="  LLM judge"):
        sample = eval_data[i]
        answer = predictions[i]
        retrieved = retrieve(sample["question"], index, passages, embedder)
        context = "\n".join(f"[{j+1}] {p['text']}" for j, p in enumerate(retrieved))

        score = judge_one(client, sample["question"], context, answer)
        per_sample.append({
            "index": i,
            "question": sample["question"],
            "answer": answer,
            "ground_truth_label": sample["label"],
            "faithfulness": score.faithfulness,
            "relevance": score.relevance,
            "reasoning": score.reasoning,
        })

        # Checkpoint after every sample — a mid-run quota exhaustion
        # shouldn't throw away the judgments already paid for in API calls.
        checkpoint_path = os.path.join(RESULTS_DIR, "llm_judge_results.json")
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({
                "judge_model": JUDGE_MODEL,
                "n_judged": len(per_sample),
                "n_planned": n,
                "mean_faithfulness": round(float(np.mean([s["faithfulness"] for s in per_sample])), 3),
                "mean_relevance": round(float(np.mean([s["relevance"] for s in per_sample])), 3),
                "per_sample": per_sample,
            }, f, indent=2, ensure_ascii=False)

        time.sleep(13.0)  # free tier is 5 requests/minute for gemini-2.5-flash

    mean_faithfulness = round(float(np.mean([s["faithfulness"] for s in per_sample])), 3)
    mean_relevance = round(float(np.mean([s["relevance"] for s in per_sample])), 3)

    summary = {
        "judge_model": JUDGE_MODEL,
        "n_judged": n,
        "mean_faithfulness": mean_faithfulness,
        "mean_relevance": mean_relevance,
        "per_sample": per_sample,
    }

    out_path = os.path.join(RESULTS_DIR, "llm_judge_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nMean faithfulness (1-5) : {mean_faithfulness}")
    print(f"Mean relevance (1-5)    : {mean_relevance}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
