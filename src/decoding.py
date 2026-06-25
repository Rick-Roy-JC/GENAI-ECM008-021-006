# pyright: reportMissingImports=false

"""
src/decoding.py

Shared prompt + decoding logic for the Clinical RAG generator, used by
both src/maybe_class_fix.py (the experiment that validated this approach)
and src/gradio_app.py (the demo, which serves the validated fix rather
than the original beam-search decoder).

See src/maybe_class_fix.py's module docstring for the full diagnosis:
Flan-T5-base's beam search never emits "maybe", and even directly scoring
"maybe" as a candidate shows the model assigns it a much lower prior
likelihood than "yes"/"no" — so this module scores all 3 candidates via
teacher-forced log-likelihood and calibrates out that prior bias by
subtracting each candidate's content-free baseline likelihood
(Zhao et al. 2021, "Calibrate Before Use").
"""

import torch

CANDIDATES = ["yes", "no", "maybe"]

# One worked example showing "maybe" is a legitimate, expected answer —
# the original prompt only described it as an option, never demonstrated it.
FEWSHOT_EXAMPLE = (
    "Example Context: A small pilot study suggested a possible benefit, but "
    "the effect size was inconsistent across subgroups and the sample size "
    "was too small to draw firm conclusions.\n"
    "Example Question: Does the treatment improve outcomes?\n"
    "Example Answer: maybe\n\n"
)

NEUTRAL_PROMPT = (
    "Based on the medical context provided, answer the question "
    "with only one word: yes, no, or maybe.\n\n"
    "Context 1: N/A\n\n"
    "Question: N/A\n\n"
    "Answer with only yes, no, or maybe:"
)


def build_prompt(query, retrieved, few_shot=True):
    """retrieved: list of passage dicts with a 'text' key (no scores needed)."""
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


def get_candidate_ids(tokenizer, device):
    return {c: tokenizer(c, return_tensors="pt").input_ids[0].to(device) for c in CANDIDATES}


def score_candidates(prompt, tokenizer, llm, device, candidate_ids):
    """Returns {candidate: summed teacher-forced log-likelihood}."""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    scores = {}
    with torch.inference_mode():
        for candidate, target_ids in candidate_ids.items():
            labels = target_ids.unsqueeze(0)
            out = llm(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
            scores[candidate] = -out.loss.item() * labels.shape[1]
    return scores


def get_baseline_scores(tokenizer, llm, device, candidate_ids):
    return score_candidates(NEUTRAL_PROMPT, tokenizer, llm, device, candidate_ids)


def generate_calibrated(prompt, tokenizer, llm, device, candidate_ids, baseline_scores, alpha=1.0):
    """
    Returns (best_candidate, raw_scores, calibrated_scores).

    Calibration: subtract alpha * each candidate's content-free baseline
    log-likelihood before taking the argmax, isolating (some or all of) the
    evidence the retrieved context actually contributes from the model's
    intrinsic class bias.

    alpha=0.0 is raw likelihood (no calibration); alpha=1.0 is full
    calibration (Zhao et al. 2021). Intermediate values trade off how much
    of the "maybe" bias gets corrected against how much "yes" recall gets
    sacrificed — see src/maybe_class_fix.py's alpha sweep for the curve.
    """
    raw_scores = score_candidates(prompt, tokenizer, llm, device, candidate_ids)
    calibrated = {c: raw_scores[c] - alpha * baseline_scores[c] for c in raw_scores}
    best = max(calibrated, key=calibrated.get)
    return best, raw_scores, calibrated
