# Clinical RAG — Full Evaluation Report

## Executive Summary

- **Retrieval is strong and not the bottleneck**: Recall@1=0.96, MRR=0.975 — the system finds the right source passage almost every time.
- **Maybe-class F1 = 0.00 is a model prior, not a search bug**: beam search and raw likelihood scoring both fail to ever predict "maybe"; full contextual calibration fixes it (Maybe F1 0.0 → 0.3182) but the correction is a threshold effect, not a smooth dial — partial calibration (alpha 0.25-0.5) helps nothing.
- **The fix has a real cost**: yes-class recall collapses (0.4783 → 0.0435) under full calibration — a genuine precision/recall trade-off to weigh before deploying, not a free win.
- **LLM-judged quality (Gemini 2.5 Flash, n=25)**: mean faithfulness 2.84/5, mean relevance 4.88/5 — and faithfulness tracks correctness directly (4.273/5 on correct answers vs 1.714/5 on incorrect ones).
- **Bottom line**: this is a retrieval-solid, generation-limited system — the interesting engineering is in diagnosing *why* generation underperforms (decoding bias, model prior, prompt confounds) rather than in the retrieval stack itself.

## Retrieval Quality

- Mean top-1 cosine similarity: **0.9421**
- Mean top-3 cosine similarity: **0.9209**
- Recall@1 (gold passage retrieved): **0.96**
- Recall@3: **0.99**
- Recall@5: **0.99**
- MRR: **0.975**

Retrieval is the strongest part of this pipeline — PubMedBERT embeddings find the question's own source passage in the top-1 result 96% of the time. Generation quality, not retrieval, is the bottleneck on overall accuracy.

## Generation — Baseline (Flan-T5-base, beam search)

- Exact match accuracy: **37.0%**
- ROUGE-L: **0.0009**
- Per-class F1:
  - yes: **0.4356** (support=46)
  - no: **0.3659** (support=37)
  - maybe: **0.0** (support=17)

Maybe-class F1 is 0.00 here by design — this is the documented baseline bug, fixed and compared in the section below.

## LLM-Judged Faithfulness & Relevance (Gemini 2.5 Flash)

Judged all 25/25 planned samples with `gemini-2.5-flash`.

| Metric | Mean | Std Dev | Min | Max |
|---|---|---|---|---|
| Faithfulness (1-5) | **2.84** | 1.953 | 1 | 5 |
| Relevance (1-5) | **4.88** | 0.588 | 2 | 5 |

### Faithfulness/Relevance vs. Correctness

| Group | n | Mean Faithfulness | Mean Relevance |
|---|---|---|---|
| Correct predictions | 11 | **4.273** | 5.0 |
| Incorrect predictions | 14 | **1.714** | 4.786 |

Faithfulness tracks correctness closely: correct predictions average **4.273/5** faithfulness vs. only **1.714/5** for incorrect ones — when the model gets the answer wrong, the judge also tends to flag the generation as ungrounded in the retrieved context, not just factually off. Relevance barely differs between the two groups (5.0 vs 4.786), which makes sense for yes/no/maybe questions — a one-word answer is almost always "relevant" to the question asked even when it's the wrong word, so relevance doesn't discriminate correctness here the way faithfulness does.


Only **2 of 25** judged samples had a ground-truth label of "maybe" (random subsampling on a small judge budget from a class that's 17/100 overall) — too few to draw a separate conclusion from, but consistent with the rest of this report: both were predicted incorrectly (the judged predictions came from the original beam-search baseline, which never predicts "maybe" at all — see the calibration fix below).

## Maybe-Class F1 Fix — Calibration Strength Sweep

Test set class support (n=100): yes=46, no=37, maybe=17. Maybe support is only 17/100 — an F1 move from 0.00 to 0.32 is roughly 7 examples flipping correct, not a large-sample trend.

| Condition | Accuracy | Macro F1 | Yes F1 | No F1 | Maybe F1 | Yes Recall |
|---|---|---|---|---|---|---|
| base_beam_baseline | 37.0% | 0.2672 | 0.4356 | 0.3659 | 0.0 | 0.4783 |
| base_likelihood_fewshot | 34.0% | 0.2399 | 0.2857 | 0.434 | 0.0 | 0.2391 |
| base_calibrated_alpha_0.25 | 35.0% | 0.2196 | 0.1587 | 0.5 | 0.0 | 0.1087 |
| base_calibrated_alpha_0.5 | 35.0% | 0.2065 | 0.1034 | 0.5161 | 0.0 | 0.0652 |
| base_calibrated_alpha_0.75 | 34.0% | 0.2403 | 0.0769 | 0.4959 | 0.1481 | 0.0435 |
| base_calibrated_alpha_1.0 | 35.0% | 0.2962 | 0.08 | 0.4906 | 0.3182 | 0.0435 |

**Caveat on base_beam_baseline vs base_likelihood_fewshot (37.0% vs 34.0% accuracy):** not a clean decode-only comparison — the beam baseline uses the original prompt (no few-shot example), while every likelihood-based condition includes one. Some of the drop is the few-shot example shifting the model's answer distribution (no-recall jumps 0.41→0.62, yes-recall drops 0.48→0.24), not purely the decoding method change. Isolating the two would need a beam+few-shot cell, not run here.


Diagnosis: beam search never generated "maybe" on any test example. Directly scoring "maybe" as a candidate (raw likelihood, no calibration, alpha=0) *still* produced maybe F1 = 0.00 — confirming this is a prior bias baked into Flan-T5-base's weights, not a beam-search artifact, since the model assigns "maybe" a much lower likelihood even given zero input context.

Sweeping calibration strength alpha reveals this is **not** a smooth trade-off curve. Yes-recall degrades immediately and monotonically from the very first step (0.478 → 0.109 by alpha=0.25), but maybe-F1 stays at *exactly* 0.00 all the way through alpha=0.5 — it only becomes nonzero at alpha=0.75. Partial calibration (alpha 0.25-0.5) is strictly worse than both the baseline and full calibration: it pays the yes-recall cost without buying any maybe-class benefit at all. The correction needed behaves more like a threshold than a dial — "maybe" only starts winning the argmax once its bias is almost fully subtracted out, not gradually as alpha increases.

By macro-F1, the best operating point in this sweep is **base_calibrated_alpha_1.0** (macro F1=0.2962, maybe F1=0.3182, yes recall=0.0435) — only a modest improvement over the original beam-search baseline (macro F1=0.2672), and it gets there by nearly abandoning yes-class recall in exchange for sometimes catching "maybe". Whether that's an acceptable trade depends entirely on what a deployment values more — this sweep makes that an explicit, informed choice instead of a hidden side effect of one harsh fix.

_A further condition (Flan-T5-large, to test whether a bigger model needs this fix at all) was planned but dropped — Hugging Face's CDN was unreliable in this environment and the ~3GB download never completed across 5 attempts._

## Limitations

- **LLM-judge sample size is small (n=25 of 100 evaluated)**: Gemini's free tier (5 requests/minute, plus a low daily quota encountered mid-run) made judging the full test set impractical in this environment; `src/llm_judge.py` checkpoints and resumes so the remaining 75 could be judged given more API budget or time.
- **Single LLM judge, no inter-rater agreement check**: faithfulness and relevance scores come from one model (Gemini 2.5 Flash) with no second judge or human spot-check to validate agreement — the scores should be read as directional, not as calibrated ground truth.
- **The calibration sweep's per-class support is small**: maybe support is 17/100 in the full test set, so the headline Maybe F1 0.00→0.32 move is ~7 examples flipping correct, not a large-sample trend; the qualitative finding (it's a threshold, not a smooth trade-off) is more robust than any single decimal in the table.
- **The judge's own maybe-class subsample is tiny (n=2 of 25)**: not enough to say anything specific about judge behavior on maybe cases beyond what the main classification metrics already show.

