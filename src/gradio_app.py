# pyright: reportMissingImports=false

"""
src/gradio_app.py

Gradio demo for the Clinical RAG system — built for HF Spaces deployment.

Generation uses the calibrated-likelihood decoder from src/decoding.py
(validated in src/maybe_class_fix.py) instead of the original beam search,
since beam search never produced a "maybe" answer on any test example.

Run locally : python src/gradio_app.py
Spaces entry: the repo-root app.py wraps this module's `demo`.
"""

import os
import sys
import json
import pickle
import datetime

import torch
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoding import (
    CANDIDATES,
    build_prompt,
    get_candidate_ids,
    get_baseline_scores,
    generate_calibrated,
)

INDEX_DIR  = "data/index"
EMBEDDER_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
LLM_NAME      = "google/flan-t5-base"
TOP_K_DEFAULT = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load everything once at process start ──────────────────────────────────
_faiss_path = os.path.join(INDEX_DIR, "pubmedqa_pubmedbert.index")
_passages_path = os.path.join(INDEX_DIR, "passages_pubmedbert.pkl")

if not (os.path.exists(_faiss_path) and os.path.exists(_passages_path)):
    # The index/passage store are gitignored (binary, derived data) — a
    # fresh clone (e.g. a new HF Space) needs to build them once on first
    # boot from data/processed/*.json, which IS committed.
    print("FAISS index not found — building it from data/processed/*.json...")
    import build_index
    build_index.main()

print("Loading FAISS index and passages...")
_index = faiss.read_index(_faiss_path)
with open(_passages_path, "rb") as f:
    _passages = pickle.load(f)

print(f"Loading embedder: {EMBEDDER_NAME}")
_embedder = SentenceTransformer(EMBEDDER_NAME, device=str(DEVICE))

print(f"Loading generator: {LLM_NAME}")
_tokenizer = T5Tokenizer.from_pretrained(LLM_NAME)
_llm = T5ForConditionalGeneration.from_pretrained(LLM_NAME).to(DEVICE)
_llm.eval()

_candidate_ids = get_candidate_ids(_tokenizer, DEVICE)
_baseline_scores = get_baseline_scores(_tokenizer, _llm, DEVICE, _candidate_ids)
print(f"Content-free baseline log-likelihoods: {_baseline_scores}")
print(f"Ready. Index has {_index.ntotal} passages.")


def _retrieve(query, top_k):
    q_emb = _embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = _index.search(q_emb, top_k)
    return [
        (_passages[idx], float(dist))
        for dist, idx in zip(distances[0], indices[0]) if idx != -1
    ]


def answer_question(query, top_k):
    if not query or not query.strip():
        return "—", {}, "Enter a question first."

    retrieved = _retrieve(query, int(top_k))
    prompt = build_prompt(query, [p for p, _ in retrieved], few_shot=True)
    best, raw_scores, calibrated_scores = generate_calibrated(
        prompt, _tokenizer, _llm, DEVICE, _candidate_ids, _baseline_scores
    )

    # Gradio's gr.Label wants nonnegative weights — softmax the calibrated
    # scores just for display, the argmax decision above already happened.
    import math
    exp_scores = {c: math.exp(v) for c, v in calibrated_scores.items()}
    total = sum(exp_scores.values())
    confidence = {c: exp_scores[c] / total for c in CANDIDATES}

    passages_md = "\n\n".join(
        f"**[{i+1}] similarity={score:.3f}** (source: {p.get('source', p.get('source_id', 'unknown'))})\n\n{p['text'][:500]}"
        for i, (p, score) in enumerate(retrieved)
    )

    return best.upper(), confidence, passages_md


def add_document(text, source):
    if not text or not text.strip():
        return f"Index has {_index.ntotal} passages. Enter some text to add."

    words, chunks, start = text.split(), [], 0
    while start < len(words):
        end = min(start + 150, len(words))
        chunks.append(" ".join(words[start:end]))
        start += 120
    if not chunks:
        return f"Index has {_index.ntotal} passages. Nothing to add."

    embeddings = _embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    _index.add(embeddings)
    timestamp = datetime.datetime.now().isoformat()
    for i, chunk in enumerate(chunks):
        _passages.append({
            "text": chunk,
            "source": source.strip() if source and source.strip() else "manual_demo",
            "date": timestamp,
            "chunk_idx": i,
            "label": "updated",
            "split": "live_update",
        })

    # In-memory only — the hosted demo is multi-user, so we don't persist
    # this to disk (that would let one visitor's input affect everyone else's).
    return f"Added {len(chunks)} chunks. Index now has {_index.ntotal} passages (session-only, not saved to disk)."


with gr.Blocks(title="Clinical RAG — PubMedQA") as demo:
    gr.Markdown(
        "# 🏥 Clinical RAG — PubMedQA\n"
        "Retrieval-augmented biomedical QA: PubMedBERT embeddings + FAISS retrieval "
        "+ Flan-T5-base generation, with calibrated-likelihood decoding to fix a "
        "yes/no answer bias (see the **About** tab for details)."
    )

    with gr.Tab("Ask a Question"):
        with gr.Row():
            with gr.Column(scale=2):
                query_box = gr.Textbox(
                    label="Clinical question",
                    placeholder="e.g. Does smoking increase the risk of lung cancer?",
                    lines=2,
                )
                top_k_slider = gr.Slider(1, 5, value=TOP_K_DEFAULT, step=1, label="Top-K passages to retrieve")
                ask_btn = gr.Button("Get Answer", variant="primary")
            with gr.Column(scale=1):
                answer_box = gr.Textbox(label="Answer", interactive=False)
                confidence_label = gr.Label(label="Calibrated confidence")
        passages_box = gr.Markdown(label="Retrieved passages")

        ask_btn.click(
            answer_question,
            inputs=[query_box, top_k_slider],
            outputs=[answer_box, confidence_label, passages_box],
        )

    with gr.Tab("Add Knowledge"):
        gr.Markdown(
            "Inject a new document into the live FAISS index without rebuilding it — "
            "demonstrates continuous knowledge updating. **Session-only**: changes "
            "here aren't saved to disk on the hosted demo."
        )
        new_text = gr.Textbox(label="New medical text", lines=4)
        new_source = gr.Textbox(label="Source label", value="manual_demo")
        update_btn = gr.Button("Add to Index")
        update_status = gr.Textbox(label="Status", interactive=False)
        update_btn.click(add_document, inputs=[new_text, new_source], outputs=[update_status])

    with gr.Tab("About"):
        gr.Markdown(
            "## Pipeline\n"
            "- **Retrieval**: PubMedBERT (`pritamdeka/S-PubMedBert-MS-MARCO`) embeddings, "
            "FAISS `IndexFlatIP`, 768-dim, cosine similarity.\n"
            "- **Generation**: `google/flan-t5-base`, decoded via calibrated log-likelihood "
            "scoring over {yes, no, maybe} rather than free-form beam search.\n\n"
            "## Why calibrated likelihood decoding?\n"
            "The original beam-search decoder never generated \"maybe\" on any of the "
            "100 PubMedQA test examples (maybe-class F1 = 0.00), even though ~17% of "
            "test labels are \"maybe\". Directly scoring the three candidate answers "
            "showed the model assigns \"maybe\" a much lower likelihood even with *no* "
            "input at all — a prior bias from training, not just a beam-search artifact. "
            "Calibration (Zhao et al. 2021, *Calibrate Before Use*) subtracts each "
            "candidate's content-free baseline likelihood before picking the argmax, "
            "isolating the evidence the retrieved context actually contributes. "
            "See `results/maybe_class_fix_summary.txt` for the before/after comparison "
            "against this fix and against Flan-T5-large.\n\n"
            "## Evaluation\n"
            "Retrieval quality, classification metrics, and an LLM-judged "
            "faithfulness/relevance score (Gemini 2.5 Flash) are in "
            "`results/final_evaluation.json` and `results/llm_judge_results.json`."
        )


if __name__ == "__main__":
    demo.launch()
