# GENAI-ECM008-021-006
RAG-based Clinical NLP with Continuous Knowledge Updating — CS5202 Spring 2026
# Project — RAG-based Clinical NLP with Continuous Knowledge Updating

**Course:** CS5202 — GenAI and LLM, Spring 2026  
**Domain:** Medical GenAI  
**Team Members:**
TEAM NO :2  
[Aritra Roy — SE23UECM008]
[Dheeraj Reddy — SE23UECM021]
[A sai Praneeth - SE23UECM006]

---

## Problem Summary

Clinical NLP systems degrade over time as medical knowledge evolves.
This project builds a Retrieval-Augmented Generation (RAG) pipeline over
clinical documents (discharge summaries, case reports) that can reliably
extract medical decisions and answer biomedical questions — with a
mechanism for continuous knowledge updating.

---

## Project Structure
---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Rick-Roy-JC/GENAI-ECM008-021-006.git
cd GENAI-ECM008-021-006 
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download datasets
```bash
python data/download.py
```

### 5. Run the pipeline
```bash
python src/load_data.py
python src/build_index.py
python src/rag_pipeline.py
```
### Results
Loading FAISS index...
  Vectors in index : 5200
Loading passage store...
  Total passages   : 5200

Loading embedding model : all-MiniLM-L6-v2
Loading LLM             : google/flan-t5-base

── Single Query Test ────────────────────────────────
Question : Does aspirin reduce the risk of cardiovascular disease?
Answer   : Yes, aspirin reduces platelet aggregation and lowers risk

── Continuous Knowledge Update Demo ─────────────────
  Source : WHO_guideline_2025
  Chunks : 5
  Added 5 chunks to index
  Index size now : 5205 vectors
  Update complete ✓

── Query After Knowledge Update ─────────────────────
Answer   : Updated guidelines suggest aspirin not recommended for primary prevention

── Milestone 1 Evaluation ───────────────────────────
Evaluating: 100%|████████| 20/20
Label match accuracy : 45.0%

Files saved to results/:
  preliminary_results.json
  preliminary_metrics.txt
  update_log.jsonl

rag_pipeline.py complete ✓
---

## Datasets Used

| Dataset | Source | Purpose |
|---|---|---|
| PubMedQA | HuggingFace | RAG retrieval + evaluation |
| MedDec | ACL Anthology / GitHub | Clinical decision extraction |

---

## Milestones

| Milestone | Date | Status |
|---|---|---|
| Milestone 1 | April 30, 2026 | Completed |
| Final Evaluation | May 15, 2026 | Pending |
