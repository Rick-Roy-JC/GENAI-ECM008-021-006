# GENAI-ECM008-021-006
RAG-based Clinical NLP with Continuous Knowledge Updating — CS5202 Spring 2026
# Project 19 — RAG-based Clinical NLP with Continuous Knowledge Updating

**Course:** CS5202 — GenAI and LLM, Spring 2026  
**Domain:** Medical GenAI  
**Team Members:** [Name 1 — Roll No], [Name 2 — Roll No], ...

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
git clone https://github.com/your-username/project-19-<rollno>.git
cd project-19-<rollno>
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
| Milestone 1 | April 30, 2026 | In Progress |
| Final Evaluation | May 15, 2026 | Pending |
