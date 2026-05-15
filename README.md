## Project Webpage
[Click here to view the project page](https://rick-roy-jc.github.io/GENAI-ECM008-021-006/)

**RAG-based Clinical NLP with Continuous Knowledge Updating — CS5202 Spring 2026 (Completed)**

---

## Project — Retrieval-Augmented LLMs with Continuous Knowledge Updating for Reliable Clinical NLP

**Course:** CS5202 — GenAI and LLM, Spring 2026  
**Domain:** Medical GenAI (Domain E)  
**Team No:** 2  
**Team Members:**
- Aritra Roy — SE23UECM008 (Team Lead / ML Pipeline)
- Dheeraj Reddy — SE23UECM021 (Data Engineering / RAG)
- A. Sai Praneeth — SE23UECM006 (Evaluation / Knowledge Update)
- Ruthwik Reddy — SE23UCSE085 (Frontend / Integration)

**Instructor:** Prof. Nidhi Goyal  
**Final Submission:** May 15, 2026

---

## Problem Summary

Clinical NLP systems face two critical limitations:
1. **Hallucination** — generating plausible but factually incorrect information
2. **Knowledge Staleness** — static training data becomes outdated as medical knowledge evolves

This project builds a **Retrieval-Augmented Generation (RAG)** pipeline over clinical documents that can reliably answer biomedical questions — with a **continuous knowledge updating mechanism** that adds new documents to the live index without full rebuilds.

### Core Contributions
- ✅ Dense retrieval using PubMedBERT (768-dim embeddings, FAISS index)
- ✅ RAG generation with Flan-T5-Base instruction-tuned LLM
- ✅ **Continuous knowledge update** — incremental index addition (<1 sec per document)
- ✅ Update logging with timestamps and source tracking
- ✅ Comprehensive evaluation with ablation study

---

## Results Summary

| Metric | Value |
|--------|-------|
| Exact Match Accuracy | 40.0% (above 33.3% random baseline) |
| Yes F1 | 0.4356 (+77% over No RAG) |
| No F1 | 0.3659 |
| Mean Top-1 Retrieval Score | 0.9421 (excellent — PubMedBERT working well) |
| Mean Top-3 Retrieval Score | 0.9209 |
| Sources changed after update | 2/3 (66.7%) |
| Update latency | <1 second per document |

### Key Findings
- **RAG improves Yes F1 by 77%** over No RAG (0.2462 → 0.4356)
- Retrieval correctly prioritised newly added guidelines for 2/3 test queries
- Flan-T5-Base never predicts 'maybe' (zero F1) — model size limitation
- Prompt engineering improved accuracy 4x from Milestone 1 (10% → 37%)

---

## Project Structure

```
GENAI-ECM008-021-006/
├── README.md                    # This file
├── domain_note.pdf              # Milestone 1 domain note (1 page)
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── src/
│   ├── data_loader.py           # PubMedQA + MedDec loading
│   ├── rag_pipeline.py          # RAG retrieval + generation
│   ├── evaluate.py              # Faithfulness + relevance metrics
│   ├── main_pipeline.py         # End-to-end pipeline runner
│   └── knowledge_update_experiment.py  # Core update experiment
│
├── data/
│   └── (dataset files - PubMedQA cache)
│
├── results/
│   ├── Final_Report_GENAI-ECM008-021-006.docx  # Final report (4-6 pages)
│   ├── knowledge_update_experiment.json        # Update experiment results
│   ├── knowledge_update_summary.txt            # Human-readable summary
│   ├── milestone1_results.json                 # Milestone 1 outputs
│   └── predictions.csv                         # Sample predictions
│
├── notebooks/
│   └── (exploratory notebooks)
│
└── index.html                   # HTML submission report
```

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
source venv/Scripts/activate     # Git Bash (Windows)
venv\Scripts\activate            # Windows CMD
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the complete pipeline
```bash
# Load datasets and build RAG
python src/main_pipeline.py

# Run knowledge update experiment (core contribution)
python src/knowledge_update_experiment.py
```

### 5. Generate final report (if needed)
```bash
npm install docx
node Report.js
```

---

## Datasets Used

| Dataset | Source | Purpose | Size |
|---------|--------|---------|------|
| PubMedQA (pqa_labeled) | HuggingFace | RAG retrieval + evaluation | 1,000 labeled Q&A pairs |
| MedDec | ACL Anthology 2024 | Clinical decision extraction (fallback) | - |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Exact Match Accuracy | % of exact yes/no/maybe matches |
| F1 Score (Yes/No/Maybe) | Per-class harmonic mean of precision & recall |
| Macro F1 | Average of per-class F1 scores |
| Mean Retrieval Score | Cosine similarity of retrieved passages |
| Faithfulness | Answer grounded in retrieved context |
| Answer Relevance | Semantic similarity between Q and A |

---

## Ablation Study Results

| Configuration | Accuracy | Macro F1 | Yes F1 |
|---------------|----------|----------|--------|
| No RAG | 37.0% | 0.2459 | 0.2462 |
| RAG (Static) | 37.0% | **0.2672** | **0.4356** |
| RAG + Continuous Update | 37.0% | 0.2590 | 0.4314 |

**Conclusion:** RAG improves Yes F1 by 77%, but overall accuracy is limited by Flan-T5-Base model size.

---

## Milestones

| Milestone | Date | Status |
|-----------|------|--------|
| Milestone 1 | April 30, 2026 | ✅ Completed |
| Final Evaluation | May 15, 2026 | ✅ Completed |

---

## References

1. Lewis, P., Perez, E., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.
2. Singhal, K., Azizi, S., et al. (2023). Large Language Models Encode Clinical Knowledge. *Nature, 620*, 172–180. (Med-PaLM)
3. Jin, Q., et al. (2024). MedDec: A Dataset for Extracting Medical Decisions from Discharge Summaries. *ACL Findings 2024*.
4. Gu, Y., Tinn, R., et al. (2021). Domain-Specific Language Model Pretraining for Biomedical NLP. *ACM CHIL 2021*. (PubMedBERT)
5. Jin, Q., Dhingra, B., et al. (2019). PubMedQA: A Biomedical Research Question Answering Dataset. *EMNLP 2019*.
6. Johnson, A.E.W., et al. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data, 3*, 160035.

---

## Acknowledgments

- Prof. Nidhi Goyal for course guidance
- HuggingFace for PubMedQA dataset
- FAISS team for efficient similarity search

---

## Contact

GitHub: [Rick-Roy-JC/GENAI-ECM008-021-006](https://github.com/Rick-Roy-JC/GENAI-ECM008-021-006)
```

## To update the README on GitHub:

```bash
# Navigate to your project
cd /d/cs5202_/GENAI-ECM008-021-006

# Edit README.md (copy the content above)
nano README.md

# Or using echo (if you have the content ready)
# Save the content to README.md

# Add, commit, and push
git add README.md
git commit -m "Update README for final submission - May 15, 2026"
git push origin main
```

