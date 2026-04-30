# pyright: reportMissingImports=false

# ================================
# 1. Install dependencies
# ================================
# Run once in terminal (not inside this Python file):
# pip install datasets sentence-transformers faiss-cpu transformers

# ================================
# 2. Imports
# ================================
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# ================================
# 3. Load SMALL datasets (fast)
# ================================
print("Loading datasets...")

pubmed = load_dataset("pubmed_qa", "pqa_labeled", split="train[:200]")

# Try MedDec (optional)
try:
    meddec = load_dataset("med_decision_dataset", split="train[:200]")
    use_meddec = True
except:
    print("MedDec not available, continuing with PubMedQA only")
    use_meddec = False

# ================================
# 4. Preprocess
# ================================
print("Processing data...")

def process_pubmed(example):
    context = " ".join(example["context"]["contexts"])
    question = example["question"]
    return context + " Question: " + question

pubmed_data = [process_pubmed(x) for x in pubmed]

meddec_data = []
if use_meddec:
    for x in meddec:
        if "text" in x:
            meddec_data.append(x["text"])
        elif "sentence" in x:
            meddec_data.append(x["sentence"])

# Combine
all_texts = pubmed_data + meddec_data

# Reduce size further if needed
all_texts = all_texts[:300]

print("Total documents:", len(all_texts))

# ================================
# 5. Create embeddings
# ================================
print("Creating embeddings...")

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = embed_model.encode(
    all_texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

print("Embeddings shape:", embeddings.shape)

# ================================
# 6. Build FAISS index
# ================================
print("Building FAISS index...")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("Index size:", index.ntotal)

# ================================
# 7. Retrieval function
# ================================
def retrieve(query, k=3):
    query_embedding = embed_model.encode([query])
    D, I = index.search(query_embedding, k)
    return [all_texts[i] for i in I[0]]

# ================================
# 8. Load LLM (lightweight)
# ================================
print("Loading LLM...")

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",  # smaller = faster
    max_length=200
)

# ================================
# 9. QA Function
# ================================
def answer_query(query):
    docs = retrieve(query)

    context = " ".join(docs)

    prompt = f"""
    Answer the medical question using the context.

    Context:
    {context}

    Question:
    {query}
    """

    result = generator(prompt)[0]["generated_text"]
    return result

# ================================
# 10. Continuous Update
# ================================
def update_knowledge(new_texts):
    global all_texts

    new_embeddings = embed_model.encode(
        new_texts,
        convert_to_numpy=True
    )

    index.add(new_embeddings)
    all_texts.extend(new_texts)

    print("Updated index size:", index.ntotal)

# ================================
# 11. Test the system
# ================================
print("\n===== TESTING SYSTEM =====")

query = "What are treatments for diabetes?"

answer = answer_query(query)

print("\nQuery:", query)
print("\nAnswer:", answer)

# ================================
# 12. Test update feature
# ================================
print("\n===== TESTING UPDATE =====")

new_docs = [
    "Insulin therapy is commonly used for managing diabetes.",
    "Lifestyle changes such as diet and exercise are important in diabetes treatment."
]

update_knowledge(new_docs)

# Test again
answer_updated = answer_query(query)

print("\nUpdated Answer:", answer_updated)