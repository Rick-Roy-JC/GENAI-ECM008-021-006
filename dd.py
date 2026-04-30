# verify_data.py - run this to confirm everything works
from pathlib import Path

from datasets import load_from_disk

project_root = Path(__file__).resolve().parent
candidates = [
    project_root / "data" / "raw" / "pubmedqa",
    project_root / "data" / "data" / "raw" / "pubmedqa",
]

dataset_path = next((p for p in candidates if p.exists()), None)
if dataset_path is None:
    checked = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(f"PubMedQA dataset not found. Checked:\n{checked}")

ds = load_from_disk(str(dataset_path))
sample = ds["train"][0]

print("Question:", sample["question"])
print("Context keys:", list(sample["context"].keys()))
print("Answer:", sample["final_decision"])
print("\nData loading verified")