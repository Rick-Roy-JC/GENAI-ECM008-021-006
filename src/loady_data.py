"""
src/load_data.py
Loads PubMedQA, cleans it, splits it, saves to data/processed/
Run: python src/load_data.py
"""

import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# Fix random seeds
random.seed(42)
np.random.seed(42)

# Create output folder
os.makedirs("data/processed", exist_ok=True)


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = " ".join(text.split())
    return text


def main():

    # ── Load dataset directly from HuggingFace ────────────────────────────
    print("Downloading PubMedQA from HuggingFace...")
    print("(First run takes 1-2 minutes, then it caches automatically)\n")

    try:
        raw_data = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    except TypeError as exc:
        # Work around stale/incompatible HF cache metadata on some Windows setups.
        print(f"Default HuggingFace cache failed ({exc}). Retrying with project cache...")
        cache_dir = os.path.join("data", "hf_cache")
        os.makedirs(cache_dir, exist_ok=True)
        raw_data = load_dataset(
            "qiaojin/PubMedQA",
            "pqa_labeled",
            cache_dir=cache_dir
        )

    print(f"Splits found     : {list(raw_data.keys())}")
    print(f"Train size       : {len(raw_data['train'])}")
    if "test" in raw_data:
        print(f"Test size        : {len(raw_data['test'])}")
    else:
        print("Test size        : N/A (split not provided by dataset)")
    print(f"Columns          : {raw_data['train'].column_names}\n")

    # ── Process each split ────────────────────────────────────────────────
    all_processed = []

    for split_name in raw_data.keys():
        split_data = raw_data[split_name]
        print(f"Processing {split_name} — {len(split_data)} samples")

        for i, sample in enumerate(tqdm(split_data, desc=f"  {split_name}")):

            # Extract question
            question = clean_text(sample.get("question", ""))

            # Extract context — it's a dict with a "contexts" key
            context_dict = sample.get("context", {})
            context_list = context_dict.get("contexts", [])
            context = " ".join([clean_text(c) for c in context_list])

            # Extract answer and label
            answer = clean_text(sample.get("long_answer", ""))
            label  = sample.get("final_decision", "")
            pid    = sample.get("pubid", str(i))

            # Skip empty samples
            if not question or not context:
                continue

            all_processed.append({
                "id":       str(pid),
                "question": question,
                "context":  context,
                "answer":   answer,
                "label":    label,
                "split":    split_name
            })

    print(f"\nTotal valid samples : {len(all_processed)}")

    # ── Split into train / val / test ─────────────────────────────────────
    train_data = [s for s in all_processed if s["split"] == "train"]
    test_data = [s for s in all_processed if s["split"] == "test"]

    random.shuffle(train_data)

    if test_data:
        # If provider includes test, create val from train only.
        cut = int(len(train_data) * 0.9)
        val_data = train_data[cut:]
        train_data = train_data[:cut]
    else:
        # If provider has only train, create train/val/test from train.
        train_cut = int(len(train_data) * 0.8)
        val_cut = int(len(train_data) * 0.9)
        val_data = train_data[train_cut:val_cut]
        test_data = train_data[val_cut:]
        train_data = train_data[:train_cut]

    print(f"Train : {len(train_data)}")
    print(f"Val   : {len(val_data)}")
    print(f"Test  : {len(test_data)}")

    # ── Print label distribution ──────────────────────────────────────────
    df = pd.DataFrame(train_data)
    print(f"\nLabel distribution (train):")
    print(df["label"].value_counts().to_string())

    # ── Save to disk ──────────────────────────────────────────────────────
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = f"data/processed/{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {path} ({len(data)} samples)")

    # Save 5 sample rows for quick inspection
    with open("data/processed/sample_5.json", "w", encoding="utf-8") as f:
        json.dump(train_data[:5], f, indent=2, ensure_ascii=False)
    print("Saved data/processed/sample_5.json")

    print("\nload_data.py complete.")


if __name__ == "__main__":
    main()