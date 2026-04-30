"""
src/check_env.py
Run this to verify all packages installed correctly.
"""

import sys
print(f"Python version: {sys.version}")
print()

packages = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("faiss", "faiss"),
    ("sentence_transformers", "sentence-transformers"),
    ("langchain", "langchain"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("rouge_score", "rouge-score"),
    ("nltk", "nltk"),
    ("tqdm", "tqdm"),
]

all_good = True
for import_name, package_name in packages:
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "installed")
        print(f"  ✓  {package_name:<25} {version}")
    except ImportError:
        print(f"  ✗  {package_name:<25} NOT FOUND")
        all_good = False

print()
if all_good:
    print("All packages verified. Environment is ready!")
else:
    print("Some packages missing. Re-run pip install for the ones marked ✗")