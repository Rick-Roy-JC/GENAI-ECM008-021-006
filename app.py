"""
Repo-root entry point for HF Spaces (Spaces expects app.py at the root).
The actual app lives in src/gradio_app.py so it can be run and tested
locally the same way as every other script in this project.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from gradio_app import demo

if __name__ == "__main__":
    demo.launch()
