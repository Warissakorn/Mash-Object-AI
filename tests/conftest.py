"""Make the repo root and src/ importable in tests without installation."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

for path in (ROOT, SRC):
    if path not in sys.path:
        sys.path.insert(0, path)
