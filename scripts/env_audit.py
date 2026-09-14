"""Run this FIRST in any new runtime, before installing anything."""
import sys

print("=== Environment Audit ===")
print(f"Python : {sys.version.split()[0]}")

problems = []

major, minor = sys.version_info[:2]
if (major, minor) != (3, 10):
    problems.append(f"Python {major}.{minor} detected. ML work requires Python 3.10.")

try:
    import numpy
    print(f"NumPy  : {numpy.__version__}")
    if not numpy.__version__.startswith("1.26"):
        problems.append("NumPy is not 1.26.x. This may cause binary crashes.")
except ImportError:
    print("NumPy  : not installed (OK for a fresh CPU runtime)")

try:
    import torch
    print(f"Torch  : {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("Torch  : not installed (OK for CPU statistics runtime)")

print("=========================")
if problems:
    for p in problems:
        print(f"WARNING: {p}")
    sys.exit(1)
print("Environment looks OK.")
