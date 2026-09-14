"""Run this FIRST in any new runtime, before installing anything.

Usage:
  python scripts/env_audit.py        -> CPU statistics runtime (lenient)
  python scripts/env_audit.py --ml   -> GPU training runtime (strict)
"""
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--ml", action="store_true",
                    help="strict checks for the GPU training environment")
args = parser.parse_args()

print("=== Environment Audit ===")
print(f"Python : {sys.version.split()[0]}")
major, minor = sys.version_info[:2]

problems = []

try:
    import numpy
    print(f"NumPy  : {numpy.__version__}")
except ImportError:
    numpy = None
    print("NumPy  : not installed")

try:
    import yaml
    print(f"PyYAML : {yaml.__version__}")
except ImportError:
    print("PyYAML : not installed")
    problems.append("PyYAML is missing. Run: pip install -r requirements-cpu.txt")

if not args.ml:
    # CPU statistics runtime: any Python, just needs numpy + yaml.
    if numpy is None:
        problems.append("NumPy is missing. Run: pip install -r requirements-cpu.txt")
else:
    # GPU training runtime: the validated stack.
    if (major, minor) != (3, 10):
        problems.append(f"Python {major}.{minor} detected. ML requires exactly Python 3.10.")
    if numpy is None or not numpy.__version__.startswith("1.26"):
        problems.append("NumPy is not 1.26.x. ML requires numpy==1.26.4.")
    try:
        import torch
        print(f"Torch  : {torch.__version__}  CUDA: {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            problems.append("Torch installed but CUDA not available.")
    except ImportError:
        print("Torch  : not installed")
        problems.append("Torch 2.0.1+cu118 not installed.")

print("=========================")
if problems:
    for p in problems:
        print(f"WARNING: {p}")
    sys.exit(1)
print("Environment looks OK.")
