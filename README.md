# SIH26053 / FoveaMap

Foveated LiDAR perception and 2.5D mapping for autonomous navigation.

## Current status
Data/split/cache foundation and SPVCNN execution are validated.
F0 trainer, F1 hybrid model, and adaptive mapper are NOT yet implemented.
See the project handover document for the full state and plan.

## Two-environment rule
- **Statistics (CPU):** `pip install -r requirements-cpu.txt`
- **Training (GPU, Python 3.10):** install torch 2.0.1+cu118, then
  `pip install -r requirements-gpu.txt`, then install the saved TorchSparse
  wheel from Drive (toolchain/torchsparse_t4_torch201_cu118, SHA256
  5cb55e2e...440c61c3). Never pip-install torchsparse directly.

## Golden rules
- Do NOT rebuild the 80 GB TAR cache.
- Do NOT trust pre-sanitation statistics (f0_*_exact*.json files).
- Run `python scripts/env_audit.py` before anything else in a new runtime.
