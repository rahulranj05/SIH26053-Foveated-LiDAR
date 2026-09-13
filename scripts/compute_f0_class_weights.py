from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


NUM_CLASSES = 23
IGNORE_INDEX = 0


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate and compare stable class-weight candidates from exact F0 counts."
    )
    p.add_argument("--distribution", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--recommended",
        choices=("sqrt_inverse", "log_inverse", "effective_num"),
        default="sqrt_inverse",
    )
    p.add_argument("--max-weight", type=float, default=4.0)
    p.add_argument("--min-weight", type=float, default=0.25)
    p.add_argument("--log-offset", type=float, default=1.02)
    p.add_argument("--effective-beta", type=float, default=0.999999)
    return p.parse_args()


def normalize_nonzero(weights: np.ndarray, counts: np.ndarray) -> np.ndarray:
    active = (counts > 0)
    active[IGNORE_INDEX] = False
    if active.any():
        weights[active] /= weights[active].mean()
    weights[~active] = 0.0
    weights[IGNORE_INDEX] = 0.0
    return weights


def clamp_active(weights, counts, low, high):
    active = (counts > 0)
    active[IGNORE_INDEX] = False
    weights[active] = np.clip(weights[active], low, high)
    weights[~active] = 0.0
    weights[IGNORE_INDEX] = 0.0
    return weights


def sqrt_inverse(counts, low, high):
    probs = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    w = np.zeros_like(probs)
    active = probs > 0
    w[active] = 1.0 / np.sqrt(probs[active])
    w = normalize_nonzero(w, counts)
    return clamp_active(w, counts, low, high)


def log_inverse(counts, low, high, offset):
    probs = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    w = np.zeros_like(probs)
    active = probs > 0
    w[active] = 1.0 / np.log(offset + probs[active])
    w = normalize_nonzero(w, counts)
    return clamp_active(w, counts, low, high)


def effective_num(counts, low, high, beta):
    c = counts.astype(np.float64)
    w = np.zeros_like(c)
    active = c > 0
    # Stable form of (1-beta)/(1-beta^n)
    w[active] = (1.0 - beta) / (1.0 - np.power(beta, c[active]))
    w = normalize_nonzero(w, counts)
    return clamp_active(w, counts, low, high)


def main():
    args = parse_args()
    with open(args.distribution, "r", encoding="utf-8") as f:
        dist = json.load(f)

    combined = dist["combined"]
    counts = np.array(
        [int(combined[str(i)]["points"]) for i in range(NUM_CLASSES)],
        dtype=np.int64,
    )
    names = [combined[str(i)]["name"] for i in range(NUM_CLASSES)]

    candidates = {
        "sqrt_inverse": sqrt_inverse(
            counts.copy(), args.min_weight, args.max_weight
        ),
        "log_inverse": log_inverse(
            counts.copy(), args.min_weight, args.max_weight, args.log_offset
        ),
        "effective_num": effective_num(
            counts.copy(), args.min_weight, args.max_weight, args.effective_beta
        ),
    }

    recommended = candidates[args.recommended]

    payload = {
        "source_distribution": str(args.distribution),
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
        "recommended_method": args.recommended,
        "note": (
            "Weights are normalized to mean 1.0 over supported non-ignore classes "
            "before clamping. Class 0 and zero-support classes receive weight 0."
        ),
        "candidates": {},
        "recommended": {},
    }

    print("=" * 94)
    print("F0 EXACT CLASS-WEIGHT CANDIDATES")
    print("=" * 94)
    print(
        f"{'ID':>3}  {'Class':25s} {'Points':>14s} "
        f"{'sqrt':>8s} {'log':>8s} {'eff':>8s}"
    )
    print("-" * 94)
    for i in range(NUM_CLASSES):
        print(
            f"{i:3d}  {names[i]:25s} {counts[i]:14,d} "
            f"{candidates['sqrt_inverse'][i]:8.4f} "
            f"{candidates['log_inverse'][i]:8.4f} "
            f"{candidates['effective_num'][i]:8.4f}"
        )

    for method, arr in candidates.items():
        payload["candidates"][method] = {
            str(i): {
                "name": names[i],
                "points": int(counts[i]),
                "weight": float(arr[i]),
            }
            for i in range(NUM_CLASSES)
        }

    payload["recommended"] = {
        str(i): {
            "name": names[i],
            "points": int(counts[i]),
            "weight": float(recommended[i]),
        }
        for i in range(NUM_CLASSES)
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    temp.replace(output)

    active = (counts > 0)
    active[IGNORE_INDEX] = False
    saturated = int((recommended[active] >= args.max_weight - 1e-12).sum())
    print("\nRecommended:", args.recommended)
    print(f"Active supported classes: {int(active.sum())}")
    print(f"Classes at max clamp    : {saturated}")
    print(f"Saved                   : {output}")
    print("\nPASS")


if __name__ == "__main__":
    main()
