from __future__ import annotations

import argparse
import hashlib
import json
import random
import tarfile
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_args():
    p = argparse.ArgumentParser(
        description="Verify F0 TAR cache integrity against its sidecar SHA256 metadata."
    )
    p.add_argument("--cache-dir", required=True)
    p.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of samples to hash. Use 0 for every sample.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--verify-all-tar-members",
        action="store_true",
        help="Also require the TAR member set to exactly match all sidecar scan/label names.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    sidecars = sorted(cache_dir.glob("f0_shard_*.json"))
    if not sidecars:
        raise RuntimeError(f"No shard sidecars found: {cache_dir}")

    rows = []
    sidecar_by_shard = {}
    for path in sidecars:
        with path.open("r", encoding="utf-8") as f:
            sidecar = json.load(f)
        shard_id = int(sidecar["shard_id"])
        tar_path = cache_dir / f"f0_shard_{shard_id:05d}.tar"
        if not tar_path.is_file():
            raise FileNotFoundError(f"Missing TAR: {tar_path}")
        expected_size = int(sidecar["tar_size_bytes"])
        actual_size = tar_path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"Size mismatch {tar_path.name}: expected {expected_size}, got {actual_size}"
            )
        sidecar_by_shard[shard_id] = (sidecar, tar_path)
        for sample in sidecar["samples"]:
            rows.append((shard_id, sample))

    print("=" * 80)
    print("SIH26053 - F0 TAR CACHE VERIFICATION")
    print("=" * 80)
    print(f"Cache directory : {cache_dir}")
    print(f"Shards          : {len(sidecars):,}")
    print(f"Samples indexed : {len(rows):,}")

    if args.verify_all_tar_members:
        print("\nChecking complete TAR member sets...")
        for i, (shard_id, (sidecar, tar_path)) in enumerate(sidecar_by_shard.items(), 1):
            expected = set()
            for s in sidecar["samples"]:
                expected.add(s["scan_arcname"])
                expected.add(s["label_arcname"])
            with tarfile.open(tar_path, "r") as tar:
                actual = {m.name for m in tar.getmembers() if m.isfile()}
            if actual != expected:
                missing = sorted(expected - actual)[:10]
                extra = sorted(actual - expected)[:10]
                raise RuntimeError(
                    f"{tar_path.name}: member-set mismatch; "
                    f"missing={missing}, extra={extra}"
                )
            if i % 20 == 0 or i == len(sidecar_by_shard):
                print(f"  member sets: {i}/{len(sidecar_by_shard)} PASS")

    if args.samples == 0 or args.samples >= len(rows):
        selected = rows
    else:
        rng = random.Random(args.seed)
        selected = rng.sample(rows, args.samples)

    grouped = {}
    for shard_id, sample in selected:
        grouped.setdefault(shard_id, []).append(sample)

    checked = 0
    print(f"\nHash-verifying {len(selected):,} samples...")
    for shard_id, samples in sorted(grouped.items()):
        _, tar_path = sidecar_by_shard[shard_id]
        with tarfile.open(tar_path, "r") as tar:
            for sample in samples:
                for kind in ("scan", "label"):
                    arcname = sample[f"{kind}_arcname"]
                    member = tar.getmember(arcname)
                    stream = tar.extractfile(member)
                    if stream is None:
                        raise RuntimeError(f"Could not read {arcname}")
                    payload = stream.read()
                    actual_hash = sha256(payload)
                    expected_hash = sample[f"{kind}_sha256"]
                    expected_size = int(sample[f"{kind}_size"])
                    if len(payload) != expected_size:
                        raise RuntimeError(
                            f"{arcname}: size {len(payload)} != {expected_size}"
                        )
                    if actual_hash != expected_hash:
                        raise RuntimeError(
                            f"{arcname}: SHA256 mismatch\n"
                            f"expected={expected_hash}\nactual  ={actual_hash}"
                        )
                checked += 1
                if checked % 100 == 0 or checked == len(selected):
                    print(f"  {checked:,}/{len(selected):,} samples PASS")

    print("\n" + "=" * 80)
    print("F0 TAR CACHE VERIFICATION: PASS")
    print("=" * 80)
    print("The verified TAR payload bytes exactly match the SHA256 values recorded")
    print("from the original raw scan/label bytes during cache construction.")


if __name__ == "__main__":
    main()
