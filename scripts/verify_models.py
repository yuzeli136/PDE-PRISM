#!/usr/bin/env python3
"""Verify checkpoint presence, SHA256 integrity, loadability, and output width."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pde_prism import TASKS  # noqa: E402
from pde_prism.model_loader import default_model_dir, load_models, resolve_device, verify_checkpoint_files  # noqa: E402
from pde_prism.predictor import predict_one_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=default_model_dir())
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--skip-load", action="store_true", help="Only check files and hashes")
    args = parser.parse_args()
    results = verify_checkpoint_files(args.model_dir, check_hashes=True)
    failed = [row for row in results if not row["present"] or not row["checksum_ok"]]
    for row in results:
        print(f"repeat {row['repeat_id']}: present={row['present']} checksum_ok={row['checksum_ok']}")
    if failed:
        print(f"FAILED: {len(failed)} checkpoint file checks failed", file=sys.stderr)
        return 1
    if not args.skip_load:
        device = resolve_device(args.device)
        for member in range(10):
            model = load_models(args.model_dir, device, [member])[0]
            output = predict_one_model(model, ["CCO"], device, batch_size=2, quiet=True)
            if output.shape != (1, len(TASKS)):
                raise RuntimeError(f"repeat {member}: output shape {output.shape}, expected (1, 11)")
            print(f"repeat {member}: load_ok=True output_tasks={output.shape[1]}")
    print("PASS: 10/10 final checkpoints verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

