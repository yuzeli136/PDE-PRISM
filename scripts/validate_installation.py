#!/usr/bin/env python3
"""Minimal installation and one-molecule ensemble prediction validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chemprop  # noqa: E402
import numpy as np  # noqa: E402
import rdkit  # noqa: E402
import torch  # noqa: E402

from pde_prism import TASKS  # noqa: E402
from pde_prism.ensemble import aggregate  # noqa: E402
from pde_prism.model_loader import default_model_dir, load_models, resolve_device  # noqa: E402
from pde_prism.predictor import predict_ensemble  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=default_model_dir())
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    print(f"chemprop={chemprop.__version__} torch={torch.__version__} rdkit={rdkit.__version__}")
    device = resolve_device(args.device)
    models = load_models(args.model_dir, device)
    raw = predict_ensemble(models, ["CCO"], device, batch_size=2, quiet=True)
    if raw.shape != (10, 1, len(TASKS)) or not np.isfinite(raw).all():
        raise RuntimeError(f"Unexpected ensemble output: shape={raw.shape}")
    result = aggregate(raw)
    if len(result.columns) != 22:
        raise RuntimeError("Expected 11 mean and 11 SD columns")
    print("PASS: valid SMILES produced finite PDE1-PDE11 ensemble means and SDs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

