"""Checkpoint discovery, integrity checks, device selection, and model loading."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from chemprop import models

from .utils import sha256_file


def default_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models"


def resolve_device(requested: str) -> torch.device:
    value = requested.lower()
    if value not in {"auto", "cpu", "cuda"}:
        raise ValueError("--device must be one of: auto, cpu, cuda")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device("cuda" if value == "cuda" or (value == "auto" and torch.cuda.is_available()) else "cpu")


def read_manifest(model_dir: Path) -> dict:
    path = model_dir / "model_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Model manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_paths(model_dir: Path, members: list[int] | None = None) -> list[Path]:
    manifest = read_manifest(model_dir)
    wanted = set(range(10) if members is None else members)
    entries = sorted(manifest["checkpoints"], key=lambda row: int(row["repeat_id"]))
    selected = [model_dir / row["path"] for row in entries if int(row["repeat_id"]) in wanted]
    if len(selected) != len(wanted):
        raise RuntimeError(f"Manifest does not define every requested ensemble member: {sorted(wanted)}")
    missing = [str(path) for path in selected if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model checkpoints:\n" + "\n".join(missing))
    return selected


def verify_checkpoint_files(model_dir: Path, check_hashes: bool = True) -> list[dict]:
    manifest = read_manifest(model_dir)
    results = []
    for entry in sorted(manifest["checkpoints"], key=lambda row: int(row["repeat_id"])):
        path = model_dir / entry["path"]
        present = path.exists()
        actual = sha256_file(path) if present and check_hashes else None
        results.append({
            "repeat_id": int(entry["repeat_id"]),
            "path": str(path),
            "present": present,
            "checksum_ok": present and (not check_hashes or actual == entry["sha256"]),
            "actual_sha256": actual,
        })
    return results


def load_models(model_dir: Path, device: torch.device, members: list[int] | None = None):
    loaded = []
    for path in checkpoint_paths(model_dir, members):
        model = models.MPNN.load_from_file(path, map_location=device).to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        loaded.append(model)
    return loaded

