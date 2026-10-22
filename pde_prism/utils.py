"""Shared validation, logging, and file helpers."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import TASKS


def configure_logging(quiet: bool = False, verbose: bool = False) -> None:
    level = logging.ERROR if quiet else (logging.DEBUG if verbose else logging.INFO)
    logging.basicConfig(level=level, format="%(message)s", force=True)


def validate_task(task: str) -> str:
    value = str(task).upper()
    if value not in TASKS:
        raise ValueError(f"Unknown PDE task '{task}'. Choose one of: {', '.join(TASKS)}")
    return value


def probability(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1; received {value}")
    return value


def nonnegative(value: float, name: str) -> float:
    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0; received {value}")
    return value


def load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".tsv", ".txt"}:
        frame = pd.read_csv(path, sep="\t")
    else:
        raise ValueError("Input must be CSV or TSV and contain a 'smiles' column")
    candidates = {str(column).lower(): column for column in frame.columns}
    if "smiles" not in candidates:
        raise ValueError("Input is missing the required 'smiles' column")
    actual = candidates["smiles"]
    if actual != "smiles":
        frame = frame.rename(columns={actual: "smiles"})
    return frame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_models(value: str) -> list[int]:
    if value.lower() == "all":
        return list(range(10))
    try:
        members = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--models must be 'all' or comma-separated integers 0-9") from exc
    if not members or len(set(members)) != len(members) or any(item not in range(10) for item in members):
        raise ValueError("--models must contain unique model numbers from 0 to 9")
    return members


def ordered_columns(frame: pd.DataFrame, preferred: Iterable[str]) -> pd.DataFrame:
    first = [column for column in preferred if column in frame.columns]
    return frame[first + [column for column in frame.columns if column not in first]]

