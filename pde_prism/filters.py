"""Safe family-profile conditions and medicinal-chemistry filter validation."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .utils import probability, validate_task


CONDITION_RE = re.compile(r"^\s*(PDE(?:10|11|[1-9]))\s*(>=|<=|>|<|==)\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*$", re.I)


@dataclass(frozen=True)
class Condition:
    task: str
    operator: str
    cutoff: float
    source: str = "condition"

    def apply(self, frame: pd.DataFrame) -> pd.Series:
        series = frame[f"{self.task}_mean"]
        return {
            ">=": series.ge,
            ">": series.gt,
            "<=": series.le,
            "<": series.lt,
            "==": series.eq,
        }[self.operator](self.cutoff)


def parse_condition(text: str, source: str = "condition") -> Condition:
    match = CONDITION_RE.fullmatch(str(text))
    if not match:
        raise ValueError(f"Invalid {source} '{text}'. Expected, for example, PDE7>=0.85")
    return Condition(validate_task(match.group(1)), match.group(2), probability(float(match.group(3)), source), source)


def parse_task_cutoff(text: str, mode: str) -> Condition:
    if ":" not in str(text):
        raise ValueError(f"Invalid --{mode} '{text}'. Expected TASK:PROBABILITY, e.g. PDE4:0.30")
    task, raw = str(text).split(":", 1)
    try:
        cutoff = probability(float(raw), f"--{mode}")
    except ValueError as exc:
        raise ValueError(f"Invalid --{mode} '{text}': {exc}") from exc
    return Condition(validate_task(task), ">=" if mode == "include" else "<=", cutoff, mode)


PROPERTY_MAP = {
    "mw_min": ("MW", "ge"), "mw_max": ("MW", "le"),
    "logp_min": ("LogP", "ge"), "logp_max": ("LogP", "le"),
    "hbd_max": ("HBD", "le"), "hba_max": ("HBA", "le"),
    "rotb_max": ("RotB", "le"), "tpsa_max": ("TPSA", "le"),
    "qed_min": ("QED", "ge"), "fsp3_min": ("Fsp3", "ge"),
    "lipinski_violations_max": ("Lipinski_violations", "le"),
}


def property_filters_from_args(args) -> dict[str, float]:
    return {name: float(getattr(args, name)) for name in PROPERTY_MAP if getattr(args, name, None) is not None}


def apply_property_filters(frame: pd.DataFrame, filters: dict[str, float]) -> pd.Series:
    result = pd.Series(True, index=frame.index)
    for name, cutoff in filters.items():
        column, operation = PROPERTY_MAP[name]
        result &= getattr(frame[column], operation)(cutoff)
    return result

