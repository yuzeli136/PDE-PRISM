"""Deterministic PDE-PRISM screening engine."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .descriptors import calculate_descriptors
from .filters import Condition, apply_property_filters


@dataclass
class ScreenResult:
    output: pd.DataFrame
    report: dict


def apply_screening(
    frame: pd.DataFrame,
    task: str | None,
    threshold: float | None,
    min_sd: float | None,
    max_sd: float | None,
    include: list[Condition],
    exclude: list[Condition],
    conditions: list[Condition],
    prediction_margin: float | None,
    property_filters: dict[str, float],
    include_descriptors: bool,
    keep_all: bool,
    sort_by: str,
    ascending: bool,
    top: int | None,
) -> ScreenResult:
    result = frame.copy()
    valid = result["valid_smiles"].fillna(False).astype(bool)

    need_descriptors = include_descriptors or bool(property_filters)
    if need_descriptors:
        rows = [calculate_descriptors(value) if ok else {} for value, ok in zip(result["canonical_smiles"], valid)]
        result = pd.concat([result, pd.DataFrame(rows, index=result.index)], axis=1)

    pass_target = valid.copy()
    if task is not None and threshold is not None:
        pass_target &= result[f"{task}_mean"] >= threshold
    for condition in include + conditions:
        pass_target &= condition.apply(result).fillna(False)

    pass_sd = valid.copy()
    if task is not None:
        if min_sd is not None:
            pass_sd &= result[f"{task}_sd"] >= min_sd
        if max_sd is not None:
            pass_sd &= result[f"{task}_sd"] <= max_sd

    pass_offtarget = valid.copy()
    for condition in exclude:
        pass_offtarget &= condition.apply(result).fillna(False)

    pass_margin = valid.copy()
    if prediction_margin is not None:
        pass_margin &= result["prediction_margin"] >= prediction_margin

    pass_property = valid.copy()
    if property_filters:
        pass_property &= apply_property_filters(result, property_filters).fillna(False)

    result["pass_target_threshold"] = pass_target
    result["pass_sd_filter"] = pass_sd
    result["pass_offtarget_filter"] = pass_offtarget
    result["pass_margin_filter"] = pass_margin
    result["pass_profile_filter"] = pass_target & pass_offtarget & pass_margin
    result["pass_property_filter"] = pass_property
    result["pass_all_filters"] = valid & pass_target & pass_sd & pass_offtarget & pass_margin & pass_property

    report = {
        "input_compounds": int(len(result)),
        "valid_compounds": int(valid.sum()),
        "invalid_smiles": int((~valid).sum()),
        "after_probability_and_include_condition_filters": int((valid & pass_target).sum()),
        "after_sd_filter": int((valid & pass_target & pass_sd).sum()),
        "after_include_exclude_constraints": int((valid & pass_target & pass_sd & pass_offtarget).sum()),
        "after_prediction_margin_filter": int((valid & pass_target & pass_sd & pass_offtarget & pass_margin).sum()),
        "after_medicinal_chemistry_filter": int(result["pass_all_filters"].sum()),
    }

    selected = result.copy() if keep_all else result.loc[result["pass_all_filters"]].copy()
    sort_column = {
        "target_mean": "target_probability",
        "target_sd": "target_sd",
        "prediction_margin": "prediction_margin",
        "highest_offtarget": "highest_offtarget_probability",
        "qed": "QED",
        "mw": "MW",
    }[sort_by]
    if sort_column not in selected.columns:
        raise ValueError(f"--sort-by {sort_by} requires a target task or descriptor calculation")
    selected = selected.sort_values(sort_column, ascending=ascending, kind="mergesort", na_position="last")
    if top is not None and not keep_all:
        selected = selected.head(top)
    passing_count = int(result["pass_all_filters"].sum())
    report["before_top_n"] = passing_count
    report["final_hits"] = min(passing_count, top) if top is not None else passing_count
    report["rows_written"] = int(len(selected))
    return ScreenResult(selected, report)
