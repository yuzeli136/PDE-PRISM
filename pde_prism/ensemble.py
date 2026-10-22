"""Ensemble aggregation and family-profile summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import TASKS


def aggregate(predictions: np.ndarray, save_individual: bool = False) -> pd.DataFrame:
    if predictions.ndim != 3 or predictions.shape[2] != len(TASKS):
        raise ValueError("Predictions must have shape (models, molecules, 11)")
    mean = predictions.mean(axis=0)
    sd = predictions.std(axis=0, ddof=1) if predictions.shape[0] > 1 else np.zeros_like(mean)
    data: dict[str, np.ndarray] = {}
    for task_index, task in enumerate(TASKS):
        data[f"{task}_mean"] = mean[:, task_index]
        data[f"{task}_sd"] = sd[:, task_index]
        if save_individual:
            for model_index in range(predictions.shape[0]):
                data[f"{task}_model_{model_index}"] = predictions[model_index, :, task_index]
    return pd.DataFrame(data)


def add_target_profile(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    result = frame.copy()
    others = [name for name in TASKS if name != task]
    other_values = result[[f"{name}_mean" for name in others]].to_numpy(dtype=float)
    highest_index = np.nanargmax(other_values, axis=1)
    highest_value = other_values[np.arange(len(result)), highest_index]
    result["target_task"] = task
    result["target_probability"] = result[f"{task}_mean"]
    result["target_sd"] = result[f"{task}_sd"]
    result["highest_offtarget_task"] = [others[index] for index in highest_index]
    result["highest_offtarget_probability"] = highest_value
    result["prediction_margin"] = result["target_probability"] - result["highest_offtarget_probability"]
    return result

