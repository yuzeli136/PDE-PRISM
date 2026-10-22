"""Official Myerson 1.0.1 atom attribution across ensemble members."""

from __future__ import annotations

import hashlib
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from chemprop import featurizers
from myerson.chemprop_explain import MyersonSamplingClassExplainer
from rdkit import Chem

from . import TASKS
from .model_loader import load_models
from .visualization import draw_myerson


def _stable_seed(smiles: str, task: str, member: int, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}|{smiles}|{task}|{member}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def explain_smiles(
    smiles: str,
    task: str,
    model_dir: Path,
    device,
    members: list[int],
    samples: int,
    seed: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Myerson explanation requires one valid, connected molecule")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    mol = Chem.MolFromSmiles(canonical)
    target_index = TASKS.index(task)
    molgraph = featurizers.SimpleMoleculeMolGraphFeaturizer()(mol)
    columns: dict[str, np.ndarray] = {
        "atom_index": np.arange(mol.GetNumAtoms(), dtype=int),
        "atom_symbol": np.array([atom.GetSymbol() for atom in mol.GetAtoms()], dtype=object),
    }
    predictions = []
    values_by_model = []
    # Load one member at a time, matching the manuscript implementation and
    # keeping GPU memory bounded for heterogeneous checkpoint sizes.
    for member in members:
        model = load_models(model_dir, device, [member])[0]
        explainer = MyersonSamplingClassExplainer(
            molgraph=molgraph,
            coalition_function=model,
            seed=_stable_seed(canonical, task, member, seed),
            number_of_samples=samples,
            disable_tqdm=True,
        )
        predictions.append(float(explainer.pred.detach().cpu().numpy()[target_index]))
        values = explainer.sample_all_myerson_values().astype(float)[:, target_index]
        if len(values) != mol.GetNumAtoms():
            raise RuntimeError("Myerson atom alignment failed")
        columns[f"model_{member}_attribution"] = values
        values_by_model.append(values)
        del explainer, model
        gc.collect()
        if getattr(device, "type", None) == "cuda":
            import torch
            torch.cuda.empty_cache()
    matrix = np.stack(values_by_model, axis=0)
    columns["mean_attribution"] = matrix.mean(axis=0)
    columns["sd_attribution"] = matrix.std(axis=0, ddof=1) if len(members) > 1 else np.zeros(mol.GetNumAtoms())
    columns["canonical_smiles"] = np.repeat(canonical, mol.GetNumAtoms())
    columns["task"] = np.repeat(task, mol.GetNumAtoms())
    columns["ensemble_prediction_mean"] = np.repeat(np.mean(predictions), mol.GetNumAtoms())
    columns["ensemble_prediction_sd"] = np.repeat(np.std(predictions, ddof=1) if len(predictions) > 1 else 0.0, mol.GetNumAtoms())
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "myerson_attribution.csv"
    png_path = output_dir / "myerson_attribution.png"
    pd.DataFrame(columns).to_csv(csv_path, index=False)
    draw_myerson(canonical, columns["mean_attribution"], png_path, task)
    return csv_path, png_path
