"""Memory-bounded Chemprop inference."""

from __future__ import annotations

import logging

import numpy as np
import torch
from chemprop import data, featurizers
from rdkit import Chem
from tqdm.auto import tqdm

from . import TASKS


def _dataset(smiles: list[str]):
    return data.MoleculeDataset(
        [data.MoleculeDatapoint(Chem.MolFromSmiles(smi)) for smi in smiles],
        featurizer=featurizers.SimpleMoleculeMolGraphFeaturizer(),
    )


def predict_one_model(model, smiles: list[str], device: torch.device, batch_size: int, quiet: bool = False) -> np.ndarray:
    if not smiles:
        return np.empty((0, len(TASKS)), dtype=float)
    chunks = []
    ranges = range(0, len(smiles), batch_size)
    for start in tqdm(ranges, desc="Predicting batches", disable=quiet, leave=False):
        batch_smiles = list(smiles[start:start + batch_size])
        original_count = len(batch_smiles)
        # A singleton batch can fail if a checkpoint contains batch normalization.
        if original_count == 1:
            batch_smiles.append(batch_smiles[0])
        loader = data.build_dataloader(_dataset(batch_smiles), batch_size=len(batch_smiles), shuffle=False, num_workers=0)
        with torch.no_grad():
            for batch in loader:
                batch.bmg.to(device)
                output = model(batch.bmg, batch.V_d, batch.X_d)
                chunks.append(output.detach().cpu().numpy().astype(float)[:original_count])
    result = np.concatenate(chunks, axis=0)
    if result.shape != (len(smiles), len(TASKS)):
        raise RuntimeError(f"Unexpected prediction shape {result.shape}; expected {(len(smiles), len(TASKS))}")
    return result


def predict_ensemble(models, smiles: list[str], device: torch.device, batch_size: int, quiet: bool = False) -> np.ndarray:
    outputs = []
    for index, model in enumerate(models):
        logging.info("Predicting with ensemble member %d/%d", index + 1, len(models))
        outputs.append(predict_one_model(model, smiles, device, batch_size, quiet=quiet))
    return np.stack(outputs, axis=0)

