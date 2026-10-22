"""SMILES validation and optional, explicit RDKit standardization."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.error")


@dataclass(frozen=True)
class ProcessedSmiles:
    canonical_smiles: str | None
    valid: bool
    status: str


def process_smiles(value: object, standardize: bool = False) -> ProcessedSmiles:
    if value is None or str(value).strip() == "":
        return ProcessedSmiles(None, False, "invalid_smiles")
    mol = Chem.MolFromSmiles(str(value).strip())
    if mol is None:
        return ProcessedSmiles(None, False, "invalid_smiles")
    try:
        if standardize:
            # Opt-in conservative RDKit normalization. Training inputs were verified
            # as canonical isomeric SMILES; no standardization is applied by default.
            mol = rdMolStandardize.Cleanup(mol)
            mol = rdMolStandardize.FragmentParent(mol)
            mol = rdMolStandardize.Uncharger().uncharge(mol)
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        checked = Chem.MolFromSmiles(canonical)
        if checked is None:
            raise ValueError("canonicalization failed")
    except Exception:
        return ProcessedSmiles(None, False, "standardization_failed" if standardize else "invalid_smiles")
    return ProcessedSmiles(canonical, True, "ok")
