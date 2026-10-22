"""RDKit medicinal-chemistry descriptors and Lipinski Rule-of-Five counts."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


DESCRIPTOR_COLUMNS = ["MW", "LogP", "HBD", "HBA", "RotB", "TPSA", "QED", "Fsp3", "Lipinski_violations"]


def calculate_descriptors(smiles: str) -> dict[str, float | int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {column: float("nan") for column in DESCRIPTOR_COLUMNS}
    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    values = {
        "MW": mw,
        "LogP": logp,
        "HBD": hbd,
        "HBA": hba,
        "RotB": int(Lipinski.NumRotatableBonds(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "QED": float(QED.qed(mol)),
        "Fsp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
    }
    # Standard Lipinski Rule of Five: MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10.
    values["Lipinski_violations"] = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    return values

