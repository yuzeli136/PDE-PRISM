"""Myerson atom-attribution visualization using the manuscript color logic."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from rdkit import Chem
from rdkit.Chem import Draw


def draw_myerson(smiles: str, values: np.ndarray, output: Path, task: str) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Cannot visualize invalid SMILES")
    limit = float(max(np.max(np.abs(values)), 1e-12))
    cmap = plt.get_cmap("RdBu_r")
    norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    atom_colors = {index: tuple(cmap(norm(float(value)))[:3]) for index, value in enumerate(values)}
    image = Draw.MolToImage(
        mol,
        size=(1200, 700),
        highlightAtoms=list(range(mol.GetNumAtoms())),
        highlightAtomColors=atom_colors,
        highlightAtomRadii={index: 0.42 for index in range(mol.GetNumAtoms())},
        kekulize=True,
    )
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=200)
    ax.imshow(image)
    ax.axis("off")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    bar = fig.colorbar(scalar, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label(f"{task} mean Myerson attribution", fontsize=10)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

