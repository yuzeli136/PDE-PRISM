# PDE-PRISM v1.0

**Family-wide PDE Activity Profiling and Screening Toolkit**

PDE-PRISM is a ready-to-use ensemble of ten Chemprop models that predicts model-estimated activity probabilities for PDE1–PDE11 from molecular SMILES. It supports family-wide profiling, large-library screening, inter-model disagreement filtering, family-profile constraints, medicinal-chemistry filters, and atom-level Myerson attribution. No retraining is required.

> Predictions are computational prioritization outputs, not experimental IC50 values. Ensemble SD is disagreement among ten independently trained models, not calibrated uncertainty or confidence. `prediction_margin` is a model-predicted family-profile difference, not experimentally measured selectivity.

## Model information

| Item | Value |
|---|---|
| Model name | PDE-PRISM |
| Release version | 1.0.0 |
| Framework | Chemprop MPNN multi-label classification |
| Model type | Chemprop MPNN multi-label binary classifier |
| Ensemble size | 10 |
| Output task order | PDE1, PDE2, PDE3, PDE4, PDE5, PDE6, PDE7, PDE8, PDE9, PDE10, PDE11 |
| Chemprop version | 2.2.3 |
| PyTorch checkpoint environment | 2.11.0+cu130 |
| RDKit version | 2025.09.6 |
| Training structure column | `smiles_canon` |

Default inference preprocessing parses SMILES with RDKit and generates canonical isomeric SMILES with `canonical=True` and `isomericSmiles=True`. Optional standardization applies RDKit `Cleanup`, `FragmentParent`, and `Uncharger` only when `--standardize` is supplied.

## Installation

Python 3.11 is recommended because it matches the verified Myerson environment.

```bash
conda env create -f environment.yml
conda activate pde-prism-1.0
python scripts/verify_models.py --device cpu
python scripts/validate_installation.py --device cpu
```

The checkpoints use Chemprop 2.2.3, PyTorch 2.11.0+cu130, and RDKit 2025.09.6. CPU inference is supported. Install a PyTorch build appropriate for the local CUDA driver when GPU inference is required.


```bash
git lfs install
git lfs track "models/repeat_*/model.pt"
python scripts/verify_models.py
```

## Input

Prediction and screening accept CSV or TSV files containing a `smiles` column. Extra columns—including `compound_id`, `ID`, and `name`—are retained. Invalid SMILES do not terminate a run: they receive `valid_smiles=False` and `prediction_status=invalid_smiles`.

Default preprocessing is RDKit parsing followed by canonical isomeric SMILES generation, consistent with the verified `smiles_canon` training inputs. `--standardize` is opt-in and applies RDKit `Cleanup`, `FragmentParent`, and `Uncharger` before canonicalization. `--deduplicate` keeps the first row per canonical SMILES; duplicates are never silently removed by default.

## Quick start

Full PDE1–PDE11 prediction:

```bash
python pde_prism.py predict --input compounds.csv --output predictions.csv
```

Family-profile screening:

```bash
python pde_prism.py screen \
  --input compounds.csv --output PDE7_hits.csv \
  --task PDE7 --threshold 0.85 --max-sd 0.15 \
  --exclude PDE4:0.30 --exclude PDE5:0.30 \
  --selectivity-margin 0.40
```

Multi-PDE predicted-active screening without a separate primary task:

```bash
python pde_prism.py screen --input compounds.csv --output dual_hits.csv \
  --include PDE4:0.80 --include PDE7:0.80
```

Complete screening example:

```bash
python pde_prism.py screen \
  --input library.csv --output PDE7_hits.csv \
  --task PDE7 --threshold 0.85 --max-sd 0.15 \
  --exclude PDE4:0.30 --exclude PDE5:0.30 \
  --selectivity-margin 0.40 \
  --mw-min 150 --mw-max 500 \
  --logp-min -0.5 --logp-max 5.0 \
  --hbd-max 3 --hba-max 7 --rotb-max 9 --tpsa-max 140 \
  --qed-min 0.30 --fsp3-min 0.20 \
  --lipinski-violations-max 1 \
  --top 100 --sort-by target_mean \
  --filter-report
```

Myerson attribution (all ten ensemble members by default):

```bash
python pde_prism.py explain \
  --smiles "CCOc1ccc2nc(S(N)(=O)=O)sc2c1" \
  --task PDE7 --output explanation/
```

Myerson outputs `myerson_attribution.csv` with aligned per-model atom values, ensemble mean, and ensemble SD, plus `myerson_attribution.png`. Myerson attribution represents model-associated contributions to the predicted output. It does not establish protein–ligand interactions or a binding mechanism.

## Screening execution order

For reproducibility, `screen` always executes:

1. SMILES validation
2. Optional standardization
3. Ten-model ensemble prediction
4. Target, include, and general probability conditions
5. Target-task ensemble SD filter
6. Off-target exclusions
7. Prediction-margin filter
8. Medicinal-chemistry filters
9. Sorting
10. Top-N selection

`--keep-all` writes all input rows and filter pass/fail columns; therefore `--top` does not truncate its audit output, although the report records the Top-N final-hit count. Otherwise only final hits are written. `--filter-report` writes dynamic counts at every stage.

## Parameters

### Core screening parameters

| Parameter | Meaning | Default |
|---|---|---|
| `--task PDE7` | Primary PDE prediction task | None; required unless `--include` is used |
| `--threshold 0.85` | Minimum ensemble-mean activity probability for `--task` | 0.0 when task is supplied |
| `--top 100` | Keep Top-N after all filters and sorting | Disabled |
| `--sort-by` | `target_mean`, `target_sd`, `prediction_margin`, `highest_offtarget`, `qed`, or `mw` | `target_mean` |
| `--ascending` | Sort ascending instead of descending | Off |

### Ensemble parameters

| Parameter | Meaning | Default |
|---|---|---|
| `--min-sd` | Minimum target-task ensemble SD | Disabled |
| `--max-sd` | Maximum target-task ensemble SD | Disabled |
| `--save-individual` | Write every member prediction | Off |

Ensemble SD quantifies disagreement among the ten independently trained models. It is not a calibrated uncertainty probability or prediction confidence.

### Family-profile parameters

| Parameter | Meaning | Default |
|---|---|---|
| `--include PDE4:0.80` | Require PDE4 mean ≥ 0.80; repeatable | Disabled |
| `--exclude PDE5:0.30` | Require PDE5 mean ≤ 0.30; repeatable | Disabled |
| `--condition "PDE7>=0.85"` | Safe `PDE task + operator + probability` expression; repeatable | Disabled |
| `--selectivity-margin 0.40` | Require `target mean − highest other-PDE mean ≥ 0.40` | Disabled |

Supported condition operators are `>=`, `>`, `<=`, `<`, and `==`. Conditions are parsed without `eval()`.

### Medicinal-chemistry parameters

| Parameter | Meaning |
|---|---|
| `--mw-min`, `--mw-max` | Molecular-weight limits |
| `--logp-min`, `--logp-max` | Crippen LogP limits |
| `--hbd-max`, `--hba-max` | H-bond donor/acceptor maxima |
| `--rotb-max` | Rotatable-bond maximum |
| `--tpsa-max` | Topological polar surface area maximum |
| `--qed-min` | Minimum QED (0–1) |
| `--fsp3-min` | Minimum fraction sp3 (0–1) |
| `--lipinski-violations-max` | Maximum Rule-of-Five violations |
| `--include-descriptors` | Write descriptors even when no property filter is active |

All property filters are disabled by default. The Rule-of-Five violation count uses MW > 500, LogP > 5, HBD > 5, and HBA > 10.

### Execution parameters

| Parameter | Meaning | Default |
|---|---|---|
| `--device auto|cpu|cuda` | Compute device | `auto` |
| `--batch-size 256` | Molecules per inference batch | 256 |
| `--model-dir PATH` | Model manifest/weight directory | Repository `models/` |
| `--keep-all` | Retain all rows with filter flags | Off |
| `--filter-report [JSON]` | Write dynamic stage counts | Off |
| `--deduplicate` | Keep first canonical-SMILES occurrence | Off |
| `--standardize` | Apply explicit RDKit standardization | Off |
| `--target-only-output` | Omit non-target family profile | Off |
| `--quiet` | Suppress progress | Off |
| `--verbose` | Show diagnostic traceback | Off |

Run `python pde_prism.py <command> --help` for the complete command-specific interface.

## Output interpretation

The default family-wide output contains `PDE1_mean/PDE1_sd` through `PDE11_mean/PDE11_sd`. Screening additionally reports `target_task`, `target_probability`, `target_sd`, `highest_offtarget_task`, `highest_offtarget_probability`, and `prediction_margin`.


## Command-line reference

Unified entry point:

```text
python pde_prism.py {predict,screen,explain,info} ...
```

### `predict`

```text
python pde_prism.py predict --input INPUT --output OUTPUT
  [--save-individual] [--standardize] [--deduplicate]
  [--model-dir MODELS] [--device auto|cpu|cuda] [--batch-size N]
  [--quiet | --verbose]
```

Reads CSV/TSV with a `smiles` column and writes the complete PDE1–PDE11 ensemble mean/SD profile. `--save-individual` adds 110 per-member columns.

### `screen`

```text
python pde_prism.py screen --input INPUT --output OUTPUT
  [--task PDE1..PDE11] [--threshold P]
  [--min-sd SD] [--max-sd SD]
  [--include TASK:P]... [--exclude TASK:P]...
  [--condition EXPR]... [--selectivity-margin P]
  [--mw-min X] [--mw-max X] [--logp-min X] [--logp-max X]
  [--hbd-max N] [--hba-max N] [--rotb-max N] [--tpsa-max X]
  [--qed-min P] [--fsp3-min P] [--lipinski-violations-max N]
  [--include-descriptors] [--save-individual]
  [--sort-by FIELD] [--ascending] [--top N]
  [--keep-all] [--filter-report [JSON]] [--target-only-output]
  [--standardize] [--deduplicate]
  [--model-dir MODELS] [--device auto|cpu|cuda] [--batch-size N]
  [--quiet | --verbose]
```

At least `--task` or one `--include` is required. If only `--include` is supplied, the first included task is used for target-profile fields and sorting. Probability-like cutoffs must be in [0, 1]. SD limits must be nonnegative and `min <= max`.

`--selectivity-margin` is retained as a user-facing option name, but the output and implementation call this `prediction_margin`. It is a model-predicted family-profile margin, not experimental selectivity.

### `explain`

```text
python pde_prism.py explain (--smiles SMILES | --input INPUT)
  --task PDE1..PDE11 --output DIRECTORY
  [--models all|0,1,...] [--samples N] [--seed N]
  [--device auto|cpu|cuda] [--model-dir MODELS]
```

Runs the official Myerson 1.0.1 `MyersonSamplingClassExplainer`. All ten members are used by default. Each molecule receives aligned atom-level per-model values, mean/SD attribution, and a PNG visualization.

### `info`

```text
python pde_prism.py info [--model-dir MODELS] [--device auto|cpu|cuda] [--verify-checksums]
```

Reports versions, task order, device, and checkpoint availability. `--verify-checksums` reads and hashes all weights.

### Exit codes

- `0`: success
- `1`: model verification/status failure
- `2`: invalid arguments, missing dependency/file, or runtime error

Use `--verbose` to expose a Python traceback for debugging. The default interface prints a concise error.

## Testing and release status

```bash
python -m unittest discover -s tests -v
python scripts/verify_models.py --device cpu
python scripts/validate_installation.py --device cpu
```

We thank the Chemprop developers and contributors for making their molecular machine learning framework openly available.