"""Command-line interface for PDE-PRISM v1.0."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import TASKS, __version__
from .ensemble import add_target_profile, aggregate
from .filters import parse_condition, parse_task_cutoff, property_filters_from_args
from .model_loader import default_model_dir, load_models, read_manifest, resolve_device, verify_checkpoint_files
from .myerson import explain_smiles
from .predictor import predict_ensemble
from .preprocessing import process_smiles
from .screening import apply_screening
from .utils import configure_logging, load_input, nonnegative, ordered_columns, probability, validate_task, write_json


def add_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", type=Path, default=default_model_dir(), help="Directory containing model_manifest.json and repeat folders")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device (default: auto)")
    parser.add_argument("--batch-size", type=int, default=256, help="Molecules per inference batch (default: 256)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages")
    parser.add_argument("--verbose", action="store_true", help="Show detailed diagnostic errors")


def add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True, help="Input CSV/TSV containing a smiles column")
    parser.add_argument("--standardize", action="store_true", help="Apply opt-in RDKit Cleanup, FragmentParent, and Uncharger before canonicalization")
    parser.add_argument("--deduplicate", action="store_true", help="Keep the first row for each canonical SMILES")


def add_property_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mw-min", type=float)
    parser.add_argument("--mw-max", type=float)
    parser.add_argument("--logp-min", type=float)
    parser.add_argument("--logp-max", type=float)
    parser.add_argument("--hbd-max", type=int)
    parser.add_argument("--hba-max", type=int)
    parser.add_argument("--rotb-max", type=int)
    parser.add_argument("--tpsa-max", type=float)
    parser.add_argument("--qed-min", type=float)
    parser.add_argument("--fsp3-min", type=float)
    parser.add_argument("--lipinski-violations-max", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pde_prism.py", description="PDE-PRISM v1.0: family-wide PDE activity profiling and screening")
    parser.add_argument("--version", action="version", version=f"PDE-PRISM {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    predict = sub.add_parser("predict", help="Predict PDE1-PDE11 activity probabilities with the ten-model ensemble")
    add_input_args(predict)
    add_execution_args(predict)
    predict.add_argument("--output", type=Path, required=True, help="Output prediction CSV")
    predict.add_argument("--save-individual", action="store_true", help="Add all per-model predictions")
    predict.set_defaults(handler=run_predict)

    screen = sub.add_parser("screen", help="Screen a library using family-wide predictions and optional property filters")
    add_input_args(screen)
    add_execution_args(screen)
    add_property_args(screen)
    screen.add_argument("--output", type=Path, required=True, help="Output screening CSV")
    screen.add_argument("--task", choices=list(TASKS), help="Primary PDE prediction task")
    screen.add_argument("--threshold", type=float, help="Minimum ensemble-mean probability for --task")
    screen.add_argument("--min-sd", type=float, help="Minimum ensemble SD for --task")
    screen.add_argument("--max-sd", type=float, help="Maximum ensemble SD for --task")
    screen.add_argument("--include", action="append", default=[], metavar="TASK:PROB", help="Require TASK mean >= probability; repeatable")
    screen.add_argument("--exclude", action="append", default=[], metavar="TASK:PROB", help="Require TASK mean <= probability; repeatable")
    screen.add_argument("--condition", action="append", default=[], metavar="EXPR", help="Safe condition such as PDE7>=0.85; repeatable")
    screen.add_argument("--selectivity-margin", type=float, dest="prediction_margin", help="Minimum model-predicted family-profile margin")
    screen.add_argument("--top", type=int, help="Keep Top-N after filtering and sorting")
    screen.add_argument("--sort-by", choices=["target_mean", "target_sd", "prediction_margin", "highest_offtarget", "qed", "mw"], default="target_mean")
    screen.add_argument("--ascending", action="store_true", help="Sort in ascending order (default: descending)")
    screen.add_argument("--include-descriptors", action="store_true", help="Write RDKit medicinal-chemistry descriptors")
    screen.add_argument("--save-individual", action="store_true", help="Add all per-model predictions")
    screen.add_argument("--keep-all", action="store_true", help="Write all rows with per-stage pass/fail flags")
    screen.add_argument("--target-only-output", action="store_true", help="Omit non-target PDE profile columns")
    screen.add_argument("--filter-report", nargs="?", const="AUTO", metavar="JSON", help="Write a JSON filter report; omit path for <output>_report.json")
    screen.set_defaults(handler=run_screen)

    explain = sub.add_parser("explain", help="Compute official Myerson atom attribution for one or more molecules")
    source = explain.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles", help="One SMILES string")
    source.add_argument("--input", type=Path, help="CSV/TSV containing a smiles column")
    explain.add_argument("--task", choices=list(TASKS), required=True)
    explain.add_argument("--output", type=Path, required=True, help="Output directory")
    explain.add_argument("--models", default="all", help="'all' (default) or comma-separated member IDs 0-9")
    explain.add_argument("--samples", type=int, default=1000, help="Sampled Myerson permutations per model (default: 1000)")
    explain.add_argument("--seed", type=int, default=20260825)
    explain.add_argument("--save-individual", action="store_true", help="Retained for explicitness; per-model values are always written")
    add_execution_args(explain)
    explain.set_defaults(handler=run_explain)

    info = sub.add_parser("info", help="Show model, task, device, and checkpoint status")
    info.add_argument("--model-dir", type=Path, default=default_model_dir())
    info.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    info.add_argument("--verify-checksums", action="store_true")
    info.add_argument("--quiet", action="store_true")
    info.add_argument("--verbose", action="store_true")
    info.set_defaults(handler=run_info)
    return parser


def validate_common(args) -> None:
    if hasattr(args, "batch_size") and args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if getattr(args, "quiet", False) and getattr(args, "verbose", False):
        raise ValueError("--quiet and --verbose cannot be used together")


def prepare_frame(frame: pd.DataFrame, standardize: bool, deduplicate: bool) -> pd.DataFrame:
    result = frame.copy()
    processed = [process_smiles(value, standardize=standardize) for value in result["smiles"]]
    result["canonical_smiles"] = [item.canonical_smiles for item in processed]
    result["valid_smiles"] = [item.valid for item in processed]
    result["prediction_status"] = [item.status for item in processed]
    if deduplicate:
        valid = result["valid_smiles"]
        duplicate = valid & result.duplicated("canonical_smiles", keep="first")
        result = result.loc[~duplicate].copy()
    return result.reset_index(drop=True)


def add_predictions(frame: pd.DataFrame, args) -> pd.DataFrame:
    result = frame.copy()
    valid_indices = result.index[result["valid_smiles"]].tolist()
    smiles = result.loc[valid_indices, "canonical_smiles"].tolist()
    device = resolve_device(args.device)
    logging.info("Loading 10 ensemble models on %s...", device)
    models = load_models(args.model_dir, device)
    predictions = predict_ensemble(models, smiles, device, args.batch_size, quiet=args.quiet)
    summary = aggregate(predictions, save_individual=args.save_individual)
    summary.index = valid_indices
    return pd.concat([result, summary.reindex(result.index)], axis=1)


def run_predict(args) -> int:
    validate_common(args)
    frame = prepare_frame(load_input(args.input), args.standardize, args.deduplicate)
    result = add_predictions(frame, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    logging.info("Wrote %d rows to %s (%d invalid SMILES)", len(result), args.output, int((~result["valid_smiles"]).sum()))
    return 0


def validate_screen_args(args):
    validate_common(args)
    args.task = validate_task(args.task) if args.task else None
    if args.threshold is not None:
        args.threshold = probability(args.threshold, "--threshold")
        if args.task is None:
            raise ValueError("--threshold requires --task")
    if args.task is not None and args.threshold is None:
        args.threshold = 0.0
    args.min_sd = nonnegative(args.min_sd, "--min-sd") if args.min_sd is not None else None
    args.max_sd = nonnegative(args.max_sd, "--max-sd") if args.max_sd is not None else None
    if (args.min_sd is not None or args.max_sd is not None) and args.task is None:
        raise ValueError("--min-sd/--max-sd require --task")
    if args.min_sd is not None and args.max_sd is not None and args.min_sd > args.max_sd:
        raise ValueError("--min-sd cannot exceed --max-sd")
    if args.prediction_margin is not None:
        args.prediction_margin = probability(args.prediction_margin, "--selectivity-margin")
    if args.top is not None and args.top <= 0:
        raise ValueError("--top must be > 0")
    if args.mw_min is not None and args.mw_max is not None and args.mw_min > args.mw_max:
        raise ValueError("--mw-min cannot exceed --mw-max")
    if args.logp_min is not None and args.logp_max is not None and args.logp_min > args.logp_max:
        raise ValueError("--logp-min cannot exceed --logp-max")
    for name in ["hbd_max", "hba_max", "rotb_max", "tpsa_max", "lipinski_violations_max"]:
        if getattr(args, name) is not None and getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    if args.qed_min is not None:
        args.qed_min = probability(args.qed_min, "--qed-min")
    if args.fsp3_min is not None:
        args.fsp3_min = probability(args.fsp3_min, "--fsp3-min")
    include = [parse_task_cutoff(value, "include") for value in args.include]
    exclude = [parse_task_cutoff(value, "exclude") for value in args.exclude]
    conditions = [parse_condition(value) for value in args.condition]
    if args.task is None and not include:
        raise ValueError("screen requires --task, or at least one --include condition")
    profile_task = args.task or include[0].task
    return include, exclude, conditions, profile_task


def run_screen(args) -> int:
    include, exclude, conditions, profile_task = validate_screen_args(args)
    frame = prepare_frame(load_input(args.input), args.standardize, args.deduplicate)
    result = add_predictions(frame, args)
    valid = result["valid_smiles"]
    profiled = add_target_profile(result.loc[valid].copy(), profile_task)
    profile_columns = ["target_task", "target_probability", "target_sd", "highest_offtarget_task", "highest_offtarget_probability", "prediction_margin"]
    result = pd.concat([result, profiled[profile_columns].reindex(result.index)], axis=1)
    property_filters = property_filters_from_args(args)
    screened = apply_screening(
        result, args.task, args.threshold, args.min_sd, args.max_sd,
        include, exclude, conditions, args.prediction_margin, property_filters,
        args.include_descriptors or args.sort_by in {"qed", "mw"}, args.keep_all, args.sort_by, args.ascending, args.top,
    )
    screened.report["parameters"] = {
        "task": args.task, "profile_task": profile_task, "threshold": args.threshold,
        "min_sd": args.min_sd, "max_sd": args.max_sd,
        "include": args.include, "exclude": args.exclude, "condition": args.condition,
        "prediction_margin": args.prediction_margin, "property_filters": property_filters,
        "deduplicate": args.deduplicate, "standardize": args.standardize,
        "sort_by": args.sort_by, "ascending": args.ascending, "top": args.top,
        "filter_order": ["SMILES validation", "optional standardization", "ensemble prediction", "probability/include/condition", "ensemble SD", "off-target exclusion", "prediction margin", "medicinal chemistry", "sorting", "Top-N"],
    }
    output = screened.output
    if args.target_only_output:
        drop = [f"{task}_{suffix}" for task in TASKS if task != profile_task for suffix in ("mean", "sd")]
        if args.save_individual:
            drop += [f"{task}_model_{index}" for task in TASKS if task != profile_task for index in range(10)]
        output = output.drop(columns=[column for column in drop if column in output], errors="ignore")
    preferred = list(load_input(args.input).columns) + ["canonical_smiles", "valid_smiles", "prediction_status"]
    preferred += [item for task in TASKS for item in (f"{task}_mean", f"{task}_sd")]
    preferred += ["target_task", "target_probability", "target_sd", "highest_offtarget_task", "highest_offtarget_probability", "prediction_margin"]
    preferred += ["MW", "LogP", "HBD", "HBA", "RotB", "TPSA", "QED", "Fsp3", "Lipinski_violations", "pass_target_threshold", "pass_sd_filter", "pass_offtarget_filter", "pass_margin_filter", "pass_profile_filter", "pass_property_filter", "pass_all_filters"]
    output = ordered_columns(output, preferred)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    if args.filter_report:
        report_path = args.output.with_name(args.output.stem + "_report.json") if args.filter_report == "AUTO" else Path(args.filter_report)
        write_json(report_path, screened.report)
    logging.info("%d compounds retained; wrote %d rows to %s", screened.report["final_hits"], len(output), args.output)
    return 0


def run_explain(args) -> int:
    validate_common(args)
    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    from .utils import parse_models
    members = parse_models(args.models)
    device = resolve_device(args.device)
    values = [args.smiles] if args.smiles is not None else load_input(args.input)["smiles"].tolist()
    for index, value in enumerate(values):
        processed = process_smiles(value)
        if not processed.valid:
            logging.error("Skipping invalid SMILES at input row %d", index)
            continue
        destination = args.output if len(values) == 1 else args.output / f"molecule_{index + 1:04d}"
        csv_path, png_path = explain_smiles(processed.canonical_smiles, args.task, args.model_dir, device, members, args.samples, args.seed, destination)
        logging.info("Wrote %s and %s", csv_path, png_path)
    return 0


def run_info(args) -> int:
    validate_common(args)
    manifest = read_manifest(args.model_dir)
    status = verify_checkpoint_files(args.model_dir, check_hashes=args.verify_checksums)
    available = sum(item["present"] and item["checksum_ok"] for item in status)
    device = resolve_device(args.device)
    print(f"PDE-PRISM v{__version__}")
    print(f"Framework: Chemprop {manifest['chemprop_version']}; PyTorch {manifest['pytorch_version']}")
    print(f"Prediction tasks: {', '.join(manifest['task_order'])}")
    print(f"Ensemble: {manifest['ensemble_size']} models")
    print(f"Available device: {device}")
    print(f"Model status: {available}/{manifest['ensemble_size']} checkpoints available" + (" and verified" if args.verify_checksums else ""))
    print("PDE7 note: The PDE7 prediction task contains curated PDE7A- and PDE7B-associated training records and is not a PDE7A-specific predictor.")
    return 0 if available == manifest["ensemble_size"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(getattr(args, "quiet", False), getattr(args, "verbose", False))
    try:
        return int(args.handler(args))
    except Exception as exc:
        if getattr(args, "verbose", False):
            raise
        logging.error("Error: %s", exc)
        return 2
