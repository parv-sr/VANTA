"""VANTA-RN input-tail study: validation-selected numeric transforms.

From repository root:
 uv run python3 experiments/rn_interlude/preprocessing_experiment.py --dataset ames --stage search
 uv run python3 experiments/rn_interlude/preprocessing_experiment.py --dataset ames --stage confirm

Search evaluates validation only; confirmation uses disjoint seeds.
Numeric policies are fit on TRAIN-standardized values; model/target/categories unchanged.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "datasets_ex03"))
sys.path.insert(0, str(HERE))
from datasets import load_dataset, preprocess
from training_budget_test import train_neural, evaluate_prediction
from robust_numeric import VARIANTS, fit_policy, apply_policy

OUT = HERE / "results" / "preprocessing"
CONFIG = {
    "ames": dict(batch_size=128, lr=0.001, latent_dim=24, gamma_init=0.1, epochs=650),
    "california": dict(batch_size=128, lr=0.002, latent_dim=24, gamma_init=0.1, epochs=450),
    "concrete": dict(batch_size=128, lr=0.001, latent_dim=24, gamma_init=0.1, epochs=1000),
    "airbnb": dict(batch_size=128, lr=0.002, latent_dim=24, gamma_init=0.1, epochs=350),
}
SEARCH_SEEDS = (301, 302)
CONFIRM_SEEDS = (303, 304, 305)


def make_numeric_variant(base, variant):
    """Build new tensors without modifying EX03's baseline and category encodings."""
    policy = fit_policy(base["splits"]["train"]["numeric"], variant)
    changed = dict(base)
    changed["splits"] = {}
    for split, part in base["splits"].items():
        numeric = apply_policy(part["numeric"], policy)
        _, cat, target = part["tensor"].tensors
        transformed = dict(part)
        transformed["numeric"] = numeric
        transformed["tensor"] = TensorDataset(torch.from_numpy(numeric), cat, target)
        # Classical arrays are irrelevant here: only VANTA-RN is trained.
        changed["splits"][split] = transformed
    return changed, policy


def run_one(bundle, dataset, seed, variant, stage, cfg, args):
    folder = OUT / dataset / stage / f"seed_{seed}" / variant
    folder.mkdir(parents=True, exist_ok=True)
    metadata = dict(stage=stage, seed=seed, variant=variant, dataset=dataset,
                    config=cfg, patience=args.patience, eval_every=args.eval_every,
                    weight_decay=args.weight_decay)
    config_file, result_file = folder / "config.json", folder / "metrics.json"
    if config_file.exists() and json.loads(config_file.read_text()) != metadata:
        raise RuntimeError(f"Existing run configuration changed: {folder}")
    if result_file.exists():
        return json.loads(result_file.read_text())
    base = preprocess(bundle, seed)
    data, policy = make_numeric_variant(base, variant)
    config_file.write_text(json.dumps(metadata, indent=2))
    (folder / "numeric_policy.json").write_text(json.dumps(policy, indent=2))
    train_args = argparse.Namespace(device=args.device, epochs=cfg["epochs"],
                    patience=args.patience, eval_every=args.eval_every,
                    weight_decay=args.weight_decay)
    predictor, details = train_neural(data, "VANTA-RN", cfg, seed,
                                      train_args, folder, vanta_params=0)
    val = evaluate_prediction(data["splits"]["val"]["actual"], predictor("val"))
    row = dict(dataset=dataset, variant=variant, seed=seed, stage=stage,
               best_epoch=details["best_epoch"],
               total_updates=details["total_updates"],
               learned_gamma=details["learned_gamma"],
               **{f"val_{k}": v for k, v in val.items()})
    if stage == "confirm":
        part = data["splits"]["test"]
        prediction = predictor("test")
        scores = evaluate_prediction(part["actual"], prediction)
        row.update({f"test_{k}": v for k, v in scores.items()})
        pd.DataFrame(dict(row_index=part["indices"],
                          actual=part["actual"], prediction=prediction)).to_csv(
                          folder / "test_predictions.csv", index=False)
    result_file.write_text(json.dumps(row, indent=2))
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=CONFIG, required=True)
    p.add_argument("--stage", choices=("search", "confirm"), required=True)
    p.add_argument("--search-seeds", nargs="+", type=int, default=list(SEARCH_SEEDS))
    p.add_argument("--confirm-seeds", nargs="+", type=int, default=list(CONFIRM_SEEDS))
    p.add_argument("--patience", type=int, default=120)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if set(args.search_seeds).intersection(args.confirm_seeds):
        p.error("Search and confirmation seeds must be disjoint")
    if len(set(args.search_seeds)) != len(args.search_seeds) or len(set(args.confirm_seeds)) != len(args.confirm_seeds):
        p.error("Seeds must be unique within each phase")
    if min(args.threads, args.patience, args.eval_every) <= 0:
        p.error("Invalid positive training settings")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(args.threads)
    study = OUT / args.dataset
    study.mkdir(parents=True, exist_ok=True)
    selected = study / "selection.json"
    cfg = CONFIG[args.dataset]
    bundle = load_dataset(args.dataset)
    if args.stage == "search":
        seeds, variants = args.search_seeds, list(VARIANTS)
        rows = [run_one(bundle, args.dataset, s, v, "search", cfg, args)
                for s in seeds for v in variants]
        df = pd.DataFrame(rows)
        df.to_csv(study / "search_metrics.csv", index=False)
        ranked = df.groupby("variant").val_RMSE.mean().sort_values()
        winner = ranked.index[0]
        selection = dict(variant=winner, search_seeds=seeds, config=cfg,
                         patience=args.patience, eval_every=args.eval_every,
                         weight_decay=args.weight_decay)
        if selected.exists() and json.loads(selected.read_text()) != selection:
            raise RuntimeError("Existing selection differs; archive before changing search")
        selected.write_text(json.dumps(selection, indent=2))
        print("\nValidation-only selection:\n" + ranked.to_string())
        print("Selected:", winner)
    else:
        if not selected.exists():
            p.error("Run --stage search first")
        selection = json.loads(selected.read_text())
        if selection["config"] != cfg or selection["search_seeds"] != args.search_seeds:
            p.error("Search seeds/config differ from selected study")
        if (selection["patience"], selection["eval_every"], selection["weight_decay"]) != (
              args.patience, args.eval_every, args.weight_decay):
            p.error("Training settings differ from selected study")
        seeds = args.confirm_seeds
        variants = list(dict.fromkeys(["standard", selection["variant"]]))
        rows = [run_one(bundle, args.dataset, s, v, "confirm", cfg, args)
                for s in seeds for v in variants]
        df = pd.DataFrame(rows)
        df.to_csv(study / "confirmation_metrics.csv", index=False)
        agg = df.groupby("variant").agg(n=("seed", "size"), R2=("test_R2", "mean"),
            R2_sd=("test_R2", "std"), RMSE=("test_RMSE", "mean"),
            RMSE_sd=("test_RMSE", "std"), MAE=("test_MAE", "mean"),
            MAE_sd=("test_MAE", "std"))
        agg.to_csv(study / "confirmation_summary.csv")
        print("\nConfirmation:\n" + agg.to_string())
        if len(variants) > 1:
            pivot = df.pivot(index="seed", columns="variant", values="test_RMSE")
            paired = pivot[selection["variant"]] - pivot["standard"]
            print("\nPaired RMSE delta (selected minus original; negative is better):\n" + paired.to_string())
            print("Mean delta:", float(paired.mean()))
    print("Results:", study)


if __name__ == "__main__":
    main()
