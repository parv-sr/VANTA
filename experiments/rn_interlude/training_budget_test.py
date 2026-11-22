"""VANTA-RN interlude: validation-selected training-budget comparison.

Run from repository root:
 uv run python3 experiments/rn_interlude/training_budget_test.py --stage search --dataset concrete
 uv run python3 experiments/rn_interlude/training_budget_test.py --stage confirm --dataset concrete

Search uses seeds 101,102; confirmation uses seeds 42,43,44. The test
partition is never examined during search. Results live below
experiments/rn_interlude/results/standoff/<dataset>/.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[2]
EX03 = ROOT / "experiments" / "datasets_ex03"
RN = ROOT / "VANTA_RN"
sys.path.insert(0, str(EX03))
sys.path.insert(0, str(RN))

from datasets import load_dataset, preprocess  # noqa: E402
from model import NormalizedFeatureAttentionRegressor  # noqa: E402
import run_exp as ex03  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "standoff"
SEARCH_SEEDS = [101, 102]
CONFIRM_SEEDS = [42, 43, 44]

# Enlarged epochs, with validation early stopping. Model selection is per dataset.
BUDGETS = {"concrete": 1000, "ames": 650, "california": 450, "airbnb": 350}
VANTA_CONFIGS = {
    "rn_bs128_lr1e3_d24": dict(batch_size=128, lr=0.001, latent_dim=24, gamma_init=0.1),
    "rn_bs128_lr2e3_d24": dict(batch_size=128, lr=0.002, latent_dim=24, gamma_init=0.1),
    "rn_bs64_lr1e3_d24": dict(batch_size=64, lr=0.001, latent_dim=24, gamma_init=0.1),
    "rn_bs128_lr1e3_d48": dict(batch_size=128, lr=0.001, latent_dim=48, gamma_init=0.1),
}
MLP_CONFIGS = {
    "mlp_bs128_lr1e3": dict(batch_size=128, lr=0.001),
    "mlp_bs128_lr2e3": dict(batch_size=128, lr=0.002),
    "mlp_bs64_lr1e3": dict(batch_size=64, lr=0.001),
}
XGB_CONFIGS = {
    "xgb_d4": dict(max_depth=4, learning_rate=0.04, min_child_weight=5, reg_lambda=5.0),
    "xgb_d6": dict(max_depth=6, learning_rate=0.04, min_child_weight=5, reg_lambda=5.0),
}
CAT_CONFIGS = {
    "cat_d5": dict(depth=5, learning_rate=0.04, l2_leaf_reg=4.0),
    "cat_d7": dict(depth=7, learning_rate=0.04, l2_leaf_reg=4.0),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_vanta(data, cfg):
    n = len(data["numeric"])
    f = len(data["features"])
    return NormalizedFeatureAttentionRegressor(
        num_features=f,
        numeric_positions=list(range(n)),
        categorical_positions=list(range(n, f)),
        categorical_cardinalities=data["cardinalities"],
        latent_dim=cfg["latent_dim"],
        gamma_init=cfg["gamma_init"],
        learnable_gamma=True,
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_mlp(data, budget):
    # Use the actual MLP class and parameter matching convention from EX03.
    model, hidden_width = ex03.build_mlp(data, budget)
    return model, hidden_width


def evaluate_prediction(y, pred):
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mse = mean_squared_error(y, pred)
    return {"R2": float(r2_score(y, pred)), "RMSE": float(math.sqrt(mse)), "MAE": float(mean_absolute_error(y, pred))}


@torch.inference_mode()
def neural_predict(model, tensor_dataset, data, device):
    model.eval()
    chunks = []
    for xn, xc, _ in DataLoader(tensor_dataset, batch_size=1024, shuffle=False):
        chunks.append(model(xn.to(device), xc.to(device)).detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64) * data["target_std"] + data["target_mean"]


@torch.inference_mode()
def normalized_mse(model, tensor_dataset, device):
    model.eval()
    sse, n = 0.0, 0
    for xn, xc, target in DataLoader(tensor_dataset, batch_size=2048):
        err = model(xn.to(device), xc.to(device)) - target.to(device)
        sse += float(torch.sum(err.square()).item())
        n += len(target)
    return sse / n


def train_neural(data, model_name, cfg, seed, args, folder, vanta_params):
    set_seed(seed)
    if model_name == "VANTA-RN":
        model = build_vanta(data, cfg)
        hidden_width = None
    else:
        model, hidden_width = build_mlp(data, vanta_params)
    model = model.to(args.device)
    params = count_params(model)
    train_ds = data["splits"]["train"]["tensor"]
    val_ds = data["splits"]["val"]["tensor"]
    batches_per_epoch = math.ceil(len(train_ds) / cfg["batch_size"])
    print(f"{model_name}: params={params:,}, batches/epoch={batches_per_epoch}, epochs <= {args.epochs}", flush=True)
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=args.weight_decay)
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    history = []
    updates = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xn, xc, y in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xn.to(args.device), xc.to(args.device))
            loss = torch.mean((pred - y.to(args.device)) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
        if epoch != 1 and epoch % args.eval_every != 0 and epoch != args.epochs:
            continue
        train_loss = normalized_mse(model, train_ds, args.device)
        val_loss = normalized_mse(model, val_ds, args.device)
        gamma = float(model.gamma.item()) if hasattr(model, "gamma") else float("nan")
        history.append(dict(epoch=epoch, updates=updates, train_normalized_MSE=train_loss,
                            val_normalized_MSE=val_loss, gamma=gamma))
        if val_loss < best_val - 1e-7:
            best_val, best_state, best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch
        if epoch == 1 or epoch % 100 == 0:
            print(f" epoch={epoch:4d}, updates={updates:6d}, val_MSE={val_loss:.5f}, gamma={gamma:.3f}", flush=True)
        if epoch - best_epoch >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
    details = dict(best_epoch=best_epoch, total_updates=updates, parameters=params,
                   hidden_width=hidden_width, learned_gamma=(float(model.gamma.item()) if hasattr(model, "gamma") else None))
    torch.save(dict(state_dict=model.state_dict(), config=cfg, model=model_name,
                    version="VANTA-RN-v0.1" if model_name == "VANTA-RN" else "embedding-mlp",
                    seed=seed, metadata=details, numeric=data["numeric"],
                    categorical=data["categorical"], cardinalities=data["cardinalities"],
                    vocabularies=data["vocabularies"], target_mean=data["target_mean"],
                    target_std=data["target_std"]), folder / "checkpoint.pt")
    predict = lambda split: neural_predict(model, data["splits"][split]["tensor"], data, args.device)
    return predict, details


def train_xgb(data, cfg, seed, args, folder):
    train, val = data["splits"]["train"], data["splits"]["val"]
    model = XGBRegressor(n_estimators=args.trees, objective="reg:squarederror",
                         tree_method="hist", subsample=0.9, colsample_bytree=0.9,
                         random_state=seed, n_jobs=args.threads, eval_metric="rmse",
                         early_stopping_rounds=100, **cfg)
    model.fit(train["classical"], train["actual"],
              eval_set=[(val["classical"], val["actual"])], verbose=False)
    model.save_model(str(folder / "model.json"))
    return (lambda split: model.predict(data["splits"][split]["classical"])), dict(best_iteration=int(model.best_iteration))


def train_cat(data, cfg, seed, args, folder):
    train, val = data["splits"]["train"], data["splits"]["val"]
    model = CatBoostRegressor(iterations=args.trees, loss_function="RMSE",
                              cat_features=data["categorical"], random_seed=seed,
                              thread_count=args.threads, allow_writing_files=False,
                              verbose=False, **cfg)
    model.fit(train["catboost"], train["actual"],
              eval_set=(val["catboost"], val["actual"]),
              early_stopping_rounds=100, use_best_model=True, verbose=False)
    model.save_model(str(folder / "model.cbm"))
    return (lambda split: model.predict(data["splits"][split]["catboost"])), dict(best_iteration=int(model.best_iteration_))


def run_one(dataset_name, data, model, name, cfg, seed, args, output_root, vanta_params):
    folder = output_root / model / f"seed_{seed}" / name
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "result.json"
    fingerprint = dict(dataset=dataset_name, stage=args.stage, model=model, name=name, seed=seed,
                       config=cfg, epochs=args.epochs, patience=args.patience,
                       eval_every=args.eval_every, weight_decay=args.weight_decay,
                       trees=args.trees, vanta_parameters=vanta_params)
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["fingerprint"] != fingerprint:
            raise RuntimeError(f"Configuration mismatch in {folder}. Use --output for a new study.")
        print(f"Skipping completed {model} {name}, seed {seed}", flush=True)
        return saved["metrics"]
    print(f"\n{dataset_name} / {model} / {name} / seed={seed}", flush=True)
    if model in ("VANTA-RN", "MLP"):
        predict, details = train_neural(data, model, cfg, seed, args, folder, vanta_params)
    elif model == "XGBoost":
        predict, details = train_xgb(data, cfg, seed, args, folder)
    elif model == "CatBoost":
        predict, details = train_cat(data, cfg, seed, args, folder)
    else:
        raise ValueError(model)
    # SEARCH: No test inference, no test metric, no test inspection.
    splits = ["val"] if args.stage == "search" else ["val", "test"]
    metrics = dict(dataset=dataset_name, model=model, config_name=name, seed=seed, **details)
    for split in splits:
        pred = np.asarray(predict(split), dtype=np.float64)
        truth = data["splits"][split]["actual"]
        metrics.update({f"{split}_{key}": value for key, value in evaluate_prediction(truth, pred).items()})
        if split == "test":
            pd.DataFrame({"row_index": data["splits"][split]["indices"],
                          "actual": truth, "prediction": pred}).to_csv(folder / "test_predictions.csv", index=False)
    marker.write_text(json.dumps(dict(fingerprint=fingerprint, metrics=metrics), indent=2))
    return metrics


def run_search(dataset, data_by_seed, args, output_root):
    # Select VANTA architecture first, then match MLP parameter count to that selected architecture.
    selection = {}
    all_results = []
    configs_by_model = {"VANTA-RN": VANTA_CONFIGS, "MLP": MLP_CONFIGS,
                        "XGBoost": XGB_CONFIGS, "CatBoost": CAT_CONFIGS}
    for model in ("VANTA-RN", "MLP", "XGBoost", "CatBoost"):
        group = []
        for config_name, config in configs_by_model[model].items():
            for seed, data in data_by_seed.items():
                reference_cfg = (config if model == "VANTA-RN" else selection["VANTA-RN"]["config"])
                budget = count_params(build_vanta(data, reference_cfg))
                row = run_one(dataset, data, model, config_name, config, seed, args, output_root, budget)
                group.append(row)
                all_results.append(row)
        table = pd.DataFrame(group).groupby("config_name")["val_RMSE"].agg(["mean", "std", "count"])
        table.to_csv(output_root / f"search_{model.replace('-', '_')}_ranking.csv")
        winner = str(table["mean"].idxmin())
        selection[model] = dict(name=winner, config=configs_by_model[model][winner],
                                mean_val_RMSE=float(table.loc[winner, "mean"]))
        print(f"SELECTED {model}: {winner}, validation RMSE={table.loc[winner, 'mean']:,.3f}", flush=True)
    selection["study"] = dict(search_seeds=list(data_by_seed), epochs=args.epochs,
                              patience=args.patience, trees=args.trees, eval_every=args.eval_every,
                              weight_decay=args.weight_decay)
    selected_file = output_root / "selected.json"
    if selected_file.exists() and (output_root / "confirmation_metrics.csv").exists():
        previous = json.loads(selected_file.read_text())
        if previous != selection:
            raise RuntimeError("Selection changed after confirmation. Use a separate --output directory.")
    selected_file.write_text(json.dumps(selection, indent=2))
    pd.DataFrame(all_results).to_csv(output_root / "search_metrics.csv", index=False)


def run_confirm(dataset, data_by_seed, args, output_root):
    path = output_root / "selected.json"
    if not path.exists():
        raise RuntimeError("Run --stage search first")
    selection = json.loads(path.read_text())
    expected = selection["study"]
    current = dict(epochs=args.epochs, patience=args.patience, trees=args.trees,
                   eval_every=args.eval_every, weight_decay=args.weight_decay)
    if any(current[k] != expected[k] for k in current):
        raise RuntimeError(f"Budget differs from search: {expected}. Reuse search settings.")
    if set(data_by_seed) & set(expected["search_seeds"]):
        raise RuntimeError("Confirmation seeds must not overlap search seeds")
    results = []
    for model in ("VANTA-RN", "MLP", "XGBoost", "CatBoost"):
        name = selection[model]["name"]
        cfg = selection[model]["config"]
        for seed, data in data_by_seed.items():
            budget = count_params(build_vanta(data, selection["VANTA-RN"]["config"]))
            row = run_one(dataset, data, model, name, cfg, seed, args, output_root, budget)
            results.append(row)
    df = pd.DataFrame(results)
    df.to_csv(output_root / "confirmation_metrics.csv", index=False)
    summary = []
    for model, group in df.groupby("model"):
        row = dict(model=model, n=len(group))
        for metric in ("R2", "RMSE", "MAE"):
            values = group[f"test_{metric}"].to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        summary.append(row)
    summary_df = pd.DataFrame(summary).sort_values("RMSE_mean")
    summary_df.to_csv(output_root / "confirmation_summary.csv", index=False)
    print("\nCONFIRMATION: TEST mean +/- sample SD\n" + summary_df.to_string(index=False), flush=True)
    pivot = df.pivot(index="seed", columns="model", values="test_RMSE")
    paired = []
    for baseline in ("MLP", "XGBoost", "CatBoost"):
        differences = pivot[baseline] - pivot["VANTA-RN"]
        paired.append(dict(comparison=f"VANTA-RN vs {baseline}", seeds=len(differences),
                           RN_wins=int((differences > 0).sum()),
                           mean_RMSE_advantage=float(differences.mean()),
                           sd_RMSE_advantage=float(differences.std(ddof=1))))
    pd.DataFrame(paired).to_csv(output_root / "paired_comparisons.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(BUDGETS), required=True)
    parser.add_argument("--stage", choices=["search", "confirm"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--trees", type=int, default=2500)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    args.epochs = args.epochs or BUDGETS[args.dataset]
    args.seeds = args.seeds or (SEARCH_SEEDS if args.stage == "search" else CONFIRM_SEEDS)
    if len(args.seeds) != len(set(args.seeds)) or min(args.epochs, args.patience, args.eval_every, args.trees, args.threads) <= 0:
        parser.error("Invalid budgets or duplicate seeds")
    torch.set_num_threads(args.threads)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = args.output / args.dataset
    output_root.mkdir(parents=True, exist_ok=True)
    bundle = load_dataset(args.dataset)
    data_by_seed = {seed: preprocess(bundle, seed) for seed in args.seeds}
    if args.stage == "search":
        run_search(args.dataset, data_by_seed, args, output_root)
    else:
        run_confirm(args.dataset, data_by_seed, args, output_root)


if __name__ == "__main__":
    main()

