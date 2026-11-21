
"""VANTA EX03 — Cross-Dataset Generalization Benchmark."""

import argparse
import copy
import importlib.util
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from torch import nn
from torch.utils.data import DataLoader

from datasets import REGISTRY, load_dataset, preprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"

LATENT_DIM = 24
EMBED_DIM = 8

MODEL_NAMES = [
    "VANTA-R",
    "Embedding MLP",
    "Ridge",
    "XGBoost",
    "CatBoost",
]


# ---------------------------------------------------------
# Load the actual VANTA-R architecture.
# ---------------------------------------------------------

def load_vanta_class():
    path = ROOT / "VANTA_R" / "model.py"

    spec = importlib.util.spec_from_file_location(
        "vanta_r_ex03", path
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.ResidualFeatureAttentionRegressor


VantaR = load_vanta_class()


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_count(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def score(y, pred, train_std):
    mse = mean_squared_error(y, pred)
    rmse = math.sqrt(mse)

    return {
        "R2": float(r2_score(y, pred)),
        "RMSE": float(rmse),
        "MAE": float(mean_absolute_error(y, pred)),
        "MSE": float(mse),
        "NRMSE": float(rmse / train_std),
    }


def save_history(train, val, folder, title,
                 xlabel="Epoch", ylabel="Loss"):
    folder.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame({
        "step": np.arange(1, len(val) + 1),
        "train": train,
        "validation": val,
    })

    frame.to_csv(folder / "history.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(frame["step"], frame["train"], label="Train")
    plt.plot(frame["step"], frame["validation"], label="Validation")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(folder / "loss_curve.png", dpi=180)
    plt.close()


def collect_scores(predict_fn, data, model_name,
                   params=None, details=None, folder=None):
    result = {
        "model": model_name,
        "parameters": params,
        "details": details or {},
    }

    for split in ("train", "val", "test"):
        part = data["splits"][split]
        prediction = np.asarray(
            predict_fn(split),
            dtype=np.float64,
        ).reshape(-1)

        if len(prediction) != len(part["actual"]):
            raise RuntimeError("Prediction length mismatch.")

        measures = score(
            part["actual"],
            prediction,
            data["target_std"],
        )

        for key, value in measures.items():
            result[f"{split}_{key}"] = value

        if split == "test" and folder is not None:
            pd.DataFrame({
                "row_index": part["indices"],
                "actual": part["actual"],
                "prediction": prediction,
            }).to_csv(
                folder / "test_predictions.csv",
                index=False,
            )

    return result


# ---------------------------------------------------------
# Neural models
# ---------------------------------------------------------

class EmbeddingMLP(nn.Module):
    def __init__(self, num_numeric, cardinalities,
                 hidden_dim):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(c, EMBED_DIM, padding_idx=0)
            for c in cardinalities
        ])

        input_dim = (
            num_numeric + len(cardinalities) * EMBED_DIM
        )

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_num, x_cat):
        chunks = [x_num]

        for i, embedding in enumerate(self.embeddings):
            chunks.append(embedding(x_cat[:, i]))

        x = torch.cat(chunks, dim=1)
        return self.network(x).squeeze(-1)


def build_vanta(data):
    n_num = len(data["numeric"])
    n_cat = len(data["categorical"])
    n_features = n_num + n_cat

    return VantaR(
        num_features=n_features,
        numeric_positions=list(range(n_num)),
        categorical_positions=list(
            range(n_num, n_features)
        ),
        categorical_cardinalities=data["cardinalities"],
        latent_dim=LATENT_DIM,
        gamma_init=0.1,
        learnable_gamma=True,
    )


def build_mlp(data, target_params):
    n_num = len(data["numeric"])
    cards = data["cardinalities"]

    input_dim = n_num + len(cards) * EMBED_DIM
    embedding_params = sum(cards) * EMBED_DIM

    # Embeddings + input-to-hidden + hidden-to-output.
    width = round(
        (target_params - embedding_params - 1)
        / (input_dim + 2)
    )

    width = max(1, width)

    return EmbeddingMLP(n_num, cards, width), width


@torch.no_grad()
def neural_mse(model, loader, device):
    model.eval()

    total = 0.0
    count = 0

    for xn, xc, y in loader:
        prediction = model(
            xn.to(device),
            xc.to(device),
        )

        error = (
            prediction - y.to(device)
        ).square()

        total += error.sum().item()
        count += len(y)

    return total / count


@torch.no_grad()
def neural_predict(model, dataset, data, device):
    model.eval()

    loader = DataLoader(dataset, batch_size=2048)
    predictions = []

    for xn, xc, _ in loader:
        prediction = model(
            xn.to(device),
            xc.to(device),
        )

        predictions.append(
            prediction.cpu().numpy()
        )

    normalized = np.concatenate(predictions)

    return (
        normalized * data["target_std"]
        + data["target_mean"]
    )


def train_neural(name, seed, data, args, folder):
    seed_everything(seed)

    reference = build_vanta(data)
    vanta_params = parameter_count(reference)

    if name == "VANTA-R":
        model = reference
        hidden_dim = None
    else:
        model, hidden_dim = build_mlp(
            data, vanta_params
        )

    model = model.to(args.device)
    params = parameter_count(model)

    print(
        f"  {name}: {params:,} parameters"
        f" | VANTA reference: {vanta_params:,}"
    )

    if hidden_dim is not None:
        print(f"  MLP hidden width: {hidden_dim}")

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        data["splits"]["train"]["tensor"],
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    eval_loaders = {
        split: DataLoader(
            data["splits"][split]["tensor"],
            batch_size=2048,
            shuffle=False,
        )
        for split in ("train", "val", "test")
    }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    train_history = []
    val_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        for xn, xc, y in train_loader:
            xn = xn.to(args.device)
            xc = xc.to(args.device)
            y = y.to(args.device)

            optimizer.zero_grad(set_to_none=True)

            prediction = model(xn, xc)
            loss = criterion(prediction, y)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss: {name}, seed {seed}"
                )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

        train_mse = neural_mse(
            model, eval_loaders["train"], args.device
        )

        val_mse = neural_mse(
            model, eval_loaders["val"], args.device
        )

        train_history.append(train_mse)
        val_history.append(val_mse)

        if val_mse < best_loss - 1e-7:
            best_loss = val_mse
            best_state = copy.deepcopy(
                model.state_dict()
            )
            best_epoch = epoch

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"  Epoch {epoch:3d} | "
                f"train={train_mse:.5f} | "
                f"val={val_mse:.5f}"
            )

        if epoch - best_epoch >= args.patience:
            print(f"  Early stopping at {epoch}")
            break

    model.load_state_dict(best_state)

    save_history(
        train_history,
        val_history,
        folder,
        f"{name} | seed {seed}",
        ylabel="Normalized MSE",
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": name,
            "seed": seed,
            "best_epoch": best_epoch,
            "parameters": params,
            "hidden_dim": hidden_dim,
            "numeric": data["numeric"],
            "categorical": data["categorical"],
            "cardinalities": data["cardinalities"],
            "vocabularies": data["vocabularies"],
            "target_mean": data["target_mean"],
            "target_std": data["target_std"],
        },
        folder / "checkpoint.pt",
    )

    result = collect_scores(
        lambda split: neural_predict(
            model,
            data["splits"][split]["tensor"],
            data,
            args.device,
        ),
        data,
        name,
        params=params,
        details={
            "best_epoch": best_epoch,
            "hidden_dim": hidden_dim,
            "best_val_normalized_MSE": best_loss,
        },
        folder=folder,
    )

    return result


# ---------------------------------------------------------
# Ridge: validation-selected regularization
# ---------------------------------------------------------

def train_ridge(data, folder):
    print("  Fitting Ridge")

    train = data["splits"]["train"]
    val = data["splits"]["val"]

    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]

    best_model = None
    best_alpha = None
    best_rmse = float("inf")

    sweep = []

    for alpha in alphas:
        model = Ridge(
            alpha=alpha,
            solver="lsqr",
        )

        model.fit(
            train["classical"], train["actual"]
        )

        pred = model.predict(val["classical"])

        rmse = math.sqrt(
            mean_squared_error(val["actual"], pred)
        )

        sweep.append({
            "alpha": alpha,
            "validation_RMSE": rmse,
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_alpha = alpha

    pd.DataFrame(sweep).to_csv(
        folder / "ridge_alpha_sweep.csv",
        index=False,
    )

    return collect_scores(
        lambda split: best_model.predict(
            data["splits"][split]["classical"]
        ),
        data,
        "Ridge",
        params=int(best_model.coef_.size + 1),
        details={"best_alpha": best_alpha},
        folder=folder,
    )


# ---------------------------------------------------------
# XGBoost
# ---------------------------------------------------------

def train_xgboost(seed, data, args, folder):
    print("  Training XGBoost")

    train = data["splits"]["train"]
    val = data["splits"]["val"]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.trees,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=5,
        reg_lambda=5.0,
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=60,
        random_state=seed,
        n_jobs=args.threads,
    )

    model.fit(
        train["classical"],
        train["actual"],
        eval_set=[
            (train["classical"], train["actual"]),
            (val["classical"], val["actual"]),
        ],
        verbose=False,
    )

    history = model.evals_result()

    save_history(
        history["validation_0"]["rmse"],
        history["validation_1"]["rmse"],
        folder,
        f"XGBoost | seed {seed}",
        xlabel="Boosting round",
        ylabel="RMSE (target units)",
    )

    model.save_model(str(folder / "model.json"))

    return collect_scores(
        lambda split: model.predict(
            data["splits"][split]["classical"]
        ),
        data,
        "XGBoost",
        details={
            "best_iteration": int(model.best_iteration),
            "max_depth": 5,
        },
        folder=folder,
    )


# ---------------------------------------------------------
# CatBoost
# ---------------------------------------------------------

def train_catboost(seed, data, args, folder):
    print("  Training CatBoost")

    train = data["splits"]["train"]
    val = data["splits"]["val"]

    model = CatBoostRegressor(
        iterations=args.trees,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        eval_metric="RMSE",
        cat_features=data["categorical"],
        random_seed=seed,
        thread_count=args.threads,
        allow_writing_files=False,
        verbose=False,
    )

    model.fit(
        train["catboost"],
        train["actual"],
        eval_set=(
            val["catboost"],
            val["actual"],
        ),
        early_stopping_rounds=60,
        use_best_model=True,
        verbose=False,
    )

    history = model.get_evals_result()

    save_history(
        history["learn"]["RMSE"],
        history["validation"]["RMSE"],
        folder,
        f"CatBoost | seed {seed}",
        xlabel="Boosting round",
        ylabel="RMSE (target units)",
    )

    model.save_model(str(folder / "model.cbm"))

    return collect_scores(
        lambda split: model.predict(
            data["splits"][split]["catboost"]
        ),
        data,
        "CatBoost",
        details={
            "best_iteration": int(model.best_iteration_),
            "tree_count": int(model.tree_count_),
            "depth": 6,
        },
        folder=folder,
    )


# ---------------------------------------------------------
# Aggregate and visualize
# ---------------------------------------------------------

METRICS = ["R2", "RMSE", "MAE", "MSE", "NRMSE"]


def summarize(rows):
    df = pd.DataFrame(rows)

    output = []

    for (dataset, model), group in df.groupby(
        ["dataset", "model"], sort=False
    ):
        row = {
            "dataset": dataset,
            "model": model,
            "runs": len(group),
            "parameters": group["parameters"].dropna().iloc[0]
            if group["parameters"].notna().any()
            else np.nan,
        }

        for metric in METRICS:
            values = group[f"test_{metric}"].to_numpy(
                dtype=float
            )

            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = (
                values.std(ddof=1)
                if len(values) > 1
                else np.nan
            )

        output.append(row)

    return pd.DataFrame(output)


def paired_comparisons(df):
    rows = []

    for dataset, group in df.groupby("dataset"):
        baseline = group[
            group["model"] == "VANTA-R"
        ][[
            "seed", "test_RMSE", "test_R2"
        ]].rename(columns={
            "test_RMSE": "vanta_rmse",
            "test_R2": "vanta_r2",
        })

        for model in MODEL_NAMES:
            if model == "VANTA-R":
                continue

            other = group[
                group["model"] == model
            ][["seed", "test_RMSE", "test_R2"]]

            paired = baseline.merge(
                other,
                on="seed",
                how="inner",
            )

            if paired.empty:
                continue

            # Positive RMSE advantage means VANTA is better.
            advantage_rmse = (
                paired["test_RMSE"]
                - paired["vanta_rmse"]
            )

            advantage_r2 = (
                paired["vanta_r2"]
                - paired["test_R2"]
            )

            rows.append({
                "dataset": dataset,
                "comparison": f"VANTA-R vs {model}",
                "paired_seeds": len(paired),
                "VANTA_wins": int(
                    (advantage_rmse > 0).sum()
                ),
                "RMSE_advantage_mean": advantage_rmse.mean(),
                "RMSE_advantage_std": advantage_rmse.std(ddof=1),
                "R2_advantage_mean": advantage_r2.mean(),
                "R2_advantage_std": advantage_r2.std(ddof=1),
            })

    return pd.DataFrame(rows)


def comparison_plots(summary):
    for dataset, group in summary.groupby("dataset"):
        folder = OUT / dataset

        # Preserve consistent model order.
        group = group.set_index("model").reindex(
            [m for m in MODEL_NAMES if m in group["model"].values]
        )

        for metric, xlabel, filename in [
            ("RMSE", "Test RMSE (lower is better)",
             "rmse_mean_std.png"),
            ("R2", "Test R² (higher is better)",
             "r2_mean_std.png"),
        ]:
            means = group[f"{metric}_mean"].to_numpy()
            stds = group[f"{metric}_std"].fillna(0).to_numpy()

            plt.figure(figsize=(10, 5))
            plt.barh(
                group.index.tolist(),
                means,
                xerr=stds,
                capsize=4,
            )

            plt.gca().invert_yaxis()
            plt.xlabel(xlabel)
            plt.title(
                f"{dataset}: mean ± sample SD across seeds"
            )
            plt.grid(axis="x", alpha=0.2)
            plt.tight_layout()
            plt.savefig(
                folder / filename,
                dpi=180,
            )
            plt.close()


def print_summary(summary):
    for dataset, group in summary.groupby(
        "dataset", sort=False
    ):
        print("\n" + "=" * 90)
        print(f"DATASET: {dataset.upper()}")
        print("=" * 90)

        for _, row in group.iterrows():
            print(
                f"{row['model']:16s} | "
                f"R2 {row['R2_mean']:+.5f} "
                f"± {row['R2_std']:.5f} | "
                f"RMSE {row['RMSE_mean']:,.3f} "
                f"± {row['RMSE_std']:,.3f} | "
                f"MAE {row['MAE_mean']:,.3f} "
                f"± {row['MAE_std']:,.3f}"
            )


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(REGISTRY),
        default=list(REGISTRY),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
    )

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--trees", type=int, default=1200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if min(
        args.epochs,
        args.batch_size,
        args.patience,
        args.trees,
        args.threads,
    ) <= 0:
        parser.error("Training settings must be positive.")

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate seeds are not allowed.")

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    torch.set_num_threads(args.threads)

    OUT.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "datasets_ex03",
        "datasets": args.datasets,
        "seeds": args.seeds,
        "latent_dim": LATENT_DIM,
        "embedding_dim": EMBED_DIM,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "lr": args.lr,
        "trees": args.trees,
        "split_protocol": "varying 80/10/10 splits",
        "airbnb_group_split": "host_id",
    }

    manifest_path = OUT / "manifest.json"
    metrics_path = OUT / "metrics.csv"

    if manifest_path.exists():
        existing_manifest = json.loads(
            manifest_path.read_text()
        )

        if existing_manifest != manifest:
            raise RuntimeError(
                "Existing results use different experiment "
                "settings. Move the results directory before "
                "starting a new configuration."
            )

        if not args.resume:
            raise RuntimeError(
                "Results already exist. Pass --resume to "
                "continue without repeating finished runs."
            )

    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2)
        )

    if metrics_path.exists():
        rows = pd.read_csv(metrics_path).to_dict(
            orient="records"
        )
    else:
        rows = []

    completed = {
        (r["dataset"], int(r["seed"]), r["model"])
        for r in rows
    }

    print("=" * 72)
    print("VANTA — EXPERIMENT 03: CROSS-DATASET GENERALIZATION")
    print("=" * 72)
    print("Device:", args.device)
    print("Datasets:", args.datasets)
    print("Seeds:", args.seeds)
    print("Models:", MODEL_NAMES)

    for dataset_name in args.datasets:
        bundle = load_dataset(dataset_name)

        for seed in args.seeds:
            print("\n" + "#" * 72)
            print(f"{dataset_name.upper()} | SEED {seed}")
            print("#" * 72)

            if all(
                (dataset_name, seed, name) in completed
                for name in MODEL_NAMES
            ):
                print("All models already completed. Skipping.")
                continue

            seed_everything(seed)
            data = preprocess(bundle, seed)

            for split in ("train", "val", "test"):
                print(
                    f"{split}: "
                    f"{len(data['splits'][split]['actual']):,}"
                )

            for name in MODEL_NAMES:
                key = (dataset_name, seed, name)

                if key in completed:
                    print(f"Skipping completed model: {name}")
                    continue

                print("\n" + "-" * 65)
                print(f"{dataset_name} | seed {seed} | {name}")
                print("-" * 65)

                safe_name = name.lower().replace(
                    " ", "_"
                ).replace("-", "_")

                folder = (
                    OUT / dataset_name
                    / f"seed_{seed}"
                    / safe_name
                )

                folder.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                if name in ("VANTA-R", "Embedding MLP"):
                    result = train_neural(
                        name, seed, data, args, folder
                    )

                elif name == "Ridge":
                    result = train_ridge(data, folder)

                elif name == "XGBoost":
                    result = train_xgboost(
                        seed, data, args, folder
                    )

                elif name == "CatBoost":
                    result = train_catboost(
                        seed, data, args, folder
                    )

                result["dataset"] = dataset_name
                result["seed"] = seed

                # Store details as serialized JSON so the
                # CSV remains portable and easy to inspect.
                result["details"] = json.dumps(
                    result["details"]
                )

                rows.append(result)
                completed.add(key)

                # Persist after every successfully completed model.
                pd.DataFrame(rows).to_csv(
                    metrics_path,
                    index=False,
                )

                print(
                    f"RESULT | {name} | "
                    f"R2={result['test_R2']:.5f} | "
                    f"RMSE={result['test_RMSE']:,.3f}"
                )

        # Generate available results after each dataset too.
        current = pd.DataFrame(rows)

        summary = summarize(current)
        comparisons = paired_comparisons(current)

        summary.to_csv(
            OUT / "summary.csv", index=False
        )

        comparisons.to_csv(
            OUT / "paired_comparisons.csv",
            index=False,
        )

        comparison_plots(summary)
        print_summary(
            summary[summary["dataset"] == dataset_name]
        )

    print("\n" + "=" * 72)
    print("EXPERIMENT 03 COMPLETE")
    print("=" * 72)
    print("Results:", OUT)


if __name__ == "__main__":
    main()

