
"""
VANTA-RN Hyperparameter Investigation.

Stage 1: Validation-only search (default seed 101).
Stage 2: Confirmation across seeds 42, 43, 44.

No test metrics are calculated during search.

Usage:
    uv run python3 experiments/rn_interlude/tune.py \
        --stage search --dataset concrete

    uv run python3 experiments/rn_interlude/tune.py \
        --stage confirm --dataset concrete
"""

import argparse
import copy
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from torch import nn
from torch.utils.data import DataLoader

from common import (
    OUT,
    EX03,
    EX04,
    load_dataset,
    preprocess,
    make_model,
    seed_all,
    model_parameters,
)


PRESETS = {
    "reference_long": dict(
        latent_dim=24,
        lr=0.001,
        batch_size=512,
        gamma_init=0.1,
        weight_decay=1e-4,
    ),
    "batch_128": dict(
        latent_dim=24,
        lr=0.001,
        batch_size=128,
        gamma_init=0.1,
        weight_decay=1e-4,
    ),
    "lr_high": dict(
        latent_dim=24,
        lr=0.002,
        batch_size=512,
        gamma_init=0.1,
        weight_decay=1e-4,
    ),
    "lr_low": dict(
        latent_dim=24,
        lr=0.0005,
        batch_size=512,
        gamma_init=0.1,
        weight_decay=1e-4,
    ),
    "latent_48": dict(
        latent_dim=48,
        lr=0.001,
        batch_size=512,
        gamma_init=0.1,
        weight_decay=1e-4,
    ),
    "gamma_05": dict(
        latent_dim=24,
        lr=0.001,
        batch_size=512,
        gamma_init=0.5,
        weight_decay=1e-4,
    ),
    "wd_high": dict(
        latent_dim=24,
        lr=0.001,
        batch_size=512,
        gamma_init=0.1,
        weight_decay=1e-3,
    ),
}

DEFAULT_EPOCHS = {
    "concrete": 600,
    "ames": 400,
    "california": 300,
    "airbnb": 250,
}


# ----------------------------------------------------------
# Evaluation
# ----------------------------------------------------------

@torch.no_grad()
def predict_normalized(model, dataset, device):

    model.eval()
    chunks = []

    loader = DataLoader(
        dataset,
        batch_size=2048,
    )

    for xn, xc, _ in loader:

        output = model(
            xn.to(device),
            xc.to(device),
        )

        chunks.append(
            output.cpu().numpy()
        )

    return np.concatenate(chunks).astype(np.float64)


def split_scores(pred_normalized, actual, data):

    prediction = (
        pred_normalized * data["target_std"]
        + data["target_mean"]
    )

    mse = float(
        mean_squared_error(actual, prediction)
    )

    return {
        "R2": float(
            r2_score(actual, prediction)
        ),
        "RMSE": math.sqrt(mse),
        "MAE": float(
            mean_absolute_error(actual, prediction)
        ),
    }


# ----------------------------------------------------------
# Train one configuration
# ----------------------------------------------------------

def train_one(
    data,
    dataset_name,
    seed,
    name,
    cfg,
    budget,
    stage,
    device,
):

    folder = (
        OUT / "tuning" / dataset_name
        / stage / f"seed_{seed}" / name
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "dataset": dataset_name,
        "seed": seed,
        "preset": name,
        "configuration": cfg,
        "budget": budget,
        "stage": stage,
    }

    meta_file = folder / "configuration.json"
    metrics_file = folder / "metrics.json"

    if meta_file.exists():
        if json.loads(meta_file.read_text()) != metadata:
            raise RuntimeError(
                f"Existing run has different settings: {folder}"
            )

    if metrics_file.exists():
        print(f"Skipping completed run: {folder}")
        return json.loads(metrics_file.read_text())

    meta_file.write_text(
        json.dumps(metadata, indent=2)
    )

    seed_all(seed)

    model = make_model(
        data,
        latent_dim=cfg["latent_dim"],
        gamma_init=cfg["gamma_init"],
    ).to(device)

    parameters = model_parameters(model)

    train_ds = data["splits"]["train"]["tensor"]
    val_ds = data["splits"]["val"]["tensor"]

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    criterion = nn.MSELoss()

    history = []

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    updates = 0

    print(
        f"\n{dataset_name} | {stage} | "
        f"seed {seed} | {name} | "
        f"params={parameters:,}"
    )

    for epoch in range(1, budget["epochs"] + 1):

        model.train()
        grad_norms = []

        for xn, xc, y in train_loader:

            xn = xn.to(device)
            xc = xc.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            prediction = model(xn, xc)
            loss = criterion(prediction, y)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Nonfinite loss: "
                    f"{dataset_name}/{seed}/{name}"
                )

            loss.backward()

            grad = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            if not torch.isfinite(grad):
                raise RuntimeError(
                    "Nonfinite gradient norm"
                )

            grad_norms.append(float(grad.item()))

            optimizer.step()
            updates += 1

        # Evaluate every few epochs to reduce CPU overhead.
        if (
            epoch != 1
            and epoch % budget["eval_every"] != 0
            and epoch != budget["epochs"]
        ):
            continue

        pred_train = predict_normalized(
            model, train_ds, device
        )

        pred_val = predict_normalized(
            model, val_ds, device
        )

        y_train = train_ds.tensors[2].numpy()
        y_val = val_ds.tensors[2].numpy()

        train_mse = float(
            np.mean((pred_train - y_train) ** 2)
        )

        val_mse = float(
            np.mean((pred_val - y_val) ** 2)
        )

        gamma = float(
            model.gamma.detach().item()
        )

        history.append({
            "epoch": epoch,
            "optimizer_updates": updates,
            "train_MSE": train_mse,
            "val_MSE": val_mse,
            "gamma": gamma,
            "mean_preclip_grad_norm": float(
                np.mean(grad_norms)
            ),
        })

        if val_mse < best_val - 1e-7:

            best_val = val_mse
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

        if epoch == 1 or epoch % 50 == 0:

            print(
                f" epoch={epoch:4d} "
                f"updates={updates:6d} "
                f"train={train_mse:.5f} "
                f"val={val_mse:.5f} "
                f"gamma={gamma:.3f}"
            )

        if epoch - best_epoch >= budget["patience"]:
            print(
                f" Early stopping at epoch {epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No valid checkpoint selected"
        )

    model.load_state_dict(best_state)
    model.eval()

    hist = pd.DataFrame(history)

    hist.to_csv(
        folder / "history.csv",
        index=False,
    )

    # ------------------------------------------------------
    # Loss curve against number of optimizer updates
    # ------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        hist["optimizer_updates"],
        hist["train_MSE"],
        label="Train",
    )

    plt.plot(
        hist["optimizer_updates"],
        hist["val_MSE"],
        label="Validation",
    )

    plt.xlabel("Optimizer updates")
    plt.ylabel("Normalized MSE")
    plt.title(
        f"{dataset_name} | {name} | seed {seed}"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        folder / "loss_curve.png",
        dpi=170,
    )
    plt.close()

    # ------------------------------------------------------
    # Gamma trajectory
    # ------------------------------------------------------

    plt.figure(figsize=(9, 4))

    plt.plot(
        hist["optimizer_updates"],
        hist["gamma"],
    )

    plt.xlabel("Optimizer updates")
    plt.ylabel("Learned gamma")
    plt.title(
        f"Gamma trajectory: {name} | seed {seed}"
    )
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        folder / "gamma_curve.png",
        dpi=170,
    )
    plt.close()

    # ------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------

    checkpoint = {
        "state_dict": model.state_dict(),
        "dataset": dataset_name,
        "seed": seed,
        "configuration": cfg,
        "model_version": "VANTA-RN-v0.1",
        "best_epoch": best_epoch,
        "cardinalities": data["cardinalities"],
        "vocabularies": data["vocabularies"],
        "numeric": data["numeric"],
        "categorical": data["categorical"],
        "target_mean": data["target_mean"],
        "target_std": data["target_std"],
    }

    torch.save(
        checkpoint,
        folder / "checkpoint.pt",
    )

    # ------------------------------------------------------
    # Validation metrics
    # ------------------------------------------------------

    val_pred = predict_normalized(
        model, val_ds, device
    )

    val_scores = split_scores(
        val_pred,
        data["splits"]["val"]["actual"],
        data,
    )

    result = {
        "dataset": dataset_name,
        "stage": stage,
        "seed": seed,
        "preset": name,
        "parameters": parameters,
        "best_epoch": best_epoch,
        "optimizer_updates": updates,
        "gamma": float(
            model.gamma.detach().item()
        ),
        "val_RMSE": val_scores["RMSE"],
        "val_R2": val_scores["R2"],
        "val_MAE": val_scores["MAE"],
    }

    # ------------------------------------------------------
    # Test set: ONLY during confirmation
    # ------------------------------------------------------

    if stage == "confirm":

        test_pred = predict_normalized(
            model,
            data["splits"]["test"]["tensor"],
            device,
        )

        test_scores = split_scores(
            test_pred,
            data["splits"]["test"]["actual"],
            data,
        )

        result.update({
            f"test_{key}": value
            for key, value in test_scores.items()
        })

        pd.DataFrame({
            "row_index": data["splits"]["test"]["indices"],
            "actual": data["splits"]["test"]["actual"],
            "prediction": (
                test_pred * data["target_std"]
                + data["target_mean"]
            ),
        }).to_csv(
            folder / "test_predictions.csv",
            index=False,
        )

    metrics_file.write_text(
        json.dumps(result, indent=2)
    )

    print(
        f" BEST epoch={best_epoch}, "
        f"val RMSE={result['val_RMSE']:,.3f}"
        + (
            f", test R2={result['test_R2']:.5f}"
            if stage == "confirm"
            else ""
        )
    )

    return result


# ----------------------------------------------------------
# Compare confirmation against EX03/EX04
# ----------------------------------------------------------

def print_confirmation(dataset_name, confirmed, seeds):

    print(
        "\nCONFIRMATION — TEST mean +/- sample SD"
    )

    old_sources = [
        (
            EX03 / "results" / "metrics.csv",
            [
                "VANTA-R",
                "Embedding MLP",
                "XGBoost",
                "CatBoost",
            ],
        ),
        (
            EX04 / "results" / "metrics.csv",
            ["VANTA-RN"],
        ),
    ]

    combined = pd.DataFrame(confirmed)

    for path, models in old_sources:

        if not path.exists():
            continue

        past = pd.read_csv(path)

        subset = past[
            (past["dataset"] == dataset_name)
            & (past["seed"].isin(seeds))
            & (past["model"].isin(models))
        ]

        combined = pd.concat(
            [combined, subset],
            ignore_index=True,
        )

    for name, group in combined.groupby(
        "model",
        dropna=False,
    ):

        if pd.isna(name):
            name = "VANTA-RN tuned"

        def stats(metric):
            values = (
                group[metric]
                .dropna()
                .to_numpy(dtype=float)
            )

            mean = float(values.mean())

            std = (
                float(values.std(ddof=1))
                if len(values) > 1
                else float("nan")
            )

            return mean, std

        r2_mean, r2_std = stats("test_R2")
        rm_mean, rm_std = stats("test_RMSE")

        print(
            f" {str(name):18s} "
            f"R2={r2_mean:+.5f} +/- {r2_std:.5f} "
            f"RMSE={rm_mean:,.3f} +/- {rm_std:,.3f}"
        )

    combined.to_csv(
        OUT / "tuning" / dataset_name
        / "comparison_rows.csv",
        index=False,
    )


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        choices=["search", "confirm"],
        required=True,
    )

    parser.add_argument(
        "--dataset",
        choices=sorted(DEFAULT_EPOCHS),
        required=True,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--presets",
        nargs="+",
        choices=list(PRESETS),
        default=[
            "reference_long",
            "batch_128",
            "lr_high",
            "latent_48",
            "gamma_05",
        ],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    if args.seeds is None:
        args.seeds = (
            [101]
            if args.stage == "search"
            else [42, 43, 44]
        )

    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Seeds must be unique")

    if min(
        args.threads,
        args.eval_every,
        args.patience,
    ) <= 0:
        parser.error(
            "Threads, evaluation interval and "
            "patience must be positive."
        )

    torch.set_num_threads(args.threads)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"VANTA-RN hyperparameter {args.stage} | "
        f"{args.dataset} | seeds={args.seeds} | "
        f"device={device}"
    )

    study_dir = (
        OUT / "tuning" / args.dataset
    )

    study_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_path = study_dir / "selected.json"

    if args.stage == "search":

        budget = {
            "epochs": (
                args.epochs
                or DEFAULT_EPOCHS[args.dataset]
            ),
            "patience": args.patience,
            "eval_every": args.eval_every,
        }

        names = args.presets

        if any(value <= 0 for value in budget.values()):
            parser.error("Training budget must be positive")

    else:

        if not selected_path.exists():
            parser.error(
                "Run --stage search for this dataset first."
            )

        selection = json.loads(
            selected_path.read_text()
        )

        budget = selection["budget"]

        if (
            args.epochs is not None
            and args.epochs != budget["epochs"]
        ):
            parser.error(
                "Confirmation must use the search "
                "training budget."
            )

        names = [selection["preset"]]

        PRESETS[names[0]] = selection["configuration"]

        if set(args.seeds) & set(
            selection["search_seeds"]
        ):
            parser.error(
                "Confirmation seeds must differ "
                "from search seeds."
            )

    bundle = load_dataset(args.dataset)

    results = []

    for seed in args.seeds:

        data = preprocess(bundle, seed)

        for name in names:

            result = train_one(
                data,
                args.dataset,
                seed,
                name,
                PRESETS[name],
                budget,
                args.stage,
                device,
            )

            results.append(result)

    frame = pd.DataFrame(results)

    frame.to_csv(
        study_dir / f"{args.stage}_metrics.csv",
        index=False,
    )

    if args.stage == "search":

        grouped = frame.groupby("preset")[
            "val_RMSE"
        ].agg([
            "mean",
            "std",
            "count",
        ])

        print(
            "\nVALIDATION RANKING "
            "(test not evaluated)"
        )

        print(
            grouped.sort_values("mean").to_string()
        )

        winner = grouped["mean"].idxmin()

        selection = {
            "preset": winner,
            "configuration": PRESETS[winner],
            "budget": budget,
            "search_seeds": args.seeds,
        }

        if selected_path.exists():

            previous = json.loads(
                selected_path.read_text()
            )

            if (
                previous != selection
                and (
                    study_dir / "confirm_metrics.csv"
                ).exists()
            ):
                raise RuntimeError(
                    "Previous confirmation exists. "
                    "Move the old results before reselection."
                )

        selected_path.write_text(
            json.dumps(selection, indent=2)
        )

        print(
            f"\nSelected using validation only: {winner}"
        )

    else:
        print_confirmation(
            args.dataset,
            results,
            args.seeds,
        )

    print("\nSaved to:", study_dir)


if __name__ == "__main__":
    main()

