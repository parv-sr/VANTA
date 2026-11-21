
"""
VANTA-G: First training experiment on Kaggle Diamonds.

Run:
    uv run python3 train.py

Outputs saved beside this file:
    loss_curve.png
    attention_checkpoint.pt
    feature_gates.csv

Comparable to VANTA-R with:
    latent_dim = 24
    seed = 42
    80/10/10 split
    AdamW, MSE, gradient clipping, early stopping
"""

import argparse
import copy
import random
from pathlib import Path

import kagglehub
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import GatedFeatureAttentionRegressor


ROOT = Path(__file__).resolve().parent

TARGET = "price"

FEATURES = [
    "carat", "cut", "color", "clarity",
    "depth", "table", "x", "y", "z"
]

NUMERIC = [
    "carat", "depth", "table", "x", "y", "z"
]

CATEGORICAL = [
    "cut", "color", "clarity"
]


# ---------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------

def load_dataset(csv_path=None):
    if csv_path:
        path = Path(csv_path).expanduser().resolve()
    else:
        path = ROOT / "diamonds.csv"

        if not path.exists():
            print("Downloading Diamonds from Kaggle...")
            path = Path(
                kagglehub.dataset_download(
                    "shivam2503/diamonds",
                    path="diamonds.csv"
                )
            )

    print(f"Loading dataset: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    missing = set(FEATURES + [TARGET]) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[FEATURES + [TARGET]].copy()

    for col in NUMERIC + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    original_rows = len(df)

    # Same filtering used in VANTA-R.
    valid = (
        df[TARGET].gt(0)
        & df[["x", "y", "z"]].gt(0).all(axis=1)
        & df[["x", "y", "z"]].lt(15).all(axis=1)
    )

    df = df.loc[valid].reset_index(drop=True)

    print(f"Original rows: {original_rows:,}")
    print(f"Usable rows:   {len(df):,}")

    return df


# ---------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------

def prepare_data(df, seed):

    train, remaining = train_test_split(
        df,
        test_size=0.20,
        random_state=seed
    )

    val, test = train_test_split(
        remaining,
        test_size=0.50,
        random_state=seed
    )

    # Fit transformations using training data only.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train_numeric = imputer.fit_transform(train[NUMERIC])
    scaler.fit(train_numeric)

    # Learned categorical embeddings need integer IDs.
    # Reserve ID 0 for unseen/missing categories.
    vocabularies = {}

    for col in CATEGORICAL:
        categories = (
            train[col]
            .dropna()
            .astype(str)
            .unique()
        )

        vocabularies[col] = {
            value: i + 1
            for i, value in enumerate(sorted(categories))
        }

    y_mean = float(train[TARGET].mean())
    y_std = float(train[TARGET].std(ddof=0))

    if not np.isfinite(y_std) or y_std <= 0:
        raise ValueError("Invalid target standard deviation.")

    def transform(frame):

        numeric = imputer.transform(frame[NUMERIC])
        numeric = scaler.transform(numeric).astype(np.float32)

        categorical = np.column_stack([
            frame[col]
            .astype("string")
            .map(vocabularies[col])
            .fillna(0)
            .to_numpy(dtype=np.int64)
            for col in CATEGORICAL
        ])

        actual = frame[TARGET].to_numpy(dtype=np.float64)

        target = (
            (actual - y_mean) / y_std
        ).astype(np.float32)

        dataset = TensorDataset(
            torch.from_numpy(numeric),
            torch.from_numpy(categorical),
            torch.from_numpy(target),
        )

        return dataset, actual

    train_ds, train_y = transform(train)
    val_ds, val_y = transform(val)
    test_ds, test_y = transform(test)

    return {
        "train": (train_ds, train_y),
        "val": (val_ds, val_y),
        "test": (test_ds, test_y),
        "y_mean": y_mean,
        "y_std": y_std,
        "vocabularies": vocabularies,
        "imputer": imputer,
        "scaler": scaler,
    }


# ---------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------

@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for x_num, x_cat, y in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        prediction = model(x_num, x_cat)

        total_loss += (
            (prediction - y).square().sum().item()
        )

        total_samples += len(y)

    return total_loss / total_samples


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    predictions = []

    for x_num, x_cat, _ in loader:
        output = model(
            x_num.to(device),
            x_cat.to(device),
        )

        predictions.append(output.cpu().numpy())

    return np.concatenate(predictions)


def report_metrics(name, prediction, actual, mean, std):

    prediction = prediction * std + mean

    mse = mean_squared_error(actual, prediction)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, prediction)
    r2 = r2_score(actual, prediction)

    print(f"\n{name} RESULTS")
    print("-" * 40)
    print(f"MSE  : {mse:,.2f} USD^2")
    print(f"RMSE : ${rmse:,.2f}")
    print(f"MAE  : ${mae:,.2f}")
    print(f"R2   : {r2:.5f}")


# ---------------------------------------------------------
# 4. Training
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--latent-dim", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--gamma-init", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)

    args = parser.parse_args()

    if min(
        args.latent_dim,
        args.epochs,
        args.batch_size,
        args.patience,
    ) <= 0:
        parser.error("Model dimensions and training settings must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 55)
    print("VANTA-G: GATED RESIDUAL FEATURE ATTENTION")
    print("=" * 55)
    print(f"Device: {device}")

    df = load_dataset(args.csv)

    if args.max_rows > 0 and args.max_rows < len(df):
        df = df.sample(
            n=args.max_rows,
            random_state=args.seed
        ).reset_index(drop=True)

    data = prepare_data(df, args.seed)

    train_ds, train_y = data["train"]
    val_ds, val_y = data["val"]
    test_ds, test_y = data["test"]

    print(
        f"Train: {len(train_ds):,} | "
        f"Validation: {len(val_ds):,} | "
        f"Test: {len(test_ds):,}"
    )

    cardinalities = [
        len(data["vocabularies"][col]) + 1
        for col in CATEGORICAL
    ]

    # -----------------------------------------------------
    # Build VANTA-G
    # -----------------------------------------------------

    model = GatedFeatureAttentionRegressor(
        num_features=len(FEATURES),

        numeric_positions=[
            FEATURES.index(col) for col in NUMERIC
        ],

        categorical_positions=[
            FEATURES.index(col) for col in CATEGORICAL
        ],

        categorical_cardinalities=cardinalities,

        latent_dim=args.latent_dim,
        gamma_init=args.gamma_init,
    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Latent dimension: {args.latent_dim}")
    print(f"Trainable parameters: {parameters:,}")
    print(f"Initial gamma: {args.gamma_init}")

    # -----------------------------------------------------
    # Data loaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True
    )

    train_eval_loader = DataLoader(
        train_ds,
        batch_size=2048
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=2048
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=2048
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    loss_fn = nn.MSELoss()

    train_history = []
    val_history = []

    best_loss = float("inf")
    best_epoch = 0
    best_state = None

    print("\nStarting training...")
    print("Epoch | Train MSE | Val MSE | Mean Gamma")
    print("-" * 55)

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    for epoch in range(1, args.epochs + 1):

        model.train()

        for x_num, x_cat, y in train_loader:

            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            prediction = model(x_num, x_cat)
            loss = loss_fn(prediction, y)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite loss encountered. "
                    "Try reducing the learning rate."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

        # Evaluate on full train and validation splits.
        train_mse = evaluate_loss(
            model, train_eval_loader, device
        )

        val_mse = evaluate_loss(
            model, val_loader, device
        )

        train_history.append(train_mse)
        val_history.append(val_mse)

        if val_mse < best_loss - 1e-7:
            best_loss = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )

        if epoch == 1 or epoch % 10 == 0:
            mean_gamma = model.gamma.detach().mean().item()

            print(
                f"{epoch:5d} | "
                f"{train_mse:9.5f} | "
                f"{val_mse:7.5f} | "
                f"{mean_gamma:+.5f}"
            )

        if epoch - best_epoch >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # -----------------------------------------------------
    # Restore best model
    # -----------------------------------------------------

    if best_state is None:
        raise RuntimeError("No valid training checkpoint was found.")

    model.load_state_dict(best_state)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation MSE: {best_loss:.6f}")

    # -----------------------------------------------------
    # Report metrics
    # -----------------------------------------------------

    for name, loader, actual in [
        ("TRAIN", train_eval_loader, train_y),
        ("VALIDATION", val_loader, val_y),
        ("TEST", test_loader, test_y)
    ]:

        predictions = predict(model, loader, device)

        report_metrics(
            name,
            predictions,
            actual,
            data["y_mean"],
            data["y_std"]
        )

    # -----------------------------------------------------
    # Inspect learned feature gates
    # -----------------------------------------------------

    gamma_values = (
        model.gamma.detach().cpu().numpy()
    )

    gates_df = pd.DataFrame({
        "feature": FEATURES,
        "gamma": gamma_values
    })

    print("\nLEARNED FEATURE GATES")
    print("-" * 40)
    print(gates_df.to_string(index=False))

    gates_path = ROOT / "feature_gates.csv"
    gates_df.to_csv(gates_path, index=False)

    # -----------------------------------------------------
    # Save loss curve
    # -----------------------------------------------------

    epochs = np.arange(
        1, len(train_history) + 1
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train_history,
        label="Train MSE"
    )

    plt.plot(
        epochs,
        val_history,
        label="Validation MSE"
    )

    plt.axvline(
        best_epoch,
        linestyle="--",
        color="gray",
        label=f"Best epoch: {best_epoch}"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Normalized MSE")
    plt.title("VANTA-G: Diamonds Regression")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = ROOT / "loss_curve.png"

    plt.savefig(
        plot_path,
        dpi=180
    )

    plt.close()

    # -----------------------------------------------------
    # Save model and preprocessing state
    # -----------------------------------------------------

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": "VANTA-G",
        "features": FEATURES,
        "numeric_columns": NUMERIC,
        "categorical_columns": CATEGORICAL,
        "cardinalities": cardinalities,
        "vocabularies": data["vocabularies"],
        "numeric_imputer": data["imputer"].statistics_.tolist(),
        "numeric_mean": data["scaler"].mean_.tolist(),
        "numeric_scale": data["scaler"].scale_.tolist(),
        "target_mean": data["y_mean"],
        "target_std": data["y_std"],
        "latent_dim": args.latent_dim,
        "gamma_init": args.gamma_init,
        "gamma_values": gamma_values.tolist(),
        "best_epoch": best_epoch,
        "seed": args.seed
    }

    checkpoint_path = ROOT / "attention_checkpoint.pt"

    torch.save(
        checkpoint,
        checkpoint_path
    )

    print(f"\nLoss curve saved: {plot_path}")
    print(f"Feature gates saved: {gates_path}")
    print(f"Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    main()

