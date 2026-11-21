
"""
VANTA - Experiment 01: Strong regression baselines.

Models:
    1. VANTA-R (existing checkpoint)
    2. Multiple Linear Regression
    3. Polynomial Ridge Regression
    4. Parameter-matched embedding MLP
    5. XGBoost
    6. CatBoost

Reuses VANTA_R/train.py for identical data splitting
and train-fitted preprocessing.

Run from the VANTA project root:
    uv run python3 experiments/baselines_ex01/run_baselines.py
"""

import copy
import json
import math
import random
import shutil
import sys
from pathlib import Path

import joblib
import kagglehub
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
)
from torch import nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VANTA_R = ROOT / "VANTA_R"
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(VANTA_R))

import train as vanta_train
from model import ResidualFeatureAttentionRegressor


SEED = 42
EPOCHS = 200
BATCH_SIZE = 512
LR = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 25
EMBED_DIM = 8

THREADS = 4
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

FEATURES = vanta_train.FEATURES
NUMERIC = vanta_train.NUMERIC
CATEGORICAL = vanta_train.CATEGORICAL


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )


def metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)

    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "R2": float(r2_score(actual, predicted)),
    }


def plot_curve(train, val, title, filename, ylabel="RMSE ($)"):
    plt.figure(figsize=(9, 5))

    steps = np.arange(1, len(val) + 1)

    if train is not None and len(train):
        plt.plot(
            np.arange(1, len(train) + 1),
            train,
            label="Train",
        )

    plt.plot(steps, val, label="Validation")
    plt.xlabel("Epoch / boosting round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=180)
    plt.close()


def show_scores(name, result):
    print(f"\n{name}")
    print("-" * 56)

    for split in ["train", "val", "test"]:
        m = result[split]

        print(
            f"{split.upper():5s} | "
            f"R2: {m['R2']:.5f} | "
            f"RMSE: ${m['RMSE']:,.2f} | "
            f"MAE: ${m['MAE']:,.2f} | "
            f"MSE: {m['MSE']:,.2f}"
        )


# ---------------------------------------------------------
# 1. Load the exact VANTA-R dataset and preprocessing
# ---------------------------------------------------------

def load_experiment():
    checkpoint_path = VANTA_R / "attention_checkpoint.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing VANTA-R checkpoint: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if int(checkpoint["seed"]) != SEED:
        raise ValueError(
            "The reference checkpoint used another split seed."
        )

    df = vanta_train.load_dataset()
    data = vanta_train.prepare_data(df, SEED)

    cardinalities = [
        len(data["vocabularies"][col]) + 1
        for col in CATEGORICAL
    ]

    # Prevent accidental comparisons using different schemas.
    if checkpoint["features"] != FEATURES:
        raise ValueError("Feature order differs from checkpoint.")

    if checkpoint["cardinalities"] != cardinalities:
        raise ValueError("Categorical cardinalities differ.")

    if checkpoint["vocabularies"] != data["vocabularies"]:
        raise ValueError("Categorical vocabularies differ.")

    if not np.isclose(
        checkpoint["target_mean"], data["y_mean"]
    ):
        raise ValueError("Target mean differs from checkpoint.")

    if not np.isclose(
        checkpoint["target_std"], data["y_std"]
    ):
        raise ValueError("Target scale differs from checkpoint.")

    arrays = {}
    actual = {}

    for split in ["train", "val", "test"]:
        dataset, y_actual = data[split]

        x_num, x_cat, y_scaled = dataset.tensors

        arrays[split] = {
            "numeric": x_num.numpy(),
            "categorical": x_cat.numpy(),
            "target_scaled": y_scaled.numpy(),
        }

        actual[split] = y_actual

    return checkpoint, data, arrays, actual


# ---------------------------------------------------------
# 2. Encode categorical inputs for classical baselines
# ---------------------------------------------------------

def prepare_classical_features(arrays):
    encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float64,
    )

    encoder.fit(arrays["train"]["categorical"])

    output = {}

    for split in ["train", "val", "test"]:
        numeric = arrays[split]["numeric"].astype(np.float64)

        categorical = encoder.transform(
            arrays[split]["categorical"]
        )

        output[split] = np.concatenate(
            [numeric, categorical], axis=1
        )

    return output, encoder


def prepare_catboost_features(arrays):
    """
    Preserve categorical features as categories for CatBoost.

    Category IDs are converted to strings so CatBoost doesn't
    treat them as ordered continuous quantities.
    """
    output = {}

    for split in ["train", "val", "test"]:
        numeric = arrays[split]["numeric"]
        categorical = arrays[split]["categorical"]

        frame = pd.DataFrame(index=range(len(numeric)))

        for i, name in enumerate(NUMERIC):
            frame[name] = numeric[:, i]

        for i, name in enumerate(CATEGORICAL):
            frame[name] = categorical[:, i].astype(str)

        output[split] = frame[FEATURES]

    return output


# ---------------------------------------------------------
# 3. Neural baseline: learned categorical embedding MLP
# ---------------------------------------------------------

class EmbeddingMLP(nn.Module):
    def __init__(self, cardinalities, hidden_dim):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(card, EMBED_DIM, padding_idx=0)
            for card in cardinalities
        ])

        input_dim = (
            len(NUMERIC) + len(cardinalities) * EMBED_DIM
        )

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_num, x_cat):
        embeddings = [
            embedding(x_cat[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]

        x = torch.cat([x_num] + embeddings, dim=1)

        return self.network(x).squeeze(-1)


def matched_mlp_width(target_params, cardinalities):
    embedding_params = sum(cardinalities) * EMBED_DIM

    input_dim = (
        len(NUMERIC) + len(cardinalities) * EMBED_DIM
    )

    # For one hidden layer:
    # params = embedding_params
    #        + input_dim * hidden
    #        + hidden (first bias)
    #        + hidden (output weights)
    #        + 1 (output bias)

    width = round(
        (target_params - embedding_params - 1)
        / (input_dim + 2)
    )

    return max(1, width)


@torch.no_grad()
def neural_predict(model, dataset, target_mean, target_std):
    model.eval()

    loader = DataLoader(dataset, batch_size=2048)
    outputs = []

    for x_num, x_cat, _ in loader:
        predictions = model(
            x_num.to(DEVICE),
            x_cat.to(DEVICE),
        )
        outputs.append(predictions.cpu().numpy())

    scaled = np.concatenate(outputs)

    return scaled * target_std + target_mean


@torch.no_grad()
def neural_mse(model, dataset):
    model.eval()

    loader = DataLoader(dataset, batch_size=2048)

    total = 0.0
    count = 0

    for x_num, x_cat, y in loader:
        x_num = x_num.to(DEVICE)
        x_cat = x_cat.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x_num, x_cat)

        total += (pred - y).square().sum().item()
        count += len(y)

    return total / count


def train_mlp(data, cardinalities, target_params):
    print("\n[4/6] Training parameter-matched MLP")

    set_seed(SEED)

    hidden = matched_mlp_width(
        target_params, cardinalities
    )

    model = EmbeddingMLP(
        cardinalities, hidden
    ).to(DEVICE)

    params = count_parameters(model)

    print(f"Hidden width: {hidden}")
    print(f"MLP parameters: {params:,}")
    print(f"VANTA-R parameters: {target_params:,}")
    print(f"Parameter ratio: {params / target_params:.3f}")

    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        data["train"][0],
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    train_history = []
    val_history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        for x_num, x_cat, y in train_loader:
            x_num = x_num.to(DEVICE)
            x_cat = x_cat.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            prediction = model(x_num, x_cat)
            loss = criterion(prediction, y)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite MLP loss")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

        train_loss = neural_mse(
            model, data["train"][0]
        )

        val_loss = neural_mse(
            model, data["val"][0]
        )

        # Convert normalized MSE into dollar RMSE.
        train_history.append(
            math.sqrt(train_loss) * data["y_std"]
        )

        val_history.append(
            math.sqrt(val_loss) * data["y_std"]
        )

        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Train RMSE ${train_history[-1]:,.2f} | "
                f"Val RMSE ${val_history[-1]:,.2f}"
            )

        if epoch - best_epoch >= PATIENCE:
            print(f"Early stopping: {epoch}")
            break

    model.load_state_dict(best_state)

    plot_curve(
        train_history,
        val_history,
        "Parameter-Matched MLP",
        "mlp_loss_curve.png",
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_dim": hidden,
            "cardinalities": cardinalities,
            "embedding_dim": EMBED_DIM,
            "seed": SEED,
        },
        OUT / "mlp_checkpoint.pt",
    )

    return model, params, best_epoch


# ---------------------------------------------------------
# 4. Main experiment
# ---------------------------------------------------------

def main():
    set_seed(SEED)

    print("=" * 65)
    print("VANTA - EXPERIMENT 01: STRONG BASELINES")
    print("=" * 65)
    print(f"Device: {DEVICE}")

    checkpoint, data, arrays, actual = load_experiment()

    cardinalities = checkpoint["cardinalities"]
    latent_dim = int(checkpoint["latent_dim"])

    print(f"Latent dimension of VANTA-R: {latent_dim}")

    print(
        "Dataset sizes:",
        *(f"{s}={len(actual[s]):,}" for s in ["train", "val", "test"])
    )

    X, encoder = prepare_classical_features(arrays)
    X_catboost = prepare_catboost_features(arrays)

    y_train = actual["train"]
    y_val = actual["val"]

    results = {}
    predictions_test = pd.DataFrame({
        "actual_price": actual["test"]
    })

    def record(name, predictor, complexity, details=None):
        result = {"complexity": complexity}

        if details is not None:
            result["details"] = details

        for split in ["train", "val", "test"]:
            prediction = np.asarray(
                predictor(split),
                dtype=np.float64
            ).reshape(-1)

            result[split] = metrics(
                actual[split], prediction
            )

            if split == "test":
                predictions_test[name] = prediction

        results[name] = result
        show_scores(name, result)

    # -----------------------------------------------------
    # Model 1: Existing VANTA-R checkpoint
    # -----------------------------------------------------

    print("\n[1/6] Loading VANTA-R reference")

    vanta = ResidualFeatureAttentionRegressor(
        num_features=len(FEATURES),
        numeric_positions=[
            FEATURES.index(c) for c in NUMERIC
        ],
        categorical_positions=[
            FEATURES.index(c) for c in CATEGORICAL
        ],
        categorical_cardinalities=cardinalities,
        latent_dim=latent_dim,
        gamma_init=0.1,
        learnable_gamma=True,
    ).to(DEVICE)

    vanta.load_state_dict(
        checkpoint["model_state_dict"]
    )

    vanta_params = count_parameters(vanta)

    record(
        "VANTA-R",
        lambda split: neural_predict(
            vanta,
            data[split][0],
            data["y_mean"],
            data["y_std"],
        ),
        f"{vanta_params} parameters",
        {"best_epoch": checkpoint["best_epoch"]},
    )

    reference_curve = VANTA_R / "loss_curve.png"

    if reference_curve.exists():
        shutil.copy2(
            reference_curve,
            OUT / "vanta_r_reference_curve.png",
        )

    # -----------------------------------------------------
    # Model 2: Ordinary Multiple Linear Regression
    # -----------------------------------------------------

    print("\n[2/6] Fitting ordinary MLR")

    mlr = LinearRegression()
    mlr.fit(X["train"], y_train)

    record(
        "MLR",
        lambda split: mlr.predict(X[split]),
        f"{len(mlr.coef_) + 1} coefficients",
    )

    # -----------------------------------------------------
    # Model 3: Polynomial Ridge Regression
    # -----------------------------------------------------

    print("\n[3/6] Fitting degree-2 Polynomial Ridge")

    poly = PolynomialFeatures(
        degree=2,
        include_bias=False,
    )

    X_poly = {
        "train": poly.fit_transform(X["train"]),
        "val": poly.transform(X["val"]),
        "test": poly.transform(X["test"]),
    }

    poly_scaler = StandardScaler()

    X_poly["train"] = poly_scaler.fit_transform(
        X_poly["train"]
    )

    for split in ["val", "test"]:
        X_poly[split] = poly_scaler.transform(
            X_poly[split]
        )

    alphas = [0.01, 0.1, 1, 10, 100, 1000]

    best_ridge = None
    best_rmse = float("inf")
    best_alpha = None

    ridge_curve = []

    for alpha in alphas:
        candidate = Ridge(alpha=alpha)
        candidate.fit(X_poly["train"], y_train)

        val_pred = candidate.predict(X_poly["val"])

        val_rmse = math.sqrt(
            mean_squared_error(y_val, val_pred)
        )

        ridge_curve.append(val_rmse)

        print(
            f"alpha={alpha:8g} | "
            f"Validation RMSE=${val_rmse:,.2f}"
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_alpha = alpha
            best_ridge = candidate

    plt.figure(figsize=(8, 5))
    plt.semilogx(alphas, ridge_curve, marker="o")
    plt.xlabel("Ridge regularization alpha")
    plt.ylabel("Validation RMSE ($)")
    plt.title("Polynomial Ridge - Alpha Sweep")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUT / "ridge_alpha_sweep.png",
        dpi=180,
    )
    plt.close()

    record(
        "Polynomial Ridge",
        lambda split: best_ridge.predict(X_poly[split]),
        f"{X_poly['train'].shape[1] + 1} coefficients",
        {
            "degree": 2,
            "best_alpha": best_alpha,
        },
    )

    joblib.dump(
        {
            "encoder": encoder,
            "polynomial": poly,
            "polynomial_scaler": poly_scaler,
            "ridge": best_ridge,
        },
        OUT / "polynomial_ridge.joblib",
    )

    # -----------------------------------------------------
    # Model 4: Parameter-matched embedding MLP
    # -----------------------------------------------------

    mlp, mlp_params, mlp_epoch = train_mlp(
        data,
        cardinalities,
        vanta_params,
    )

    record(
        "Embedding MLP",
        lambda split: neural_predict(
            mlp,
            data[split][0],
            data["y_mean"],
            data["y_std"],
        ),
        f"{mlp_params} parameters",
        {
            "best_epoch": mlp_epoch,
            "embedding_dim": EMBED_DIM,
        },
    )

    # -----------------------------------------------------
    # Model 5: XGBoost
    # -----------------------------------------------------

    print("\n[5/6] Training XGBoost")

    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=5,
        reg_lambda=5.0,
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=60,
        random_state=SEED,
        n_jobs=THREADS,
    )

    xgb_model.fit(
        X["train"],
        y_train,
        eval_set=[
            (X["train"], y_train),
            (X["val"], y_val),
        ],
        verbose=False,
    )

    xgb_history = xgb_model.evals_result()

    plot_curve(
        xgb_history["validation_0"]["rmse"],
        xgb_history["validation_1"]["rmse"],
        "XGBoost - Training History",
        "xgboost_loss_curve.png",
    )

    xgb_trees = int(xgb_model.best_iteration) + 1

    record(
        "XGBoost",
        lambda split: xgb_model.predict(X[split]),
        f"{xgb_trees} selected rounds; max depth 5",
        {
            "best_iteration": int(xgb_model.best_iteration),
            "max_depth": 5,
        },
    )

    xgb_model.save_model(
        str(OUT / "xgboost.json")
    )

    # -----------------------------------------------------
    # Model 6: CatBoost
    # -----------------------------------------------------

    print("\n[6/6] Training CatBoost")

    cat_model = CatBoostRegressor(
        iterations=1500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        eval_metric="RMSE",
        cat_features=CATEGORICAL,
        random_seed=SEED,
        thread_count=THREADS,
        allow_writing_files=False,
        verbose=False,
    )

    cat_model.fit(
        X_catboost["train"],
        y_train,
        eval_set=(
            X_catboost["val"],
            y_val,
        ),
        early_stopping_rounds=60,
        use_best_model=True,
        verbose=False,
    )

    cat_history = cat_model.get_evals_result()

    plot_curve(
        cat_history.get("learn", {}).get("RMSE", []),
        cat_history["validation"]["RMSE"],
        "CatBoost - Training History",
        "catboost_loss_curve.png",
    )

    record(
        "CatBoost",
        lambda split: cat_model.predict(
            X_catboost[split]
        ),
        f"{cat_model.tree_count_} trees; depth 6",
        {
            "best_iteration": int(cat_model.best_iteration_),
            "tree_count": int(cat_model.tree_count_),
            "depth": 6,
        },
    )

    cat_model.save_model(
        str(OUT / "catboost.cbm")
    )

    # -----------------------------------------------------
    # 5. Save comparison results
    # -----------------------------------------------------

    print("\n" + "=" * 85)
    print("FINAL COMPARISON - TEST SET")
    print("=" * 85)

    rows = []

    for name, result in results.items():
        row = {
            "model": name,
            "complexity": result["complexity"],
        }

        for split in ["train", "val", "test"]:
            for metric, value in result[split].items():
                row[f"{split}_{metric}"] = value

        rows.append(row)

    summary = pd.DataFrame(rows)

    display_columns = [
        "model",
        "test_R2",
        "test_RMSE",
        "test_MAE",
        "test_MSE",
        "complexity",
    ]

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:,.5f}",
        )
    )

    summary.to_csv(
        OUT / "metrics.csv",
        index=False,
    )

    with open(OUT / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    predictions_test.to_csv(
        OUT / "test_predictions.csv",
        index=False,
    )

    # -----------------------------------------------------
    # 6. Final visual comparison
    # -----------------------------------------------------

    names = list(results.keys())

    rmse_values = [
        results[name]["test"]["RMSE"]
        for name in names
    ]

    r2_values = [
        results[name]["test"]["R2"]
        for name in names
    ]

    plt.figure(figsize=(10, 5))

    plt.barh(names, rmse_values)
    plt.gca().invert_yaxis()

    plt.xlabel("Test RMSE ($) - lower is better")
    plt.title("VANTA Experiment 01 - RMSE Comparison")
    plt.tight_layout()

    plt.savefig(
        OUT / "comparison_rmse.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(10, 5))

    plt.barh(names, r2_values)
    plt.gca().invert_yaxis()

    plt.xlabel("Test R² - higher is better")
    plt.title("VANTA Experiment 01 - R² Comparison")
    plt.tight_layout()

    plt.savefig(
        OUT / "comparison_r2.png",
        dpi=180,
    )
    plt.close()

    print(f"\nExperiment complete. Results saved to:\n{OUT}")


if __name__ == "__main__":
    main()

