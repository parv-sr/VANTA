
"""
VANTA-RN v0.1 — Cross-dataset normalization experiment.

Reuses EX03 preprocessing and training code.

Default:
    Ames, seed 43

Run:
    uv run python3 VANTA_RN/train.py

Three Ames seeds:
    uv run python3 VANTA_RN/train.py --seeds 42 43 44

All datasets:
    uv run python3 VANTA_RN/train.py \
        --datasets concrete california ames airbnb \
        --seeds 42 43 44

Results:
    experiments/normalization_ex04/results/
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model import NormalizedFeatureAttentionRegressor


ROOT = Path(__file__).resolve().parents[1]
EX03 = ROOT / "experiments" / "datasets_ex03"
EX04 = ROOT / "experiments" / "normalization_ex04"

sys.path.insert(0, str(EX03))

import run_exp as ex03
from datasets import REGISTRY, load_dataset, preprocess


ORIGINAL_VANTA_CLASS = ex03.VantaR

MODEL_NAME = "VANTA-RN"

BASELINE_METRICS = EX03 / "results" / "metrics.csv"

MODEL_ORDER = [
    "VANTA-RN",
    "VANTA-R",
    "Embedding MLP",
    "Ridge",
    "XGBoost",
    "CatBoost",
]


# ---------------------------------------------------------
# 1. Dataset and model validation
# ---------------------------------------------------------

def verify_reference(data):
    """
    Ensure RN inherits exactly the same trainable
    parameters and state structure as VANTA-R.
    """

    ex03.VantaR = ORIGINAL_VANTA_CLASS
    reference = ex03.build_vanta(data)

    ex03.VantaR = NormalizedFeatureAttentionRegressor
    normalized = ex03.build_vanta(data)

    ref_params = ex03.parameter_count(reference)
    rn_params = ex03.parameter_count(normalized)

    if ref_params != rn_params:
        raise RuntimeError(
            "Parameter count mismatch: "
            f"VANTA-R={ref_params}, VANTA-RN={rn_params}"
        )

    ref_keys = set(reference.state_dict())
    rn_keys = set(normalized.state_dict())

    if ref_keys != rn_keys:
        raise RuntimeError(
            "VANTA-R and VANTA-RN state structures differ."
        )

    # Catch incompatible inherited model attributes
    # before committing to a complete training run.
    dataset = data["splits"]["train"]["tensor"]
    x_num, x_cat, _ = dataset.tensors

    with torch.inference_mode():
        output = normalized(
            x_num[:2],
            x_cat[:2],
        )

        attention = normalized.attention_matrix(
            x_num[:2],
            x_cat[:2],
        )

    F = len(data["features"])

    if output.shape != (2,):
        raise RuntimeError(
            f"Unexpected output shape: {output.shape}"
        )

    if attention.shape != (2, F, F):
        raise RuntimeError(
            f"Unexpected attention shape: {attention.shape}"
        )

    if not torch.isfinite(output).all():
        raise RuntimeError("Non-finite initial predictions.")

    if attention.abs().max().item() > 1 / math.sqrt(F) + 1e-5:
        raise RuntimeError(
            "Normalized attention exceeds theoretical bound."
        )

    print(
        f"Architecture verified: "
        f"{rn_params:,} parameters in both variants."
    )

    return rn_params


def verify_original_split(dataset_name, seed, data):
    """
    Confirm that the original EX03 test predictions
    correspond to the same rows and ordering.
    """
    path = (
        EX03 / "results"
        / dataset_name
        / f"seed_{seed}"
        / "vanta_r"
        / "test_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing original predictions: {path}"
        )

    original = pd.read_csv(path)

    expected = data["splits"]["test"]["indices"]

    if not np.array_equal(
        original["row_index"].to_numpy(),
        expected,
    ):
        raise RuntimeError(
            "Test row indices differ from EX03."
        )

    if not np.allclose(
        original["actual"].to_numpy(),
        data["splits"]["test"]["actual"],
    ):
        raise RuntimeError(
            "Test targets differ from EX03."
        )

    return original


# ---------------------------------------------------------
# 2. Checkpoint versioning
# ---------------------------------------------------------

def label_checkpoint(folder):
    path = folder / "checkpoint.pt"

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    checkpoint["model"] = MODEL_NAME
    checkpoint["model_version"] = "VANTA-RN-v0.1"
    checkpoint["qk_normalization"] = "smooth_l2"
    checkpoint["attention_divisor"] = "sqrt(num_features)"

    torch.save(checkpoint, path)

    gamma = checkpoint["state_dict"].get("gamma")

    if gamma is not None:
        print(
            "Learned gamma:",
            gamma.detach().reshape(-1).tolist()
        )


# ---------------------------------------------------------
# 3. Outlier diagnostics
# ---------------------------------------------------------

def report_outliers(dataset_name, seed, folder, original):
    rn_path = folder / "test_predictions.csv"

    rn = pd.read_csv(rn_path)

    comparison = original[
        ["row_index", "actual", "prediction"]
    ].merge(
        rn[["row_index", "prediction"]],
        on="row_index",
        suffixes=("_original", "_RN"),
        validate="one_to_one",
    )

    if len(comparison) != len(original):
        raise RuntimeError("Prediction comparison is incomplete.")

    print("\nPrediction diagnostics")
    print("-" * 70)

    for column, label in [
        ("prediction_original", "VANTA-R"),
        ("prediction_RN", "VANTA-RN"),
    ]:
        squared_error = (
            comparison[column] - comparison["actual"]
        ) ** 2

        total = squared_error.sum()

        max_share = (
            100 * squared_error.max() / total
            if total > 0 else 0.0
        )

        print(
            f"{label:10s} | "
            f"range [{comparison[column].min():,.2f}, "
            f"{comparison[column].max():,.2f}] | "
            f"largest squared-error share: {max_share:.2f}%"
        )

    if dataset_name == "ames" and seed == 43:
        selected = comparison[
            comparison["row_index"].isin([2180, 956])
        ]

        print("\nPreviously problematic Ames observations:")
        print(
            selected.to_string(
                index=False,
                float_format=lambda x: f"{x:,.2f}",
            )
        )

    comparison.to_csv(
        folder / "original_vs_RN_predictions.csv",
        index=False,
    )


# ---------------------------------------------------------
# 4. Aggregated comparisons
# ---------------------------------------------------------

def summarize(results_dir):
    rn_path = results_dir / "metrics.csv"

    if not rn_path.exists():
        return

    rn = pd.read_csv(rn_path)
    baseline = pd.read_csv(BASELINE_METRICS)

    # Only compare identical dataset/seed combinations.
    covered = rn[["dataset", "seed"]].drop_duplicates()

    baseline = baseline.merge(
        covered,
        on=["dataset", "seed"],
        how="inner",
    )

    combined = pd.concat(
        [rn, baseline],
        ignore_index=True,
        sort=False,
    )

    summary_rows = []

    for (dataset, model), group in combined.groupby(
        ["dataset", "model"], sort=False
    ):
        row = {
            "dataset": dataset,
            "model": model,
            "runs": len(group),
        }

        for metric in ["R2", "RMSE", "MAE", "MSE"]:
            values = group[
                f"test_{metric}"
            ].to_numpy(dtype=float)

            row[f"{metric}_mean"] = float(
                values.mean()
            )

            row[f"{metric}_std"] = (
                float(values.std(ddof=1))
                if len(values) > 1
                else np.nan
            )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        results_dir / "summary.csv",
        index=False,
    )

    paired_rows = []

    for dataset in rn["dataset"].unique():
        rn_group = rn[rn["dataset"] == dataset]

        for competitor in MODEL_ORDER[1:]:
            baseline_group = baseline[
                (baseline["dataset"] == dataset)
                & (baseline["model"] == competitor)
            ]

            paired = rn_group.merge(
                baseline_group,
                on=["dataset", "seed"],
                suffixes=("_RN", "_baseline"),
            )

            if paired.empty:
                continue

            # Positive RMSE advantage:
            # RN has lower error than competitor.
            delta_rmse = (
                paired["test_RMSE_baseline"]
                - paired["test_RMSE_RN"]
            )

            # Positive R2 advantage:
            # RN has higher R2 than competitor.
            delta_r2 = (
                paired["test_R2_RN"]
                - paired["test_R2_baseline"]
            )

            paired_rows.append({
                "dataset": dataset,
                "comparison": f"VANTA-RN vs {competitor}",
                "seeds": len(paired),
                "RN_wins": int((delta_rmse > 0).sum()),
                "RMSE_advantage_mean": delta_rmse.mean(),
                "RMSE_advantage_std": delta_rmse.std(
                    ddof=1
                ) if len(paired) > 1 else np.nan,
                "R2_advantage_mean": delta_r2.mean(),
                "R2_advantage_std": delta_r2.std(
                    ddof=1
                ) if len(paired) > 1 else np.nan,
            })

    pd.DataFrame(paired_rows).to_csv(
        results_dir / "paired_comparisons.csv",
        index=False,
    )

    for dataset, group in summary.groupby("dataset"):
        print("\n" + "=" * 90)
        print(f"DATASET: {dataset.upper()}")
        print("=" * 90)

        for name in MODEL_ORDER:
            match = group[group["model"] == name]

            if match.empty:
                continue

            r = match.iloc[0]

            print(
                f"{name:16s} | "
                f"R2 {r['R2_mean']:+.5f} "
                f"± {r['R2_std']:.5f} | "
                f"RMSE {r['RMSE_mean']:,.3f} "
                f"± {r['RMSE_std']:,.3f} | "
                f"n={int(r['runs'])}"
            )

        ordered = group.set_index("model").reindex(
            [
                name for name in MODEL_ORDER
                if name in group["model"].values
            ]
        )

        folder = results_dir / dataset
        folder.mkdir(parents=True, exist_ok=True)

        for metric, filename, xlabel in [
            ("RMSE", "rmse_comparison.png",
             "Test RMSE (lower is better)"),
            ("R2", "r2_comparison.png",
             "Test R2 (higher is better)"),
        ]:
            plt.figure(figsize=(10, 5))

            means = ordered[
                f"{metric}_mean"
            ].to_numpy()

            errors = ordered[
                f"{metric}_std"
            ].fillna(0).to_numpy()

            plt.barh(
                ordered.index.tolist(),
                means,
                xerr=errors,
                capsize=4,
            )

            plt.gca().invert_yaxis()
            plt.xlabel(xlabel)
            plt.title(
                f"{dataset.upper()} — mean ± sample SD"
            )
            plt.grid(axis="x", alpha=0.2)
            plt.tight_layout()

            plt.savefig(
                folder / filename,
                dpi=180,
            )
            plt.close()


# ---------------------------------------------------------
# 5. Main experiment
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(REGISTRY),
        default=["ames"],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[43],
    )

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=4)

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=EX04 / "results",
    )

    args = parser.parse_args()

    if min(
        args.epochs,
        args.batch_size,
        args.patience,
        args.threads,
    ) <= 0:
        parser.error("Training settings must be positive.")

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate seeds are not allowed.")

    if not BASELINE_METRICS.exists():
        raise FileNotFoundError(
            "Experiment 03 metrics.csv is missing."
        )

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    torch.set_num_threads(args.threads)

    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = results_dir / "metrics.csv"
    config_file = results_dir / "config.json"

    # This configuration deliberately excludes the seed
    # and dataset list, so we can expand the experiment
    # incrementally without mixing hyperparameters.
    configuration = {
        "version": "VANTA-RN-v0.1",
        "normalization": "smooth_l2",
        "attention_divisor": "sqrt_F",
        "latent_dim": ex03.LATENT_DIM,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": 1e-4,
    }

    if config_file.exists():
        old_config = json.loads(
            config_file.read_text()
        )

        if old_config != configuration:
            raise RuntimeError(
                "Existing experiment has different settings. "
                "Use --results-dir for a new configuration."
            )
    else:
        config_file.write_text(
            json.dumps(configuration, indent=2)
        )

    if metrics_file.exists():
        rows = pd.read_csv(metrics_file).to_dict(
            orient="records"
        )
    else:
        rows = []

    completed = {
        (str(r["dataset"]), int(r["seed"]))
        for r in rows
    }

    print("=" * 70)
    print("VANTA-RN v0.1 — NORMALIZATION EXPERIMENT")
    print("=" * 70)
    print("Device:", args.device)
    print("Datasets:", args.datasets)
    print("Seeds:", args.seeds)
    print("Normalization: smooth L2")
    print("Attention divisor: sqrt(F)")

    for dataset_name in args.datasets:
        bundle = load_dataset(dataset_name)

        for seed in args.seeds:
            if (dataset_name, seed) in completed:
                print(
                    f"\nSkipping completed: "
                    f"{dataset_name}, seed {seed}"
                )
                continue

            print("\n" + "#" * 70)
            print(f"{dataset_name.upper()} | SEED {seed}")
            print("#" * 70)

            data = preprocess(bundle, seed)

            verify_reference(data)

            original = verify_original_split(
                dataset_name, seed, data
            )

            folder = (
                results_dir
                / dataset_name
                / f"seed_{seed}"
                / "vanta_rn"
            )

            folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Reuse the actual EX03 training function.
            # It handles random seeds, AdamW, early stopping,
            # gradient clipping, histories and predictions.
            ex03.VantaR = NormalizedFeatureAttentionRegressor

            try:
                result = ex03.train_neural(
                    "VANTA-R",
                    seed,
                    data,
                    args,
                    folder,
                )
            finally:
                ex03.VantaR = ORIGINAL_VANTA_CLASS

            result["model"] = MODEL_NAME
            result["dataset"] = dataset_name
            result["seed"] = seed

            label_checkpoint(folder)

            report_outliers(
                dataset_name,
                seed,
                folder,
                original,
            )

            print(
                f"\nRESULT | VANTA-RN | "
                f"R2={result['test_R2']:.5f} | "
                f"RMSE={result['test_RMSE']:,.3f}"
            )

            result["details"] = json.dumps(
                result["details"]
            )

            rows.append(result)
            completed.add((dataset_name, seed))

            pd.DataFrame(rows).to_csv(
                metrics_file,
                index=False,
            )

        summarize(results_dir)

    print("\n" + "=" * 70)
    print("VANTA-RN EXPERIMENT COMPLETE")
    print("=" * 70)
    print("Results saved to:", results_dir)


if __name__ == "__main__":
    main()

