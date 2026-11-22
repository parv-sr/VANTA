"""VANTA-RN residual-pathway ablation — inference only; no parameter updates.

Run from VANTA repository root after the standoff confirmation:

    uv run python3 experiments/rn_interlude/residual_ablation.py --dataset ames
    uv run python3 experiments/rn_interlude/residual_ablation.py --dataset california

Uses the *selected tuned* VANTA-RN checkpoint from the standoff for each
seed. Computes fixed, post-hoc changes to the scalar residual gate and an
exact source/receiver decomposition of the signed interaction contributions.

IMPORTANT: Checkpoints were co-trained with a nonzero residual pathway. A
post-hoc direct-only/interaction-only evaluation does NOT measure the
performance of an independently trained ablated architecture. These are
mechanistic diagnostics, not a new hyperparameter search.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EX03 = ROOT / "experiments" / "datasets_ex03"
sys.path.insert(0, str(EX03))
sys.path.insert(0, str(HERE))

from datasets import load_dataset, preprocess  # noqa: E402
from trace_explosions import load_checkpoint_model  # noqa: E402

DEFAULT_RESULTS = HERE / "results" / "standoff"
DEFAULT_OUTPUT = HERE / "results" / "residual_ablation"
DEFAULT_MULTIPLIERS = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5]


def get_selected_checkpoint(standoff_root: Path, dataset: str, seed: int) -> Path:
    study = standoff_root / dataset
    selection_file = study / "selected.json"
    if not selection_file.is_file():
        raise FileNotFoundError(
            f"Missing {selection_file}. Complete the standoff search/confirmation "
            "or pass --standoff-root to its results directory."
        )
    selection = json.loads(selection_file.read_text())
    config_name = selection["VANTA-RN"]["name"]
    path = study / "VANTA-RN" / f"seed_{seed}" / config_name / "checkpoint.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing trained standoff checkpoint: {path}")
    return path


def calc_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if len(y) != len(p):
        raise ValueError("Target/prediction length mismatch")
    if not np.isfinite(p).all():
        raise FloatingPointError("Nonfinite predictions")
    errors = y - p
    squared = errors ** 2
    return {
        "R2": float(r2_score(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": float(mean_absolute_error(y, p)),
        "worst_error_share_pct": float(100.0 * np.max(squared) / np.sum(squared))
        if float(np.sum(squared)) > 0 else 0.0,
    }


@torch.inference_mode()
def compute_components(model, xn: torch.Tensor, xc: torch.Tensor, data: dict) -> dict:
    """Numerically evaluate all components, with float64 *after* Q/K projection.

    For normalized q/k and scalar v:
      direct_n = w^T v
      C_ij = gamma * w_i * (q_i^T k_j / sqrt(F)) * v_j
      branch_n = sum_ij C_ij
      receiver_i = sum_j C_ij
      source_j   = sum_i C_ij
      y_n = b + direct_n + branch_n

    We never change model parameters or training data.
    """
    q, k, v = model.project(xn, xc)
    q = q.to(torch.float64)
    k = k.to(torch.float64)
    v = v.to(torch.float64)
    w = model.readout.weight.reshape(-1).to(torch.float64)
    b = model.readout.bias.reshape(()).to(torch.float64)
    gamma = model.gamma.reshape(()).to(torch.float64)
    F = len(data["features"])
    if v.shape[1] != F:
        raise RuntimeError("Feature count differs from model output")

    # Value/source -> key summary -> per-feature receiving output.
    # These two decompositions sum to precisely the same signed scalar.
    key_summary = (k * v.unsqueeze(-1)).sum(dim=1)  # (batch,d)
    receiver_n = gamma * w.unsqueeze(0) * (q * key_summary.unsqueeze(1)).sum(-1) / math.sqrt(F)
    weighted_queries = (q * w.view(1, -1, 1)).sum(dim=1)  # (batch,d)
    source_n = gamma * v * (k * weighted_queries.unsqueeze(1)).sum(-1) / math.sqrt(F)
    branch_n = receiver_n.sum(-1)
    direct_by_feature_n = v * w.unsqueeze(0)
    direct_n = direct_by_feature_n.sum(-1)

    if not torch.allclose(branch_n, source_n.sum(-1), rtol=1e-10, atol=1e-10):
        raise RuntimeError("Source and receiver contribution sums disagree")

    mean = float(data["target_mean"])
    std = float(data["target_std"])
    full_n = b + direct_n + branch_n
    fwd_n = model(xn, xc).to(torch.float64)
    # Float32 forward and float64 decomposition can differ slightly due to
    # floating-point summation order; more than 2e-4 indicates a real mismatch.
    if not torch.allclose(full_n, fwd_n, rtol=2e-4, atol=2e-4):
        max_diff = (full_n - fwd_n).abs().max().item()
        raise RuntimeError(f"Decomposition disagrees with forward() (max normalized diff {max_diff:.3g})")

    def np1(t):
        return t.detach().cpu().numpy()

    return {
        "intercept": mean + std * float(b.item()),
        "direct": mean + std * np1(b + direct_n),
        "interaction_only": mean + std * np1(b + branch_n),
        "full": mean + std * np1(full_n),
        "attention": std * np1(branch_n),
        "direct_deviation": std * np1(direct_n),
        "source_contributions": std * np1(source_n),
        "receiver_contributions": std * np1(receiver_n),
        "direct_feature_contributions": std * np1(direct_by_feature_n),
    }


def analyze_split(model, data: dict, split: str, device: torch.device,
                  multipliers: list[float], top: int, output: Path,
                  dataset_name: str, seed: int, checkpoint: Path, batch_size: int = 512):
    part = data["splits"][split]
    xn_all, xc_all, _ = part["tensor"].tensors
    fields = ["direct", "interaction_only", "full", "attention", "direct_deviation",
              "source_contributions", "receiver_contributions", "direct_feature_contributions"]
    chunks: dict[str, list] = {field: [] for field in fields}
    intercept = None
    for start in range(0, len(xn_all), batch_size):
        result = compute_components(model,
            xn_all[start:start + batch_size].to(device),
            xc_all[start:start + batch_size].to(device), data)
        if intercept is None:
            intercept = result["intercept"]
        for field in fields:
            chunks[field].append(result[field])
    all_components = {field: np.concatenate(chunks[field], axis=0) for field in fields}
    actual = np.asarray(part["actual"], dtype=np.float64)
    row_indices = np.asarray(part["indices"])

    full = all_components["full"]
    direct = all_components["direct"]
    attention = all_components["attention"]
    interaction_only = all_components["interaction_only"]
    if not np.allclose(full, direct + attention, rtol=1e-11, atol=1e-7):
        raise RuntimeError("Direct plus attention != full prediction")
    if not np.allclose(full, direct + interaction_only - intercept, rtol=1e-11, atol=1e-7):
        raise RuntimeError("Both component paths do not reconstruct prediction")

    folder = output / dataset_name / f"seed_{seed}" / split
    folder.mkdir(parents=True, exist_ok=True)

    per_row = pd.DataFrame({
        "row_index": row_indices,
        "actual": actual,
        "full_prediction": full,
        "direct_only_prediction": direct,
        "interaction_only_prediction": interaction_only,
        "intercept_prediction": float(intercept),
        "direct_deviation": all_components["direct_deviation"],
        "attention_contribution": attention,
        "signed_error_full": full - actual,
        "abs_error_full": np.abs(full - actual),
        "squared_error_full": (full - actual) ** 2,
        "abs_attention_fraction_of_total_deviation": (
            np.abs(attention) / (np.abs(all_components["direct_deviation"]) + np.abs(attention) + 1e-12)
        ),
        "cancellation_index": (
            (np.abs(all_components["direct_deviation"]) + np.abs(attention))
            / (np.abs(all_components["direct_deviation"] + attention) + 1e-8)
        ),
    })
    per_row.to_csv(folder / "all_rows.csv", index=False)

    # Fixed (pre-declared) multipliers; do NOT select a winning gate on test metrics.
    metric_rows = []
    for scale in multipliers:
        pred = direct + scale * attention
        row = dict(dataset=dataset_name, seed=seed, split=split,
                   variant=("direct_only" if scale == 0 else "trained_full" if scale == 1 else f"gate_x{scale:g}"),
                   gate_multiplier=scale, gamma_effective=float(model.gamma.item()) * scale,
                   n=len(actual), checkpoint=str(checkpoint), **calc_metrics(actual, pred))
        metric_rows.append(row)
    # The interaction-only row keeps the original learned readout bias plus
    # dataset target mean; it does not include the learned direct values.
    metric_rows.append(dict(dataset=dataset_name, seed=seed, split=split,
                            variant="interaction_only_intercept_anchored",
                            gate_multiplier=np.nan, gamma_effective=float(model.gamma.item()),
                            n=len(actual), checkpoint=str(checkpoint),
                            **calc_metrics(actual, interaction_only)))
    metric_rows.append(dict(dataset=dataset_name, seed=seed, split=split,
                            variant="intercept_only", gate_multiplier=np.nan,
                            gamma_effective=0.0, n=len(actual), checkpoint=str(checkpoint),
                            **calc_metrics(actual, np.full_like(actual, intercept))))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(folder / "variant_metrics.csv", index=False)

    # Report attribution from the *trained full* checkpoint only.
    feature_rows = []
    for j, feature in enumerate(data["features"]):
        recv = all_components["receiver_contributions"][:, j]
        src = all_components["source_contributions"][:, j]
        direct_feature = all_components["direct_feature_contributions"][:, j]
        feature_rows.append(dict(feature=feature, feature_type=("numerical" if j < len(data["numeric"]) else "categorical"),
                                 mean_abs_receiver=float(np.mean(np.abs(recv))),
                                 mean_abs_source=float(np.mean(np.abs(src))),
                                 mean_abs_direct=float(np.mean(np.abs(direct_feature))),
                                 mean_signed_receiver=float(np.mean(recv)),
                                 mean_signed_source=float(np.mean(src))))
    pd.DataFrame(feature_rows).sort_values("mean_abs_source", ascending=False).to_csv(
        folder / "feature_summary.csv", index=False)

    bad_positions = np.argsort(per_row["squared_error_full"].to_numpy())[::-1][:top]
    per_row.iloc[bad_positions].to_csv(folder / "worst_rows.csv", index=False)
    for pos in bad_positions:
        r = int(row_indices[pos])
        fdf = pd.DataFrame({
            "feature": data["features"],
            "type": ["numerical"] * len(data["numeric"]) + ["categorical"] * len(data["categorical"]),
            "source_attention": all_components["source_contributions"][pos],
            "receiver_attention": all_components["receiver_contributions"][pos],
            "direct_feature": all_components["direct_feature_contributions"][pos],
        })
        fdf["abs_source_attention"] = fdf["source_attention"].abs()
        fdf["abs_receiver_attention"] = fdf["receiver_attention"].abs()
        fdf.sort_values("abs_source_attention", ascending=False).to_csv(
            folder / f"row_{r}_feature_contributions.csv", index=False)

    full_metrics = next(row for row in metric_rows if row["variant"] == "trained_full")
    direct_metrics = next(row for row in metric_rows if row["variant"] == "direct_only")
    top1_share = full_metrics["worst_error_share_pct"]
    print(f"  seed {seed:3d} {split:4s} | full R2={full_metrics['R2']:+.5f}, "
          f"RMSE={full_metrics['RMSE']:,.2f} | direct-only RMSE={direct_metrics['RMSE']:,.2f} | "
          f"worst row share={top1_share:.1f}%")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ames", "concrete", "california", "airbnb"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument("--multipliers", nargs="+", type=float, default=DEFAULT_MULTIPLIERS)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--standoff-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or args.top < 1 or args.batch_size < 1 or args.threads < 1:
        parser.error("Seeds must be unique; top, batch-size, threads must be positive")
    if not np.isfinite(args.multipliers).all() or not any(np.isclose(a, 0.0) for a in args.multipliers) or not any(np.isclose(a, 1.0) for a in args.multipliers):
        parser.error("Multipliers must be finite and must include 0 and 1")
    args.multipliers = sorted(set(float(round(m, 8)) for m in args.multipliers))
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    bundle = load_dataset(args.dataset)
    splits = ["val", "test"] if args.split == "both" else [args.split]
    metrics = []
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        data = preprocess(bundle, seed)
        path = get_selected_checkpoint(args.standoff_root, args.dataset, seed)
        model, ckpt = load_checkpoint_model(path, data, device)
        print(f"\n{args.dataset.upper()} | seed {seed} | "
              f"gamma={float(model.gamma.item()):.4f} | checkpoint={path.name}")
        for split in splits:
            df = analyze_split(model, data, split, device, args.multipliers,
                               args.top, args.output, args.dataset, seed, path, args.batch_size)
            metrics.append(df)
    raw = pd.concat(metrics, ignore_index=True)
    folder = args.output / args.dataset
    raw.to_csv(folder / "variant_metrics_per_seed.csv", index=False)
    # Standard deviation means variability over different splits *and* seeds.
    summary = raw.groupby(["split", "variant", "gate_multiplier"], dropna=False).agg(
        n=("seed", "count"), R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
        worst_share_mean_pct=("worst_error_share_pct", "mean")
    ).reset_index()
    summary.to_csv(folder / "variant_summary.csv", index=False)
    print("\nAggregate metrics (do NOT select a gate from test scores):")
    for split in splits:
        group = summary[summary["split"] == split]
        print("\n" + group[["variant", "n", "R2_mean", "R2_sd", "RMSE_mean", "RMSE_sd"]].to_string(
            index=False, float_format=lambda x: f"{x:,.4f}"))
    print(f"\nSaved files under: {folder}")
    print("NOTE: This is post-hoc intervention on co-trained weights, not a retrained architecture ablation.")
    print("NOTE: Data splits 42-44 have been explored before. Test-set sweeps here are exploratory diagnostics, not hyperparameter selection.")


if __name__ == "__main__":
    main()

