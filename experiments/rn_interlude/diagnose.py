
"""
VANTA-RN final diagnostic.

Loads saved EX04 checkpoints. Does not train anything.

Run:
    uv run python3 experiments/rn_interlude/diagnose.py
"""

import argparse
import math

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
)

from common import (
    OUT,
    load_dataset,
    preprocess,
    load_ex04_model,
)


def inspect(data, model, split, device, batch_size=256):

    part = data["splits"][split]
    x_num, x_cat, _ = part["tensor"].tensors

    outputs = []

    F = len(data["features"])
    scale = math.sqrt(F)

    with torch.inference_mode():

        for start in range(0, len(x_num), batch_size):

            end = start + batch_size

            xn = x_num[start:end].to(device)
            xc = x_cat[start:end].to(device)

            # Raw projections, before normalization.
            q_raw, k_raw, values = model.project_raw(
                xn, xc
            )

            # Normalized projections.
            q = model.smooth_l2_normalize(q_raw)
            k = model.smooth_l2_normalize(k_raw)

            # Associative aggregation.
            summary = (
                k * values.unsqueeze(-1)
            ).sum(dim=1)

            h = (
                q * summary.unsqueeze(1)
            ).sum(dim=-1) / scale

            # Direct pathway includes regression bias.
            direct = model.readout(values).squeeze(-1)

            # Interaction pathway.
            branch = (
                h
                * model.gamma
                * model.readout.weight.squeeze(0)
            ).sum(dim=-1)

            prediction = (
                (direct + branch) * data["target_std"]
                + data["target_mean"]
            )

            direct_original = (
                direct * data["target_std"]
                + data["target_mean"]
            )

            branch_original = (
                branch * data["target_std"]
            )

            # Explicit attention matrix for diagnostic use.
            max_score = (
                q @ k.transpose(1, 2)
            ).abs().amax(dim=(1, 2)) / scale

            q_raw_norm = (
                q_raw.norm(dim=-1).amax(dim=1)
            )

            k_raw_norm = (
                k_raw.norm(dim=-1).amax(dim=1)
            )

            if xn.shape[1]:
                max_numeric_z = xn.abs().amax(dim=1)
            else:
                max_numeric_z = torch.zeros(
                    len(xn),
                    device=device,
                )

            batch_results = torch.stack([
                prediction,
                direct_original,
                branch_original,
                max_score,
                q_raw_norm,
                k_raw_norm,
                max_numeric_z,
            ], dim=1)

            outputs.append(
                batch_results.cpu().numpy()
            )

    values = np.concatenate(outputs, axis=0)

    frame = pd.DataFrame(
        values,
        columns=[
            "prediction",
            "direct_usd",
            "attention_usd",
            "max_abs_A",
            "max_raw_q_norm",
            "max_raw_k_norm",
            "max_abs_numeric_z",
        ],
    )

    frame.insert(
        0, "row_index", part["indices"]
    )

    frame.insert(
        1, "actual", part["actual"]
    )

    frame["error"] = (
        frame["prediction"] - frame["actual"]
    )

    frame["squared_error"] = frame["error"] ** 2

    return frame


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "ames",
            "california",
            "airbnb",
            "concrete",
        ],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
    )

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    device = torch.device("cpu")

    result_dir = OUT / "diagnostics"
    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = []

    for dataset in args.datasets:

        bundle = load_dataset(dataset)

        for seed in args.seeds:

            data = preprocess(bundle, seed)

            model, checkpoint_path = load_ex04_model(
                data, dataset, seed, device
            )

            frame = inspect(
                data,
                model,
                args.split,
                device,
            )

            folder = (
                result_dir / dataset / f"seed_{seed}"
            )

            folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            frame.to_csv(
                folder / f"{args.split}_decomposition.csv",
                index=False,
            )

            worst = frame.nlargest(
                args.top,
                "squared_error",
            )

            worst.to_csv(
                folder / "worst_predictions.csv",
                index=False,
            )

            gamma = float(
                model.gamma.detach().item()
            )

            total = frame["squared_error"].sum()

            worst_share = (
                100.0
                * frame["squared_error"].max()
                / total
            )

            branch_rms = float(
                np.sqrt(np.mean(
                    frame["attention_usd"] ** 2
                ))
            )

            base_centered = (
                frame["direct_usd"]
                - data["target_mean"]
            )

            base_rms = float(
                np.sqrt(np.mean(base_centered ** 2))
            )

            r2 = float(
                r2_score(
                    frame["actual"],
                    frame["prediction"],
                )
            )

            rmse = float(
                np.sqrt(mean_squared_error(
                    frame["actual"],
                    frame["prediction"],
                ))
            )

            print(
                f"\n{dataset.upper()} | "
                f"seed {seed} | {args.split} | "
                f"gamma={gamma:.4f}"
            )

            print(
                f"R2={r2:.5f}, "
                f"RMSE={rmse:,.3f}; "
                f"largest error share={worst_share:.2f}%"
            )

            print(
                f"Branch RMS={branch_rms:,.3f}; "
                f"centered direct RMS={base_rms:,.3f}"
            )

            print(
                f"max |A_ij|="
                f"{frame['max_abs_A'].max():.5f}; "
                f"bound={1/math.sqrt(len(data['features'])):.5f}"
            )

            print(
                f"p99 max raw ||Q_i||="
                f"{frame['max_raw_q_norm'].quantile(.99):.3f}; "
                f"p99 max raw ||K_i||="
                f"{frame['max_raw_k_norm'].quantile(.99):.3f}"
            )

            print(
                worst[[
                    "row_index",
                    "actual",
                    "prediction",
                    "direct_usd",
                    "attention_usd",
                ]].to_string(
                    index=False,
                    float_format=lambda v: f"{v:,.2f}",
                )
            )

            if (
                dataset == "ames"
                and seed == 43
                and args.split == "test"
            ):
                for row_idx in (2180, 956):

                    hits = np.where(
                        data["splits"]["test"]["indices"]
                        == row_idx
                    )[0]

                    if len(hits):

                        z = data["splits"]["test"][
                            "numeric"
                        ][hits[0]]

                        largest = np.argsort(
                            np.abs(z)
                        )[-6:][::-1]

                        print(
                            f"Row {row_idx} strongest "
                            f"numeric z-scores: "
                            + ", ".join(
                                f"{data['numeric'][i]}={z[i]:+.2f}"
                                for i in largest
                            )
                        )

            summary.append({
                "dataset": dataset,
                "seed": seed,
                "split": args.split,
                "gamma": gamma,
                "R2": r2,
                "RMSE": rmse,
                "attention_RMS": branch_rms,
                "direct_centered_RMS": base_rms,
                "worst_squared_error_pct": worst_share,
                "p99_max_q_raw_norm": float(
                    frame["max_raw_q_norm"].quantile(.99)
                ),
                "p99_max_k_raw_norm": float(
                    frame["max_raw_k_norm"].quantile(.99)
                ),
                "max_abs_A": float(
                    frame["max_abs_A"].max()
                ),
                "checkpoint": str(checkpoint_path),
            })

    pd.DataFrame(summary).to_csv(
        result_dir / "diagnostics_summary.csv",
        index=False,
    )

    print(f"\nSaved diagnostics to {result_dir}")


if __name__ == "__main__":
    main()

