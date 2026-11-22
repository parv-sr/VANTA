"""Dissect extreme VANTA-RN predictions without retraining or modifying weights.

Run after the standoff confirmation:
 uv run python3 experiments/rn_interlude/trace_explosions.py --dataset ames --seed 43

Use --checkpoint /path/to/checkpoint.pt to trace another compatible VANTA-RN run.
Identifies (a) large standardized inputs, (b) direct/attention branch effects,
(c) per-feature counterfactual effects, and (d) parameter-group gradients.
Counterfactuals are diagnostic interventions, NOT causal feature attributions.
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
from sklearn.metrics import r2_score, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
EX03 = ROOT / "experiments" / "datasets_ex03"
RN = ROOT / "VANTA_RN"
sys.path.insert(0, str(EX03))
sys.path.insert(0, str(RN))
from datasets import load_dataset, preprocess  # noqa: E402
from model import NormalizedFeatureAttentionRegressor  # noqa: E402


@torch.no_grad()
def parts(model, xn, xc, data):
    q_raw, k_raw, v = model.project_raw(xn, xc)
    q = model.smooth_l2_normalize(q_raw)
    k = model.smooth_l2_normalize(k_raw)
    summary = (k * v.unsqueeze(-1)).sum(dim=1)
    h = (q * summary.unsqueeze(1)).sum(-1) / math.sqrt(model.num_features)
    w = model.readout.weight.squeeze(0)
    direct = model.readout(v).squeeze(-1)
    contributions = h * w * model.gamma
    interaction = contributions.sum(dim=-1)
    prediction = (direct + interaction) * data["target_std"] + data["target_mean"]
    return {
        "prediction": prediction,
        "direct": direct * data["target_std"] + data["target_mean"],
        "interaction": interaction * data["target_std"],
        "per_output_feature": contributions * data["target_std"],
        "raw_q_norm": q_raw.norm(dim=-1),
        "raw_k_norm": k_raw.norm(dim=-1),
        "normalized_q_norm": q.norm(dim=-1),
        "normalized_k_norm": k.norm(dim=-1),
        "max_abs_attention": (q @ k.transpose(-1, -2)).abs().amax(dim=(1,2)) / math.sqrt(model.num_features),
        "values": v,
        "h": h,
    }


def load_checkpoint_model(checkpoint_path, data, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "state_dict" not in ckpt:
        raise ValueError("Not a supported VANTA-RN checkpoint")
    cfg = ckpt.get("config", {})
    latent_dim = cfg.get("latent_dim", ckpt.get("latent_dim", 24))
    n_numeric = len(data["numeric"])
    f = len(data["features"])
    model = NormalizedFeatureAttentionRegressor(
        num_features=f,
        numeric_positions=list(range(n_numeric)),
        categorical_positions=list(range(n_numeric, f)),
        categorical_cardinalities=data["cardinalities"],
        latent_dim=latent_dim, gamma_init=0.1, learnable_gamma=True,
    )
    for key in ("numeric", "categorical", "cardinalities", "vocabularies"):
        if key in ckpt and ckpt[key] != data[key]:
            raise RuntimeError(f"Dataset preprocessing mismatch: {key}")
    for key in ("target_mean", "target_std"):
        if key in ckpt and not np.isclose(ckpt[key], data[key]):
            raise RuntimeError(f"Target preprocessing mismatch: {key}")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device).eval(), ckpt


def group_of_parameter(name):
    if name.startswith("query_"):
        return "Q projection"
    if name.startswith("key_"):
        return "K projection"
    if name.startswith("category_value_embeddings"):
        return "categorical values"
    if name.startswith("category_embeddings"):
        return "categorical tokens"
    if name.startswith("readout"):
        return "readout"
    if name == "gamma":
        return "gamma"
    return "other"


def parameter_diagnostics(model, xn, xc, data):
    """Local derivatives of *one prediction* w.r.t. each parameter group."""
    model.zero_grad(set_to_none=True)
    normalized_prediction = model(xn, xc).squeeze(0)
    normalized_prediction.backward()
    groups = {}
    individual = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        group = group_of_parameter(name)
        g = param.grad.detach()
        p = param.detach()
        entry = groups.setdefault(group, dict(grad_l2_sq=0.0, grad_times_param_l1=0.0, max_abs_parameter=0.0))
        entry["grad_l2_sq"] += float(torch.sum(g.square()))
        entry["grad_times_param_l1"] += float(torch.sum(torch.abs(g*p)))
        entry["max_abs_parameter"] = max(entry["max_abs_parameter"], float(p.abs().max()))
        individual.append(dict(parameter=name, group=group,
                               grad_l2=float(torch.linalg.vector_norm(g)),
                               max_abs_parameter=float(p.abs().max()),
                               grad_times_param_l1=float((g*p).abs().sum())))
    result = []
    for group, vals in groups.items():
        result.append(dict(group=group, grad_l2=math.sqrt(vals["grad_l2_sq"]),
                           grad_times_param_l1=vals["grad_times_param_l1"],
                           max_abs_parameter=vals["max_abs_parameter"]))
    return pd.DataFrame(result).sort_values("grad_times_param_l1", ascending=False), pd.DataFrame(individual).sort_values("grad_times_param_l1", ascending=False)


def evaluate_all(model, data, split, device, batch_size=256):
    part = data["splits"][split]
    xn_all, xc_all, _ = part["tensor"].tensors
    rows = []
    with torch.no_grad():
        for start in range(0, len(xn_all), batch_size):
            xn, xc = xn_all[start:start+batch_size].to(device), xc_all[start:start+batch_size].to(device)
            a = parts(model, xn, xc, data)
            for i in range(len(xn)):
                rows.append(dict(row_index=int(part["indices"][start+i]),
                                 actual=float(part["actual"][start+i]),
                                 prediction=float(a["prediction"][i]),
                                 direct=float(a["direct"][i]),
                                 interaction=float(a["interaction"][i]),
                                 max_abs_attention=float(a["max_abs_attention"][i]),
                                 max_raw_q_norm=float(a["raw_q_norm"][i].max()),
                                 max_raw_k_norm=float(a["raw_k_norm"][i].max()),
                                 max_abs_value=float(a["values"][i].abs().max()),
                                 max_numeric_abs_z=float(xn[i].abs().max()) if xn.shape[1] else 0.0))
    frame = pd.DataFrame(rows)
    frame["error"] = frame["prediction"] - frame["actual"]
    frame["squared_error"] = frame["error"]**2
    return frame


def feature_counterfactuals(model, data, part, position, device):
    """Replace each feature with train numerical mean (z=0), or unknown cat (id=0)."""
    num = len(data["numeric"])
    xn_all, xc_all, _ = part["tensor"].tensors
    xn = xn_all[position:position+1].to(device)
    xc = xc_all[position:position+1].to(device)
    with torch.no_grad():
        original = parts(model, xn, xc, data)
        rows = []
        for j, feature in enumerate(data["features"]):
            xn_mod = xn.clone()
            xc_mod = xc.clone()
            if j < num:
                xn_mod[0,j] = 0.0
            else:
                xc_mod[0,j-num] = 0
            changed = parts(model, xn_mod, xc_mod, data)
            rows.append(dict(feature=feature, original_input=float(xn[0,j]) if j < num else int(xc[0,j-num]),
                             input_type="numerical" if j < num else "categorical",
                             change_in_prediction=float(original["prediction"][0]-changed["prediction"][0]),
                             change_in_direct=float(original["direct"][0]-changed["direct"][0]),
                             change_in_interaction=float(original["interaction"][0]-changed["interaction"][0]),
                             feature_readout_interaction=float(original["per_output_feature"][0,j]),
                             raw_q_norm=float(original["raw_q_norm"][0,j]),
                             raw_k_norm=float(original["raw_k_norm"][0,j]),
                             normalized_q_norm=float(original["normalized_q_norm"][0,j]),
                             normalized_k_norm=float(original["normalized_k_norm"][0,j])))
    return pd.DataFrame(rows).sort_values("change_in_prediction", key=lambda s: s.abs(), ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ames","concrete","california","airbnb"], default="ames")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "attribution")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    data = preprocess(load_dataset(args.dataset), args.seed)
    if args.checkpoint is None:
        selected_path = Path(__file__).resolve().parent / "results" / "standoff" / args.dataset / "selected.json"
        if not selected_path.exists():
            parser.error("Missing standoff selected.json. Run search and confirmation, or pass --checkpoint")
        selected = json.loads(selected_path.read_text())
        cfg_name = selected["VANTA-RN"]["name"]
        args.checkpoint = (Path(__file__).resolve().parent / "results" / "standoff" /
                           args.dataset / "VANTA-RN" / f"seed_{args.seed}" / cfg_name / "checkpoint.pt")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint unavailable: {args.checkpoint}")
    model, checkpoint = load_checkpoint_model(args.checkpoint, data, device)
    folder = args.output / args.dataset / f"seed_{args.seed}" / args.split
    folder.mkdir(parents=True, exist_ok=True)
    frame = evaluate_all(model, data, args.split, device)
    frame.to_csv(folder / "all_prediction_decompositions.csv", index=False)
    top = frame.nlargest(args.top, "squared_error")
    total = float(frame["squared_error"].sum())
    print(f"\n{args.dataset} seed {args.seed} {args.split}: R2={r2_score(frame['actual'],frame['prediction']):.5f}, RMSE={math.sqrt(mean_squared_error(frame['actual'],frame['prediction'])):,.2f}")
    print(f"gamma = {float(model.gamma.detach()):.5f}; highest single-error share = {100*float(top.iloc[0]['squared_error'])/total:.2f}%")
    print("Largest-error observations:\n"+top[["row_index","actual","prediction","direct","interaction","max_abs_attention"]].to_string(index=False))
    print("\nGlobal parameter magnitudes:")
    for name, p in model.named_parameters():
        print(f" {name:45s} maxabs={float(p.detach().abs().max()):.6g} l2={float(p.detach().norm()):.6g}")
    part = data["splits"][args.split]
    for row in top.itertuples():
        position = int(np.where(part["indices"] == row.row_index)[0][0])
        df = feature_counterfactuals(model, data, part, position, device)
        df.to_csv(folder / f"row_{row.row_index}_feature_counterfactuals.csv", index=False)
        xn_all, xc_all, _ = part["tensor"].tensors
        grp, per_parameter = parameter_diagnostics(model, xn_all[position:position+1], xc_all[position:position+1], data)
        grp.to_csv(folder / f"row_{row.row_index}_parameter_groups.csv", index=False)
        per_parameter.to_csv(folder / f"row_{row.row_index}_parameters.csv", index=False)
        print(f"\nRow {row.row_index}, top feature replacement effects:\n"+df.head(8).to_string(index=False))
        print("Parameter group local sensitivities (normalized-target units):\n"+grp.to_string(index=False))
    print(f"\nSaved diagnostics in {folder}")
    print("CAUTION: gradient-based group sensitivities depend on parameterization and scale; feature replacements are not causal attributions.")


if __name__ == "__main__":
    main()

