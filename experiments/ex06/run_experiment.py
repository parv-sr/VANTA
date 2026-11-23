"""Experiment 6: cross-dataset latent width and numerical attention-value maps.

Run from repository root. Four files aggregate all datasets, stages and seeds.
No original model or regression-head implementation is changed.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_ex06_support", HERE / "support.py")
if _spec is None or _spec.loader is None:
    raise ImportError("Cannot import Experiment 6 support module")
protocol = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = protocol
_spec.loader.exec_module(protocol)
ROOT = protocol.ROOT


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def score(y, prediction, train_std: float) -> dict:
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if (y.ndim != 1 or y.shape != prediction.shape or len(y) < 2 or
            not np.isfinite(prediction).all() or not np.isfinite(y).all() or train_std <= 0):
        raise ValueError("Invalid metric inputs")
    squared = (prediction - y) ** 2
    total, variance = squared.sum(), ((y - y.mean()) ** 2).sum()
    mse = float(squared.mean())
    return dict(MSE=mse, RMSE=math.sqrt(mse), MAE=float(abs(prediction-y).mean()),
        R2=float(1 - total / variance) if variance > 0 else float("nan"), NRMSE=math.sqrt(mse)/train_std,
        worst1_sse_share=float(squared.max()/total) if total else 0.,
        worst5_sse_share=float(np.sort(squared)[-5:].sum()/total) if total else 0.)


@torch.inference_mode()
def evaluate(model, data, split, device):
    model.eval()
    part, outputs = data["splits"][split], []
    for xn, xc, _ in DataLoader(part["tensor"], batch_size=2048, shuffle=False):
        prediction = model(xn.to(device), xc.to(device))
        if prediction.shape != (len(xn),):
            raise ValueError("Regressor must return one scalar per row")
        outputs.append(prediction.cpu().numpy().astype(np.float64))
    values = np.concatenate(outputs)*data["target_std"] + data["target_mean"]
    return score(part["actual"], values, data["target_std"])


@torch.inference_mode()
def diagnostics(model, data, device):
    """Training-only norm/rank diagnostics and device-resident forward timing."""
    xn, xc, _ = data["splits"]["train"]["tensor"].tensors
    n, c = xn[:min(512, len(xn))].to(device), xc[:min(512, len(xc))].to(device)
    model.eval()
    raw_q, raw_k, _ = model.project_raw(n[:128], c[:128])
    q, k, _ = model.project(n[:128], c[:128])
    row = {f"{label}_norm_p95": float(torch.quantile(t.norm(dim=-1).float(), .95).cpu())
           for label, t in (("raw_q", raw_q), ("raw_k", raw_k), ("normalized_q", q), ("normalized_k", k))}
    # Small diagnostic-only matrices; never used in the training forward path.
    a = q[:16] @ k[:16].transpose(-1, -2) / math.sqrt(len(data["features"]))
    energy = torch.linalg.svdvals(a.float()).square()
    rank = energy.sum(-1).square() / energy.square().sum(-1).clamp_min(1e-20)
    row["train_attention_effective_rank"] = float(rank.mean().cpu())
    for _ in range(3):
        model(n, c)
    sync(device)
    start = time.perf_counter()
    for _ in range(10):
        model(n, c)
    sync(device)
    row.update(inference_batch_rows=len(n), model_only_us_per_row=1e6*(time.perf_counter()-start)/(10*len(n)))
    return row


def fit_one(data, identity, cfg, device):
    seed_all(identity["init_seed"])
    model = protocol.make_model(identity["family"], identity["d"], data).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loader = DataLoader(data["splits"]["train"]["tensor"], batch_size=cfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(identity["init_seed"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best, best_epoch, best_updates, state = math.inf, 0, 0, None
    updates, curves = 0, []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    start = time.perf_counter()
    for epoch in range(1, cfg["epochs"]+1):
        model.train()
        for xn, xc, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = (model(xn.to(device), xc.to(device))-target.to(device)).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss: {identity['run_id']} epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            updates += 1
        if epoch == 1 or epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
            train, val = evaluate(model, data, "train", device), evaluate(model, data, "val", device)
            objective = val["NRMSE"]**2
            curves.append(dict(**identity, epoch=epoch, updates=updates, train_RMSE=train["RMSE"],
                train_R2=train["R2"], val_RMSE=val["RMSE"], val_R2=val["R2"],
                train_NMSE=train["NRMSE"]**2, val_NMSE=objective,
                gamma=float(model.gamma.detach().cpu())))
            if objective < best-1e-7:
                best, best_epoch, best_updates = objective, epoch, updates
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch == 1 or epoch % 100 == 0:
                print(f"  epoch={epoch} updates={updates} train_RMSE={train['RMSE']:.6g} val_RMSE={val['RMSE']:.6g}", flush=True)
            if epoch-best_epoch >= cfg["patience"]:
                break
    sync(device)
    elapsed = time.perf_counter()-start
    if state is None:
        raise RuntimeError("No finite validation checkpoint")
    model.load_state_dict(state)
    result = dict(**identity, parameters=parameters, best_epoch=best_epoch, best_updates=best_updates,
        epochs_run=epoch, updates=updates, stop_reason="epoch_cap" if epoch == cfg["epochs"] else "early_stopping",
        fit_seconds=elapsed, gamma=float(model.gamma.detach().cpu()))
    for split in ("train", "val") + (("test",) if identity["phase"] == "confirm" else ()):
        result.update({f"{split}_{key}": value for key, value in evaluate(model, data, split, device).items()})
    result.update(diagnostics(model, data, device))
    result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return result, curves


def describe_partition(bundle, data, indices):
    frame = bundle["frame"].iloc[indices["train"]]
    x, y = data["splits"]["train"]["numeric"].astype(float), data["splits"]["train"]["actual"].astype(float)
    cards = data["cardinalities"]
    return dict(train_rows=len(y), val_rows=len(indices["val"]), test_rows=len(indices["test"]),
        missing_fraction_train=float(frame[bundle["features"]].isna().to_numpy().mean()),
        extreme_numeric_fraction_train=float((abs(x)>5).mean()) if x.size else 0.,
        target_std_train=float(data["target_std"]),
        target_skew_train=float(np.mean(((y-y.mean())/data["target_std"])**3)),
        max_category_cardinality=max(cards, default=0), sum_category_cardinalities=sum(cards))


def freeze(store, cfg, dataset):
    selected = protocol.select_widths(store.results, cfg, dataset)
    old = store.manifest["selections"].get(dataset)
    if old is not None and old != selected:
        raise ValueError("Frozen validation selection cannot change")
    store.manifest["selections"][dataset] = selected
    store.save()
    return selected


def execute(cfg, output, phase, device, provenance, loader=None, data_module=None, source_commit=None):
    """Injected local data adapters support network-free integration tests."""
    if loader is None or data_module is None:
        data_module, registry = protocol.backends()
        loader = lambda name: registry.load_task(name, data_root=cfg["data_root"])
    with protocol.Store(output, cfg, provenance) as store:
        if source_commit and "git_commit_created" not in store.manifest:
            store.manifest["git_commit_created"] = source_commit
            store.save()
        for stage in (("search", "confirm") if phase == "all" else (phase,)):
            if stage == "confirm":
                if store.results.empty:
                    raise ValueError("Complete validation search first")
                for name in cfg["datasets"]:
                    freeze(store, cfg, name)
            for name in cfg["datasets"]:
                bundle = loader(name)
                metadata = dict(source_sha256=bundle["source_sha256"], source_slug=bundle["source_slug"],
                    rows=len(bundle["frame"]), features=bundle["features"], numeric=bundle["numeric"],
                    categorical=bundle["categorical"], target=bundle["target"],
                    split_policy="feature_group_disjoint" if name.startswith("wine_") else
                    "host_grouped" if name == "airbnb" else "row_random")
                previous = store.manifest["datasets"].get(name)
                if previous and previous["metadata"] != metadata:
                    raise ValueError(f"Dataset/schema drift for {name}; use a new --output")
                record = store.manifest["datasets"].setdefault(name, dict(metadata=metadata, partitions={}))
                partitions = cfg["search_splits"] if stage == "search" else cfg["confirm_splits"]
                initializations = cfg["search_inits"] if stage == "search" else cfg["confirm_inits"]
                plan = ([(m, d) for m in protocol.FAMILIES for d in cfg["widths"]] if stage == "search"
                        else protocol.confirmation_plan(store.manifest["selections"][name]))
                for partition_seed in partitions:
                    indices = protocol.split_indices(bundle, name, partition_seed, data_module.split_indices)
                    serialized = {s: np.asarray(v, dtype=int).tolist() for s, v in indices.items()}
                    split_hash = protocol.digest(serialized)
                    data = protocol.prepare_data(bundle, indices, data_module.preprocess)
                    description = describe_partition(bundle, data, indices)
                    stored = dict(indices=serialized, sha256=split_hash, **description)
                    key = str(partition_seed)
                    if key in record["partitions"] and record["partitions"][key] != stored:
                        raise ValueError(f"Partition/preprocessing drift for {name}/{key}")
                    record["partitions"][key] = stored
                    store.save()
                    if stage == "search":
                        del data["splits"]["test"]
                    for init in initializations:
                        for family, width in plan:
                            identity = dict(phase=stage, dataset=name, family=family, d=width,
                                            split_seed=partition_seed, init_seed=init)
                            identity["run_id"] = protocol.digest(identity)[:24]
                            if store.completed(identity["run_id"]):
                                continue
                            print(f"{stage} {name} {family} d={width} split={partition_seed} init={init}", flush=True)
                            try:
                                row, curves = fit_one(data, identity, cfg, device)
                                row.update(description, source_sha256=bundle["source_sha256"],
                                    split_indices_sha256=split_hash, rows=len(bundle["frame"]),
                                    features=len(bundle["features"]), numeric_features=len(bundle["numeric"]),
                                    categorical_features=len(bundle["categorical"]))
                                store.commit(row, curves)
                            except Exception as error:
                                store.manifest["errors"][identity["run_id"]] = repr(error)
                                store.save()
                                raise
                            print(f"  saved: val_RMSE={row['val_RMSE']:.6g}" +
                                (f" test_RMSE={row['test_RMSE']:.6g}" if stage == "confirm" else ""), flush=True)
                if stage == "search":
                    selected = freeze(store, cfg, name)
                    print(f"{name}: accuracy widths RN={selected['rn']['accuracy_d']} "
                          f"RNX-GELU2={selected['rnx_gelu2']['accuracy_d']}", flush=True)
            store.flush()
        print(f"Experiment 6 outputs: {output}", flush=True)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("search", "confirm", "all"), default="all")
    p.add_argument("--datasets", nargs="+", default=list(protocol.PANEL))
    p.add_argument("--widths", nargs="+", type=int, default=list(protocol.WIDTHS))
    p.add_argument("--search-splits", nargs="+", type=int, default=[701, 702])
    p.add_argument("--search-inits", nargs="+", type=int, default=[801, 802])
    p.add_argument("--confirm-splits", nargs="+", type=int, default=[703, 704, 705])
    p.add_argument("--confirm-inits", nargs="+", type=int, default=[803, 804])
    p.add_argument("--epochs", type=int, default=650)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--weight-decay", type=float, default=.0001)
    p.add_argument("--patience", type=int, default=120)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--compact-tolerance", type=float, default=.01,
                   help="Secondary compactness statistic; primary selection minimizes validation error")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "experiments/ex06/results")
    p.add_argument("--dry-run", action="store_true", help="Print fit counts without downloads or outputs")
    a = p.parse_args(argv)
    for key in ("datasets", "widths", "search_splits", "search_inits", "confirm_splits", "confirm_inits"):
        values = getattr(a, key)
        if len(set(values)) != len(values):
            p.error(f"Duplicate --{key.replace('_', '-')} values")
    if protocol.REFERENCE_D not in a.widths or min(a.widths+[a.epochs, a.batch_size, a.patience, a.eval_every, a.threads]) <= 0:
        p.error("Positive dimensions/budgets required; include reference d=24")
    if set(a.search_splits)&set(a.confirm_splits) or set(a.search_inits)&set(a.confirm_inits):
        p.error("Search/confirmation seeds must be disjoint within each seed type")
    if any(not 0 <= s < 2**32-10001 for s in a.search_splits+a.confirm_splits+a.search_inits+a.confirm_inits):
        p.error("Seeds must be nonnegative uint32 values with room for the split offset")
    if not math.isfinite(a.lr) or a.lr <= 0 or not math.isfinite(a.weight_decay) or a.weight_decay < 0:
        p.error("Invalid learning rate or weight decay")
    if not 0 <= a.compact_tolerance < .5:
        p.error("Compact tolerance must lie in [0, 0.5)")
    if "all" in a.datasets and a.datasets != ["all"]:
        p.error("Use --datasets all alone")
    return a


def main(argv=None):
    args = arguments(argv)
    if args.dry_run:
        n = 15 if args.datasets == ["all"] else len(args.datasets)
        search = n*len(args.widths)*2*len(args.search_splits)*len(args.search_inits)
        confirm = n*4*len(args.confirm_splits)*len(args.confirm_inits)
        print(f"{n} tasks; {search} search fits; at most {confirm} confirmation fits; four output files.")
        return
    _, registry = protocol.backends()
    names = list(registry.TASKS) if args.datasets == ["all"] else args.datasets
    if set(names)-set(registry.TASKS):
        raise ValueError(f"Unknown task IDs: {set(names)-set(registry.TASKS)}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this PyTorch environment")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = {k: v for k, v in vars(args).items() if k not in ("stage", "dry_run", "output", "device")}
    cfg.update(datasets=names, device=str(device), data_root=str(args.data_root.resolve()) if args.data_root else None)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=False).stdout.strip()
    except OSError:
        commit = "unavailable"
    provenance = dict(source_hashes=protocol.source_hashes(), python=platform.python_version(),
        torch=str(torch.__version__), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
        cuda=torch.version.cuda, hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine())
    print(f"Experiment 6 | {device} | source commit {commit}", flush=True)
    execute(cfg, args.output.resolve(), args.stage, device, provenance, source_commit=commit)


if __name__ == "__main__":
    main()
