"""Experiment 6 adapters, validation selection and compact transactional storage."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import FunctionType

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
PANEL = ("diamonds", "ames", "california", "wine_white", "bike_hourly", "concrete")
FAMILIES = ("rn", "rnx_gelu2")
WIDTHS = (4, 8, 16, 24, 32, 48, 64)
REFERENCE_D = 24


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def backends():
    """Import the exact registry/preprocessor, without generic-name collisions."""
    data = load_module(ROOT / "experiments/datasets_ex03/datasets.py", "_ex06_data")
    saved_path, previous = sys.path[:], sys.modules.get("datasets")
    try:
        sys.modules["datasets"] = data
        registry = load_module(ROOT / "experiments/rn_interlude/benchmark_datasets.py", "_ex06_registry")
    finally:
        sys.path[:] = saved_path
        if previous is None:
            sys.modules.pop("datasets", None)
        else:
            sys.modules["datasets"] = previous
    return data, registry


@lru_cache(maxsize=1)
def model_classes():
    rn = load_module(ROOT / "VANTA_RN/model.py", "_ex06_rn")
    rnx = load_module(ROOT / "VANTA_RNX/model.py", "_ex06_rnx")
    return rn.NormalizedFeatureAttentionRegressor, rnx.NonlinearFeatureAttentionRegressor


def make_model(family: str, width: int, data: dict):
    rn, rnx = model_classes()
    numeric, total = len(data["numeric"]), len(data["features"])
    kwargs = dict(num_features=total, numeric_positions=list(range(numeric)),
                  categorical_positions=list(range(numeric, total)),
                  categorical_cardinalities=data["cardinalities"],
                  latent_dim=width, gamma_init=0.1, learnable_gamma=True)
    if family == "rn":
        return rn(**kwargs)
    if family == "rnx_gelu2":
        return rnx(**kwargs, kernel_map="identity", value_map="gelu2")
    raise ValueError(f"Unknown family: {family}")


def split_indices(bundle: dict, name: str, seed: int, original_split):
    """Group identical wine predictors; retain existing protocols elsewhere."""
    if name not in ("wine_red", "wine_white"):
        return original_split(bundle, seed)
    keys = pd.MultiIndex.from_frame(bundle["frame"][bundle["features"]])
    groups = pd.factorize(keys, sort=False)[0]
    if len(np.unique(groups)) < 10:
        raise ValueError("Too few unique predictor groups")
    rows = np.arange(len(groups))
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train, rest = next(outer.split(rows, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 10001)
    val, test = next(inner.split(rest, groups=groups[rest]))
    indices = {"train": train, "val": rest[val], "test": rest[test]}
    sets = [set(groups[indices[k]]) for k in ("train", "val", "test")]
    if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Duplicate predictor groups cross partitions")
    return indices


def prepare_data(bundle: dict, indices: dict, preprocess):
    """Reuse EX03 train-only transformations with private function globals."""
    context = dict(preprocess.__globals__)
    context["split_indices"] = lambda _bundle, _seed: indices
    local = FunctionType(preprocess.__code__, context, preprocess.__name__,
                         preprocess.__defaults__, preprocess.__closure__)
    data = local(bundle, 0)
    for part in data["splits"].values():
        for key in ("classical", "catboost", "cat_strings"):
            part.pop(key, None)
    return data


def digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def source_hashes() -> dict:
    paths = ["experiments/ex06/support.py", "experiments/ex06/run_experiment.py",
             "VANTA_R/model.py", "VANTA_RN/model.py", "VANTA_RNX/model.py",
             "experiments/datasets_ex03/datasets.py",
             "experiments/rn_interlude/benchmark_datasets.py", "uv.lock"]
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def select_widths(results: pd.DataFrame, cfg: dict, dataset: str) -> dict:
    """Accuracy-first selection; compact-within-tolerance choice is secondary."""
    selections = {}
    for family in FAMILIES:
        rows = results[(results.phase == "search") & (results.dataset == dataset)
                       & (results.family == family)]
        expected = {(d, s, i) for d in cfg["widths"] for s in cfg["search_splits"]
                    for i in cfg["search_inits"]}
        actual = list(zip(rows.d.astype(int), rows.split_seed.astype(int), rows.init_seed.astype(int)))
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(f"Incomplete or duplicated search for {dataset}/{family}")
        if not np.isfinite(rows.val_NRMSE.to_numpy(float)).all():
            raise ValueError("Nonfinite validation scores cannot select a width")
        ranking = rows.groupby("d").val_NRMSE.mean().sort_index()
        best = int(ranking.idxmin())
        compact = int(ranking[ranking <= ranking.loc[best] * (1 + cfg["compact_tolerance"])].index.min())
        selections[family] = dict(accuracy_d=best, compact_d=compact,
            validation_NRMSE={str(int(d)): float(v) for d, v in ranking.items()})
    return selections


def confirmation_plan(selection: dict) -> list[tuple[str, int]]:
    rn_d = selection["rn"]["accuracy_d"]
    rnx_d = selection["rnx_gelu2"]["accuracy_d"]
    return sorted({("rn", REFERENCE_D), ("rn", rn_d),
                   ("rnx_gelu2", rn_d), ("rnx_gelu2", rnx_d)})


def atomic(path: Path, write):
    temporary = path.with_name(path.name + ".tmp")
    try:
        write(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_write(path: Path, value):
    atomic(path, lambda p: p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"))


class Store:
    """Four files; results.csv marks a fit complete only after its curves exist."""
    def __init__(self, root: Path, config: dict, provenance: dict):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(root, os.O_RDONLY)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError(f"Another Experiment 6 process is writing {root}") from None
        self.path = root / "manifest.json"
        try:
            if self.path.exists():
                self.manifest = json.loads(self.path.read_text())
                if self.manifest["config"] != config or self.manifest["provenance"] != provenance:
                    raise ValueError("Study settings, sources or environment changed; use a new --output")
            else:
                if any(root.iterdir()):
                    raise ValueError("New study requires an empty output directory")
                self.manifest = dict(schema_version=1, config=config, provenance=provenance,
                                     datasets={}, selections={}, errors={})
                self.save()
            self.results = self.read("results.csv", ["run_id", "phase", "dataset", "family", "d", "split_seed", "init_seed"])
            self.curves = self.read("learning_curves.csv", ["run_id", "epoch", "updates", "train_RMSE", "val_RMSE"])
            if self.results.run_id.duplicated().any():
                raise ValueError("Duplicate run IDs in results.csv")
            self.curves = self.curves[self.curves.run_id.isin(self.results.run_id)]
            self.flush()
        except Exception:
            self.close()
            raise

    def read(self, filename, columns):
        path = self.root / filename
        return pd.read_csv(path, float_precision="round_trip") if path.exists() else pd.DataFrame(columns=columns)

    def save(self):
        json_write(self.path, self.manifest)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def completed(self, run_id: str) -> bool:
        return run_id in set(self.results.run_id)

    def flush(self):
        atomic(self.root / "learning_curves.csv", lambda p: self.curves.to_csv(p, index=False))
        atomic(self.root / "results.csv", lambda p: self.results.to_csv(p, index=False))
        summary = build_summary(self.results, self.manifest["selections"])
        atomic(self.root / "summary.csv", lambda p: summary.to_csv(p, index=False))
        self.save()

    def commit(self, result: dict, curves: list[dict]):
        key = result["run_id"]
        if self.completed(key):
            raise ValueError(f"Already completed: {key}")
        def merge(old, new):
            retained = old.loc[old.run_id != key]
            added = pd.DataFrame(new)
            return added if retained.empty else pd.concat([retained, added], ignore_index=True)
        self.curves = merge(self.curves, curves)
        self.results = merge(self.results, [result])
        self.manifest["errors"].pop(key, None)
        self.flush()


def build_summary(results: pd.DataFrame, selections: dict) -> pd.DataFrame:
    """Pooled descriptive SD and SD across partition means are distinct columns."""
    rows = []
    if results.empty:
        return pd.DataFrame(columns=["record_type", "dataset"])
    metrics = ("train_RMSE", "val_NRMSE", "test_RMSE", "test_MAE", "test_R2", "test_NRMSE",
               "test_worst1_sse_share", "test_worst5_sse_share", "parameters", "fit_seconds",
               "model_only_us_per_row", "raw_q_norm_p95", "raw_k_norm_p95")
    for (phase, dataset, family, width), group in results.groupby(["phase", "dataset", "family", "d"]):
        row = dict(record_type="model", phase=phase, dataset=dataset, family=family,
                   d=int(width), n_runs=len(group), n_splits=group.split_seed.nunique())
        for metric in metrics:
            if metric in group and group[metric].notna().any():
                means = group.groupby("split_seed")[metric].mean()
                row[metric + "_mean"] = group[metric].mean()
                row[metric + "_sd_pooled"] = group[metric].std(ddof=1)
                row[metric + "_sd_split_means"] = means.std(ddof=1)
        rows.append(row)
    for dataset, chosen in selections.items():
        rn, rnx = chosen["rn"]["accuracy_d"], chosen["rnx_gelu2"]["accuracy_d"]
        contrasts = (("A_width", ("rn", 24), ("rn", rn)),
                     ("B_fixed_width", ("rn", rn), ("rnx_gelu2", rn)),
                     ("B_tuned_families", ("rn", rn), ("rnx_gelu2", rnx)))
        test = results[(results.phase == "confirm") & (results.dataset == dataset)]
        for label, reference, candidate in contrasts:
            def subset(pair):
                return test[(test.family == pair[0]) & (test.d == pair[1])]
            paired = subset(reference).merge(subset(candidate), on=["split_seed", "init_seed"],
                                               suffixes=("_ref", "_candidate"), validate="one_to_one")
            if paired.empty:
                continue
            deltas = paired[["split_seed", "init_seed"]].copy()
            for metric in ("RMSE", "MAE", "R2", "NRMSE"):
                deltas[metric + "_delta"] = paired[f"test_{metric}_candidate"] - paired[f"test_{metric}_ref"]
            deltas["RMSE_percent_delta"] = 100 * deltas.RMSE_delta / paired.test_RMSE_ref
            row = dict(record_type="paired", phase="confirm", dataset=dataset, contrast=label,
                       reference=f"{reference[0]}/d{reference[1]}",
                       candidate=f"{candidate[0]}/d{candidate[1]}", n_runs=len(paired),
                       n_splits=paired.split_seed.nunique(),
                       wins=int((deltas.RMSE_delta < 0).sum()), ties=int((deltas.RMSE_delta == 0).sum()),
                       split_wins=int((deltas.groupby("split_seed").RMSE_delta.mean() < 0).sum()))
            for metric in deltas.columns[2:]:
                row[metric + "_mean"] = deltas[metric].mean()
                row[metric + "_sd_split_means"] = deltas.groupby("split_seed")[metric].mean().std(ddof=1)
            rows.append(row)
            for split, sub in deltas.groupby("split_seed"):
                rows.append(dict(record_type="paired_split", dataset=dataset, contrast=label,
                    split_seed=int(split), n_runs=len(sub), **{m: sub[m].mean() for m in deltas.columns[2:]}))
    return pd.DataFrame(rows)
