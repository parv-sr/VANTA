"""Descriptive dataset and split-leakage audit for VANTA's wine benchmarks.

No fitting or model selection occurs in this script.  Statistics used to
characterize a split are computed from its training rows, not from its test
targets.  Duplicate-group overlap is reported for the historical random-row
splits as well as for the proposed feature-disjoint splits.

From the repository root:
  uv run python experiments/rn_interlude/wine_audit.py
  uv run python experiments/rn_interlude/wine_audit.py --datasets wine_red wine_white
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "experiments" / "datasets_ex03"), str(HERE)]
import datasets as ex03
from benchmark_datasets import TASKS, load_task


def feature_groups(bundle: dict) -> np.ndarray:
    """Stable groups by complete raw feature vector; target is not included."""
    features = bundle["features"]
    if not features:
        raise ValueError("No input features")
    frame = bundle["frame"][features]
    # Hash exact parsed feature values, including missingness. Independent of
    # target and row index; grouping is intentional for duplicated predictors.
    raw_hash = pd.util.hash_pandas_object(
        frame.astype(object).where(frame.notna(), "__MISSING__"), index=False
    ).to_numpy(dtype="uint64", copy=False)
    _, labels = np.unique(raw_hash, return_inverse=True)
    return labels.astype(np.int64)


def make_group_split(bundle: dict, seed: int) -> dict[str, np.ndarray]:
    """Train/validation/test split with no repeated feature vector across sets."""
    groups = feature_groups(bundle)
    if len(np.unique(groups)) < 10:
        raise ValueError("Too few distinct feature vectors for a group split")
    indices = np.arange(len(groups))
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, rest_rel = next(outer.split(indices, groups=groups))
    rest = indices[rest_rel]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed+10001)
    val_rel, test_rel = next(inner.split(rest, groups=groups[rest]))
    out = {"train": train_idx, "val": rest[val_rel], "test": rest[test_rel]}
    sets = {key: set(groups[value]) for key, value in out.items()}
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])
    return out


def index_sha256(indices: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for split in ("train", "val", "test"):
        a = np.asarray(indices[split], dtype="<i8")
        h.update(split.encode("ascii"))
        h.update(len(a).to_bytes(8, "little"))
        h.update(a.tobytes())
    return h.hexdigest()


def overlap_summary(indices: dict, groups: np.ndarray, *, dataset: str,
                    split_seed: int, split_method: str) -> dict:
    tr, val, te = (np.asarray(indices[s], dtype=int)
                   for s in ("train","val","test"))
    train_groups, val_groups, test_groups = (set(groups[x]) for x in (tr,val,te))
    def share(candidate, reference):
        return float(np.mean(np.isin(groups[candidate], list(reference)))) if len(candidate) else 0.
    return dict(dataset=dataset, split_method=split_method, split_seed=split_seed,
        train_rows=len(tr), val_rows=len(val), test_rows=len(te),
        val_rows_with_duplicate_in_train=share(val,train_groups),
        test_rows_with_duplicate_in_train=share(te,train_groups),
        test_rows_with_duplicate_in_val=share(te,val_groups),
        split_indices_sha256=index_sha256(indices),
        num_train_groups=len(train_groups),
        num_test_groups=len(test_groups))


def training_descriptors(bundle: dict, indices: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Descriptives for the TRAIN partition; does not inspect validation/test targets."""
    frame = bundle["frame"].iloc[np.asarray(indices["train"], dtype=int)]
    numeric = bundle["numeric"]
    target = bundle["target"]
    y = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    if numeric:
        xx = frame[numeric].apply(pd.to_numeric, errors="coerce")
        x = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(xx)
        z = StandardScaler().fit_transform(x)
        pair = pd.DataFrame(z, columns=numeric).corr()
        raw = []
        for i,col in enumerate(numeric):
            p = xx[col].to_numpy(dtype=float)
            finite = p[np.isfinite(p)]
            raw.append(dict(feature=col,missing_rate=float(np.mean(~np.isfinite(p))),
                std=float(np.std(finite)) if len(finite) else float("nan"),
                skew=float(pd.Series(finite).skew()) if len(finite)>2 else float("nan"),
                standardized_abs_gt5=float(np.mean(abs(z[:,i])>5))))
        features = pd.DataFrame(raw)
        pairs = []
        for i in range(len(numeric)):
            for j in range(i+1,len(numeric)):
                c = float(pair.iloc[i,j])
                if np.isfinite(c):
                    pairs.append(dict(feature_1=numeric[i], feature_2=numeric[j],
                                      pearson=c, absolute_pearson=abs(c)))
        correlations = pd.DataFrame(pairs, columns=["feature_1","feature_2","pearson","absolute_pearson"])
        if not correlations.empty:
            correlations = correlations.sort_values("absolute_pearson", ascending=False)
        spectrum = np.linalg.svd(z, full_matrices=False, compute_uv=False)**2
        prop = spectrum/spectrum.sum() if spectrum.sum() else spectrum
        effective_pca_rank = float(1/np.sum(prop**2)) if prop.sum() else 0.
        n_components_90 = int(np.searchsorted(prop.cumsum(),.90)+1) if prop.sum() else 0
    else:
        features=pd.DataFrame()
        correlations=pd.DataFrame()
        effective_pca_rank=0.
        n_components_90=0
    y_counts = pd.Series(y).value_counts()
    desc = dict(train_rows=len(frame),num_numeric=len(numeric),
       num_categorical=len(bundle["categorical"]),
       target_unique_values=int(len(y_counts)),
       target_most_common_fraction=float(y_counts.iloc[0]/len(y)),
       target_skew=float(pd.Series(y).skew()),
       pca_effective_rank=effective_pca_rank,
       pca_components_explaining_90pct=n_components_90,
       max_abs_pairwise_pearson=(float(correlations.absolute_pearson.max())
                                 if len(correlations) else None))
    return desc,features,correlations


def audit_task(name: str, seeds: list[int], out: Path, data_root: Path | None = None):
    bundle=load_task(name,data_root=data_root)
    frame=bundle["frame"]
    groups=feature_groups(bundle)
    y=frame[bundle["target"]].to_numpy()
    # Repeated feature vectors may have identical OR contradictory targets.
    group_sizes=np.bincount(groups)
    duplicated=np.isin(groups,np.where(group_sizes>1)[0])
    same_target=pd.DataFrame({"group":groups,"target":y}).groupby("group")["target"].nunique()
    conflicting_groups=set(same_target[same_target>1].index)
    summary=dict(dataset=name,rows=len(frame),features=len(bundle["features"]),
        numeric_features=len(bundle["numeric"]),
        categorical_features=len(bundle["categorical"]),
        source=bundle["source"],source_slug=bundle["source_slug"],
        source_sha256=bundle["source_sha256"],
        unique_feature_vectors=len(group_sizes),
        duplicate_feature_row_fraction=float(np.mean(duplicated)),
        extra_repeated_feature_rows=int(len(groups)-len(group_sizes)),
        groups_with_conflicting_targets=len(conflicting_groups),
        rows_in_conflicting_groups=int(np.isin(groups,list(conflicting_groups)).sum()))
    path=out/name
    path.mkdir(parents=True,exist_ok=True)
    overlaps=[]
    all_train=[]
    target_distributions=[]
    for seed in seeds:
        for method in ("row_random","feature_group_disjoint"):
            indices=(ex03.split_indices(bundle,seed) if method=="row_random"
                     else make_group_split(bundle,seed))
            overlaps.append(overlap_summary(indices,groups,dataset=name,
                split_seed=seed,split_method=method))
            desc, features, correlations = training_descriptors(bundle,indices)
            all_train.append(dict(dataset=name,split_seed=seed,
                                  split_method=method, **desc))
            if seed == seeds[0]:
                features.to_csv(path/f"{method}_train_feature_stats.csv",index=False)
                correlations.to_csv(path/f"{method}_train_correlations.csv",index=False)
                pd.Series(frame.iloc[indices["train"]][bundle["target"]]).value_counts().sort_index().to_csv(
                    path/f"{method}_train_target_counts.csv",header=["count"])
    pd.DataFrame(overlaps).to_csv(path/"split_overlap.csv",index=False)
    pd.DataFrame(all_train).to_csv(path/"training_descriptors.csv",index=False)
    (path/"dataset_summary.json").write_text(json.dumps(summary,indent=2))
    print(f"{name}: {len(frame)} rows; {len(group_sizes)} unique feature vectors; "
        f"{summary['duplicate_feature_row_fraction']:.1%} in repeated groups; "
        f"{len(conflicting_groups)} groups with non-identical targets")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets",nargs="+",choices=list(TASKS),default=["wine_red","wine_white"])
    parser.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    parser.add_argument("--data-root",type=Path,default=None)
    parser.add_argument("--output",type=Path,default=HERE/"results"/"wine_audit")
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds):
        parser.error("Repeated split seeds")
    args.output.mkdir(parents=True,exist_ok=True)
    summaries=[audit_task(name,args.seeds,args.output,args.data_root) for name in args.datasets]
    pd.DataFrame(summaries).to_csv(args.output/"dataset_summary.csv",index=False)
    print(f"Audit outputs: {args.output}")


if __name__ == "__main__":
    main()
