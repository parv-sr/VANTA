"""Benchmark dataset registry: 15 heterogeneous Kaggle tabular regression tasks.

All models use the same EX03 train-only preprocessing and split indices.
No target-derived columns or direct row IDs are included as predictors.
An optional local --data-root avoids repeated downloads on CAI.
"""
from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

import kagglehub
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EX03 = ROOT / "experiments" / "datasets_ex03"
sys.path.insert(0, str(EX03))
import datasets as ex03


@dataclass(frozen=True)
class Task:
    slug: str
    filename: str
    target: str
    domain: str
    characteristic: str
    categories: tuple[str, ...] = ()
    drop: tuple[str, ...] = ()
    separator: str = ","
    sheet: str | int = 0
    notes: str = ""


# Names are stable benchmark task IDs. The four EX03 tasks are special-cased below
# to retain their established datasets and preprocessing rules exactly.
TASKS: dict[str, Task] = {
    "concrete": Task("prathamtripathi/regression-with-neural-networking",
        "concrete_data.csv", "__last__", "materials", "small, all-numeric"),
    "california": Task("camnugent/california-housing-prices",
        "housing.csv", "median_house_value", "housing", "geographic, one categorical"),
    "ames": Task("shashanknecrothapa/ames-housing-dataset",
        "AmesHousing.csv", "SalePrice", "housing", "many categorical columns, extreme values"),
    "airbnb": Task("dgomonov/new-york-city-airbnb-open-data",
        "AB_NYC_2019.csv", "price", "short-term rentals",
        "heavy-tailed, host-grouped splits"),
    "diamonds": Task("shivam2503/diamonds", "diamonds.csv", "price",
        "retail", "large, mixed, geometric interactions",
        categories=("cut","color","clarity"),
        drop=("Unnamed: 0",), notes="Filters impossible x/y/z dimensions."),
    "wine_red": Task("uciml/red-wine-quality-cortez-et-al-2009",
        "winequality-red.csv", "quality", "chemistry",
        "small, all-numeric, ordinal-valued regression", separator=";"),
    "wine_white": Task("kashnitsky/mlcourse-ai",
        "winequality-white.csv", "quality", "chemistry",
        "moderate, all-numeric, ordinal-valued regression", separator=";",
        notes="Kaggle mlcourse.ai mirror is a relatively large download."),
    "insurance": Task("mirichoi0218/insurance",
        "insurance.csv", "charges", "insurance",
        "small, nonlinear categorical effects",
        categories=("sex","smoker","region")),
    "energy": Task("elikplim/eergy-efficiency-dataset",
        "ENB2012_data.csv", "Y1", "building energy",
        "small, physical load", drop=("Y2",),
        notes="Y2 cooling load excluded; benchmark target is Y1 heating load."),
    "bike_hourly": Task("lakshmi25npathi/bike-sharing-dataset",
        "hour.csv", "cnt", "mobility",
        "hourly count, strong seasonality",
        categories=("season","yr","mnth","hr","holiday","weekday","workingday","weathersit"),
        drop=("instant","dteday","casual","registered"),
        notes="Random-split interpolation, NOT future forecasting. casual+registered=cnt are leaked components."),
    "avocado": Task("neuromusic/avocado-prices",
        "avocado.csv", "AveragePrice", "agriculture/retail",
        "longitudinal market, region/type categories",
        categories=("type","region"), drop=("Unnamed: 0","Date"),
        notes="Random-split price estimation, NOT temporal extrapolation."),
    "king_county": Task("harlfoxem/housesalesprediction",
        "kc_house_data.csv", "price", "housing",
        "large, geographic, nonlinear size/age",
        categories=("zipcode",), drop=("id","date")),
    "auto_mpg": Task("uciml/autompg-dataset",
        "auto-mpg.csv", "mpg", "automotive",
        "very small, numeric and categorical",
        categories=("origin","cylinders"),
        drop=("car name","car_name","name")),
    "graduate": Task("mohansacharya/graduate-admissions",
        "Admission_Predict_Ver1.1.csv", "Chance of Admit",
        "education", "small, mostly numeric",
        drop=("Serial No.","Serial No"), notes="Admission chance is a bounded target."),
    "student_math": Task("dskagglemt/student-performance-data-set",
        "student-mat.csv", "G3", "education",
        "very small, many categorical columns",
        drop=("G1","G2"), separator=";",
        notes="Prior grades G1/G2 excluded to prevent near-target proxy dominance."),
}
assert len(TASKS) == 15


def _find_file(name: str, task: Task, data_root: Path | None):
    if data_root is not None:
        root = Path(data_root).expanduser().resolve()
        matches = [root / name / task.filename, root / task.filename]
        matches += list((root / name).rglob(task.filename)) if (root / name).exists() else []
        for p in matches:
            if p.is_file():
                return p
        raise FileNotFoundError(f"{name}: expected {task.filename} under {root} or {root/name}; "
                                "omit --data-root to use KaggleHub downloads")
    try:
        root = Path(kagglehub.dataset_download(task.slug))
    except Exception as exc:
        if name != "wine_white":
            raise
        # The mlcourse-ai Kaggle mirror may be inaccessible (HTTP 403).
        # Use the original UCI Wine Quality release as an explicit, documented
        # fallback; preserve its source URL in the returned bundle.
        root = Path.home() / ".cache" / "vanta" / "datasets" / "wine_white"
        root.mkdir(parents=True, exist_ok=True)
        target = root / "winequality-white.csv"
        if not target.is_file():
            url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                   "wine-quality/winequality-white.csv")
            tmp = root / "winequality-white.csv.tmp"
            try:
                urllib.request.urlretrieve(url, tmp)
                os.replace(tmp, target)
            except Exception as fallback_exc:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    "White Wine Kaggle source unavailable, and official UCI "
                    "fallback download failed. Check server internet access, "
                    "or place winequality-white.csv under "
                    "--data-root/wine_white/.") from fallback_exc
        return target
    candidates = list(root.rglob(task.filename))
    if len(candidates) != 1:
        # Energy Kaggle mirrors sometimes package the same data as XLSX.
        if name == "energy":
            candidates = sorted([*root.rglob("*ENB*.xlsx"), *root.rglob("*ENB*.csv"),
                                 *root.rglob("*energy*.csv")])
        if len(candidates) != 1:
            raise FileNotFoundError(f"{name}: expected {task.filename} in {root}. "
                f"Available: {[p.name for p in root.rglob('*') if p.is_file()][:40]}")
    return candidates[0]


def _read_csv(path: Path, task: Task):
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=task.sheet)
    else:
        # Kaggle mirrors sometimes convert UCI's semicolon-delimited wine CSV
        # to comma-delimited CSV. Auto-detect rather than relying on filename.
        df = pd.read_csv(path, sep=None, engine="python")
        if len(df.columns) == 1 and ("," in str(df.columns[0]) or ";" in str(df.columns[0])):
            raise ValueError(f"CSV delimiter detection failed for {path}")
    df.columns = df.columns.astype(str).str.strip()
    return df


def load_task(name: str, *, data_root: Path | None = None) -> dict:
    if name not in TASKS:
        raise ValueError(f"Unknown benchmark task {name!r}")
    task = TASKS[name]
    # Keep exact existing EX03 task semantics, including Airbnb host-group split.
    if name in ex03.REGISTRY:
        # Always use the original EX03 schema, even when --data-root is used.
        if data_root is None:
            bundle = ex03.load_dataset(name)
            source = Path(bundle["source"])
        else:
            source = _find_file(name, task, data_root)
            original_downloader = ex03.download_csv
            try:
                ex03.download_csv = lambda _: source
                bundle = ex03.load_dataset(name)
            finally:
                ex03.download_csv = original_downloader
    else:
        source = _find_file(name, task, data_root)
        df = _read_csv(source, task)
        if name == "concrete":
            if len(df.columns) != 9:
                raise ValueError("Concrete should have exactly eight inputs and one target")
            target = df.columns[-1]
        else:
            wanted = ex03.normalized_name(task.target)
            candidates = [col for col in df if ex03.normalized_name(col) == wanted]
            if len(candidates) != 1:
                raise ValueError(f"{name}: target {task.target!r} missing/ambiguous. Columns: {list(df)}")
            target = candidates[0]
        # No duplicate columns or silently-selected alternate targets.
        if df.columns.duplicated().any():
            raise ValueError(f"{name}: duplicate feature columns")
        remove = {ex03.normalized_name(x) for x in task.drop}
        remove |= {ex03.normalized_name(x) for x in ("Unnamed: 0",)}
        features = [x for x in df.columns if x != target
                    and ex03.normalized_name(x) not in remove
                    and not x.startswith("Unnamed:")]
        if name == "diamonds":
            for x in ("x","y","z"):
                df[x] = pd.to_numeric(df[x], errors="coerce")
            df = df.loc[df[["x","y","z"]].gt(0).all(axis=1) &
                        df[["x","y","z"]].lt(15).all(axis=1)].copy()
        if name == "auto_mpg":
            # Kaggle's '?' horsepower missing-value tokens are numeric missingness,
            # not a categorical level.
            if "horsepower" not in df.columns:
                raise ValueError("Auto MPG horsepower field missing")
            df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
        categorical_names = {ex03.normalized_name(c) for c in task.categories}
        categorical = [f for f in features
            if ex03.normalized_name(f) in categorical_names
            or not pd.api.types.is_numeric_dtype(df[f])]
        if name == "energy":
            # Y2 is a second target, never a predictor.
            if any(ex03.normalized_name(f) == "y2" for f in features):
                raise RuntimeError("Energy target leakage: Y2 was retained")
        if name == "bike_hourly":
            if any(ex03.normalized_name(f) in ("casual","registered") for f in features):
                raise RuntimeError("Bike-sharing target component leaked")
        numeric = [f for f in features if f not in categorical]
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
        for f in numeric:
            df[f] = pd.to_numeric(df[f], errors="coerce")
        df = df.reset_index(drop=True)
        if len(df) < 100 or not len(features):
            raise ValueError(f"{name}: invalid row/feature count {len(df)}, {len(features)}")
        bundle = dict(frame=df, target=target, numeric=numeric,
                      categorical=categorical, features=numeric+categorical,
                      source=str(source))
    bundle["task_id"] = name
    bundle["task_domain"] = task.domain
    bundle["task_characteristic"] = task.characteristic
    bundle["source_sha256"] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    bundle["source_slug"] = (
        "UCI Wine Quality official repository (fallback)"
        if name == "wine_white" and ".cache/vanta/datasets/wine_white" in str(source)
        else task.slug
    )
    return bundle


def profile_train(data: dict, bundle: dict, seed: int) -> dict:
    """Train-only descriptors, suitable for comparing dataset regimes."""
    x = data["splits"]["train"]["numeric"].astype(float)
    y = data["splits"]["train"]["actual"].astype(float)
    frame = bundle["frame"].iloc[data["splits"]["train"]["indices"]]
    xs = x[np.isfinite(x)]
    missing = (float(frame[bundle["features"]].isna().to_numpy().mean())
               if bundle["features"] else float("nan"))
    centered = y-y.mean()
    sd = max(float(y.std()), 1e-12)
    skew = float(np.mean((centered/sd)**3))
    cards = data["cardinalities"]
    return dict(seed=seed, rows=len(bundle["frame"]), train_rows=len(y),
                features=len(bundle["features"]), numeric_features=len(bundle["numeric"]),
                categorical_features=len(bundle["categorical"]),
                categorical_fraction=len(bundle["categorical"])/len(bundle["features"]),
                median_cat_cardinality=float(np.median(cards)) if cards else 0,
                max_cat_cardinality=int(max(cards)) if cards else 0,
                missing_fraction_train=missing,
                extreme_numeric_fraction_train=float(np.mean(np.abs(xs)>5)) if xs.size else 0.,
                target_skew_train=skew, target_std_train=float(data["target_std"]),
                zero_target_fraction_train=float(np.mean(y==0)),
                domain=bundle["task_domain"], characteristic=bundle["task_characteristic"],
                source_sha256=bundle["source_sha256"])
