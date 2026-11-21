
"""VANTA EX03 — real-world dataset registry and preprocessing."""

from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
)
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from torch.utils.data import TensorDataset
import torch


REGISTRY = {
    "concrete": {
        "slug": "prathamtripathi/regression-with-neural-networking",
        "file": "concrete_data.csv",
        "description": "Measured concrete compressive strength",
    },
    "california": {
        "slug": "camnugent/california-housing-prices",
        "file": "housing.csv",
        "description": "California census housing values",
    },
    "ames": {
        "slug": "shashanknecrothapa/ames-housing-dataset",
        "file": "AmesHousing.csv",
        "description": "Ames, Iowa residential property sales",
    },
    "airbnb": {
        "slug": "dgomonov/new-york-city-airbnb-open-data",
        "file": "AB_NYC_2019.csv",
        "description": "NYC Airbnb listing prices, 2019",
    },
}


def download_csv(name):
    info = REGISTRY[name]

    root = Path(kagglehub.dataset_download(info["slug"]))

    candidates = [
        p for p in root.rglob("*.csv")
        if p.name.lower() == info["file"].lower()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Expected {info['file']} in {root}. "
            f"Available CSV files: {[p.name for p in root.rglob('*.csv')]}"
        )

    return candidates[0]


def normalized_name(name):
    return "".join(
        ch.lower() for ch in str(name)
        if ch.isalnum()
    )


def find_column(df, expected):
    key = normalized_name(expected)

    for column in df.columns:
        if normalized_name(column) == key:
            return column

    raise KeyError(
        f"Column {expected!r} not found. "
        f"Available columns: {list(df.columns)}"
    )


def load_dataset(name):
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")

    path = download_csv(name)
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()

    df = df.loc[
        :,
        ~df.columns.str.startswith("Unnamed:")
    ].copy()

    if name == "concrete":
        # Original dataset: eight measured input variables
        # followed by compressive strength as the final column.
        if df.shape[1] != 9:
            raise ValueError(
                f"Unexpected Concrete schema: {list(df.columns)}"
            )

        target = df.columns[-1]
        numeric = list(df.columns[:-1])
        categorical = []

    elif name == "california":
        target = find_column(df, "median_house_value")

        categorical = [
            find_column(df, "ocean_proximity")
        ]

        numeric = [
            c for c in df.columns
            if c not in categorical + [target]
        ]

    elif name == "ames":
        target = find_column(df, "SalePrice")

        excluded = {
            "order", "pid", "id",
            "saletype", "salecondition",
        }

        features = [
            c for c in df.columns
            if c != target
            and normalized_name(c) not in excluded
        ]

        categorical = [
            c for c in features
            if (
                not pd.api.types.is_numeric_dtype(df[c])
                or normalized_name(c) == "mssubclass"
            )
        ]

        numeric = [
            c for c in features
            if c not in categorical
        ]

    elif name == "airbnb":
        target = find_column(df, "price")

        numeric = [
            find_column(df, c) for c in [
                "latitude",
                "longitude",
                "minimum_nights",
                "number_of_reviews",
                "reviews_per_month",
                "calculated_host_listings_count",
                "availability_365",
            ]
        ]

        categorical = [
            find_column(df, c) for c in [
                "neighbourhood_group",
                "neighbourhood",
                "room_type",
            ]
        ]

    df[target] = pd.to_numeric(
        df[target], errors="coerce"
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target])

    # Remove invalid/nonpositive prices, not expensive outliers.
    if name in ("airbnb", "ames", "california"):
        df = df[df[target] > 0]

    df = df.reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(f"Dataset {name} is unexpectedly small.")

    for col in numeric:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    print(f"\nDataset: {name}")
    print(f"Source: {path}")
    print(f"Rows: {len(df):,}")
    print(f"Numeric features: {len(numeric)}")
    print(f"Categorical features: {len(categorical)}")
    print(
        "Target range:",
        f"{df[target].min():,.3f}",
        "to",
        f"{df[target].max():,.3f}",
    )

    return {
        "frame": df,
        "target": target,
        "numeric": numeric,
        "categorical": categorical,
        "features": numeric + categorical,
        "source": str(path),
    }


def split_indices(bundle, seed):
    df = bundle["frame"]
    indices = np.arange(len(df))

    if normalized_name(bundle["target"]) == "price" \
            and "host_id" in df.columns:

        # Airbnb: keep all listings from a host in one split.
        groups = df["host_id"].fillna(-1).astype(str)

        first = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=seed,
        )

        train, remainder = next(
            first.split(indices, groups=groups)
        )

        second = GroupShuffleSplit(
            n_splits=1,
            test_size=0.50,
            random_state=seed + 10001,
        )

        val_rel, test_rel = next(
            second.split(
                remainder,
                groups=groups.iloc[remainder],
            )
        )

        val = remainder[val_rel]
        test = remainder[test_rel]

    else:
        train, remainder = train_test_split(
            indices,
            test_size=0.20,
            random_state=seed,
        )

        val, test = train_test_split(
            remainder,
            test_size=0.50,
            random_state=seed,
        )

    return {
        "train": np.asarray(train),
        "val": np.asarray(val),
        "test": np.asarray(test),
    }


def preprocess(bundle, seed):
    df = bundle["frame"]
    numeric = bundle["numeric"]
    categorical = bundle["categorical"]
    target = bundle["target"]

    indices = split_indices(bundle, seed)

    # --------------------------------------------------
    # Train-only numerical transformations
    # --------------------------------------------------

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    scaler = StandardScaler()

    train_num = df.iloc[indices["train"]][numeric]

    scaler.fit(
        imputer.fit_transform(train_num)
    )

    # --------------------------------------------------
    # Train-only categorical vocabulary
    # --------------------------------------------------

    vocabularies = {}

    for column in categorical:
        values = (
            df.iloc[indices["train"]][column]
            .fillna("__MISSING__")
            .astype(str)
        )

        vocabularies[column] = {
            category: i + 1
            for i, category in enumerate(
                sorted(values.unique())
            )
        }

    cardinalities = [
        len(vocabularies[column]) + 1
        for column in categorical
    ]

    # --------------------------------------------------
    # Train-only target normalization
    # --------------------------------------------------

    train_y = df.iloc[indices["train"]][target].to_numpy(
        dtype=np.float64
    )

    target_mean = float(train_y.mean())
    target_std = float(train_y.std(ddof=0))

    if target_std <= 1e-12:
        raise ValueError("Target has near-zero variance.")

    # --------------------------------------------------
    # Transform all splits
    # --------------------------------------------------

    data = {
        "target_mean": target_mean,
        "target_std": target_std,
        "cardinalities": cardinalities,
        "vocabularies": vocabularies,
        "numeric": numeric,
        "categorical": categorical,
        "features": bundle["features"],
        "splits": {},
    }

    for split, idx in indices.items():
        frame = df.iloc[idx]

        x_num = scaler.transform(
            imputer.transform(frame[numeric])
        ).astype(np.float32)

        if categorical:
            x_cat_strings = (
                frame[categorical]
                .fillna("__MISSING__")
                .astype(str)
            )

            x_cat = np.column_stack([
                x_cat_strings[col]
                .map(vocabularies[col])
                .fillna(0)
                .to_numpy(dtype=np.int64)
                for col in categorical
            ])
        else:
            x_cat_strings = pd.DataFrame(
                index=frame.index
            )

            x_cat = np.empty(
                (len(frame), 0),
                dtype=np.int64,
            )

        y_actual = frame[target].to_numpy(
            dtype=np.float64
        )

        y_scaled = (
            (y_actual - target_mean) / target_std
        ).astype(np.float32)

        data["splits"][split] = {
            "indices": idx,
            "numeric": x_num,
            "categorical": x_cat,
            "cat_strings": x_cat_strings.reset_index(
                drop=True
            ),
            "actual": y_actual,
            "tensor": TensorDataset(
                torch.from_numpy(x_num),
                torch.from_numpy(x_cat),
                torch.from_numpy(y_scaled),
            ),
        }

    # --------------------------------------------------
    # Classical / tree-based representations
    # --------------------------------------------------

    if categorical:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float64,
        )

        encoder.fit(
            data["splits"]["train"]["cat_strings"]
        )
    else:
        encoder = None

    for split, item in data["splits"].items():
        numerical_sparse = sparse.csr_matrix(
            item["numeric"].astype(np.float64)
        )

        if encoder is not None:
            categorical_sparse = encoder.transform(
                item["cat_strings"]
            )

            x_classical = sparse.hstack(
                [numerical_sparse, categorical_sparse],
                format="csr",
            )
        else:
            x_classical = numerical_sparse

        item["classical"] = x_classical

        catboost_df = pd.DataFrame({
            col: item["numeric"][:, i]
            for i, col in enumerate(numeric)
        })

        for col in categorical:
            catboost_df[col] = (
                item["cat_strings"][col].to_numpy()
            )

        item["catboost"] = catboost_df[
            bundle["features"]
        ]

    return data

