#!/usr/bin/env python3
"""Data loading, preprocessing, and utility functions for EduSynX."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

import config


def get_data_files(data_dir: Path) -> list:
    """Return list of generated CSV files."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("student_chunk_*.csv"))
    if not files:
        single = data_dir / "student_data.csv"
        if single.exists():
            files = [single]
    return files


def read_chunks(data_dir: Path, chunksize: int = 100_000):
    """Generator yielding DataFrames from all CSV chunks."""
    for file in get_data_files(data_dir):
        for chunk in pd.read_csv(file, chunksize=chunksize):
            yield chunk


def load_sample(data_dir: Path, max_rows: int = 200_000, seed: int = 42) -> pd.DataFrame:
    """Load a random sample of at most max_rows rows from chunked CSVs without loading everything."""
    files = get_data_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    rng = np.random.default_rng(seed)
    rng.shuffle(files)

    sampled = []
    remaining = max_rows

    for file in files:
        if remaining <= 0:
            break
        chunk_size = min(100_000, remaining)
        reader = pd.read_csv(file, chunksize=chunk_size)
        for chunk in reader:
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.sample(n=remaining, random_state=seed)
            sampled.append(chunk)
            remaining -= len(chunk)

    if not sampled:
        return pd.DataFrame(columns=config.ALL_COLUMNS)

    return pd.concat(sampled, ignore_index=True)


def create_preprocessor() -> ColumnTransformer:
    """Create a ColumnTransformer for numeric scaling and categorical encoding."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, config.NUMERIC_COLS),
            ("cat", categorical_transformer, config.CATEGORICAL_COLS),
        ]
    )
    return preprocessor


def encode_target(y: pd.Series) -> tuple:
    """Encode target labels and return encoded array and fitted LabelEncoder."""
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(y)
    return encoded, label_encoder


def save_object(obj, path: Path) -> None:
    """Save a Python object using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_object(path: Path):
    """Load a Python object using joblib."""
    return joblib.load(Path(path))