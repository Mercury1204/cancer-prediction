"""
data_loader.py — Data Ingestion, Cleaning, and Stratified Splitting

Loads the Wisconsin Diagnostic Breast Cancer (WDBC) dataset,
removes non-predictive columns, encodes the target variable,
and provides a reproducible stratified train-test split.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_URL = "https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv"


def load_and_clean(url: str = DATA_URL) -> pd.DataFrame:
    """Fetch the WDBC CSV and strip non-predictive columns."""
    df = pd.read_csv(url)
    cols_to_drop = [col for col in ["id", "Unnamed: 32"] if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df


def encode_target(df: pd.DataFrame, column: str = "diagnosis"):
    """Encode diagnosis labels: M (Malignant) -> 1, B (Benign) -> 0."""
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df, le


def get_splits(
    df: pd.DataFrame,
    target: str = "diagnosis",
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Return stratified 80:20 train-test partitions."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


if __name__ == "__main__":
    raw = load_and_clean()
    print(f"Loaded {raw.shape[0]} records with {raw.shape[1]} columns.")
    encoded, _ = encode_target(raw)
    X_train, X_test, y_train, y_test = get_splits(encoded)
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
