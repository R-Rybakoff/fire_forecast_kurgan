import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =========================
# PATHS
# =========================
BASELINE_PATH = "data_processed/ml_dataset.parquet"
FULL_PATH = "data_processed/ml_dataset_ndvi_weather_features.parquet"

TARGET = "y"
DATE_COL = "date"

TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"


# =========================
# UTILS
# =========================
def evaluate(y_true, y_score, name):
    roc = roc_auc_score(y_true, y_score)
    pr = average_precision_score(y_true, y_score)

    print(f"\n{name}")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC : {pr:.6f}")

    return roc, pr


def get_feature_columns(df, mode):
    if mode == "baseline":
        return [c for c in df.columns if c.startswith("ndvi_")]

    if mode == "full":
        return [
            c for c in df.columns
            if c.startswith("ndvi_")
            or c.startswith("t2m_")
            or c.startswith("precip")
            or c.startswith("wind_mean")
        ]

    raise ValueError("Unknown mode")


# =========================
# MAIN
# =========================
def main():
    print("Loading datasets...")
    df_base = pd.read_parquet(BASELINE_PATH)
    df_full = pd.read_parquet(FULL_PATH)

    for df in (df_base, df_full):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # temporal split
    train_mask_base = df_base[DATE_COL] <= TRAIN_END
    test_mask_base = df_base[DATE_COL] >= TEST_START

    train_mask_full = df_full[DATE_COL] <= TRAIN_END
    test_mask_full = df_full[DATE_COL] >= TEST_START

    # feature sets
    base_features = get_feature_columns(df_base, "baseline")
    full_features = get_feature_columns(df_full, "full")

    print(f"Baseline features: {len(base_features)}")
    print(f"Full features    : {len(full_features)}")

    # drop rows with NaN ONLY in used columns
    df_base = df_base.dropna(subset=base_features + [TARGET])
    df_full = df_full.dropna(subset=full_features + [TARGET])

    # train / test
    Xb_train = df_base.loc[train_mask_base, base_features]
    yb_train = df_base.loc[train_mask_base, TARGET]
    Xb_test = df_base.loc[test_mask_base, base_features]
    yb_test = df_base.loc[test_mask_base, TARGET]

    Xf_train = df_full.loc[train_mask_full, full_features]
    yf_train = df_full.loc[train_mask_full, TARGET]
    Xf_test = df_full.loc[test_mask_full, full_features]
    yf_test = df_full.loc[test_mask_full, TARGET]

    print("Train size (baseline):", len(Xb_train))
    print("Test  size (baseline):", len(Xb_test))
    print("Train size (full)    :", len(Xf_train))
    print("Test  size (full)    :", len(Xf_test))

    # model
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    # =========================
    # BASELINE
    # =========================
    print("\nTraining NDVI-only model...")
    model.fit(Xb_train, yb_train)
    yb_pred = model.predict_proba(Xb_test)[:, 1]
    evaluate(yb_test, yb_pred, "NDVI-only")

    # =========================
    # FULL
    # =========================
    print("\nTraining NDVI + Weather model...")
    model.fit(Xf_train, yf_train)
    yf_pred = model.predict_proba(Xf_test)[:, 1]
    evaluate(yf_test, yf_pred, "NDVI + Weather")


if __name__ == "__main__":
    main()
