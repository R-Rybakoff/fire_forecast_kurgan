import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

DATASET_PATH = "data_processed/ml_dataset_ndvi_weather_features.parquet"

TARGET = "y"
DATE_COL = "date"

TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"


def evaluate(y_true, y_score, name):
    roc = roc_auc_score(y_true, y_score)
    pr = average_precision_score(y_true, y_score)

    print(f"\n{name}")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC : {pr:.6f}")


def get_features(df):
    return [
        c for c in df.columns
        if c.startswith("ndvi_")
        or c.startswith("t2m_")
        or c.startswith("precip")
        or c.startswith("wind_mean")
    ]


def main():
    print("Loading dataset...")
    df = pd.read_parquet(DATASET_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    features = get_features(df)
    print(f"Total features: {len(features)}")

    df = df.dropna(subset=features + [TARGET])

    train_mask = df[DATE_COL] <= TRAIN_END
    test_mask = df[DATE_COL] >= TEST_START

    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, TARGET]
    X_test = df.loc[test_mask, features]
    y_test = df.loc[test_mask, TARGET]

    print("Train size:", len(X_train))
    print("Test  size:", len(X_test))
    print("Positive rate:", y_train.mean())

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        min_child_weight=20,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        objective="binary:logistic",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )

    print("\nTraining XGBoost (stable config)...")
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    evaluate(y_test, y_pred, "XGBoost (NDVI + Weather)")


if __name__ == "__main__":
    main()
