import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =========================
# PATHS
# =========================
DATASET_PATH = "data_processed/ml_dataset_ndvi_weather_features.parquet"

TARGET = "y"
DATE_COL = "date"

TRAIN_END = "2022-12-31"


# =========================
# MAIN
# =========================
def main():
    print("Loading dataset...")
    df = pd.read_parquet(DATASET_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # features
    features = [
        c for c in df.columns
        if c.startswith("ndvi_")
        or c.startswith("t2m_")
        or c.startswith("precip")
        or c.startswith("wind_mean")
    ]

    # keep only train period
    df = df[df[DATE_COL] <= TRAIN_END]
    df = df.dropna(subset=features + [TARGET])

    X = df[features]
    y = df[TARGET]

    print(f"Train size: {len(X)}")
    print(f"Positive rate: {y.mean():.6f}")

    # model (same as final)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    print("Training final model for interpretation...")
    model.fit(X, y)

    # extract coefficients
    coef = model.named_steps["clf"].coef_[0]
    coef_df = pd.DataFrame({
        "feature": features,
        "coef": coef,
        "abs_coef": np.abs(coef)
    }).sort_values("abs_coef", ascending=False)

    print("\nTop 20 features by absolute impact:")
    print(coef_df.head(20))

    # save for report
    coef_df.to_csv(
        "data_processed/logreg_feature_importance.csv",
        index=False
    )
    print("\nSaved: data_processed/logreg_feature_importance.csv")


if __name__ == "__main__":
    main()
