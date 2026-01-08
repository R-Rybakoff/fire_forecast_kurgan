import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =========================
# PATHS
# =========================
DATASET_PATH = "data_processed/ml_dataset_ndvi_weather_features.parquet"
OUT_FIG = "data_processed/logreg_top10_features.png"

TARGET = "y"
DATE_COL = "date"
TRAIN_END = "2022-12-31"


def main():
    print("Loading dataset...")
    df = pd.read_parquet(DATASET_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # feature selection
    features = [
        c for c in df.columns
        if c.startswith("ndvi_")
        or c.startswith("t2m_")
        or c.startswith("precip")
        or c.startswith("wind_mean")
    ]

    # train only (no leakage)
    df = df[df[DATE_COL] <= TRAIN_END]
    df = df.dropna(subset=features + [TARGET])

    X = df[features]
    y = df[TARGET]

    print(f"Train size: {len(X)}")

    # model (same as final)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    print("Training model for visualization...")
    model.fit(X, y)

    coef = model.named_steps["clf"].coef_[0]

    coef_df = pd.DataFrame({
        "feature": features,
        "coef": coef,
        "abs_coef": np.abs(coef)
    }).sort_values("abs_coef", ascending=False).head(10)

    coef_df = coef_df.sort_values("coef")

    # =========================
    # PLOT
    # =========================
    plt.figure(figsize=(8, 5))
    plt.barh(coef_df["feature"], coef_df["coef"])
    plt.axvline(0)
    plt.title("Top-10 признаков пожарной опасности\n(Logistic Regression, NDVI + Weather)")
    plt.xlabel("Влияние на лог-вероятность пожара")
    plt.tight_layout()

    plt.savefig(OUT_FIG, dpi=150)
    print(f"Saved figure: {OUT_FIG}")
    plt.show()


if __name__ == "__main__":
    main()
