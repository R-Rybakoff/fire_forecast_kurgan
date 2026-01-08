import pandas as pd
import geopandas as gpd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =========================
# PATHS
# =========================
DATASET_PATH = "data_processed/ml_dataset_ndvi_weather_features.parquet"

# ПОЛНАЯ сетка области (ОСНОВА КАРТЫ)
GRID_PATH = "data_interim/grids/grid_500m_kurgan.geojson"

# Сетка с фактами пожаров (y = 1)
GRID_Y_PATH = "data_processed/grid_with_y_kurgan.geojson"

OUT_GEO = "data_processed/visualization/fire_risk_with_y_kurgan.geojson"

TARGET = "y"
DATE_COL = "date"

TRAIN_END = "2022-12-31"

# желаемая дата (если нет данных — берётся ближайшая <=)
PREDICT_DATE = "2023-05-15"


# =========================
# UTILS
# =========================
def get_features(df):
    return [
        c for c in df.columns
        if c.startswith("ndvi_")
        or c.startswith("t2m_")
        or c.startswith("precip")
        or c.startswith("wind_mean")
    ]


# =========================
# MAIN
# =========================
def main():
    print("Loading ML dataset...")
    df = pd.read_parquet(DATASET_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    features = get_features(df)
    print(f"Total features: {len(features)}")

    # =========================
    # TRAIN FINAL MODEL
    # =========================
    train_df = df[df[DATE_COL] <= TRAIN_END]
    train_df = train_df.dropna(subset=features + [TARGET])

    X_train = train_df[features]
    y_train = train_df[TARGET]

    print(f"Train size: {len(X_train)}")
    print(f"Positive rate: {y_train.mean():.6f}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    print("Training final model...")
    model.fit(X_train, y_train)

    # =========================
    # SELECT VALID DATE
    # =========================
    target_date = pd.to_datetime(PREDICT_DATE)

    valid = (
        df[df[DATE_COL] <= target_date]
        .dropna(subset=features)
    )

    if valid.empty:
        raise RuntimeError("No valid data available for prediction")

    pred_date = valid[DATE_COL].max()
    print(f"Using prediction date: {pred_date.date()}")

    pred_df = valid[valid[DATE_COL] == pred_date].copy()
    print(f"Prediction rows: {len(pred_df)}")

    # =========================
    # PREDICT RISK (ТОЛЬКО ТАМ, ГДЕ МОЖНО)
    # =========================
    pred_df["fire_risk"] = model.predict_proba(
        pred_df[features]
    )[:, 1]

    risk_table = pred_df[["cell_id", "fire_risk"]]

    # =========================
    # LOAD FULL GRID (ВСЯ ОБЛАСТЬ)
    # =========================
    print("Loading full grid...")
    grid = gpd.read_file(GRID_PATH)

    # =========================
    # MERGE RISK (LEFT JOIN — КЛЮЧЕВО!)
    # =========================
    grid = grid.merge(
        risk_table,
        on="cell_id",
        how="left",
        validate="1:1"
    )

    # =========================
    # LOAD FIRE FACTS
    # =========================
    print("Loading fire facts...")
    grid_y = gpd.read_file(GRID_Y_PATH)

    fire_cells = grid_y[grid_y["y"] == 1][["cell_id"]]
    fire_cells["fire_fact"] = 1

    grid = grid.merge(
        fire_cells,
        on="cell_id",
        how="left"
    )

    grid["fire_fact"] = grid["fire_fact"].fillna(0).astype(int)

    # =========================
    # SAVE
    # =========================
    grid.to_file(OUT_GEO, driver="GeoJSON")
    print(f"Saved FULL map with facts: {OUT_GEO}")


if __name__ == "__main__":
    main()

