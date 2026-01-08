import pandas as pd

DATASET_PATH = "data_processed/ml_dataset_ndvi_weather.parquet"
OUT_PATH = "data_processed/ml_dataset_ndvi_weather_features.parquet"

# canonical weather names we want to use дальше
WEATHER_RENAME = {
    "temperature_2m_mean": "t2m_mean",
    "temperature_2m_max": "t2m_max",
    "precipitation_sum": "precip",
    "wind_speed_10m_mean": "wind_mean",
}

WEATHER_COLS = {
    "t2m_mean": {"sum": False},
    "t2m_max": {"sum": False},
    "precip": {"sum": True},
    "wind_mean": {"sum": False},
}

LAGS = [1, 3, 7]
ROLLING_WINDOWS = [3, 7]


def main():
    df = pd.read_parquet(DATASET_PATH)

    # normalize date
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # 🔑 normalize weather column names INSIDE dataset
    for old, new in WEATHER_RENAME.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # sanity check
    missing = [c for c in WEATHER_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing weather columns in dataset: {missing}"
        )

    # sort for rolling
    df = df.sort_values(["weather_id", "date"])

    # feature engineering
    for col, cfg in WEATHER_COLS.items():
        grp = df.groupby("weather_id")[col]

        # lags
        for lag in LAGS:
            df[f"{col}_lag_{lag}"] = grp.shift(lag)

        # rolling stats (strictly past)
        for win in ROLLING_WINDOWS:
            df[f"{col}_mean_{win}"] = grp.shift(1).rolling(win).mean()
            df[f"{col}_max_{win}"] = grp.shift(1).rolling(win).max()

        # rolling sum for precipitation
        if cfg["sum"]:
            df[f"{col}_sum_7"] = grp.shift(1).rolling(7).sum()

    df.to_parquet(OUT_PATH, index=False)
    print("Weather features created:", OUT_PATH)


if __name__ == "__main__":
    main()
