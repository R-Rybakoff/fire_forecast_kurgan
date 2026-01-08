import pandas as pd

BASELINE_PATH = "data_processed/ml_dataset.parquet"
WEATHER_PATH = "data_interim/weather_processed/weather_era5_openmeteo.parquet"
CELL_WEATHER_PATH = "data_interim/weather_processed/cell_to_weather.parquet"

OUT_PATH = "data_processed/ml_dataset_ndvi_weather.parquet"


def main():
    df = pd.read_parquet(BASELINE_PATH)
    weather = pd.read_parquet(WEATHER_PATH)
    cell_weather = pd.read_parquet(CELL_WEATHER_PATH)

    # normalize date
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()

    # cell_id → weather_id
    df = df.merge(
        cell_weather,
        on="cell_id",
        how="left"
    )

    # weather_id + date → weather
    df = df.merge(
        weather,
        on=["weather_id", "date"],
        how="left"
    )

    df.to_parquet(OUT_PATH, index=False)
    print("ML dataset with weather saved:", OUT_PATH)


if __name__ == "__main__":
    main()
