import pandas as pd

DATASET = "data_processed/ml_dataset_ndvi_weather_features.parquet"


def main():
    df = pd.read_parquet(DATASET)

    print("\nNaN share (top 15):")
    print(
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )

    weather_cols = [
        "t2m_mean",
        "t2m_max",
        "precip",
        "wind_mean",
    ]

    print("\nWeather ranges:")
    print(df[weather_cols].describe())

    print("\nCoverage by year:")
    print(df["date"].dt.year.value_counts().sort_index())


if __name__ == "__main__":
    main()
