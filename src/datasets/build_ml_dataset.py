import pandas as pd

 

def build_ml_dataset(
    ndvi_path: str,
    fire_path: str
) -> pd.DataFrame:
    """
    Собирает ML-датасет:
    NDVI признаки (day D) → пожар (day D+1)

    Возвращает DataFrame:
    cell_id | date | features... | fire
    """

    print("Загрузка NDVI-признаков...")
    ndvi = pd.read_parquet(ndvi_path)

    print("Загрузка пожаров...")
    fire = pd.read_parquet(fire_path)

    # Приводим типы
    ndvi["date"] = pd.to_datetime(ndvi["date"])
    fire["date"] = pd.to_datetime(fire["date"])

    ndvi["cell_id"] = ndvi["cell_id"].astype("int32")
    fire["cell_id"] = fire["cell_id"].astype("int32")

    fire["fire"] = fire["fire"].astype("int8")
 

    fire["date_prev"] = fire["date"] - pd.Timedelta(days=1)

    fire_shifted = fire[[
        "cell_id",
        "date_prev",
        "fire"
    ]].rename(columns={"date_prev": "date"})

 
    print("Объединение NDVI и fire (D → D+1)...")

    dataset = ndvi.merge(
        fire_shifted,
        on=["cell_id", "date"],
        how="left"
    )

 
    dataset["fire"] = dataset["fire"].fillna(0).astype("int8")

    return dataset

 
if __name__ == "__main__":

    ndvi_path = r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed\ndvi_full_features.parquet"
    fire_path = r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed\fires_by_grid_daily.parquet"

    out_path = r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed\ml_dataset.parquet"

    df = build_ml_dataset(ndvi_path, fire_path)

    print("Сохранение ML-датасета...")
    df.to_parquet(out_path, index=False)

    print("Готово:", out_path)

df = pd.read_parquet(
    r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed\ml_dataset.parquet"
)

print(df.shape)
print(df["fire"].value_counts(normalize=True))

