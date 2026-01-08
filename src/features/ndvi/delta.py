import pandas as pd


# ============================================================
# 1. ФУНКЦИЯ NDVI DELTA
# ============================================================

def compute_ndvi_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет признак NDVI_delta_7:
    изменение NDVI за ~7 дней.
    """

    # Сортировка для порядка (на всякий случай)
    df = df.sort_values(["cell_id", "date"]).copy()

    # Разница между текущим NDVI и прошлым
    df["ndvi_delta_7"] = df["ndvi_mean"] - df["ndvi_lag_7"]

    return df


# ============================================================
# 2. ЗАПУСК
# ============================================================

if __name__ == "__main__":

    input_path = r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed\ndvi_with_lags_and_rolling.parquet"
    output_path = r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed\ndvi_with_lags_rolling_delta.parquet"

    print("Загрузка данных...")
    df = pd.read_parquet(input_path)

    print("Расчёт NDVI_delta_7...")
    df = compute_ndvi_delta(df)

    print("Сохранение результата...")
    df.to_parquet(output_path, index=False)

    print("Готово:", output_path)
    
df = pd.read_parquet(
    r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed\ndvi_with_lags_rolling_delta.parquet"
)

print(
    df[[
        "ndvi_mean",
        "ndvi_lag_7",
        "ndvi_delta_7"
    ]].head(10)
)
