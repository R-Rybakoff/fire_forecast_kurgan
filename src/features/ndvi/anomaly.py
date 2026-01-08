import pandas as pd


# ============================================================
# 1. NDVI ANOMALY
# ============================================================

def compute_ndvi_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет признак NDVI_anomaly:
    отклонение NDVI от многолетней нормы
    для данной ячейки и месяца.
    """

    # Сортировка (на всякий случай)
    df = df.sort_values(["cell_id", "date"]).copy()

    # 1. Месяц (сезонность)
    df["month"] = df["date"].dt.month

    # 2. Многолетняя месячная норма NDVI
    month_norm = (
        df.groupby(["cell_id", "month"])["ndvi_mean"]
          .mean()
          .rename("ndvi_month_mean")
          .reset_index()
    )

    # 3. Присоединяем норму
    df = df.merge(
        month_norm,
        on=["cell_id", "month"],
        how="left"
    )

    # 4. Аномалия
    df["ndvi_anomaly"] = df["ndvi_mean"] - df["ndvi_month_mean"]

    # 5. Чистим служебные колонки
    df = df.drop(columns=["month", "ndvi_month_mean"])

    return df


# ============================================================
# 2. ЗАПУСК
# ============================================================

if __name__ == "__main__":

    input_path = r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed\ndvi_with_lags_rolling_delta.parquet"
    output_path = r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed\ndvi_full_features.parquet"

    print("Загрузка данных...")
    df = pd.read_parquet(input_path)

    print("Расчёт NDVI_anomaly...")
    df = compute_ndvi_anomaly(df)

    print("Сохранение результата...")
    df.to_parquet(output_path, index=False)

    print("Готово:", output_path)

df = pd.read_parquet(
    r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed\ndvi_full_features.parquet"
)

print(
    df[[
        "ndvi_mean",
        "ndvi_anomaly"
    ]].head(10)
)
