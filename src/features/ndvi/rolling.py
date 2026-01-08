import pandas as pd


# ============================================================
# 1. ФУНКЦИЯ ROLLING-ПРИЗНАКОВ
# ============================================================

def compute_ndvi_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет rolling-признаки NDVI:
    - ndvi_mean_14
    - ndvi_std_30

    Корректно работает с нерегулярными датами
    и не ломается на индексах.
    """

    # 1. ЖЁСТКАЯ сортировка (мы контролируем порядок строк)
    df = df.sort_values(["cell_id", "date"]).copy()

    # 2. Убираем текущий день (нет утечки будущего)
    df["ndvi_shifted"] = (
        df.groupby("cell_id")["ndvi_mean"]
          .shift(1)
    )

    # 3. Rolling mean за 14 дней
    rolling_mean_14 = (
        df.groupby("cell_id")
          .rolling(
              window="14D",
              on="date",
              min_periods=1
          )["ndvi_shifted"]
          .mean()
          .values      # ← КЛЮЧЕВОЙ ФИКС
    )

    # 4. Rolling std за 30 дней
    rolling_std_30 = (
        df.groupby("cell_id")
          .rolling(
              window="30D",
              on="date",
              min_periods=1
          )["ndvi_shifted"]
          .std()
          .values      # ← КЛЮЧЕВОЙ ФИКС
    )

    # 5. Записываем в DataFrame
    df["ndvi_mean_14"] = rolling_mean_14
    df["ndvi_std_30"] = rolling_std_30

    # 6. Убираем временную колонку
    df = df.drop(columns=["ndvi_shifted"])

    return df



# ============================================================
# 2. ЗАПУСК
# ============================================================

if __name__ == "__main__":

    input_path = r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed\ndvi_with_lags.parquet"
    output_path = r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed\ndvi_with_lags_and_rolling.parquet"

    print(" Загрузка данных...")
    df = pd.read_parquet(input_path)

    print(" Расчёт rolling-признаков...")
    df = compute_ndvi_rolling(df)

    print(" Сохранение результата...")
    df.to_parquet(output_path, index=False)

    print(" Готово:", output_path)

df = pd.read_parquet(
    r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed\ndvi_with_lags_and_rolling.parquet"
)

print(
    df[[
        "ndvi_mean",
        "ndvi_mean_14",
        "ndvi_std_30"
    ]].head(10)
)
