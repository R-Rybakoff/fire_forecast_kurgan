import pandas as pd
import numpy as np
from typing import Iterable


# ============================================================
# 1. ФУНКЦИЯ РАСЧЁТА TIME-BASED NDVI ЛАГОВ
# ============================================================

def compute_ndvi_lags(
    df: pd.DataFrame,
    lags_days=(7, 14, 30)
) -> pd.DataFrame:
    """
    Добавляет NDVI лаги по времени (дни), корректно для merge_asof.
    """

    # КРИТИЧНО: сначала date, потом cell_id
    df = df.sort_values(["date", "cell_id"]).copy()

    base = df[["cell_id", "date", "ndvi_mean"]]

    for lag in lags_days:
        lag_df = base.copy()

        # сдвиг даты вперёд
        lag_df["date"] = lag_df["date"] + pd.Timedelta(days=lag)

        lag_df = lag_df.rename(
            columns={"ndvi_mean": f"ndvi_lag_{lag}"}
        )

        # ТОЖЕ ОБЯЗАТЕЛЬНО
        lag_df = lag_df.sort_values(["date", "cell_id"])

        df = pd.merge_asof(
            df,
            lag_df,
            on="date",
            by="cell_id",
            direction="backward",
            allow_exact_matches=True
        )

    return df



# ============================================================
# 2. BATCH-PROCESSING ПО cell_id (MEMORY-SAFE)
# ============================================================

def process_ndvi_in_batches(
    ndvi_path: str,
    out_path: str,
    batch_size: int = 1500
):
    """
    Считает NDVI лаги батчами по cell_id
    и сохраняет результат на диск.
    """

    print("▶ Загрузка NDVI...")
    ndvi = pd.read_parquet(ndvi_path)

    # приведение типов (КРИТИЧНО для памяти)
    ndvi["cell_id"] = ndvi["cell_id"].astype("int32")
    ndvi["date"] = pd.to_datetime(ndvi["date"])
    ndvi["ndvi_mean"] = ndvi["ndvi_mean"].astype("float32")

    cell_ids = ndvi["cell_id"].unique()
    cell_ids.sort()

    print(f"▶ Всего cell_id: {len(cell_ids)}")

    result_chunks = []

    for i in range(0, len(cell_ids), batch_size):
        batch_ids = cell_ids[i : i + batch_size]

        chunk = ndvi[ndvi["cell_id"].isin(batch_ids)]

        chunk = compute_ndvi_lags(chunk)

        result_chunks.append(chunk)

        print(
            f"✔ Обработано cell_id: {i}–{i + len(batch_ids)} "
            f"({round((i + len(batch_ids)) / len(cell_ids) * 100, 2)}%)"
        )

    print("▶ Конкатенация результата...")
    result = pd.concat(result_chunks, ignore_index=True)

    print("▶ Сохранение parquet...")
    result.to_parquet(out_path, index=False)

    print("✅ Готово:", out_path)


# ============================================================
# 3. ЗАПУСК
# ============================================================

if __name__ == "__main__":
    process_ndvi_in_batches(
        ndvi_path=r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed\ndvi_with_y_kurgan_2019_2023.parquet",
        out_path=r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed\ndvi_with_lags.parquet",
        batch_size=1500
    )
