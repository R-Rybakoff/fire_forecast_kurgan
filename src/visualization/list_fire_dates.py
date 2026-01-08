import pandas as pd

FIRES_DATA = "data_processed/fires_by_grid_daily.parquet"

# =========================
# ЗАГРУЗКА
# =========================

fires = pd.read_parquet(FIRES_DATA)
fires["date"] = pd.to_datetime(fires["date"])

# =========================
# АГРЕГАЦИЯ ПО ДАТАМ
# =========================

fires_by_date = (
    fires
    .groupby("date")
    .size()
    .reset_index(name="fire_cells")
    .sort_values("fire_cells", ascending=False)
)

# =========================
# ВЫВОД
# =========================

if fires_by_date.empty:
    print("No fire records found.")
else:
    print("Dates with fires (sorted by number of fire cells):\n")
    print(fires_by_date.head(20).to_string(index=False))

    print("\nSummary:")
    print("Total dates with fires:", fires_by_date.shape[0])
    print("Date range:", fires_by_date["date"].min().date(), "→", fires_by_date["date"].max().date())
