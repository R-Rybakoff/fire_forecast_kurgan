import geopandas as gpd
import pandas as pd
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# ============================================================
GRID = Path("data_interim/grids/grid_weather_10km.geojson")
OUT = Path("data_interim/weather_processed/weather_era5_openmeteo.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

YEARS = [2019, 2020, 2021, 2022, 2023]

VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "precipitation_sum",
    "wind_speed_10m_mean"
]

BASE_URL = "https://archive-api.open-meteo.com/v1/era5"
MAX_WORKERS = 6        # КЛЮЧЕВО
MAX_RETRIES = 3

# ============================================================
grid = gpd.read_file(GRID).to_crs(4326)
grid["lon"] = grid.geometry.centroid.x
grid["lat"] = grid.geometry.centroid.y

if OUT.exists():
    weather_all = pd.read_parquet(OUT)
    done_ids = set(weather_all.weather_id.unique())
else:
    weather_all = pd.DataFrame()
    done_ids = set()

# ============================================================
def fetch_weather(row):
    lat, lon, wid = row.lat, row.lon, row.weather_id
    frames = []

    for year in YEARS:
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(
                    BASE_URL,
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "start_date": f"{year}-01-01",
                        "end_date": f"{year}-12-31",
                        "daily": ",".join(VARS),
                        "timezone": "UTC",
                    },
                    timeout=20,
                )

                if r.status_code == 429:
                    time.sleep(2)
                    continue

                r.raise_for_status()
                d = r.json().get("daily")
                if not d:
                    break

                df = pd.DataFrame(d)
                df["date"] = pd.to_datetime(df["time"])
                df = df.drop(columns="time")
                frames.append(df)
                break

            except Exception:
                time.sleep(2)

    if frames:
        out = pd.concat(frames)
        out["weather_id"] = wid
        return out

    return None

# ============================================================
rows = grid[~grid.weather_id.isin(done_ids)].itertuples(index=False)

records = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(fetch_weather, r) for r in rows]

    for fut in tqdm(as_completed(futures), total=len(futures), desc="ERA5-Land parallel"):
        res = fut.result()
        if res is not None:
            records.append(res)

        if len(records) >= 20:
            chunk = pd.concat(records)
            weather_all = pd.concat([weather_all, chunk], ignore_index=True)
            weather_all.to_parquet(OUT)
            records = []

# ============================================================
if records:
    weather_all = pd.concat([weather_all, pd.concat(records)], ignore_index=True)

weather_all = weather_all.rename(columns={
    "temperature_2m_mean": "t2m_mean",
    "temperature_2m_max": "t2m_max",
    "precipitation_sum": "precip",
    "wind_speed_10m_mean": "wind_mean"
})

weather_all.to_parquet(OUT)

print("DONE. Saved:", OUT)
print(weather_all.head())

