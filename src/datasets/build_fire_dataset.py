"""
Fire Forecast Kurgan — FULL PIPELINE

cell_id | date | ndvi_mean | y

Источники:
- MODIS NDVI (HDF)
- VIIRS FIRMS fires
- Регулярная сетка 500×500 м
"""

import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import box
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================

GRID_PATH = "data_processed/grid_with_y_kurgan.geojson"
MODIS_DIR = "data_raw/modis_ndvi"
FIRES_PATH = "data_processed/fires_by_grid_daily.parquet"
OUTPUT_PATH = "data_processed/dataset_fire_ndvi_2019_2023.parquet"

CRS_GRID = "EPSG:32642"
CELL_SIZE = 500

# ==========================
# STEP 1 — LOAD GRID
# ==========================

print("Loading grid...")
grid = gpd.read_file(GRID_PATH)
assert "cell_id" in grid.columns
print("Grid cells:", len(grid))
print("Grid CRS:", grid.crs)

# ==========================
# STEP 2 — READ NDVI FROM HDF
# ==========================

def read_ndvi_layer(hdf_path):
    layer = (
        f'HDF4_EOS:EOS_GRID:"{hdf_path}":'
        'MODIS_Grid_16DAY_250m_500m_VI:"250m 16 days NDVI"'
    )
    return layer


def aggregate_ndvi_for_file(hdf_path):
    date_str = os.path.basename(hdf_path).split(".")[1][1:]
    date = pd.to_datetime(date_str, format="%Y%j")

    records = []

    with rasterio.open(read_ndvi_layer(hdf_path)) as src:
        for _, row in grid.iterrows():
            try:
                out, _ = rasterio.mask.mask(
                    src,
                    [row.geometry],
                    crop=True
                )
                vals = out[0].astype("float32")
                vals[vals <= -2000] = np.nan
                vals *= 0.0001
                mean = np.nanmean(vals)
            except Exception:
                mean = np.nan

            records.append({
                "cell_id": row.cell_id,
                "date": date,
                "ndvi_mean": mean
            })

    return pd.DataFrame(records)


print("Collecting MODIS files...")
modis_files = sorted(glob.glob(f"{MODIS_DIR}/**/*.hdf", recursive=True))
print("MODIS files:", len(modis_files))

ndvi_parts = []

for f in tqdm(modis_files, desc="Processing NDVI"):
    df_part = aggregate_ndvi_for_file(f)
    ndvi_parts.append(df_part)

ndvi = pd.concat(ndvi_parts, ignore_index=True)
ndvi["cell_id"] = ndvi["cell_id"].astype("int32")
ndvi["ndvi_mean"] = ndvi["ndvi_mean"].astype("float32")

print("NDVI shape:", ndvi.shape)
print("NDVI date range:", ndvi.date.min(), "→", ndvi.date.max())

# ==========================
# STEP 3 — LOAD FIRE LABELS
# ==========================

print("Loading fire labels...")
fires = pd.read_parquet(FIRES_PATH)

fires = fires[
    (fires.date >= "2019-01-01") &
    (fires.date <= "2023-12-31")
]

fires["cell_id"] = fires["cell_id"].astype("int32")
fires["date"] = pd.to_datetime(fires["date"])

print("Fire rows:", len(fires))
print("Fire date range:", fires.date.min(), "→", fires.date.max())

# ==========================
# STEP 4 — MEMORY SAFE LABELING
# ==========================

print("Indexing fire events...")
fire_index = set(
    zip(
        fires.cell_id.values,
        fires.date.values
    )
)

print("Labeling NDVI...")
ndvi["y"] = [
    1 if (cid, d) in fire_index else 0
    for cid, d in zip(ndvi.cell_id, ndvi.date)
]

ndvi["y"] = ndvi["y"].astype("int8")

# ==========================
# STEP 5 — FINAL CHECKS
# ==========================

print("\nFINAL DATASET CHECKS")
print("Shape:", ndvi.shape)
print("Columns:", ndvi.columns.tolist())
print("Fire share:", ndvi.y.mean())
print("NDVI NaN share:", ndvi.ndvi_mean.isna().mean())
print("NDVI stats:")
print(ndvi.ndvi_mean.describe())

# ==========================
# STEP 6 — SAVE
# ==========================

ndvi.to_parquet(OUTPUT_PATH, index=False)
print("\nSaved:", OUTPUT_PATH)
