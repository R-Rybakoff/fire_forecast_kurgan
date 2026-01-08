import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd

def aggregate_ndvi(grid, raster_path, date):
    records = []

    with rasterio.open(raster_path) as src:
        for _, row in grid.iterrows():
            try:
                out, _ = rasterio.mask.mask(
                    src,
                    [row.geometry],
                    crop=True
                )
                values = out[0]
                values = values[~np.isnan(values)]
                mean = np.nanmean(values)
            except:
                mean = np.nan

            records.append({
                "cell_id": row.cell_id,
                "date": date,
                "ndvi_mean": mean
            })

    return pd.DataFrame(records)
