import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
import numpy as np

GRID_500M = Path("data_interim/grids/grid_500m_kurgan.geojson")
OUT_GRID = Path("data_interim/grids/grid_weather_10km.geojson")

gdf = gpd.read_file(GRID_500M).to_crs(4326)

# шаг ~0.1 градуса ≈ 10–11 км
STEP = 0.1

minx, miny, maxx, maxy = gdf.total_bounds

xs = np.arange(minx, maxx, STEP)
ys = np.arange(miny, maxy, STEP)

cells = []
cid = 0
for x in xs:
    for y in ys:
        cells.append({
            "weather_id": cid,
            "geometry": box(x, y, x + STEP, y + STEP)
        })
        cid += 1

weather_grid = gpd.GeoDataFrame(cells, crs=4326)
weather_grid.to_file(OUT_GRID)

print("Weather grid saved:", OUT_GRID)
print("Weather cells:", len(weather_grid))
