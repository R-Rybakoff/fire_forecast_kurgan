import geopandas as gpd
import numpy as np
from shapely.geometry import box

def build_grid(region_gdf, cell_size=500):
    """
    Строит регулярную сетку cell_size x cell_size (в метрах)
    внутри границы региона
    """
    minx, miny, maxx, maxy = region_gdf.total_bounds

    x_coords = np.arange(minx, maxx, cell_size)
    y_coords = np.arange(miny, maxy, cell_size)

    cells = []
    for x in x_coords:
        for y in y_coords:
            cells.append(box(x, y, x + cell_size, y + cell_size))

    grid = gpd.GeoDataFrame(
        geometry=cells,
        crs=region_gdf.crs
    )

    # Обрезаем сетку по границе региона
    grid = gpd.overlay(grid, region_gdf, how="intersection")

    # Добавляем id ячейки
    grid["cell_id"] = range(len(grid))

    return grid
