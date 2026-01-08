import geopandas as gpd

def assign_fires_to_grid(fires, grid):
    fires_gdf = gpd.GeoDataFrame(
        fires,
        geometry=gpd.points_from_xy(
            fires.longitude,
            fires.latitude
        ),
        crs="EPSG:4326"
    ).to_crs(grid.crs)

    joined = gpd.sjoin(
        fires_gdf,
        grid,
        predicate="within"
    )

    return joined[["cell_id", "date"]]
