import geopandas as gpd

GRID_500M = "data_interim/grids/grid_500m_kurgan.geojson"
GRID_WEATHER = "data_interim/grids/grid_weather_10km.geojson"
OUT_PATH = "data_interim/weather_processed/cell_to_weather.parquet"


def main():
    grid_500m = gpd.read_file(GRID_500M)
    grid_weather = gpd.read_file(GRID_WEATHER)

    if grid_500m.crs != grid_weather.crs:
        grid_weather = grid_weather.to_crs(grid_500m.crs)

    j = gpd.sjoin(
        grid_500m[["cell_id", "geometry"]],
        grid_weather[["weather_id", "geometry"]],
        how="left",
        predicate="within"
    )

    missing = j[j["weather_id"].isna()][["cell_id", "geometry"]]

    if len(missing) > 0:
        j2 = gpd.sjoin(
            missing,
            grid_weather[["weather_id", "geometry"]],
            how="left",
            predicate="intersects"
        )
        j2 = j2.drop_duplicates("cell_id")
        j.update(j2)

    out = (
        j.drop(columns=["geometry", "index_right"])
         .sort_values("cell_id")
    )

    out.to_parquet(OUT_PATH, index=False)
    print("Spatial join saved:", OUT_PATH)


if __name__ == "__main__":
    main()
