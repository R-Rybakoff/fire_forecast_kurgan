import geopandas as gpd
import numpy as np

# =========================
# PATHS
# =========================
GRID_PATH = "data_interim/grids/grid_500m_kurgan.geojson"
ROADS_PATH = "data_interim/infrastructure/roads_osm.geojson"
SETTLEMENTS_PATH = "data_interim/infrastructure/settlements_osm.geojson"

OUT_PATH = "data_interim/infrastructure/grid_with_infrastructure.parquet"

# =========================
# LOAD DATA
# =========================
print("Loading grid...")
grid = gpd.read_file(GRID_PATH)[["cell_id", "geometry"]]

print("Loading roads...")
roads = gpd.read_file(ROADS_PATH)[["geometry"]]

print("Loading settlements...")
settlements = gpd.read_file(SETTLEMENTS_PATH)[["geometry"]]

# =========================
# CRS TO METERS
# =========================
print("Reprojecting to EPSG:3857...")
grid = grid.to_crs(epsg=3857)
roads = roads.to_crs(epsg=3857)
settlements = settlements.to_crs(epsg=3857)

# =========================
# ROADS → POINTS (ACCELERATION)
# =========================
print("Converting roads to points...")
road_points = roads.copy()
road_points["geometry"] = road_points.geometry.centroid
road_points = road_points[["geometry"]]

# =========================
# DISTANCE TO ROADS (FAST)
# =========================
print("Calculating distance to roads (fast nearest)...")
grid = gpd.sjoin_nearest(
    grid,
    road_points,
    how="left",
    distance_col="dist_to_road"
)

grid.drop(columns=["index_right"], inplace=True)

# =========================
# DISTANCE TO SETTLEMENTS (FAST)
# =========================
print("Calculating distance to settlements (fast nearest)...")
grid = gpd.sjoin_nearest(
    grid,
    settlements,
    how="left",
    distance_col="dist_to_settlement"
)

grid.drop(columns=["index_right"], inplace=True)

# =========================
# LOG TRANSFORM
# =========================
print("Applying log transform...")
grid["log_dist_road"] = np.log1p(grid["dist_to_road"])
grid["log_dist_settlement"] = np.log1p(grid["dist_to_settlement"])

# =========================
# SAVE
# =========================
print("Saving result...")
grid.drop(columns="geometry").to_parquet(OUT_PATH)

print("Infrastructure features saved:")
print(OUT_PATH)
