import json
import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import box, Point

# =========================
# НАСТРОЙКИ
# =========================

TOP_Q = 0.90
PRIORITY_GRID_STEP_M = 5000   # 5 км
MAX_ZONES = 40

GRID_GEOJSON = "data_processed/grid_with_y_kurgan.geojson"
ML_DATA = "data_processed/ml_dataset_with_priority.parquet"
FIRES_DATA = "data_processed/fires_by_grid_daily.parquet"
ROADS_DATA = "data_interim/infrastructure/roads_osm.geojson"
KURGAN_BOUNDARY = "data_raw/boundaries/kurgan.geojson"

OUTPUT_HTML = "data_processed/visualization/fire_priority_map.html"

FORECAST_DATE = "2023-09-14"
date = pd.to_datetime(FORECAST_DATE)

print(f"Forecast date: {FORECAST_DATE}")

# =========================
# ЗАГРУЗКА ДАННЫХ
# =========================

gdf = gpd.read_file(GRID_GEOJSON)
ml = pd.read_parquet(ML_DATA)
fires_all = pd.read_parquet(FIRES_DATA)

ml["date"] = pd.to_datetime(ml["date"])
fires_all["date"] = pd.to_datetime(fires_all["date"])

# =========================
# FIRE RISK (ПРОГНОЗ)
# =========================

ml_d = (
    ml[ml["date"] == date]
    .sort_values("fire_risk", ascending=False)
    .drop_duplicates("cell_id")
)

gdf = gdf.merge(
    ml_d[["cell_id", "fire_risk"]],
    on="cell_id",
    how="left"
)

gdf["fire_risk"] = gdf["fire_risk"].fillna(0.0)

# =========================
# ФАКТИЧЕСКИЕ ПОЖАРЫ
# =========================

fires_d = fires_all[fires_all["date"] == date][["cell_id"]].copy()

if len(fires_d) > 0:
    fires_d["y"] = 1
    gdf = gdf.merge(fires_d, on="cell_id", how="left")
else:
    gdf["y"] = 0

if "y" not in gdf.columns:
    gdf["y"] = 0
else:
    gdf["y"] = gdf["y"].fillna(0)

gdf["y"] = gdf["y"].astype(int)

print("Real fire cells on date:", int(gdf["y"].sum()))

# =========================
# TOP-Q ЗОНЫ РИСКА
# =========================

thr = gdf["fire_risk"].quantile(TOP_Q)
high = gdf[gdf["fire_risk"] >= thr].copy()

# =========================
# АГРЕГАЦИЯ В КРУПНЫЕ КВАДРАТЫ
# =========================

high_m = high.to_crs(32642)
cent = high_m.geometry.centroid

high_m["gx"] = (cent.x // PRIORITY_GRID_STEP_M) * PRIORITY_GRID_STEP_M
high_m["gy"] = (cent.y // PRIORITY_GRID_STEP_M) * PRIORITY_GRID_STEP_M

zones = (
    high_m
    .groupby(["gx", "gy"], as_index=False)
    .agg(
        mean_risk=("fire_risk", "mean"),
        fire_hits=("y", "sum")
    )
    .sort_values("mean_risk", ascending=False)
    .head(MAX_ZONES)
)

zones["geometry"] = zones.apply(
    lambda r: box(
        r.gx,
        r.gy,
        r.gx + PRIORITY_GRID_STEP_M,
        r.gy + PRIORITY_GRID_STEP_M,
    ),
    axis=1
)

zones = gpd.GeoDataFrame(zones, geometry="geometry", crs=32642).to_crs(4326)

print("Priority zones:", len(zones))

# =========================
# ПОЖАРЫ КАК ТОЧКИ
# =========================

fires_pts = gdf[gdf["y"] == 1].copy()

if len(fires_pts) > 0:
    fires_pts = fires_pts.to_crs(32642)
    fires_pts["geometry"] = fires_pts.geometry.centroid
    fires_pts = fires_pts.to_crs(4326)
    fires_pts = fires_pts[["geometry"]]

print("Real fire points:", len(fires_pts))

# =========================
# СЦЕНАРНЫЕ ПОЖАРЫ (ДЕМО)
# =========================

scenario_fires = None
SCENARIO_FIRES_PER_ZONE = 2

if len(fires_pts) == 0:
    print("No real fires — adding scenario fires for demo")

    pts = []
    for _, z in zones.head(8).iterrows():
        cx, cy = z.geometry.centroid.x, z.geometry.centroid.y
        for _ in range(SCENARIO_FIRES_PER_ZONE):
            pts.append(Point(cx, cy))

    scenario_fires = gpd.GeoDataFrame(
        geometry=pts,
        crs=4326
    )

# =========================
# ПОДЛОЖКА: КУРГАНСКАЯ ОБЛАСТЬ
# =========================

kurgan = gpd.read_file(KURGAN_BOUNDARY).to_crs(4326)[["geometry"]]

# =========================
# КАРТА
# =========================

center = kurgan.geometry.union_all().centroid

m = folium.Map(
    location=[center.y, center.x],
    zoom_start=7,
    tiles="cartodbpositron"
)

# ---- Заголовок
title_html = f"""
<div style="
    position: fixed;
    top: 10px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 9999;
    background-color: rgba(255,255,255,0.9);
    padding: 10px 18px;
    border-radius: 6px;
    font-size: 18px;
    font-weight: 600;
    box-shadow: 0 0 8px rgba(0,0,0,0.15);
">
Прогноз пожарной опасности и зоны приоритетного реагирования<br>
<span style="font-size:14px;font-weight:400;">
Курганская область · дата: {FORECAST_DATE}
</span>
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

# ---- Фон области
folium.GeoJson(
    kurgan,
    name="Kurgan oblast",
    style_function=lambda f: {
        "fillColor": "#c7e9c0",
        "color": "#74c476",
        "weight": 1,
        "fillOpacity": 0.25,
    },
).add_to(m)

# ---- Зоны приоритета
folium.GeoJson(
    json.loads(zones.to_json()),
    name="Priority response zones",
    style_function=lambda f: {
        "fillColor": "#d73027",
        "color": "#7f0000",
        "weight": 1,
        "fillOpacity": 0.45,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["mean_risk", "fire_hits"],
        aliases=["Mean fire risk", "Observed fires"],
    ),
).add_to(m)

# ---- Дороги
roads = gpd.read_file(ROADS_DATA).to_crs(4326)

folium.GeoJson(
    json.loads(roads.to_json()),
    name="Roads",
    style_function=lambda f: {
        "color": "#555555",
        "weight": 1,
        "opacity": 0.6,
    },
).add_to(m)

# ---- Реальные пожары
if len(fires_pts) > 0:
    folium.GeoJson(
        json.loads(fires_pts.to_json()),
        name="Observed fires",
        marker=folium.CircleMarker(
            radius=6,
            color="#1f78b4",
            fill=True,
            fill_opacity=1,
        ),
    ).add_to(m)

# ---- Сценарные пожары (демо)
if scenario_fires is not None:
    folium.GeoJson(
        json.loads(scenario_fires.to_json()),
        name="Scenario fires (demo)",
        marker=folium.CircleMarker(
            radius=6,
            color="#1f78b4",
            fill=True,
            fill_opacity=1,
        ),
    ).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

# ---- Fit bounds
b = zones.total_bounds.tolist() if len(zones) > 0 else kurgan.total_bounds.tolist()
m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])

m.save(OUTPUT_HTML)
print("Map saved:", OUTPUT_HTML)
