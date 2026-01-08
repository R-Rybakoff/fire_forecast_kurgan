import pandas as pd
import geopandas as gpd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ============================================================
# 1. КОНСТАНТНЫЕ ПУТИ (БЕЗ ПЕРЕМЕННЫХ!)
# ============================================================

ML_DATASET_PATH = r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed\ml_dataset.parquet"
GRID_PATH = r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed\grid_with_y_kurgan.geojson"
OUTPUT_DIR = r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed"

FEATURES = [
    "ndvi_mean",
    "ndvi_lag_7",
    "ndvi_lag_14",
    "ndvi_lag_30",
    "ndvi_mean_14",
    "ndvi_std_30",
    "ndvi_delta_7",
    "ndvi_anomaly"
]


# ============================================================
# 2. ЗАГРУЗКА ML-ДАТАСЕТА
# ============================================================

print("Загрузка ML-датасета...")
df = pd.read_parquet(ML_DATASET_PATH)
df["date"] = pd.to_datetime(df["date"])

# ============================================================
# 3. ВЫБОР ДАТЫ ДЛЯ КАРТЫ (ТОЛЬКО ПОСЛЕ df!)
# ============================================================

DATE_FOR_MAP = df["date"].max()
print(f"Дата для карты риска: {DATE_FOR_MAP.date()}")

OUTPUT_PATH = (
    OUTPUT_DIR
    + f"\\risk_map_{DATE_FOR_MAP.date()}.geojson"
)

# ============================================================
# 4. ФИЛЬТРАЦИЯ ПО ДАТЕ
# ============================================================

map_df = df[df["date"] == DATE_FOR_MAP].copy()
print(f"Записей на дату: {len(map_df)}")

if map_df.empty:
    raise ValueError("Нет данных NDVI для выбранной даты")

# ============================================================
# 5. ЗАГРУЗКА СЕТКИ
# ============================================================

print("Загрузка сетки...")
grid = gpd.read_file(GRID_PATH)

# ============================================================
# 6. ОБУЧЕНИЕ BASELINE-МОДЕЛИ
# ============================================================

train_df = df[df["date"].dt.year <= 2021]

X_train = train_df[FEATURES]
y_train = train_df["fire"]

model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
        (
            "logreg",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                n_jobs=-1
            )
        )
    ]
)

print("Обучение baseline-модели...")
model.fit(X_train, y_train)

# ============================================================
# 7. РАСЧЁТ ВЕРОЯТНОСТЕЙ
# ============================================================

print("Расчёт вероятностей пожара...")
map_df["fire_risk"] = model.predict_proba(map_df[FEATURES])[:, 1]

# ============================================================
# 8. ОБЪЕДИНЕНИЕ С ГЕОМЕТРИЕЙ
# ============================================================

print("Объединение с геометрией...")
risk_map = grid.merge(
    map_df[["cell_id", "fire_risk"]],
    on="cell_id",
    how="left"
)

risk_map["fire_risk"] = risk_map["fire_risk"].fillna(0)

# ============================================================
# 9. СОХРАНЕНИЕ
# ============================================================

print("Сохранение карты риска...")
risk_map.to_file(OUTPUT_PATH, driver="GeoJSON")

print("Карта риска сохранена:", OUTPUT_PATH)
