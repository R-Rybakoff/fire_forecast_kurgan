import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ============================================================
# 1. ПАРАМЕТРЫ
# ============================================================

ML_DATASET_PATH = r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed\ml_dataset.parquet"

TOP_K_LIST = [0.001, 0.005, 0.01, 0.02]  # 0.1%, 0.5%, 1%, 2%

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
# 2. ЗАГРУЗКА ДАННЫХ
# ============================================================

print("Загрузка ML-датасета...")
df = pd.read_parquet(ML_DATASET_PATH)
df["date"] = pd.to_datetime(df["date"])

# оставляем только даты, где был пожар
fires = df[df["fire"] == 1][["cell_id", "date"]].copy()
fires = fires.sort_values("date")

print(f"Всего пожаров: {len(fires)}")

# все доступные даты NDVI
ndvi_dates = np.sort(df["date"].unique())

# ============================================================
# 3. ОБУЧЕНИЕ BASELINE-МОДЕЛИ
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
# 4. АГРЕГИРОВАННАЯ ПРОВЕРКА
# ============================================================

results = []

for _, row in tqdm(fires.iterrows(), total=len(fires)):
    fire_date = row["date"]
    fire_cell = row["cell_id"]

    # последняя дата NDVI ДО пожара
    ndvi_before = ndvi_dates[ndvi_dates < fire_date]
    if len(ndvi_before) == 0:
        continue

    ndvi_date = ndvi_before[-1]

    risk_df = df[df["date"] == ndvi_date].copy()
    if risk_df.empty:
        continue

    # считаем риск
    risk_df["fire_risk"] = model.predict_proba(risk_df[FEATURES])[:, 1]
    risk_df = risk_df.sort_values("fire_risk", ascending=False)

    total_cells = len(risk_df)

    for top_k in TOP_K_LIST:
        cutoff = int(total_cells * top_k)
        top_cells = set(risk_df.iloc[:cutoff]["cell_id"])

        hit = int(fire_cell in top_cells)

        results.append({
            "fire_date": fire_date,
            "ndvi_date": ndvi_date,
            "top_k": top_k,
            "hit": hit
        })

# ============================================================
# 5. АГРЕГАЦИЯ РЕЗУЛЬТАТОВ
# ============================================================

results_df = pd.DataFrame(results)

summary = (
    results_df
    .groupby("top_k")["hit"]
    .agg(["sum", "count", "mean"])
    .reset_index()
)

summary["Top_K_percent"] = (summary["top_k"] * 100).astype(str) + "%"
summary = summary.rename(
    columns={
        "sum": "Fires_found",
        "count": "Total_fires",
        "mean": "Recall"
    }
)

summary = summary[
    ["Top_K_percent", "Fires_found", "Total_fires", "Recall"]
]

print("\nАгрегированная проверка по всем пожарам:")
print(summary)
