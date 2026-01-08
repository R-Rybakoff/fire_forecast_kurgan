import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ============================================================
# 1. КОНСТАНТЫ
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

# ============================================================
# 3. ВЫБОР ПОЖАРА И БЛИЖАЙШЕГО NDVI ДО НЕГО
# ============================================================

# последний день с пожаром
DATE_FIRE = df[df["fire"] == 1]["date"].max()

if pd.isna(DATE_FIRE):
    raise ValueError("В датасете нет пожаров")

# ищем последнюю дату NDVI ДО пожара
ndvi_dates = df["date"].sort_values().unique()
ndvi_dates_before_fire = ndvi_dates[ndvi_dates < DATE_FIRE]

if len(ndvi_dates_before_fire) == 0:
    raise ValueError("Нет NDVI до пожара")

DATE_RISK = ndvi_dates_before_fire[-1]

print(f"Дата пожара (fire): {DATE_FIRE.date()}")
print(f"Дата NDVI для риска: {DATE_RISK.date()}")

# ============================================================
# 4. ДАННЫЕ ДЛЯ СРАВНЕНИЯ
# ============================================================

risk_df = df[df["date"] == DATE_RISK].copy()
fire_df = df[
    (df["date"] == DATE_FIRE) &
    (df["fire"] == 1)
].copy()

print(f"Ячеек с NDVI: {len(risk_df)}")
print(f"Реальных пожаров: {len(fire_df)}")

if risk_df.empty:
    raise ValueError("Пустой NDVI-срез")

# ============================================================
# 5. ОБУЧЕНИЕ BASELINE-МОДЕЛИ
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
# 6. РАСЧЁТ РИСКА
# ============================================================

print("Расчёт вероятностей пожара...")
risk_df["fire_risk"] = model.predict_proba(risk_df[FEATURES])[:, 1]
risk_df = risk_df.sort_values("fire_risk", ascending=False)

# ============================================================
# 7. СРАВНЕНИЕ
# ============================================================

results = []

total_cells = len(risk_df)
total_fires = len(fire_df)

for top_k in TOP_K_LIST:
    cutoff = int(total_cells * top_k)
    top_cells = set(risk_df.iloc[:cutoff]["cell_id"])

    found_fires = fire_df[fire_df["cell_id"].isin(top_cells)]
    recall = len(found_fires) / total_fires if total_fires > 0 else np.nan

    results.append({
        "Top_K_percent": f"{top_k*100:.1f}%",
        "Cells_checked": cutoff,
        "Fires_found": len(found_fires),
        "Total_fires": total_fires,
        "Recall": round(recall, 3)
    })

results_df = pd.DataFrame(results)

# ============================================================
# 8. ВЫВОД
# ============================================================

print("\nСравнение карты риска с реальными пожарами:")
print(results_df)
