import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

print("Загрузка ML-датасета...")
df = pd.read_parquet(
    r"D:\fire_forecast_kurgan\data_processed\ml_dataset.parquet"
)

df["date"] = pd.to_datetime(df["date"])

# ============================================================
# 2. ВРЕМЕННОЕ РАЗБИЕНИЕ
# ============================================================

train_df = df[df["date"].dt.year <= 2021]
valid_df = df[df["date"].dt.year == 2022]
test_df  = df[df["date"].dt.year == 2023]

print("Размеры выборок:")
print("train:", train_df.shape)
print("valid:", valid_df.shape)
print("test :", test_df.shape)

# ============================================================
# 3. ПРИЗНАКИ И ЦЕЛЬ
# ============================================================

features = [
    "ndvi_mean",
    "ndvi_lag_7",
    "ndvi_lag_14",
    "ndvi_lag_30",
    "ndvi_mean_14",
    "ndvi_std_30",
    "ndvi_delta_7",
    "ndvi_anomaly"
]

X_train = train_df[features]
y_train = train_df["fire"]

X_valid = valid_df[features]
y_valid = valid_df["fire"]

X_test  = test_df[features]
y_test  = test_df["fire"]

# ============================================================
# 4. PIPELINE: SCALER + LOGREG
# ============================================================

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


# ============================================================
# 5. ОБУЧЕНИЕ
# ============================================================

print("Обучение модели...")
model.fit(X_train, y_train)

# ============================================================
# 6. ОЦЕНКА
# ============================================================

def evaluate(X, y, name):
    proba = model.predict_proba(X)[:, 1]

    roc = roc_auc_score(y, proba)
    pr  = average_precision_score(y, proba)

    print(f"{name}: ROC-AUC = {roc:.4f}, PR-AUC = {pr:.6f}")

print("\nРезультаты:")
evaluate(X_train, y_train, "Train")
evaluate(X_valid, y_valid, "Valid")
evaluate(X_test,  y_test,  "Test")
