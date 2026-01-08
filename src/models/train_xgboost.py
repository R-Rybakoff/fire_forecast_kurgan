import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

print("Загрузка ML-датасета...")
df = pd.read_parquet(
    r"C:\Users\i-ryb\Desktop\fire_forecast_kurgan\data_processed\ml_dataset.parquet"
)

df["date"] = pd.to_datetime(df["date"])

# ============================================================
# 2. ВРЕМЕННОЕ РАЗБИЕНИЕ
# ============================================================

train_df = df[df["date"].dt.year <= 2021]
valid_df = df[df["date"].dt.year == 2022]
test_df  = df[df["date"].dt.year == 2023]

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
# 3. МОДЕЛЬ XGBOOST
# ============================================================

# отношение отрицательных к положительным (для дисбаланса)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",  # важно для редких событий
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

# ============================================================
# 4. ОБУЧЕНИЕ
# ============================================================

print("Обучение XGBoost...")
model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=True
)

# ============================================================
# 5. ОЦЕНКА
# ============================================================

def evaluate(X, y, name):
    proba = model.predict_proba(X)[:, 1]

    roc = roc_auc_score(y, proba)
    pr  = average_precision_score(y, proba)

    print(f"{name}: ROC-AUC = {roc:.4f}, PR-AUC = {pr:.6f}")

print("\nРезультаты XGBoost:")
evaluate(X_train, y_train, "Train")
evaluate(X_valid, y_valid, "Valid")
evaluate(X_test,  y_test,  "Test")
