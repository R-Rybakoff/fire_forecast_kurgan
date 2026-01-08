import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# =========================
# PATHS
# =========================
DATASET_PATH = "data_processed/ml_dataset_ndvi_weather.parquet"
OUT_PATH = "data_processed/ml_dataset_with_fire_risk.parquet"

TARGET = "y"
ID_COLS = ["cell_id", "date"]

# =========================
# LOAD
# =========================
print("Loading dataset...")
df = pd.read_parquet(DATASET_PATH)

# =========================
# FEATURES
# =========================
exclude = ID_COLS + [TARGET]
features = [c for c in df.columns if c not in exclude]

X = df[features]
y = df[TARGET]

# =========================
# MODEL (baseline + imputer)
# =========================
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    ))
])

# =========================
# TRAIN
# =========================
print("Training model...")
model.fit(X, y)

# =========================
# PREDICT RISK
# =========================
print("Calculating fire risk...")
df["fire_risk"] = model.predict_proba(X)[:, 1]

# =========================
# SAVE
# =========================
print("Saving dataset with fire_risk...")
df.to_parquet(OUT_PATH)

print("Saved:", OUT_PATH)
