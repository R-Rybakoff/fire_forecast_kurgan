import pandas as pd
import numpy as np

# =========================
# PATHS
# =========================
ML_DATASET_PATH = "data_processed/ml_dataset_with_fire_risk.parquet"
INFRA_PATH = "data_interim/infrastructure/grid_with_infrastructure.parquet"

OUT_PATH = "data_processed/ml_dataset_with_priority.parquet"

# =========================
# PARAMETERS
# =========================
ALPHA = 0.7   # weight for settlements
BETA = 0.3    # weight for roads

# =========================
# LOAD DATA
# =========================
print("Loading ML dataset...")
df = pd.read_parquet(ML_DATASET_PATH)

print("Loading infrastructure features...")
infra = pd.read_parquet(INFRA_PATH)

# =========================
# MERGE
# =========================
print("Merging datasets...")
df = df.merge(
    infra[["cell_id", "log_dist_settlement", "log_dist_road"]],
    on="cell_id",
    how="left"
)

# =========================
# CHECK
# =========================
assert "fire_risk" in df.columns, "fire_risk not found"
assert df["log_dist_settlement"].notna().mean() > 0.9, "Too many NaNs in settlement distance"
assert df["log_dist_road"].notna().mean() > 0.9, "Too many NaNs in road distance"

# =========================
# PRIORITY SCORE
# =========================
print("Calculating priority score...")

df["settlement_risk"] = 1.0 / (1.0 + df["log_dist_settlement"])
df["road_risk"] = 1.0 / (1.0 + df["log_dist_road"])

df["priority_score"] = (
    df["fire_risk"] *
    (ALPHA * df["settlement_risk"] + BETA * df["road_risk"])
)

# =========================
# NORMALIZE (OPTIONAL BUT NICE)
# =========================
df["priority_score"] = (
    df["priority_score"] - df["priority_score"].min()
) / (
    df["priority_score"].max() - df["priority_score"].min()
)

# =========================
# SAVE
# =========================
print("Saving dataset with priority score...")
df.to_parquet(OUT_PATH)

print("Priority score calculated and saved:")
print(OUT_PATH)

# =========================
# QUICK STATS
# =========================
print("\nPriority score stats:")
print(df["priority_score"].describe())
