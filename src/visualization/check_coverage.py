import geopandas as gpd

GRID_FULL = "data_interim/grids/grid_500m_kurgan.geojson"
GRID_RISK = "data_processed/visualization/fire_risk_with_y_kurgan.geojson"

print("Loading grids...")
g_full = gpd.read_file(GRID_FULL)
g_risk = gpd.read_file(GRID_RISK)

# =========================
# 1. BASIC COUNTS
# =========================
n_full = len(g_full)
n_risk = len(g_risk)

print("\n=== BASIC COUNTS ===")
print(f"Full grid cells : {n_full}")
print(f"Risk grid cells : {n_risk}")

# =========================
# 2. CELL_ID COVERAGE
# =========================
ids_full = set(g_full["cell_id"])
ids_risk = set(g_risk["cell_id"])

missing_ids = ids_full - ids_risk
extra_ids = ids_risk - ids_full

print("\n=== CELL_ID CHECK ===")
print(f"Missing cell_id in risk grid : {len(missing_ids)}")
print(f"Extra cell_id in risk grid   : {len(extra_ids)}")

# =========================
# 3. FIRE_RISK COVERAGE
# =========================
has_risk = g_risk["fire_risk"].notna().sum()
no_risk = g_risk["fire_risk"].isna().sum()

print("\n=== FIRE_RISK COVERAGE ===")
print(f"Cells with fire_risk : {has_risk}")
print(f"Cells without risk   : {no_risk}")
print(f"Share with risk (%)  : {has_risk / n_risk * 100:.1f}")

# =========================
# 4. GEOMETRY CHECK
# =========================
print("\n=== GEOMETRY CHECK ===")
print("Full grid bounds :", g_full.total_bounds)
print("Risk grid bounds :", g_risk.total_bounds)

# =========================
# 5. FINAL CONCLUSION
# =========================
print("\n=== CONCLUSION ===")
if len(missing_ids) == 0 and n_full == n_risk:
    print("Вся геометрия области присутствует.")
else:
    print("Геометрия области неполная (ошибка сборки).")

print("ℹ Ячейки без fire_risk = область вне применимости модели.")


