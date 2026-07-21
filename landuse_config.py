"""Shared constants for the Ayutthaya land-use Random Forest pipeline.

Imported by every LandUse-*.py script and landuse_model_io.py so the AOI,
class scheme, seeds, and table names stay consistent across the pipeline.
"""

from __future__ import annotations

# --- GCP / BigQuery ---------------------------------------------------------
GCP_PROJECT_ID = "tidy-nomad-470808-e1"
BQ_DATASET = "landuse"
BQ_LABELS_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.sample_points"
BQ_TRAINING_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.training_samples"
BQ_METRICS_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.model_metrics"
BQ_MODEL_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.rf_model_trees"

OSM_TABLE = "bigquery-public-data.geo_openstreetmap.planet_features"

# --- AOI: Ayutthaya bounding box --------------------------------------------
AOI_WEST = 100.45
AOI_EAST = 100.65
AOI_SOUTH = 14.25
AOI_NORTH = 14.45
AOI_WKT_POLYGON = (
    f"POLYGON(({AOI_WEST} {AOI_SOUTH}, {AOI_EAST} {AOI_SOUTH}, "
    f"{AOI_EAST} {AOI_NORTH}, {AOI_WEST} {AOI_NORTH}, {AOI_WEST} {AOI_SOUTH}))"
)

# --- Reproducibility ---------------------------------------------------------
SEED = 42

# --- Land-use classes ---------------------------------------------------------
# class_id -> (short_name used in code/SQL, Thai label, map color)
CLASSES = {
    1: ("water", "น้ำ", "blue"),
    2: ("urban", "เมือง", "red"),
    3: ("agri", "เกษตร", "yellow"),
    4: ("tree", "ไม้ยืนต้น", "green"),
}
CLASS_IDS = list(CLASSES.keys())
CLASS_NAMES = {cid: names[0] for cid, names in CLASSES.items()}
CLASS_LABELS_TH = {cid: names[1] for cid, names in CLASSES.items()}
CLASS_COLORS = {cid: names[2] for cid, names in CLASSES.items()}
N_PER_CLASS = 200

# SQL CASE expression mapping OSM tags (from `all_tags` on `planet_features`,
# feature_type='multipolygons') to a class_id. Validated against the AOI on
# BigQuery: water=784, urban=46787, agri=230, tree=367 candidate polygons,
# all comfortably >= N_PER_CLASS. `landuse=orchard` is treated as agri.
OSM_CLASS_CASE_SQL = """
  CASE
    WHEN natural_tag = 'water' OR waterway_tag IS NOT NULL OR landuse_tag = 'reservoir' THEN 1
    WHEN building_tag IS NOT NULL OR landuse_tag IN ('residential', 'commercial', 'industrial') THEN 2
    WHEN landuse_tag IN ('farmland', 'meadow', 'orchard', 'vineyard') THEN 3
    WHEN landuse_tag = 'forest' OR natural_tag = 'wood' THEN 4
    ELSE NULL
  END
"""

# --- Sentinel-2 features -----------------------------------------------------
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_DATE_START = "2023-01-01"
S2_DATE_END = "2023-12-31"
S2_MAX_CLOUD_PCT = 20
S2_SCALE = 10  # meters

S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
INDEX_BANDS = ["NDVI", "EVI", "NDWI", "NDBI"]
FEATURE_BANDS = S2_BANDS + INDEX_BANDS  # 10 features total

# --- Random Forest ------------------------------------------------------------
TRAIN_FRACTION = 0.7
RF_NUM_TREES = 100
MODEL_VERSION = "rf_v1"  # identifies the saved tree ensemble in BQ_MODEL_TABLE
