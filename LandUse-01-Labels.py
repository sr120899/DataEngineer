"""Step 1: build stratified label points (Y) for the Ayutthaya land-use RF model.

Pulls closed polygons (feature_type='multipolygons') from the public OSM
BigQuery dataset, tags them into 4 land-use classes (water/urban/agri/tree),
takes a seeded stratified sample of N_PER_CLASS points per class, and writes
the result to a BigQuery table for the next pipeline steps to consume.

Data source : bigquery-public-data.geo_openstreetmap.planet_features
Billing project : tidy-nomad-470808-e1

Prerequisites:
    pip install -r requirements.txt
    gcloud auth application-default login
    gcloud auth application-default set-quota-project tidy-nomad-470808-e1
"""

from __future__ import annotations

import sys

from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery

import landuse_config as cfg

# BigQuery's RAND() has no seed argument, so reproducible sampling uses a
# deterministic hash of osm_id + SEED as the ORDER BY key instead.
CLASS_NAME_CASE_SQL = "CASE " + " ".join(
    f"WHEN class_id = {cid} THEN '{name}'" for cid, name in cfg.CLASS_NAMES.items()
) + " END"

QUERY_TEMPLATE = f"""
CREATE OR REPLACE TABLE `{cfg.BQ_LABELS_TABLE}` AS
WITH aoi AS (
  SELECT ST_GEOGFROMTEXT('{cfg.AOI_WKT_POLYGON}') AS geom
),
tagged AS (
  SELECT
    -- feature_type='multipolygons' rows are relations: osm_id is NULL for
    -- almost all of them and osm_way_id carries the identifier instead.
    COALESCE(f.osm_id, f.osm_way_id) AS feature_id,
    f.geometry,
    (SELECT value FROM UNNEST(f.all_tags) WHERE key = 'natural') AS natural_tag,
    (SELECT value FROM UNNEST(f.all_tags) WHERE key = 'waterway') AS waterway_tag,
    (SELECT value FROM UNNEST(f.all_tags) WHERE key = 'landuse') AS landuse_tag,
    (SELECT value FROM UNNEST(f.all_tags) WHERE key = 'building') AS building_tag
  FROM `{cfg.OSM_TABLE}` f, aoi
  WHERE f.feature_type = 'multipolygons'
    AND ST_INTERSECTS(f.geometry, aoi.geom)
),
classified AS (
  SELECT
    feature_id,
    geometry,
    {cfg.OSM_CLASS_CASE_SQL} AS class_id
  FROM tagged
),
ranked AS (
  SELECT
    feature_id,
    class_id,
    ST_X(ST_CENTROID(geometry)) AS lon,
    ST_Y(ST_CENTROID(geometry)) AS lat,
    ROW_NUMBER() OVER (
      PARTITION BY class_id
      ORDER BY FARM_FINGERPRINT(CONCAT(CAST(feature_id AS STRING), '_{cfg.SEED}'))
    ) AS rn
  FROM classified
  WHERE class_id IS NOT NULL
)
SELECT
  feature_id AS point_id,
  lon,
  lat,
  class_id,
  {CLASS_NAME_CASE_SQL} AS class_name
FROM ranked
WHERE rn <= {cfg.N_PER_CLASS}
"""

COUNT_QUERY = f"""
SELECT class_id, class_name, COUNT(*) AS n
FROM `{cfg.BQ_LABELS_TABLE}`
GROUP BY class_id, class_name
ORDER BY class_id
"""


def ensure_dataset(client: bigquery.Client) -> None:
    dataset_ref = bigquery.DatasetReference(cfg.GCP_PROJECT_ID, cfg.BQ_DATASET)
    client.create_dataset(dataset_ref, exists_ok=True)


def build_labels(client: bigquery.Client) -> None:
    client.query(QUERY_TEMPLATE).result()


def fetch_class_counts(client: bigquery.Client):
    return list(client.query(COUNT_QUERY).result())


def main() -> None:
    client = bigquery.Client(project=cfg.GCP_PROJECT_ID)
    try:
        ensure_dataset(client)
        build_labels(client)
        counts = fetch_class_counts(client)
    except (GoogleAPIError, DefaultCredentialsError) as exc:
        print(f"BigQuery job failed: {exc}", file=sys.stderr)
        print(
            "Hint: run 'gcloud auth application-default login' and "
            f"'gcloud auth application-default set-quota-project {cfg.GCP_PROJECT_ID}'",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Wrote label points to {cfg.BQ_LABELS_TABLE}")
    total = 0
    for row in counts:
        expected = cfg.N_PER_CLASS
        flag = "" if row.n == expected else f"  <-- expected {expected}!"
        print(f"  class_id={row.class_id} ({row.class_name}): {row.n} points{flag}")
        total += row.n
    print(f"Total: {total} points")


if __name__ == "__main__":
    main()
