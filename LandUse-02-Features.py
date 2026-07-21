"""Step 2: extract Sentinel-2 features (Xs) at the label points from Step 1.

Reads the stratified sample points from BigQuery, builds a cloud-masked
Sentinel-2 median composite over the Ayutthaya AOI with 4 spectral indices
added, samples the 10 feature bands at each point, and writes the combined
label+feature table back to BigQuery for the training step.

Prerequisites:
    Run LandUse-01-Labels.py first.
    earthengine-api authenticated: `earthengine authenticate` (once per machine)
    gcloud auth application-default login / set-quota-project (for BigQuery)
"""

from __future__ import annotations

import sys

import ee
import pandas as pd
from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery

import landuse_config as cfg
from landuse_features import build_feature_image


def load_points(client: bigquery.Client) -> pd.DataFrame:
    query = f"""
    SELECT point_id, lon, lat, class_id, class_name
    FROM `{cfg.BQ_LABELS_TABLE}`
    """
    return client.query(query).to_dataframe()


def points_to_fc(points: pd.DataFrame) -> ee.FeatureCollection:
    features = [
        ee.Feature(
            ee.Geometry.Point([row.lon, row.lat]),
            {"point_id": int(row.point_id), "class_id": int(row.class_id)},
        )
        for row in points.itertuples()
    ]
    return ee.FeatureCollection(features)


def sample_features(feature_image: ee.Image, points_fc: ee.FeatureCollection) -> pd.DataFrame:
    sampled = feature_image.sampleRegions(
        collection=points_fc,
        properties=["point_id", "class_id"],
        scale=cfg.S2_SCALE,
        geometries=False,
    )
    info = sampled.getInfo()
    rows = [f["properties"] for f in info["features"]]
    return pd.DataFrame(rows)


def main() -> None:
    bq_client = bigquery.Client(project=cfg.GCP_PROJECT_ID)
    try:
        points = load_points(bq_client)
    except (GoogleAPIError, DefaultCredentialsError) as exc:
        print(f"BigQuery read failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if points.empty:
        print("No label points found. Run LandUse-01-Labels.py first.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(points)} label points from {cfg.BQ_LABELS_TABLE}")

    ee.Initialize(project=cfg.GCP_PROJECT_ID)
    aoi = ee.Geometry.Rectangle([cfg.AOI_WEST, cfg.AOI_SOUTH, cfg.AOI_EAST, cfg.AOI_NORTH])
    points_fc = points_to_fc(points)
    feature_image = build_feature_image(aoi)

    print("Sampling Sentinel-2 features at each point (this calls Earth Engine)...")
    sampled = sample_features(feature_image, points_fc)
    print(f"Sampled {len(sampled)} / {len(points)} points (some may fall on masked/cloudy pixels)")

    null_counts = sampled[cfg.FEATURE_BANDS].isna().sum()
    dropped = sampled[cfg.FEATURE_BANDS].isna().any(axis=1).sum()
    if dropped:
        print(f"Dropping {dropped} points with a null feature value:")
        for band, n in null_counts[null_counts > 0].items():
            print(f"  {band}: {n} nulls")
        sampled = sampled.dropna(subset=cfg.FEATURE_BANDS)

    # class_name isn't returned by sampleRegions (only properties we asked for);
    # bring it back for readability in the output table.
    merged = sampled.merge(points[["point_id", "class_name"]], on="point_id", how="left")
    merged = merged[["point_id", "class_id", "class_name", *cfg.FEATURE_BANDS]]

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    bq_client.load_table_from_dataframe(merged, cfg.BQ_TRAINING_TABLE, job_config=job_config).result()

    print(f"Wrote {len(merged)} training samples to {cfg.BQ_TRAINING_TABLE}")
    print(merged.groupby("class_name").size().to_string())


if __name__ == "__main__":
    main()
