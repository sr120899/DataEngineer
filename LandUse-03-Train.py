"""Step 3-5: split train/test, train the Random Forest, evaluate, save the model.

Loads the labeled feature table from BigQuery, splits it 70/30 with a seeded
random column, trains ee.Classifier.smileRandomForest(100, seed=42), reports
the confusion matrix / overall accuracy / kappa / variable importance, and
persists the trained trees to BigQuery via landuse_model_io (Step 7) so
LandUse-04-Map.py can reload the model without retraining.
"""

from __future__ import annotations

import datetime as dt
import json
import sys

import ee
from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery

import landuse_config as cfg
import landuse_model_io as model_io

CREATE_METRICS_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS `{cfg.BQ_METRICS_TABLE}` (
  model_version STRING NOT NULL,
  trained_at TIMESTAMP NOT NULL,
  train_count INT64,
  test_count INT64,
  overall_accuracy FLOAT64,
  kappa FLOAT64,
  confusion_matrix_json STRING,
  variable_importance_json STRING
)
"""


def load_training_samples(client: bigquery.Client):
    query = f"SELECT * FROM `{cfg.BQ_TRAINING_TABLE}`"
    return client.query(query).to_dataframe()


def samples_to_fc(df) -> ee.FeatureCollection:
    features = [
        ee.Feature(
            None,
            {"class_id": int(row.class_id), **{b: float(getattr(row, b)) for b in cfg.FEATURE_BANDS}},
        )
        for row in df.itertuples()
    ]
    return ee.FeatureCollection(features)


def main() -> None:
    bq_client = bigquery.Client(project=cfg.GCP_PROJECT_ID)
    try:
        df = load_training_samples(bq_client)
    except (GoogleAPIError, DefaultCredentialsError) as exc:
        print(f"BigQuery read failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print("No training samples found. Run LandUse-02-Features.py first.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(df)} training samples from {cfg.BQ_TRAINING_TABLE}")

    ee.Initialize(project=cfg.GCP_PROJECT_ID)
    fc = samples_to_fc(df).randomColumn("random", cfg.SEED)
    train = fc.filter(ee.Filter.lt("random", cfg.TRAIN_FRACTION))
    test = fc.filter(ee.Filter.gte("random", cfg.TRAIN_FRACTION))
    train_count = train.size().getInfo()
    test_count = test.size().getInfo()
    print(f"Split: {train_count} train / {test_count} test (target ratio {cfg.TRAIN_FRACTION:.0%}/{1 - cfg.TRAIN_FRACTION:.0%})")

    print(f"Training smileRandomForest(numberOfTrees={cfg.RF_NUM_TREES}, seed={cfg.SEED})...")
    classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=cfg.RF_NUM_TREES, seed=cfg.SEED
    ).train(features=train, classProperty="class_id", inputProperties=cfg.FEATURE_BANDS)

    classified_test = test.classify(classifier)
    matrix = classified_test.errorMatrix("class_id", "classification")
    accuracy = matrix.accuracy().getInfo()
    kappa = matrix.kappa().getInfo()
    raw_confusion = matrix.array().getInfo()
    raw_order = matrix.order().getInfo()
    importance = classifier.explain().get("importance").getInfo()

    # ee.ConfusionMatrix always spans class values 0..max(class_id). Our
    # class_id values are 1-4 (no 0), so raw_order/raw_confusion include a
    # phantom "class 0" row/col that's always zero — drop it before display.
    keep_idx = [i for i, cid in enumerate(raw_order) if cid in cfg.CLASS_IDS]
    class_order = [cfg.CLASS_NAMES[raw_order[i]] for i in keep_idx]
    confusion = [[raw_confusion[i][j] for j in keep_idx] for i in keep_idx]

    print("\n=== Evaluation (test set) ===")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Kappa: {kappa:.4f}")
    print(f"Confusion matrix (rows=actual, cols=predicted, order={class_order}):")
    for row in confusion:
        print("  " + " ".join(f"{v:5d}" for v in row))
    print("Variable importance:")
    for band, score in sorted(importance.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {band}: {score:.4f}")

    tree_count = model_io.save_trees_to_bq(classifier, cfg.MODEL_VERSION, client=bq_client)
    print(f"\nSaved {tree_count} trees to {cfg.BQ_MODEL_TABLE} as model_version={cfg.MODEL_VERSION!r}")

    bq_client.query(CREATE_METRICS_TABLE_SQL).result()
    metrics_row = {
        "model_version": cfg.MODEL_VERSION,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "train_count": train_count,
        "test_count": test_count,
        "overall_accuracy": accuracy,
        "kappa": kappa,
        "confusion_matrix_json": json.dumps({"class_order": class_order, "matrix": confusion}),
        "variable_importance_json": json.dumps(importance),
    }
    bq_client.load_table_from_json(
        [metrics_row], cfg.BQ_METRICS_TABLE, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    ).result()
    print(f"Logged run metrics to {cfg.BQ_METRICS_TABLE}")


if __name__ == "__main__":
    main()
