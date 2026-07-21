"""Step 7: persist/reload the trained Random Forest as decision-tree text rows.

ee.Classifier.explain() exposes the trained smileRandomForest model as a list
of Weka-format decision-tree strings (classifier.explain()['trees']). Each
tree is stored as one BigQuery row so the classifier can be rebuilt later
with ee.Classifier.decisionTreeEnsemble(trees) without retraining.
"""

from __future__ import annotations

import ee
import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import bigquery

import landuse_config as cfg

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS `{cfg.BQ_MODEL_TABLE}` (
  model_version STRING NOT NULL,
  tree_index INT64 NOT NULL,
  tree_string STRING NOT NULL
)
"""


def save_trees_to_bq(
    classifier: ee.Classifier, model_version: str, client: bigquery.Client | None = None
) -> int:
    """Write every tree of a trained classifier to BigQuery under model_version.

    Any existing rows for the same model_version are replaced. Returns the
    number of trees written.
    """
    client = client or bigquery.Client(project=cfg.GCP_PROJECT_ID)
    client.query(CREATE_TABLE_SQL).result()

    trees = classifier.explain().get("trees").getInfo()
    new_rows = pd.DataFrame(
        {
            "model_version": model_version,
            "tree_index": range(len(trees)),
            "tree_string": trees,
        }
    )

    # The project's free tier rejects DML (DELETE/UPDATE/MERGE), so instead of
    # "DELETE ... WHERE model_version = ..." we read the table, drop this
    # version's old rows in pandas, and rewrite the whole table via a load
    # job (WRITE_TRUNCATE), which isn't DML and works on the free tier.
    try:
        existing = client.query(
            f"SELECT model_version, tree_index, tree_string FROM `{cfg.BQ_MODEL_TABLE}`"
        ).to_dataframe()
        existing = existing[existing["model_version"] != model_version]
    except NotFound:
        existing = pd.DataFrame(columns=["model_version", "tree_index", "tree_string"])

    combined = pd.concat([existing, new_rows], ignore_index=True)
    client.load_table_from_dataframe(
        combined, cfg.BQ_MODEL_TABLE, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    ).result()
    return len(trees)


def load_trees_from_bq(model_version: str, client: bigquery.Client | None = None) -> list[str]:
    """Return the tree strings for model_version, ordered by tree_index."""
    client = client or bigquery.Client(project=cfg.GCP_PROJECT_ID)
    query = f"""
    SELECT tree_string
    FROM `{cfg.BQ_MODEL_TABLE}`
    WHERE model_version = @model_version
    ORDER BY tree_index
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("model_version", "STRING", model_version)]
    )
    rows = list(client.query(query, job_config=job_config).result())
    if not rows:
        raise ValueError(f"No trees found for model_version={model_version!r} in {cfg.BQ_MODEL_TABLE}")
    return [row.tree_string for row in rows]


def load_classifier_from_bq(model_version: str, client: bigquery.Client | None = None) -> ee.Classifier:
    """Rebuild a trained classifier from its saved trees, with no retraining."""
    trees = load_trees_from_bq(model_version, client=client)
    return ee.Classifier.decisionTreeEnsemble(trees)
