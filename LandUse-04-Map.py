"""Step 6: classify the full AOI and render an interactive HTML map.

Reloads the trained classifier from BigQuery (Step 7's saved decision trees
-- no retraining), classifies the whole Sentinel-2 composite, and builds a
folium map with 3 layer groups: S2 true-color, the 4-class classification,
and the training sample points (as real toggleable markers). Saved as a
standalone HTML file.

Uses plain folium (not geemap.Map/geemap.foliumap.Map): geemap's default
ipyleaflet-backed Map.to_html() bakes a frozen tile snapshot outside Jupyter
(broken pan/zoom, duplicated controls), and geemap.foliumap.Map fails to
import in this environment (xyzservices/geemap version mismatch). Plain
folium with ee.Image.getMapId() tile URLs is the same pattern already proven
working in BigQuery-Task.py / openaq_thailand_map.html in this repo.

Prerequisites:
    Run LandUse-01/02/03 first so BQ_MODEL_TABLE has a saved model.
    earthengine-api authenticated; BigQuery ADC configured.

Note: the exported HTML is NOT fully offline -- the S2 and classified layers
are Earth Engine tiles fetched live from Google's servers, so viewing the map
still requires internet access and a valid/authorized GEE project.
"""

from __future__ import annotations

import sys

import ee
import folium
from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery

import landuse_config as cfg
import landuse_model_io as model_io
from landuse_features import build_feature_image

OUTPUT_HTML = "landuse_ayutthaya_map.html"


def load_sample_points(client: bigquery.Client):
    query = f"SELECT point_id, lon, lat, class_id, class_name FROM `{cfg.BQ_LABELS_TABLE}`"
    return client.query(query).to_dataframe()


def ee_tile_layer(image: ee.Image, vis_params: dict, name: str) -> folium.TileLayer:
    map_id_dict = image.getMapId(vis_params)
    return folium.TileLayer(
        tiles=map_id_dict["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
    )


def build_map(classified: ee.Image, s2_composite: ee.Image, points_df) -> folium.Map:
    center_lat = (cfg.AOI_SOUTH + cfg.AOI_NORTH) / 2
    center_lon = (cfg.AOI_WEST + cfg.AOI_EAST) / 2
    fmap = folium.Map(location=[center_lat, center_lon], tiles="cartodbpositron")
    fmap.fit_bounds([[cfg.AOI_SOUTH, cfg.AOI_WEST], [cfg.AOI_NORTH, cfg.AOI_EAST]])

    true_color_vis = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 0.3}
    ee_tile_layer(s2_composite, true_color_vis, "Sentinel-2 True Color").add_to(fmap)

    class_ids_sorted = sorted(cfg.CLASS_IDS)
    classified_vis = {
        "min": min(class_ids_sorted),
        "max": max(class_ids_sorted),
        "palette": [cfg.CLASS_COLORS[cid] for cid in class_ids_sorted],
    }
    ee_tile_layer(classified, classified_vis, "Land Use Classification").add_to(fmap)

    for cid in class_ids_sorted:
        subset = points_df[points_df["class_id"] == cid]
        if subset.empty:
            continue
        color = cfg.CLASS_COLORS[cid]
        group = folium.FeatureGroup(name=f"Samples: {cfg.CLASS_NAMES[cid]}", show=True)
        for row in subset.itertuples():
            folium.CircleMarker(
                location=[row.lat, row.lon],
                radius=4,
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                tooltip=f"{cfg.CLASS_LABELS_TH[cid]} ({cfg.CLASS_NAMES[cid]}) point_id={row.point_id}",
            ).add_to(group)
        group.add_to(fmap)

    legend_items = "".join(
        f'<span style="color:{cfg.CLASS_COLORS[cid]}">&#9679;</span> '
        f"{cfg.CLASS_LABELS_TH[cid]} ({cfg.CLASS_NAMES[cid]})<br>"
        for cid in class_ids_sorted
    )
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border: 1px solid #999;
                border-radius: 4px; font-size: 13px; line-height: 1.6;">
      <b>Land Use / การใช้ที่ดิน</b><br>{legend_items}
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def main() -> None:
    bq_client = bigquery.Client(project=cfg.GCP_PROJECT_ID)
    try:
        points_df = load_sample_points(bq_client)
    except (GoogleAPIError, DefaultCredentialsError) as exc:
        print(f"BigQuery read failed: {exc}", file=sys.stderr)
        sys.exit(1)

    ee.Initialize(project=cfg.GCP_PROJECT_ID)

    print(f"Loading model_version={cfg.MODEL_VERSION!r} from {cfg.BQ_MODEL_TABLE} (no retraining)...")
    classifier = model_io.load_classifier_from_bq(cfg.MODEL_VERSION, client=bq_client)

    aoi = ee.Geometry.Rectangle([cfg.AOI_WEST, cfg.AOI_SOUTH, cfg.AOI_EAST, cfg.AOI_NORTH])
    feature_image = build_feature_image(aoi)

    print("Classifying the full AOI...")
    classified = feature_image.classify(classifier).clip(aoi)

    fmap = build_map(classified, feature_image, points_df)
    fmap.save(OUTPUT_HTML)
    print(f"Map written to {OUTPUT_HTML} ({len(points_df)} sample points overlaid)")


if __name__ == "__main__":
    main()
