"""Query OpenAQ air-quality data from BigQuery and render an HTML interactive map.

Data source : bigquery-public-data.openaq.global_air_quality
Billing project : tidy-nomad-470808-e1

Prerequisites:
    pip install -r requirements.txt
    gcloud auth application-default login
    gcloud auth application-default set-quota-project tidy-nomad-470808-e1
"""

from __future__ import annotations

import sys

import folium
import pandas as pd
from folium.plugins import MarkerCluster
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError

GEE_PROJECT_ID = "tidy-nomad-470808-e1"
COUNTRY_FILTER = "TH"
OUTPUT_HTML = "openaq_thailand_map.html"

# [[south, west], [north, east]] - covers Thailand's full extent so the initial
# view is consistent even if the current data doesn't reach every corner of the country.
THAILAND_BOUNDS = [[5.6, 97.3], [20.5, 105.7]]

# Latest reading per station per pollutant, restricted to Thailand, to keep the
# result set (and therefore the map) small even though the source table holds
# many millions of rows worldwide.
QUERY = """
WITH ranked AS (
  SELECT
    location, city, country, pollutant, value, unit,
    timestamp, latitude, longitude, source_name, averaged_over_in_hours,
    ROW_NUMBER() OVER (
      PARTITION BY location, pollutant ORDER BY timestamp DESC
    ) AS rn
  FROM `bigquery-public-data.openaq.global_air_quality`
  WHERE country = @country
    AND latitude IS NOT NULL
    AND longitude IS NOT NULL
)
SELECT location, city, country, pollutant, value, unit,
       timestamp, latitude, longitude, source_name, averaged_over_in_hours
FROM ranked
WHERE rn = 1
"""

# PM2.5 (ug/m3) breakpoints -> (color, label), loosely following US EPA AQI categories.
PM25_BREAKPOINTS = [
    (12.0, "green", "Good"),
    (35.4, "yellow", "Moderate"),
    (55.4, "orange", "Unhealthy (sensitive groups)"),
    (150.4, "red", "Unhealthy"),
    (250.4, "purple", "Very Unhealthy"),
    (float("inf"), "darkred", "Hazardous"),
]


def fetch_data() -> pd.DataFrame:
    """Run the parameterized BigQuery query and return the result as a DataFrame."""
    client = bigquery.Client(project=GEE_PROJECT_ID)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("country", "STRING", COUNTRY_FILTER),
        ]
    )
    return client.query(QUERY, job_config=job_config).to_dataframe()


def classify_pm25(value: float) -> tuple[str, str]:
    """Return (color, label) for a PM2.5 concentration in ug/m3."""
    for threshold, color, label in PM25_BREAKPOINTS:
        if value <= threshold:
            return color, label
    return "gray", "Unknown"


def build_popup_html(station: str, city: str, readings: pd.DataFrame) -> str:
    rows = "".join(
        f"<tr><td>{r.pollutant}</td><td>{r.value:.2f} {r.unit}</td>"
        f"<td>{r.timestamp}</td></tr>"
        for r in readings.itertuples()
    )
    return (
        f"<b>{station}</b><br>{city}<br>"
        '<table border="1" cellspacing="0" cellpadding="3" style="margin-top:4px">'
        "<tr><th>Pollutant</th><th>Value</th><th>Timestamp (UTC)</th></tr>"
        f"{rows}</table>"
    )


def build_map(df: pd.DataFrame) -> folium.Map:
    fmap = folium.Map(tiles="cartodbpositron")
    fmap.fit_bounds(THAILAND_BOUNDS)
    cluster = MarkerCluster(name="OpenAQ Stations").add_to(fmap)

    for (station, city, lat, lon), readings in df.groupby(
        ["location", "city", "latitude", "longitude"]
    ):
        pm25 = readings[readings["pollutant"] == "pm25"]
        if not pm25.empty:
            color, label = classify_pm25(pm25.iloc[0]["value"])
        else:
            color, label = "gray", "No PM2.5 data"

        popup_html = build_popup_html(station, city, readings)
        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=f"{station} ({label})",
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(cluster)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border: 1px solid #999;
                border-radius: 4px; font-size: 13px; line-height: 1.6;">
      <b>PM2.5 (ug/m3)</b><br>
      <span style="color:green">&#9679;</span> Good (&le;12)<br>
      <span style="color:#c9a900">&#9679;</span> Moderate (&le;35.4)<br>
      <span style="color:orange">&#9679;</span> Unhealthy for sensitive groups (&le;55.4)<br>
      <span style="color:red">&#9679;</span> Unhealthy (&le;150.4)<br>
      <span style="color:purple">&#9679;</span> Very Unhealthy (&le;250.4)<br>
      <span style="color:darkred">&#9679;</span> Hazardous (&gt;250.4)<br>
      <span style="color:gray">&#9679;</span> No PM2.5 data
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(fmap)
    return fmap


def main() -> None:
    try:
        df = fetch_data()
    except (GoogleAPIError, DefaultCredentialsError) as exc:
        print(f"BigQuery query failed: {exc}", file=sys.stderr)
        print(
            "Hint: run 'gcloud auth application-default login' and "
            f"'gcloud auth application-default set-quota-project {GEE_PROJECT_ID}'",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Fetched {len(df)} latest readings for country='{COUNTRY_FILTER}'.")
    if df.empty:
        print("No stations found for the given filter; nothing to map.")
        return

    fmap = build_map(df)
    fmap.save(OUTPUT_HTML)
    station_count = df["location"].nunique()
    print(f"Plotted {station_count} stations. Map written to {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
