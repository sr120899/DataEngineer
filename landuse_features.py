"""Sentinel-2 composite + spectral-index feature image, shared by Step 2
(sampling features at label points) and Step 4 (classifying the full image).
Both steps must build the identical image or the classifier trained in
Step 3-5 won't match what it's applied to.
"""

from __future__ import annotations

import ee

import landuse_config as cfg


def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Mask clouds/cirrus using the QA60 bitmask (bits 10 and 11)."""
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask)


def build_feature_image(aoi: ee.Geometry) -> ee.Image:
    """Cloud-masked S2 median composite + spectral indices, scaled to reflectance."""
    collection = (
        ee.ImageCollection(cfg.S2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(cfg.S2_DATE_START, cfg.S2_DATE_END)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg.S2_MAX_CLOUD_PCT))
        .map(mask_s2_clouds)
    )
    composite = collection.median().clip(aoi)

    # S2 SR digital numbers -> surface reflectance (0-1) so EVI's additive
    # constant is meaningful; NDVI/NDWI/NDBI are ratios so scale cancels out.
    reflectance = composite.select(cfg.S2_BANDS).multiply(0.0001)

    ndvi = reflectance.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = reflectance.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndbi = reflectance.normalizedDifference(["B11", "B8"]).rename("NDBI")
    evi = reflectance.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {
            "NIR": reflectance.select("B8"),
            "RED": reflectance.select("B4"),
            "BLUE": reflectance.select("B2"),
        },
    ).rename("EVI")

    return reflectance.addBands([ndvi, evi, ndwi, ndbi]).select(cfg.FEATURE_BANDS)
