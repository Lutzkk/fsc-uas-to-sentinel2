import ee

# Initialize Earth Engine
ee.Initialize(project="ee-lutz-training")

# AOI: bbox of the old rf_test_20241021_ps.shp (lon/lat, EPSG:4326)
aoi_geom = ee.Geometry.Rectangle([10.9528475, 47.38765747, 11.08958364, 47.43016658])

# Bands to export
reflectance_bands = [
    'B2','B3','B4','B5','B6',
    'B7','B8','B8A','B11','B12'
]

# Dates and labels
download_list = [
    {"label":"2025-02-21", "start":"2025-02-20", "end":"2025-02-22"},
    {"label": "2025-06-06", "start": "2025-06-05", "end": "2025-06-07"},
    {"label": "2025-06-09", "start": "2025-06-08", "end": "2025-06-10"},
]

# Loop through and export images
for entry in download_list:
    label = entry["label"]
    start = entry["start"]
    end = entry["end"]

    # Get list of S2 scenes matching date + AOI
    s2_list = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi_geom)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 100))
        .sort("system:time_start")
        .toList(10)  # up to 10 scenes for safety
    )

    n = s2_list.size().getInfo()

    if n == 0:
        print(f"No images found for {label}")
        continue

    print(f"{n} Image(s) found for {label}")

    # Loop through each image in the list
    for i in range(n):
        s2 = ee.Image(s2_list.get(i))

        # Get date of the image
        date_str = ee.Date(s2.get("system:time_start")).format("YYYY-MM-dd").getInfo()
        out_name = f"S2_SR_{label}_{date_str}_{i}"

        reflectance = s2.select(reflectance_bands).clip(aoi_geom)

        task = ee.batch.Export.image.toDrive(
            image=reflectance,
            description=out_name,
            folder="EarthEngineExports",
            fileNamePrefix=out_name,
            region=aoi_geom.getInfo()["coordinates"],
            scale=10,
            crs="EPSG:32632",
            maxPixels=1e13,
            fileFormat="GeoTIFF",
        )

        task.start()
        print(f"Export started → {out_name}")
