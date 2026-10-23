import ee
import geopandas as gpd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1])) #crawl up one level to project root
from utils.paths import load_paths
paths=load_paths()

ee.Authenticate()
ee.Initialize(project="FILLITOUT") #replace with your GEE project name


aoi_p=paths["data"]["aoi"]
print(f"Loading AOI from: {aoi_p}")

aoi=gpd.read_file(aoi_p).to_crs("EPSG:4326") #ee requires WGS84 for geometry

#extract geom
aoi_geom=aoi.geometry.iloc[0]

#convert to geojson dict
aoi_json=aoi_geom.__geo_interface__

#convert to ee object
aoi_ee=ee.Geometry(aoi_json)


#bands to export (no MSK_CLDPRB, the june scene is cloud masked manually in qgis)
bands=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

#Datelist for valid acquisitions
#all_tiles=True exports every scene of that day as _0, _1, june has two tiles, the other dates only one tile each
date_list=[
    {"date":"2024-09-19", "all_tiles":False},
    {"date":"2024-10-24", "all_tiles":False},
    {"date":"2025-02-21", "all_tiles":False},
    {"date":"2025-06-06", "all_tiles":True},
]

# Loop over each date and export
for entry in date_list:
    date=entry["date"]
    start = ee.Date(date)
    end = start.advance(1, "day")  # next day (exclusive)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .select(bands)
        .filterBounds(aoi_ee)
        .filterDate(start, end)
    )

    n=collection.size().getInfo()
    if n == 0:
        print(f"No image found on {date}")
        continue

    #output name as expected by feature_engineering.py, e.g. S2_2024_09_19
    name=f"S2_{date.replace('-', '_')}"

    if entry["all_tiles"]:
        image_list=collection.sort("system:time_start").toList(n)
        exports=[(ee.Image(image_list.get(i)), f"{name}_{i}") for i in range(n)]
    else:
        exports=[(collection.sort("CLOUDY_PIXEL_PERCENTAGE").first(), name)]

    for image, out_name in exports:
        image=image.clip(aoi_ee)

        print(f"Exporting {out_name}")
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=f"{out_name}_export",
            folder="fsc_drive_storage",
            fileNamePrefix=out_name,
            region=aoi_ee,
            crs="EPSG:32632",
            scale=10,
            maxPixels=1e13
        )
        task.start()
