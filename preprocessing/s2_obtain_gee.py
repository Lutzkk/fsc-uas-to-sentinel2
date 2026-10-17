import ee 
import geopandas as gpd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1])) #crawl up one level to project root
from utils.paths import load_paths
paths=load_paths()

ee.Authenticate()
ee.Initialize(project="ee-lutz-training")


aoi_p=paths["data"]["aoi"]
print(f"Loading AOI from: {aoi_p}")

aoi=gpd.read_file(aoi_p).to_crs("4326") #ee requires WGS84 for geometry

#extract geom
aoi_geom=aoi.geometry.iloc[0]

#convert to geojson dict 
aoi_json=aoi_geom.__geo_interface__

#convert to ee object 
aoi_ee=ee.Geometry(aoi_json)


#Datelist for valid acquisitions
date_list=[
    "2024-09-19",
    "2024-10-24"
]

# Loop over each date and export
for date in date_list:
    start = ee.Date(date)
    end = start.advance(1, "day")  # next day (exclusive)

    image = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .select(["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "MSK_CLDPRB"])
        .filterBounds(aoi_ee)
        .filterDate(start, end)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .first()
    )

    if image is None:
        print(f"No image found on {date}")
        continue
    
    image=image.float().clip(aoi_ee)  # Unify Datatype to float32 and clip to the AOI

    print(f"Exporting image from {date}")
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"S2_export_{date}",
        folder="fsc_drive_storage",
        fileNamePrefix=f"S2_{date}",
        region=aoi_ee,
        crs="EPSG:32632",
        scale=10,
        maxPixels=1e13
    )
    task.start()