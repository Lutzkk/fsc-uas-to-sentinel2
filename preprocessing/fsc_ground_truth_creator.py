# fsc_ground_truth.py

# Script that generates ground truth fsc data from a sentinel2 grid and binary drone scene classes.
import rasterio as rio 
import numpy as np 
import geopandas as gpd 
from pathlib import Path 

#---

# Paths 
import sys
sys.path.append(str(Path(__file__).resolve().parents[1])) #crawl up one level to project root
from utils.paths import load_paths
paths=load_paths()
aoi_p=paths["data"]["aoi"]

#s2 scene (any will do)
s2_p=paths["data"]["s2_dir"]/"S2_2024_10_24.tif"

#drone scenes (binary classified)
f=paths["data"]["binary_snow"]
d_0920=f/"binarysnow_20240920_aligned_clipped.tif"
d_1025=f/"binarysnow_20241024_aligned_clipped.tif"

d_0225=f/"binarysnow_20250225_32632_clipped.tif"
d_0604=f/"binarysnow_20250604_32632_clipped.tif"

#clip s2 scene to aoi 
aoi=gpd.read_file(aoi_p).to_crs("EPSG:32632")
with rio.open(s2_p) as src:
    b1=src.read(1)
    out_image, out_transform=rio.mask.mask(src, aoi.geometry, crop=True)
    #values ==0 sty 0 the rest becomes 1 
    out_image= np.where(out_image==0, 0, 1)


    out_meta=src.meta.copy()


# Convert to uint8
out_image = out_image.astype(np.uint8)

# Update metadata for a single-band uint8 raster
out_meta.update({
    "driver": "GTiff",
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform,
    "count": 1,
    "dtype": "uint8"
})

# Output path
out_path = paths["data"]["fsc_gt"] / "s2_grid_binary_mask.tif"

out_path.parent.mkdir(parents=True, exist_ok=True)

# Save GeoTIFF
with rio.open(out_path, "w", **out_meta) as dest:
    dest.write(out_image[0], 1)

print("Saved:", out_path)