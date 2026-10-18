import numpy as np
import rioxarray as rxr
import rasterio as rio
from skimage.morphology import binary_opening, binary_closing, disk
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1])) #crawl up one level to project root
from utils.paths import load_paths
paths=load_paths()

#(rf binary, disk size) -> 2 for 2024-10-24, 0 = no filtering (only recode 2 -> 0, used for 2024-09-20)
scenes = [
    (paths["data"]["binary_snow_raw"]/"2024_09_20/binary_snow_20240920.tif", 0),
    (paths["data"]["binary_snow_raw"]/"2024_10_24/binary_snow_20241024.tif", 2),
]

for input_path, disk_size in scenes:

    output_path = input_path.parent / f"{input_path.stem}_cleaned_disksize{disk_size}.tif"

    print(f"Processing: {input_path}")

    ds = rxr.open_rasterio(input_path, masked=True)
    arr = ds.squeeze().values

    # Convert class 2 to 0 (background)
    binary_arr = np.where(arr == 2, 0, arr)

    # Snow mask (1 = snow)
    snow_mask = (binary_arr == 1)

    # Remove small speckles
    opened = binary_opening(snow_mask, disk(disk_size))

    # Fill small holes
    closed = binary_closing(opened, disk(disk_size))

    # Back to uint8 (1 = snow, 0 = no snow)
    final_arr = np.where(closed, 1, 0).astype(np.uint8)

    final_arr[np.isnan(arr) | (arr == 0)] = 255 #0 is nodata in rf output (alpha == 0), no nodata set in metadata

    #Copy metadata from original
    with rio.open(input_path) as src:
        meta = src.meta.copy()

    meta.update({
        "count": 1,
        "dtype": "uint8",
        "nodata": 255,
        "compress": "lzw"
    })

    #Save cleaned raster
    with rio.open(output_path, "w", **meta) as dst:
        dst.write(final_arr, 1)

    print(f"Saved: {output_path}")
