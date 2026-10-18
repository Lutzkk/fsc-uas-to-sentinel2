import numpy as np
import rasterio
from skimage.filters import threshold_otsu
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1])) #crawl up one level to project root
from utils.paths import load_paths
paths=load_paths()

input_path = paths["data"]["drone_scenes"]/"2025_02_25/20250225_Zugspitze_100m_DJIM300Altum_OM.tif"
output_ndsi = paths["data"]["binary_snow_raw"]/"2025_02_25/ndsi.tif"
output_binary = paths["data"]["binary_snow_raw"]/"2025_02_25/20250225_binary_snow.tif"


with rasterio.open(input_path) as src:
    green = src.read(2).astype(np.float32)
    nir = src.read(5).astype(np.float32)
    mask = src.read(7)
    profile = src.profile

invalid = mask == 0
ndsi = (green - nir) / (green + nir + 1e-6)
ndsi[invalid] = np.nan

thresh = threshold_otsu(ndsi[~np.isnan(ndsi)])
binary = np.where(ndsi >= thresh, 1, 0).astype(np.uint8)
binary[invalid] = 255

profile.update(dtype="float32", count=1)
with rasterio.open(output_ndsi, "w", **profile) as dst:
    dst.write(ndsi, 1)

profile.update(dtype="uint8", nodata=255)
with rasterio.open(output_binary, "w", **profile) as dst:
    dst.write(binary, 1)
