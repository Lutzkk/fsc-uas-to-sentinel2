import rasterio as rio
import numpy as np
import joblib
from pathlib import Path

# ------------------------------------------------
# Paths
# ------------------------------------------------
import sys
root = Path(__file__).resolve().parents[1] #repo root
sys.path.append(str(root)) #crawl up one level to project root
from utils.paths import load_paths
paths=load_paths()
model = joblib.load(root / "model/svm_fsc_model.pkl")

input_tif = paths["data"]["processed_s2"] / "VALIDATION_s2_20250606_processed.tif"
output_tif = root / "results/fsc_prediction_svm.tif"

# Bands the model was trained on (same order as in training)
bands = list(model.feature_names_in_)

# ------------------------------------------------
# Load stacked raster
# ------------------------------------------------
with rio.open(input_tif) as src:

    print("Input raster shape:", src.count, "bands,", src.width, "x", src.height)

    # Read only needed bands, looked up by band name (band descriptions of the stacked tif)
    band_indices = [src.descriptions.index(b) + 1 for b in bands]  # convert to 1-based
    arr = src.read(band_indices)  # shape: (n_bands, H, W)

    meta = src.meta.copy()

# ------------------------------------------------
# Reshape to (N_pixels, N_features)
# ------------------------------------------------
# (bands, H, W) -> (H, W, bands)
arr = np.transpose(arr, (1, 2, 0))

H, W, B = arr.shape
arr_2d = arr.reshape(-1, B)

# ------------------------------------------------
# Predict FSC
# ------------------------------------------------
print("Running SVM prediction on", arr_2d.shape[0], "pixels")

#only predict pixels without NaN (outside AOI is NaN, SVR cant handle NaN)
valid = ~np.isnan(arr_2d).any(axis=1)
pred = np.full(arr_2d.shape[0], np.nan)
pred[valid] = model.predict(arr_2d[valid])

# Clip to valid range 0–1 (just in case)
pred = np.clip(pred, 0, 1)

# Reshape back to raster
fsc_map = pred.reshape(H, W)

# Optionally scale to uint16 FSC (0–1000 or 0–10000)
# fsc_map_scaled = (fsc_map * 1000).astype("uint16")

# ------------------------------------------------
# Write FSC GeoTIFF
# ------------------------------------------------
meta.update({
    "count": 1,
    "dtype": "float32",
    "nodata": np.nan,
    # For scaled Uint16 output:
    # "dtype": "uint16"
})

with rio.open(output_tif, "w", **meta) as dst:
    dst.write(fsc_map.astype("float32"), 1)

print("Saved FSC map to:", output_tif)
