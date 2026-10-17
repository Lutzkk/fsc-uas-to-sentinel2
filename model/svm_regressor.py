import numpy as np
import xarray as xr
import rioxarray as rxr
import pandas as pd
from pathlib import Path
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load data
import sys
root = Path(__file__).resolve().parents[1] #repo root
sys.path.append(str(root)) #crawl up one level to project root
from utils.paths import load_paths
paths=load_paths()
train = pd.read_csv(paths["data"]["dataframes"] / "train_fsc.csv")
val   = pd.read_csv(paths["data"]["dataframes"] / "val_fsc.csv")

# Best features (poster model, from svm brute force search)
# old Optuna model: ['B3', 'B4', 'B5', 'B6', 'B7','B8', 'B8A', 'B11', 'DEM']
best_features = ['B2', 'B3', 'B12']

X_train = train[best_features]
y_train = train["FSC_GT"]

X_val = val[best_features]
y_val = val["FSC_GT"]

# Hyperparameters used in svm brute force search
# old Optuna params: C=61.23124455976027, epsilon=0.024070370840014756, gamma=0.00614340621135284
best_C = 50
best_epsilon = 0.01
best_gamma = "scale"

# Final model
final_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(
        kernel="rbf",
        C=best_C,
        epsilon=best_epsilon,
        gamma=best_gamma
    ))
])

# Train
final_svm.fit(X_train, y_train)

# Validate
preds = final_svm.predict(X_val)

print("\nFinal SVM Performance:")
print("RMSE:", np.sqrt(mean_squared_error(y_val, preds)))
print("MAE :", mean_absolute_error(y_val, preds))
print("R²  :", r2_score(y_val, preds))

print("Saving final model...")
import joblib

# Save model
joblib.dump(final_svm, root / "model/svm_fsc_model.pkl")

pred_df = pd.DataFrame({
    "x": val["x"],
    "y": val["y"],
    "FSC_GT": val["FSC_GT"],
    "FSC_pred": preds
})
pred_df.to_csv(paths["data"]["dataframes"] / "val_predictions.csv", index=False)
print("\nSaved model to:", root / "model/svm_fsc_model.pkl")


import rasterio
from rasterio.transform import rowcol

def write_predictions_to_raster(preds, val_df, cube_path, out_path):
    # load raster metadata with rasterio
    with rasterio.open(cube_path) as src:
        transform = src.transform
        crs = src.crs
        height = src.height
        width = src.width

    # initialize full raster
    pred_full = np.full((height, width), np.nan, dtype="float32")

    xs = val_df["x"].values
    ys = val_df["y"].values

    # world coords → pixel coords using rasterio
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)

    pred_full[rows, cols] = preds.astype("float32")

    # write GeoTIFF correctly
    new_profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
    }

    with rasterio.open(out_path, "w", **new_profile) as dst:
        dst.write(pred_full, 1)

    print(f"\n✓ Prediction raster saved → {out_path}")


from pathlib import Path

cube_path  = paths["data"]["processed_s2"] / "VALIDATION_s2_20250606_processed.tif"
out_path   = root / "model/FSC_PRED_20250606.tif"

write_predictions_to_raster(preds, val, cube_path, out_path)




