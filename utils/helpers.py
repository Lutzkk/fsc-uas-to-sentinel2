#helpers.py

import rasterio as rio
import pandas as pd
import rasterio.mask
import geopandas as gpd
import xarray as xr
from rasterio.enums import Resampling
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from datetime import datetime
#------------------


def crop_aoi(ds_path, aoi_path):
    """
    Clip a raster dataset to the given AOI (as GeoJSON or Shapefile).
    
    Parameters:
        ds_path (str or Path): Path to the raster file to clip.
        aoi_path (str or Path): Path to the AOI file (GeoJSON, SHP).
    
    Returns:
        out_image (np.ndarray): Clipped image array.
        out_transform (Affine): Transform of the clipped image.
        out_meta (dict): Updated metadata for writing to disk if needed.
    """
    with rio.open(ds_path) as src:
        # Read and reproject AOI to match raster CRS
        aoi = gpd.read_file(aoi_path).to_crs(src.crs)
        aoi_geom = [aoi.geometry.iloc[0].__geo_interface__]  # geo-interface for masking

        # Clip raster
        out_image, out_transform = rio.mask.mask(src, aoi_geom, crop=True)
        out_meta = src.meta.copy()
        
        # Update metadata for the clipped raster
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        print(f"Clipped {ds_path} to AOI: {aoi_path}")

        return out_image, out_transform, out_meta


#------------------

def compute_spectral_index(ds, b1, b2, index_type=None):
    """
    Computes a spectral index from two bands of a raster ds. 
    always follows: (b1 - b2) / (b1 + b2)
    
    Parameters: 
        ds (xarray.DataArray or xarray.Dataset): Input raster data with bands.
        b1 (str): Name of the first band.
        b2 (str): Name of the second band.
        index_type (str, optional): Type of index to compute. If None, defaults to "b1-b2_index".
                                    Common types include "NDVI", "NDWI", etc.
    """

    b1_da, b2_da = ds[b1], ds[b2]
    index = (b1_da - b2_da) / (b1_da + b2_da)
    name = index_type.upper() if index_type else f"{b1}-{b2}_index"
    return index.rename(name)


#------------------

def mask_raster(ds, mask, invert=False):
    """
    Masks a raster dataset using a binary mask
    
    Parameters:
        ds (xarray.DataArray or xarray.Dataset): Input raster data.
        mask (xarray.DataArray): Binary mask (1 for valid, 0 for invalid).
        invert (bool): If True, inverts the mask (0 for valid, 1 for invalid).
    """
    ds, mask=xr.align(ds, mask, join="exact") #align coords
    
    #opt. invert mask 
    if invert: 
        mask=1-mask

    #apply mask 
    if isinstance(ds, xr.Dataset):
        masked = ds.where(mask==1)
    elif isinstance(ds, xr.DataArray):
        masked = ds.where(mask==1)
    else:
        raise TypeError("Input ds must be xarray Dataset or DataArray.")
    
    return masked


#---------------------------------------

def align_fsc_gt(ds, mask, resampling=Resampling.nearest):
    """ 
    Crops and aligns entire netcdf datacube to fsc ground truth

    Parameters: 
        ds(xarray.Dataset): Input dataset with features (Sentinelbands, dem derivatives, spectral indices)
        mask(xarray.DataArray): FSC Ground truth resampled to 10m spatial resolution
        resampling: Resampling method to use (default: nearest neighbor)
    """

    if not isinstance(ds, xr.Dataset):
        raise TypeError("Input Dataset must be an xarray Dataset")

    elif isinstance(ds, xr.Dataset):
        aligned = {}
        for var in ds.data_vars:
            aligned[var] = ds[var].rio.reproject_match(mask, resampling=resampling)
        return xr.Dataset(aligned)
    
#----------------------------------------
def vectormask(ds, shapefile_path):
    """
    Masks (sets to NaN) all pixels INSIDE polygons in the given shapefile.
    
    Parameters
    ----------
    ds : xr.Dataset
        Sentinel-2 dataset (with .rio accessor, same CRS & resolution).
        
    shapefile_path : str or Path
        Path to polygon shapefile containing cloud masks.

    Returns
    -------
    xr.Dataset
        Dataset with pixels inside polygons set to NaN.
    """

    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Ensure CRS matches
    if gdf.crs != ds.rio.crs:
        gdf = gdf.to_crs(ds.rio.crs)

    # Convert vector → raster mask (True = inside polygon)
    mask = ds.rio.clip(gdf.geometry, drop=False, invert=False)

    # mask is a dataset → convert to boolean
    # inside polygon = keep only pixels present in mask
    inside = ~mask.isnull().to_array().any("variable")

    # Invert because we want to MASK OUT pixels inside
    remove_mask = inside  # True where pixels should become NaN

    # Apply masking to all bands
    for var in ds.data_vars:
        arr = ds[var]
        arr = arr.where(~remove_mask, np.nan)
        ds[var] = arr

    return ds


def merge_s2_scenes(scene_a: xr.Dataset, scene_b: xr.Dataset) -> xr.Dataset:
    """
    Merge two Sentinel-2 scenes by:
      - averaging values where both scenes have valid (non-NaN) pixels
      - taking the valid pixel where only one scene has data
      - leaving NaN where both are NaN

    Parameters
    ----------
    scene_a : xr.Dataset
    scene_b : xr.Dataset

    Returns
    -------
    xr.Dataset
        Merged dataset with the same structure as the inputs.
    """

    merged = xr.Dataset()

    for var in scene_a.data_vars:
        a = scene_a[var]
        b = scene_b[var]

        a_valid = ~np.isnan(a)
        b_valid = ~np.isnan(b)

        # Average where both valid, else take whichever is valid
        merged_var = xr.where(
            a_valid & b_valid,              # condition: both valid
            (a + b) / 2,                    # average
            xr.where(a_valid, a, b)         # else take whichever is valid
        )

        merged[var] = merged_var

    # Copy coordinates and metadata
    merged = merged.assign_coords(scene_a.coords)
    merged.attrs = scene_a.attrs

    return merged





#--------------------------------------------
def flatten_dataset(ds: xr.Dataset, target="FSC_GT") -> pd.DataFrame:
    """
    Flatten an xarray Dataset (Sentinel-2 cube + features + FSC_GT) 
    into a pandas DataFrame for machine learning.

    - Keeps all predictor columns
    - Keeps x,y spatial coordinates
    - Keeps FSC_GT as target
    - Drops only pixels with NaN in predictors or target
    """

    # Convertinto a flat df
    df = ds.to_dataframe().reset_index()

    # Ensure target exists
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataset columns.")

    # Remove rows where target is missing
    df = df.dropna(subset=[target])

    # Identify predictor columns (exclude coords and target)
    predictor_cols = [
        col for col in df.columns
        if col not in ["x", "y", target] and df[col].dtype != "O"
    ]

    # Drop rows where any predictor is NaN
    df = df.dropna(subset=predictor_cols)

    return df



#--------------------------------------------
def save_dataset_as_tiffs(ds, out_dir, prefix=""):
    """Save each variable in an xarray dataset as an individual GeoTIFF."""
    for var in ds.data_vars:
        out_path = out_dir / f"{prefix}{var}.tif"
        print(f"Saving {var} to {out_path.name}")
        ds[var].rio.to_raster(out_path)


#-----------------------------------------------
def safe_subcols(df, cols):
    """Return only columns that exist and are numeric."""
    exist = [c for c in cols if c in df.columns]
    num = df[exist].select_dtypes(include=[np.number]).columns.tolist()
    return num


#-----------------------------------------------

def evaluate_feature_sets(train_df, val_df, feature_sets, out_csv, rf_params=None, random_state=42):
    rf_params = rf_params or dict(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )

    results = []

    for name, cols in feature_sets.items():
        Xtr_cols = safe_subcols(train_df, cols)
        Xva_cols = safe_subcols(val_df, cols)
        common = sorted(list(set(Xtr_cols).intersection(Xva_cols)))

        if not common:
            print(f"[skip] {name}: no common numeric columns.")
            continue

        X_train = train_df[common].values
        y_train = train_df["fsc_gt"].values
        X_val = val_df[common].values
        y_val = val_df["fsc_gt"].values

        # Train model
        model = RandomForestRegressor(**rf_params)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        results.append({
            "timestamp": datetime.utcnow().isoformat(),
            "feature_set": name,
            "n_features": len(common),
            "features": ",".join(common),
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "n_train": len(y_train),
            "n_val": len(y_val),
        })

        print(f"{name:28s} | R²={r2: .3f}  RMSE={rmse: .3f}  MAE={mae: .3f}  (k={len(common)})")

    # Save or append results
    results_df = pd.DataFrame(results)
    try:
        existing = pd.read_csv(out_csv)
        results_df = pd.concat([existing, results_df], ignore_index=True)
    except Exception:
        pass

    results_df.to_csv(out_csv, index=False)
    print(f"Results saved to: {out_csv}")
    return results_df