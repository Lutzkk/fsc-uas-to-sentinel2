#dataloader.py
#Functions to load data from disk

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1])) #crawl up one level to project root
import rasterio as rio 
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling
import numpy as np

#--------------------------

def load_sentinel_scene(path, target_res=10, ref_band="B2", resampling=Resampling.bilinear, save_path=None):
    """
    Load a Sentinel-2 scene and resample all bands to a target resolution.
    """

    path = Path(path)
    da = rxr.open_rasterio(path, masked=True)

    # ORIGINAL band name assumption
    band_names = [
        "B2", "B3", "B4", "B5", "B6", "B7",
        "B8", "B8A", "B11", "B12", "MSK_CLDPRB"
    ][: da.sizes["band"]]

    # Convert to Dataset
    ds = da.to_dataset(dim="band")
    ds = ds.assign_coords(band=band_names)
    ds = ds.rename({k: v for k, v in zip(ds.data_vars, band_names)})

    #
    if "MSK_CLDPRB" in ds:
        print("Dropping MSK_CLDPRB band")
        ds = ds.drop_vars("MSK_CLDPRB")

    # update band_names 
    band_names = list(ds.data_vars)

    # Choose reference band
    ref = ds[ref_band]

    # Resample all bands that remain
    aligned = xr.Dataset()
    for b in band_names:
        aligned[b] = ds[b].rio.reproject_match(ref, resampling=resampling)

    # Save if needed
    if save_path is not None:
        save_path = Path(save_path)
        aligned.to_array("band").rio.to_raster(save_path, compress="LZW", tiled=True)
        print(f"Saved resampled scene to {save_path}")

    return aligned

def crop_s2_to_binary_mask(s2_ds: xr.Dataset, binary_mask: xr.DataArray) -> xr.Dataset:
    """
    Crop a loaded Sentinel-2 xarray.Dataset to the pixels where
    the binary mask == 1. All mask==0 pixels become NaN.
    """
    # Ensure same projection & transform
    if not s2_ds.rio.crs == binary_mask.rio.crs:
        raise ValueError("CRS mismatch between sentinel dataset and binary mask.")

    # Optionally align array shapes using reproject_match (if needed)
    if (s2_ds.sizes["y"] != binary_mask.sizes["y"]) or (s2_ds.sizes["x"] != binary_mask.sizes["x"]):
        binary_mask = binary_mask.rio.reproject_match(s2_ds)

    # Mask: keep only pixels where mask == 1
    masked = s2_ds.where(binary_mask == 1)

    # Optional: crop to bounding box of mask
    masked = masked.rio.clip_box(
        minx=float(binary_mask.rio.bounds()[0]),
        miny=float(binary_mask.rio.bounds()[1]),
        maxx=float(binary_mask.rio.bounds()[2]),
        maxy=float(binary_mask.rio.bounds()[3])
    )

    return masked

#--------------------------

def minmax_np(arr: np.ndarray) -> np.ndarray:
    min_val = np.nanmin(arr, axis=(-2, -1), keepdims=True)
    max_val = np.nanmax(arr, axis=(-2, -1), keepdims=True)
    range_val = np.where((max_val - min_val) == 0, 1, max_val - min_val)
    return (arr - min_val) / range_val

def minmax_normalize_array(arr: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(
        minmax_np,                # function to apply
        arr,                      # xarray object
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,           # loop over non-core dims (like band)
        dask="parallelized",      # support Dask if needed
        output_dtypes=[arr.dtype] # preserve dtype
    )

def minmax_normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Min-Max normalize each band in a multi-band dataset."""
    normalized=xr.Dataset()
    for var in ds.data_vars:
        normalized[var]=minmax_normalize_array(ds[var])
    return normalized
#-------------------------

#global percentile normalization across multiple datasets following guide from esa:
# https://medium.com/sentinel-hub/how-to-normalize-satellite-images-for-deep-learning-d5b668c885af
def compute_global_percentiles(datasets, bands=None, p_low=1, p_high=99):
    """
    Compute global per-band percentiles across multiple xarray Datasets.
    """
    if bands is None:
        bands = list(datasets[0].data_vars)

    percentiles = {}

    for band in bands:
        arrs = [ds[band].values for ds in datasets]
        stacked = np.concatenate([a.reshape(-1) for a in arrs])
        stacked = stacked[~np.isnan(stacked)]  # remove NaNs
        
        p1 = np.percentile(stacked, p_low)
        p99 = np.percentile(stacked, p_high)

        percentiles[band] = (p1, p99)

    return percentiles

def apply_percentile_normalization(ds, pct_dict):
    """
    Apply global percentile normalization to a Dataset.
    Clamps values below/above p1/p99.
    """
    normalized = xr.Dataset()

    for band in ds.data_vars:
        p1, p99 = pct_dict[band]

        arr = ds[band]
        arr_clipped = xr.apply_ufunc(
            lambda x, p1=p1, p99=p99: np.clip((x - p1) / (p99 - p1), 0, 1),
            arr,
            dask="parallelized",
            output_dtypes=[arr.dtype]
        )
        normalized[band] = arr_clipped

    return normalized

#--------------------------

def load_align_rasters(ref_path, raster_paths, names, resampling=Resampling.nearest, save_paths=None):
    """
    Align multiple rasters (including reference) to the grid of ref_path,
    and return them as an xarray.Dataset with named variables.
    """
    
    if len(raster_paths) != len(names):
        raise ValueError("`raster_paths` and `names` must have the same length.")

    ref = rxr.open_rasterio(ref_path, masked=True).squeeze()

    dataset = {}

    for i, path in enumerate(raster_paths):
        arr = rxr.open_rasterio(path, masked=True).squeeze()
        aligned = arr.rio.reproject_match(ref, resampling=resampling)
        dataset[names[i]] = aligned

        if save_paths is not None:
            if len(save_paths) != len(raster_paths):
                raise ValueError("`save_paths` must match number of raster_paths.")
            aligned.rio.to_raster(save_paths[i])

    return xr.Dataset(dataset)

#--------------------------

def fsc_with_s2_mask(
    drone_binary,
    s2_binary_mask,
    s2_template,
    coverage_threshold=0.9,
    output=None
):
    """
    Compute FSC only where the S2 binary mask == 1
    AND the drone coverage inside each S2 pixel >= threshold.
    """

    #Ensure drone mask is DataArray
    if isinstance(drone_binary, (str, Path)):
        drone_binary = rxr.open_rasterio(drone_binary, masked=True).squeeze()

    #Ensure S2 mask + template are DataArrays
    if isinstance(s2_binary_mask, (str, Path)):
        s2_binary_mask = rxr.open_rasterio(s2_binary_mask, masked=True).squeeze()

    if isinstance(s2_template, (str, Path)):
        s2_template = (
            rxr.open_rasterio(s2_template, masked=True)
            .sel(band=1)      # or any S2 band you want
            .squeeze()
        )

    #Build drone coverage (1 where drone has data)
    drone_coverage = xr.where(~xr.ufuncs.isnan(drone_binary), 1, 0)

    #Reproject both to the S2 grid with average aggregation
    snow_frac  = drone_binary.rio.reproject_match(s2_template, resampling=Resampling.average)
    cover_frac = drone_coverage.rio.reproject_match(s2_template, resampling=Resampling.average)

    # Compute FSC = snow / coverage
    fsc = snow_frac / cover_frac

    # Mask by drone coverage threshold
    fsc = fsc.where(cover_frac >= coverage_threshold)

    # Mask by S2 binary mask
    fsc = fsc.where(s2_binary_mask == 1)

    # Optional save
    if output is not None:
        fsc.rio.to_raster(output, compress="LZW")
        print(f"Saved FSC → {output}")

    return fsc

#--------------------------

def save_dataset_as_tiff(ds: xr.Dataset, output_path: Path):
    """
    Save an xarray.Dataset as a multiband GeoTIFF with correct band names stored
    in raster band descriptions, using rasterio.
    """

    band_names = list(ds.data_vars.keys())
    da = ds.to_array("band")  # shape: (bands, y, x)
    arr = da.values.astype("float32")

    # Extract spatial information
    transform = ds.rio.transform()
    crs = ds.rio.crs
    height, width = ds.rio.height, ds.rio.width
    count = arr.shape[0]

    # Write with rio
    with rio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype="float32",
        crs=crs,
        transform=transform,
        compress="LZW",
        tiled=True
    ) as dst:

        # Write each band and set description
        for i, name in enumerate(band_names, start=1):
            dst.write(arr[i-1], i)
            dst.set_band_description(i, name)

    print(f"Saved dataset with band names → {output_path}")