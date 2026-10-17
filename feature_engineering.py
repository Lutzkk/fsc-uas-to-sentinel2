from pathlib import Path 
import sys
import pandas as pd
from utils import paths
from utils.dataloader import compute_global_percentiles, apply_percentile_normalization, fsc_with_s2_mask, load_align_rasters, load_sentinel_scene, minmax_normalize_dataset, crop_s2_to_binary_mask, save_dataset_as_tiff
from utils.paths import load_paths
from utils.helpers import merge_s2_scenes, compute_spectral_index, mask_raster, save_dataset_as_tiffs, spatial_stratified_split, flatten_dataset, evaluate_feature_sets, vectormask
import rioxarray as rxr
import xarray as xr
import numpy as np

import time
from rasterio.enums import Resampling
from multiprocessing import cpu_count

def main():
    sys.path.append(str(Path(__file__).resolve().parents[1]))  # crawl up to project root

    paths = load_paths()

    # Define paths to drone scenes
    
    drone_path_20240920 = paths["data"]["binary_snow"] / "binarysnow_20240920_clipped.tif"
    drone_path_20241025 = paths["data"]["binary_snow"] / "binarysnow_20241025_clipped.tif"
    drone_path_20250225 = paths["data"]["binary_snow"] / "binarysnow_20250225_clipped.tif"
    drone_path_20250604 = paths["data"]["binary_snow"] / "binarysnow_20250604_clipped.tif" #VALIDATION SCENE
    #20250604 is val scene
    
    #define fsc grid path 
    fsc_grid_p=paths["data"]["fsc_gt"] / "s2_grid_binary_mask.tif"

    #define paths to sentinel2 scenes
    s2_0920_p=paths["data"]["s2_dir"]/"S2_2024_09_19.tif"
    s2_1024_p=paths["data"]["s2_dir"]/"S2_2024_10_24.tif"
    s2_0221_p=paths["data"]["s2_dir"]/"S2_2025_02_21.tif"



    # Define paths to Elevation Data
    dem_p=paths["data"]["elevation"]/"Zugspitze_DJIM300L1_snowfree_DSM.tif" #epsg:32632, doesnt fully match the ground truth area 
    slope_p=paths["data"]["elevation"]/"slope.tif" #epsg:32632, 1m res, contains stripes
    aspect_p=paths["data"]["elevation"]/"aspect.tif" #epsg:32632, 1m res, contains stripes

    # Define paths to forest mask 
    forest_p=paths["data"]["forest"]/"forest_alps_clipped_32632.tif" #binary raster indicating forested areas (10m res, epsg:32632)

    #---------------------------

    #PREPROCESSING: FSC GT CREATION, DRONE SCENE ALIGNMENT

    #load and align rasters to the 2024-09-20 scene
    print("Loading binary snow masks (drone scenes)...")
    snow_0920 = rxr.open_rasterio(drone_path_20240920, masked=True).squeeze()
    snow_1025 = rxr.open_rasterio(drone_path_20241025, masked=True).squeeze()
    snow_0225 = rxr.open_rasterio(drone_path_20250225, masked=True).squeeze()
    snow_0604=rxr.open_rasterio(drone_path_20250604, masked=True).squeeze()
    print("Binary snow masks loaded.")
    fsc_grid=rxr.open_rasterio(fsc_grid_p, masked=True).squeeze()

    #compute fsc ground truth from aligned rasters 

    #output paths 
    out_0920 = paths["data"]["fsc_gt"] / "fsc_20240920.tif"
    out_1024 = paths["data"]["fsc_gt"] / "fsc_20241024.tif"
    out_0225 = paths["data"]["fsc_gt"] / "fsc_20250225.tif"
    out_0604 = paths["data"]["fsc_gt"] / "VALIDATION_fsc_20250604.tif"

    fsc_0920=fsc_with_s2_mask(snow_0920, fsc_grid, s2_0920_p, coverage_threshold=0.9, output=out_0920)
    fsc_1024=fsc_with_s2_mask(snow_1025, fsc_grid, s2_1024_p, coverage_threshold=0.9, output=out_1024)
    fsc_0225=fsc_with_s2_mask(snow_0225, fsc_grid, s2_0221_p, coverage_threshold=0.9, output=out_0225)
    fsc_0604=fsc_with_s2_mask(snow_0604, fsc_grid, s2_0221_p, coverage_threshold=0.9, output=out_0604) #feb 0221 because 0604 is designed from two s2 scenes (circumvent cloud)

    #------------------------------

    print("Starting Sentinel-2 Data Preprocessing...")
    #SENTINEL-2 SCENE LOADING, FOREST MASKING  AND NORMALIZATION
    s2_0920 = load_sentinel_scene(s2_0920_p) #loads and resamples to 10m (deafualt bilinear)
    s2_1024 = load_sentinel_scene(s2_1024_p) #loads and resamples to 10m (deafualt bilinear)
    s2_0221= load_sentinel_scene(s2_0221_p) #loads and resamples to 10m (deafualt bilinear)

    #crop to aoi 
    s2_0920=crop_s2_to_binary_mask(s2_0920, fsc_grid)
    s2_1024=crop_s2_to_binary_mask(s2_1024, fsc_grid)
    s2_0221=crop_s2_to_binary_mask(s2_0221, fsc_grid)

    #test export 
    save_dataset_as_tiff(s2_0920, paths["data"]["processed_s2"] / "s2_20240920_test.tif")


    print("Computing global 1-99 percentile normalization parameters from training scenes...")
    pct = compute_global_percentiles([s2_0920, s2_1024, s2_0221])

    print("Applying percentile normalization to training scenes...")
    s2_0920 = apply_percentile_normalization(s2_0920, pct)
    s2_1024 = apply_percentile_normalization(s2_1024, pct)
    s2_0221 = apply_percentile_normalization(s2_0221, pct)
    print(pct)

    # Spectral Index Calculation
    print("Calculating spectral indices...")
    print("Calculating NDSI...")
    
    s2_0920["NDSI"]=compute_spectral_index(s2_0920, b1="B3", b2="B11", index_type="NDSI")
    s2_1024["NDSI"]=compute_spectral_index(s2_1024, b1="B3", b2="B11", index_type="NDSI")
    s2_0221["NDSI"]=compute_spectral_index(s2_0221, b1="B3", b2="B11", index_type="NDSI")

    print("Calculating NDVI...")
    s2_0920["NDVI"]=compute_spectral_index(s2_0920, b1="B8", b2="B4", index_type="NDVI")
    s2_1024["NDVI"]=compute_spectral_index(s2_1024, b1="B8", b2="B4", index_type="NDVI")
    s2_0221["NDVI"]=compute_spectral_index(s2_0221, b1="B8", b2="B4", index_type="NDVI")

    print("Calculating NDWI...")
    s2_0920["NDWI"]=compute_spectral_index(s2_0920, b1="B3", b2="B8", index_type="NDWI")
    s2_1024["NDWI"]=compute_spectral_index(s2_1024, b1="B3", b2="B8", index_type="NDWI")
    s2_0221["NDWI"]=compute_spectral_index(s2_0221, b1="B3", b2="B8", index_type="NDWI")

    #-----------------------------
    print("Loading DEM Features")
    dem=rxr.open_rasterio(dem_p, masked=True).squeeze()
    dem_10m = dem.rio.reproject_match(s2_0920["B2"], resampling=Resampling.average)
    dem_10m=crop_s2_to_binary_mask(dem_10m, fsc_grid)   

    #attach to all scenes
    s2_0920["DEM"]=dem_10m
    s2_1024["DEM"]=dem_10m
    s2_0221["DEM"]=dem_10m


    #-----------------------------
    print("Generating Validation Scene out of two S2 scenes of the same date...")
    s2_0606_0_p=paths["data"]["s2_dir"]/"S2_2025_06_06_0.tif"
    s2_0606_1_p=paths["data"]["s2_dir"]/"S2_2025_06_06_1.tif"
    s2_0606_0_mask_p=paths["data"]["s2_dir"]/"masking/S2_2025-06-06_0_mask.shp"
    s2_0606_1_mask_p=paths["data"]["s2_dir"]/"masking/S2_2025-06-06_1_mask.shp"

    print("Loading both S2 Scenes, clipping and normalizing them...")
    s2_0606_0=load_sentinel_scene(s2_0606_0_p)
    s2_0606_1=load_sentinel_scene(s2_0606_1_p)
    s2_0606_0=crop_s2_to_binary_mask(s2_0606_0, fsc_grid)
    s2_0606_1=crop_s2_to_binary_mask(s2_0606_1, fsc_grid)
    s2_0606_0 = apply_percentile_normalization(s2_0606_0, pct)
    s2_0606_1 = apply_percentile_normalization(s2_0606_1, pct)

    print("Applying vector masks to both scenes...")

    s2_0606_0_masked=vectormask(s2_0606_0, s2_0606_0_mask_p)
    s2_0606_1_masked=vectormask(s2_0606_1, s2_0606_1_mask_p)

    print("Merging both masked scenes into one validation scene...")
    s2_merged = merge_s2_scenes(s2_0606_0_masked, s2_0606_1_masked)
    print(s2_merged["B2"].min().item(), s2_merged["B2"].max().item())
    print(s2_merged["B3"].min().item(), s2_merged["B3"].max().item())
    print(s2_merged["B4"].min().item(), s2_merged["B4"].max().item())
    print("Calculating spectral indices for validation scene...")
    s2_merged["NDSI"]=compute_spectral_index(s2_merged, b1="B3", b2="B11", index_type="NDSI")
    s2_merged["NDVI"]=compute_spectral_index(s2_merged, b1="B8", b2="B4", index_type="NDVI")
    s2_merged["NDWI"]=compute_spectral_index(s2_merged, b1="B3", b2="B8", index_type="NDWI")
    s2_merged["DEM"]=dem_10m

    print("Sentinel-2 Data Preprocessing Complete.")

    print("Attaching FSC GT to processed Sentinel-2 scenes...")
    s2_0920["FSC_GT"]=fsc_0920
    s2_1024["FSC_GT"]=fsc_1024
    s2_0221["FSC_GT"]=fsc_0225
    s2_merged["FSC_GT"]=fsc_0604.where(s2_merged["B2"].notnull())



    print("Saving processed Sentinel-2 scenes to disk...")

    save_dataset_as_tiff(s2_0920, paths["data"]["processed_s2"] / "s2_20240920_processed.tif")
    save_dataset_as_tiff(s2_1024, paths["data"]["processed_s2"] / "s2_20241024_processed.tif")
    save_dataset_as_tiff(s2_0221, paths["data"]["processed_s2"] / "s2_20250221_processed.tif")
    save_dataset_as_tiff(s2_merged, paths["data"]["processed_s2"] / "VALIDATION_s2_20250606_processed.tif")


    #----------------------------
    print("Flattening features and predictors for ML Pipeline")

    df_0920 = flatten_dataset(s2_0920)
    df_1024 = flatten_dataset(s2_1024)
    df_0221 = flatten_dataset(s2_0221)
    df_0920["scene_id"] = "20240920"
    df_1024["scene_id"] = "20241024"
    df_0221["scene_id"] = "20250221"

    df_train = pd.concat([df_0920, df_1024, df_0221], ignore_index=True)

    df_val = flatten_dataset(s2_merged)
    df_val["scene_id"]  = "20250606"
    out_dir = paths["data"]["dataframes"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save training data
    train_path = out_dir / "train_fsc.csv"
    df_train.to_csv(train_path, index=False)
    print(f"Saved training dataframe → {train_path}")

    # Save validation data
    val_path = out_dir / "val_fsc.csv"
    df_val.to_csv(val_path, index=False)
    print(f"Saved validation dataframe → {val_path}")

    
if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"\n Script completed in {end - start:.2f} seconds.")
