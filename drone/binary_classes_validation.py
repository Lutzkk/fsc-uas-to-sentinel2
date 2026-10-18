import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, f1_score, precision_score, recall_score
from pathlib import Path
import csv
import sys
sys.path.append(str(Path(__file__).resolve().parents[1])) #crawl up one level to project root
from utils.paths import load_paths
paths=load_paths()

VAL_DIR = paths["data"]["validation_polygons"]

scenes = [
    {"name": "20240920 (RF/RGB)",      "raster": paths["data"]["binary_snow_raw"] / "2024_09_20/binary_snow_20240920_cleaned_disksize0.tif", "polygons": VAL_DIR / "20240920_validation_polys_32632.shp"},
    {"name": "20241025 (RF/RGB)",      "raster": paths["data"]["binary_snow_raw"] / "2024_10_24/binary_snow_20241024_cleaned_disksize2.tif", "polygons": VAL_DIR / "20241025_validation_polys_32632.shp"},
    {"name": "20250225 (NDSI)",        "raster": paths["data"]["binary_snow_raw"] / "2025_02_25/20250225_binary_snow.tif", "polygons": VAL_DIR / "20250225_validation_polys_32632.shp"},
    {"name": "20250604 (NDSI+thermal)","raster": paths["data"]["binary_snow_raw"] / "2025_06_04/binary_snow_map_20250604.tif", "polygons": VAL_DIR / "20250604_validation_polys_32632.shp"},
]

OUT_CSV = VAL_DIR / "validation_results.csv"
OUT_TXT = VAL_DIR / "validation_results.txt"

csv_rows = []

with open(OUT_TXT, "w") as txt:

 for scene in scenes:
    print("\n" + "="*60)
    print(f"SCENE: {scene['name']}")
    print("="*60)

    gdf = gpd.read_file(scene["polygons"])

    # --- check for invalid ids ---
    invalid = gdf[~gdf["id"].isin([0, 1])]
    if not invalid.empty:
        print(f"  !! WARNING: {len(invalid)} polygon(s) with id not in [0,1]:")
        print(invalid[["id", "geometry"]].to_string())

    with rasterio.open(scene["raster"]) as src:
        raster_data = src.read(1, masked=True)  # masked array, nodata auto-masked
        transform = src.transform
        height, width = src.shape
        raster_crs = src.crs

        print(f"  Raster unique values: {np.unique(raster_data.compressed())}")

        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        ref_raster = rasterize(
            [(geom, val) for geom, val in zip(gdf.geometry, gdf["id"])],
            out_shape=(height, width),
            transform=transform,
            fill=255,
            dtype=np.uint8
        )

    has_ref   = ref_raster != 255
    not_nodata = ~raster_data.mask

    valid = has_ref & not_nodata
    print(f"  Valid pixels for validation: {valid.sum()}")

    if valid.sum() == 0:
        print("  !! No valid pixels — check CRS alignment or polygon extent.")
        continue

    y_true = ref_raster[valid].astype(int)
    y_pred = raster_data.data[valid].astype(int)

    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    oa  = np.sum(y_true == y_pred) / len(y_true)
    kappa = cohen_kappa_score(y_true, y_pred)

    print(f"\n  Confusion matrix (rows=true, cols=pred) [0=no-snow, 1=snow]:")
    print(f"            pred 0   pred 1")
    print(f"  true 0  {cm[0,0]:>8}  {cm[0,1]:>8}")
    print(f"  true 1  {cm[1,0]:>8}  {cm[1,1]:>8}")
    print(f"\n  Overall Accuracy : {oa:.4f}")
    print(f"  Cohen's Kappa    : {kappa:.4f}")
    print(f"\n  Per-class report:")
    report = classification_report(y_true, y_pred, target_names=["no-snow", "snow"], digits=4)
    print(report)

    #write to txt
    txt.write(f"SCENE: {scene['name']}\n")
    txt.write(f"Valid pixels for validation: {valid.sum()}\n")
    txt.write(f"Confusion matrix (rows=true, cols=pred) [0=no-snow, 1=snow]:\n{cm}\n")
    txt.write(f"Overall Accuracy : {oa:.4f}\nCohen's Kappa    : {kappa:.4f}\n{report}\n\n")

    #collect row for csv
    csv_rows.append({
        "scene": scene["name"],
        "n_pixels": int(valid.sum()),
        "OA": round(oa, 4),
        "kappa": round(kappa, 4),
        "precision_snow": round(precision_score(y_true, y_pred), 4),
        "recall_snow": round(recall_score(y_true, y_pred), 4),
        "f1_snow": round(f1_score(y_true, y_pred), 4),
        "tn": cm[0,0], "fp": cm[0,1], "fn": cm[1,0], "tp": cm[1,1],
    })

#write csv
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\nSaved: {OUT_CSV}")
print(f"Saved: {OUT_TXT}")