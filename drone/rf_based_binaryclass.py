# rf_snow_binary.py

import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import mapping
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from sklearn.ensemble import RandomForestClassifier
from rasterio.windows import Window
from rasterio.features import rasterize
from joblib import Parallel, delayed
import os


class RandomForestSnowClassifier:
    def __init__(self, rgb_path, training_path):
        self.rgb_path = rgb_path
        self.training_path = training_path
        self.target_crs = "EPSG:32632"

    def ensure_reprojection(self):
        # Reproject raster if needed
        with rasterio.open(self.rgb_path) as src:
            if src.crs.to_string() != self.target_crs:
                out_path = self.rgb_path.replace(".tif", "_reproj.tif")
                if os.path.exists(out_path):
                    print(f"Using existing reprojected raster: {out_path}")
                    self.rgb_path = out_path
                else:
                    print("Reprojecting raster to EPSG:32632...")
                    transform, width, height = calculate_default_transform(
                        src.crs, self.target_crs, src.width, src.height, *src.bounds)
                    profile = src.profile.copy()
                    profile.update({
                        'crs': self.target_crs,
                        'transform': transform,
                        'width': width,
                        'height': height,
                        'BIGTIFF': 'YES',
                        'compress':'lzw'
                    })

                    with rasterio.open(out_path, 'w', **profile) as dst:
                        for i in range(1, src.count + 1):
                            reproject(
                                source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=self.target_crs,
                                resampling=Resampling.nearest
                            )
                    self.rgb_path = out_path
                    print(f"Saved reprojected raster to {out_path}")
            else:
                print("Raster is already in EPSG:32632.")

        # Reproject shapefile if needed
        gdf = gpd.read_file(self.training_path)
        if gdf.crs.to_string() != self.target_crs:
            out_shp = self.training_path.replace(".shp", "_32632.shp")
            if os.path.exists(out_shp):
                print(f"Using existing reprojected shapefile: {out_shp}")
                self.training_path = out_shp
            else:
                print("Reprojecting shapefile to EPSG:32632...")
                gdf = gdf.to_crs(self.target_crs)
                gdf.to_file(out_shp)
                self.training_path = out_shp
                print(f"Saved reprojected shapefile to {out_shp}")
        else:
            print("Shapefile is already in EPSG:32632.")


    def load_rgb(self):
        print("Loading RGB and alpha bands...")
        with rasterio.open(self.rgb_path) as src:
            self.rgb = src.read([1, 2, 3])
            self.alpha = src.read(4)
            self.meta = src.meta.copy()
            self.transform = src.transform
            self.height = src.height
            self.width = src.width
            self.crs = src.crs

    def load_training(self):
        print("Loading training data and rasterizing label raster...")
        gdf = gpd.read_file(self.training_path)

        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['id']))

        self.label_raster = rasterize(
            shapes=shapes,
            out_shape=(self.height, self.width),
            transform=self.transform,
            fill=np.nan,
            dtype='float32'
        )

    def extract_samples(self):
        print("Extracting training samples...")
        X = self.rgb.reshape(3, -1).T
        y = self.label_raster.flatten()
        alpha = self.alpha.flatten()
        mask = (~np.isnan(y)) & (alpha==255)    # Only include pixels with valid labels and full alpha


        self.X_train = X[mask]
        self.y_train = y[mask].astype(int)
        self.X_full = X

    def train_rf(self):
        print("Training random forest classifier...")
        self.clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        self.clf.fit(self.X_train, self.y_train)

    def classify_and_binarize(self, block_size=2048):  # <-- FIXED: now inside the class
        print("Classifying raster in chunks and binarizing result...")
        self.classified = np.zeros((self.height, self.width), dtype=np.uint8)
        self.binary = np.zeros((self.height, self.width), dtype=np.uint8)

        with rasterio.open(self.rgb_path) as src:
            for row in range(0, self.height, block_size):
                for col in range(0, self.width, block_size):
                    win_width = min(block_size, self.width - col)
                    win_height = min(block_size, self.height - row)

                    win = Window(col_off=col, row_off=row,
                                 width=win_width, height=win_height)

                    r = src.read(1, window=win)
                    g = src.read(2, window=win)
                    b = src.read(3, window=win)
                    alpha = src.read(4, window=win)

                    rgb_block = np.stack([r, g, b], axis=-1).reshape(-1, 3)
                    pred = self.clf.predict(rgb_block)
                    pred = pred.reshape((win_height, win_width)).astype(np.uint8)

                    self.classified[row:row + win_height, col:col + win_width] = pred

                    binary = np.where(np.isin(pred, [1, 2]), 1, 2)
                    binary[alpha == 0] = 0
                    self.binary[row:row + win_height, col:col + win_width] = binary

                    print(f"Processed block at ({row}, {col}) with shape {win_height}x{win_width}")

    def save(self, classified_out, binary_out):
        print(f"Saving outputs to:\n  - {classified_out}\n  - {binary_out}")
        meta = self.meta.copy()
        meta.update(dtype='uint8', count=1)

        with rasterio.open(classified_out, 'w', **meta) as dst:
            dst.write(self.classified, 1)

        with rasterio.open(binary_out, 'w', **meta) as dst:
            dst.write(self.binary, 1)

    def run(self, classified_out, binary_out):
        print("Starting classification pipeline...\n")
        #self.ensure_reprojection()
        self.load_rgb()
        self.load_training()
        self.extract_samples()
        self.train_rf()
        self.classify_and_binarize()
        self.save(classified_out, binary_out)
        print("\nPipeline completed successfully.")

from pathlib import Path
import rasterio
import geopandas as gpd

def check_case(name, rgb_path, training_path, classified_out, binary_out):
    print("\n" + "="*80)
    print(f"CHECK: {name}")
    print("="*80)

    rgb_path = Path(rgb_path)
    training_path = Path(training_path)
    classified_out = Path(classified_out)
    binary_out = Path(binary_out)

    # --- existence ---
    print(f"RGB exists:      {rgb_path} -> {rgb_path.exists()}")
    print(f"Training exists: {training_path} -> {training_path.exists()}")

    # --- output dirs ---
    print(f"Classified dir exists: {classified_out.parent} -> {classified_out.parent.exists()}")
    print(f"Binary dir exists:     {binary_out.parent} -> {binary_out.parent.exists()}")

    # --- raster sanity ---
    if rgb_path.exists():
        with rasterio.open(rgb_path) as src:
            print(f"RGB CRS: {src.crs}")
            print(f"RGB size: {src.width} x {src.height}")
            print(f"RGB count (bands): {src.count}")
            if src.count < 4:
                print("!! ERROR: raster has <4 bands. Your code reads alpha band 4.")
            else:
                # check alpha stats quickly
                alpha = src.read(4, window=((0, min(512, src.height)), (0, min(512, src.width))))
                unique = set(alpha.flatten().tolist())
                print(f"Alpha sample unique values (first 512x512): {sorted(list(unique))[:10]}{' ...' if len(unique)>10 else ''}")

    # --- training sanity ---
    if training_path.exists():
        gdf = gpd.read_file(training_path)
        print(f"Training CRS: {gdf.crs}")
        print(f"Training rows: {len(gdf)}")
        print(f"Training columns: {list(gdf.columns)}")
        if "id" not in gdf.columns:
            print("!! ERROR: training shapefile has no 'id' column (required for labels).")
        if gdf.crs is None:
            print("!! ERROR: training CRS is None (needs EPSG:32632 or reproject).")

    # --- output path warnings ---
    for p in [classified_out, binary_out]:
        if p.exists():
            print(f"WARNING: output already exists and will be overwritten: {p}")

if __name__ == "__main__":    
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1])) #crawl up one level to project root
    from utils.paths import load_paths
    paths=load_paths()

    #2024-09-20
    classifier = RandomForestSnowClassifier(
        rgb_path=str(paths["data"]["drone_scenes"]/"2024_09_20/Zugspitze_092024_DJIM300L1_100m_RGB.tif"),
        training_path=str(paths["data"]["training_data"]/"zugspitze_092024_training_4326.shp")
    )
    classifier.run(
        classified_out=paths["data"]["binary_snow_raw"]/"2024_09_20/rf_classified_snow_20240920.tif",
        binary_out=paths["data"]["binary_snow_raw"]/"2024_09_20/binary_snow_20240920.tif"
    )

    #2024-10-24
    classifier = RandomForestSnowClassifier(
        rgb_path=str(paths["data"]["drone_scenes"]/"2024_10_24/Zugspitze_102024_DJIM300L1_100m_RGB.tif"),
        training_path=str(paths["data"]["training_data"]/"zugspitze_102024_training_4326.shp")
    )
    classifier.run(
        classified_out=paths["data"]["binary_snow_raw"]/"2024_10_24/rf_classified_snow_20241024.tif",
        binary_out=paths["data"]["binary_snow_raw"]/"2024_10_24/binary_snow_20241024.tif"
    )
