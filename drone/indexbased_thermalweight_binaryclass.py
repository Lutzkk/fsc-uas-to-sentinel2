import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from skimage.filters import threshold_otsu
import os
from pathlib import Path

class NDSIWithThermal:
    def __init__(self, input_path, output_ndsi, output_binary):
        self.input_path = input_path
        self.output_ndsi = output_ndsi
        self.output_binary = output_binary
        self.target_crs = "EPSG:32632"

    def ensure_reprojection(self):
        with rasterio.open(self.input_path) as src:
            if src.crs.to_string() != self.target_crs:
                out_path = self.input_path.replace(".tif", "_reproj.tif")
                if os.path.exists(out_path):
                    print(f"Using existing reprojected raster: {out_path}")
                    self.input_path = out_path
                else:
                    print("Reprojecting raster to EPSG:32632")
                    transform, width, height = calculate_default_transform(
                        src.crs, self.target_crs, src.width, src.height, *src.bounds)
                    profile = src.profile.copy()
                    profile.update({
                        'crs': self.target_crs,
                        'transform': transform,
                        'width': width,
                        'height': height,
                        'BIGTIFF': 'YES',
                        'compress': 'lzw'
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
                    self.input_path = out_path
                    print(f"Saved reprojected raster to {out_path}")
            else:
                print("Raster is already in EPSG:32632.")

    def run(self):
        print("Opening multispectral image")
        with rasterio.open(self.input_path) as src:
            green = src.read(2).astype(np.float32)
            nir = src.read(5).astype(np.float32)
            thermal=src.read(6).astype(np.float32)
            profile = src.profile

        print("Masking invalid pixels")
        invalid_mask = (green == 65535) & (nir == 65535)

        print("normalizing thermal data")
        thermal_valid=thermal[~invalid_mask]
        min_temp=np.percentile(thermal_valid, 10)
        max_temp=np.percentile(thermal_valid, 90)
        thermal = np.clip(thermal, min_temp, max_temp)
        thermal_normalized= (thermal - min_temp) / (max_temp - min_temp)

        print("Calculating pseudo NDSI")
        ndsi = (green - nir) / (green + nir + 1e-6)
        alpha=0.3 #value between 0 and 1 -> 0 == no thermal correction, 1 == full thermal correction
        ndsi_enhanced=ndsi + alpha *(1-thermal_normalized)

        print("Applying Otsu thresholding")
        valid_pixels = ndsi_enhanced[~invalid_mask]
        threshold = threshold_otsu(valid_pixels)

        print("Generating binary snow mask")
        binary = np.where(ndsi_enhanced >= threshold, 1, 0).astype(np.uint8)
        binary[invalid_mask] = 255

        print("Saving outputs")
        ndsi_masked = np.where(invalid_mask, np.nan, ndsi_enhanced)

        profile.update(dtype='float32', count=1)
        with rasterio.open(self.output_ndsi, 'w', **profile) as dst:
            dst.write(ndsi_masked.astype(np.float32), 1)

        profile.update(dtype='uint8', nodata=255)
        with rasterio.open(self.output_binary, 'w', **profile) as dst:
            dst.write(binary, 1)

        with rasterio.open(self.output_binary) as src:
            print("Unique values in binary output:", np.unique(src.read(1)))

        print("Done")

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1])) #crawl up one level to project root
    from utils.paths import load_paths
    paths=load_paths()
    processor = NDSIWithThermal(
        input_path=str(paths["data"]["drone_scenes"]/"2025_06_04/20250604_Zugspitze_100m_DJIM300Altum_OM.tif"),
        output_ndsi=paths["data"]["binary_snow_raw"]/"2025_06_04/ndsi_20250604.tif",
        output_binary=paths["data"]["binary_snow_raw"]/"2025_06_04/binary_snow_map_20250604.tif"
    )
    # processor.ensure_reprojection()
    processor.run()
    print("Done with the first NDSI processing.")

