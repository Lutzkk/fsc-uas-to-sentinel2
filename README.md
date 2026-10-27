# From UAS to Satellite: Fractional Snow Cover Mapping of the Zugspitzplateau

Estimating fractional snow cover (FSC) at Sentinel-2 resolution (10 m) on the Zugspitzplateau, using very-high-resolution drone imagery as ground truth. This repository contains the scripts used to generate the results presented in the poster "From UAS to Satellite: Fractional Snow Cover Mapping of the Zugspitzplateau" ([PDF](docs/poster_fsc_mapping.pdf)). 

**This is not a modular data pipeline. The scripts are of exploratory nature and were written for a single project. No Data is included (copyright).** 

Maybe someone finds it useful as reference or for building a more general pipeline from this. 

The project was part of my work at the [Earth Observation Research Cluster](https://remote-sensing.org/), Department of Remote Sensing, Institute of Geography and Geology, University of Würzburg, Germany.

## Research Questions

1: Are three drone scenes enough to train a reliable Machine Learning Model to predict fractional snow cover at Sentinel-2 (10m) resolution for the Zugspitzplateau?

2: Which features are important for the Machine Learning model and how do Support Vector Regressor and Random Forest Regressor compare?

## Study area and data

- **Area:** Zugspitzplateau, ~0.51 km², 2206–2572 m a.s.l.
- **UAS:** DJI Matrice 300 RTK with a Zenmuse L1 (RGB, ~2.6 cm) and a MicaSense Altum (multispectral + thermal, ~4.3 cm)
- **Satellite:** Sentinel-2 L2A via Google Earth Engine
- **Elevation:** snow-free DSM from a summer UAS LiDAR flight, aggregated to 10 m

| Scene | UAS flight | Sentinel-2 | Snow classification | Role |
|---|---|---|---|---|
| Sep 2024 | 2024-09-19 (RGB) | 2024-09-19 | Random Forest on RGB | train |
| Oct 2024 | 2024-10-25 (RGB) | 2024-10-24 | Random Forest on RGB + morphological filter | train |
| Feb 2025 | 2025-02-25 (Altum) | 2025-02-21 | pseudo-NDSI (green/NIR) + Otsu thresholding | train |
| Jun 2025 | 2025-06-04 (Altum) | 2025-06-06 (two tiles, manually cloud-masked and merged) | pseudo-NDSI + thermal weighting (α = 0.3) + Otsu | test |

## Repository structure

```
config/paths.yaml        data root + paths to all inputs/outputs
utils/                   loading, resampling, masking, FSC aggregation helpers
preprocessing/           Sentinel-2 download, S2 grid mask of the AOI, FSC ground truth + features
drone/                   binary snow classification of the UAS scenes + accuracy check
model/                   feature subset search (RF, SVR), final model (not included, can be sent on request)
docs/                    poster (PDF)
```

## Workflow

Python scripts use  `config/paths.yaml` to access the data. `utils/` contains helper functions for general gis operations used multiple times. 

The workflow is as follows:

1. **Sentinel-2 download** (Earth Engine exports to Google Drive)
   - `preprocessing/s2_obtain_gee.py` (2024/2025 scenes)
   - move exports from Drive to data/s2/
2. **Binary snow maps** from the UAS scenes
   - `drone/rf_based_binaryclass.py` → `drone/speckling_noise_remove.py` (Sep, Oct)
   - `drone/ndsi.py` (Feb)
   - `drone/indexbased_thermalweight_binaryclass.py` (Jun)
   - `drone/binary_classes_validation.py`: accuracy against independent validation polygons
3. **Sentinel-2 grid mask** of the AOI: `preprocessing/fsc_ground_truth_creator.py`
4. **Ground truth + features**: `preprocessing/feature_engineering.py`
   aggregates the binary maps to FSC on the S2 grid (only pixels with ≥ 90 % drone coverage), computes NDSI/NDVI/NDWI, adds the DSM and writes `train_fsc.csv` / `val_fsc.csv`
5. **Feature subset search**: all 16,278 combinations of 3–14 features
   - `model/rf_feature_subset_search.py`
   - `model/svm_feature_subset_search.py`
6. **Final model**
   - `model/svm_regressor.py`: SVR with B2, B3, B12
   - `model/svm_crossval.py`: leave-one-scene-out cross-validation
   - `model/fsc_prediction_svm.py`: FSC map from the trained model

Manual inputs that no script creates: RF training polygons, validation polygons and the cloud masks for the June Sentinel-2 tiles (all drawn in QGIS).

## Results

### **Binary snow maps** (validation polygons, ~50 per scene)

| Scene | Method | Misclassified / validation pixels | Overall accuracy | Kappa |
|---|---|---|---|---|
| Sep 2024 | RF | 486 / 11,529,945 | 0.99996 | 0.9884 |
| Oct 2024 | RF + filter | 34,990 / 4,576,331 | 0.99235 | 0.9757 |
| Feb 2025 | pseudo-NDSI | 1,996 / 6,352,018 | 0.99969 | 0.9921 |
| Jun 2025 | pseudo-NDSI + thermal | 828 / 2,244,413 | 0.99963 | 0.9993 |

### **FSC regression** (trained on Sep/Oct/Feb, tested on Jun (3 965 pixels))

| Model | Features | R² | RMSE |
|---|---|---|---|
| SVR (final) | B2, B3, B12 | 0.77 | 0.20 | 
| RF (final) | B2, B3, B12 | 0.72 | 0.22 |


### Research Question 1 (Are three scenes enough?)
- Both models achieve stable performances across feature subsets (R² ~0.6-0.7; best: SVR: 0.77, RF: 0.72) 
- Both models reproduce spatial snow patterns in mixed pixels
- Three scenes can be enough, depending on the amount of mixed pixels

### Research Question 2 (Feature importance)
- Random Forest more robust to feature selection than the SVR
- SVR strongly depends on NDSI
- DEM rarely used in either model (for top 5% of models)

## Limitations

- **Same pixels in every scene.** All scenes cover the same 0.51 km², so the June test is a test in time, not in space. No spatial heterogeneity.
- **Unbalanced training data.** Two of three training scenes are almost completely snow-covered (mean FSC 0.998 and 0.95); only October has mixed conditions, which are especially important for the regressors.
- **Inconsistent pixel shift** between drone scenes and Satellite scenes.


## Environment

Python 3.10 with rasterio, rioxarray, xarray, geopandas, scikit-learn, scikit-image, earthengine-api.

```
conda env create -f environment.yml
conda activate fsc_env
```

## Note on AI use

Parts of the code were written with the help of AI tools (code suggestions, debugging).

## Author

Lutz Kleemann (EAGLE M.Sc.)
Earth Observation Research Cluster, Department of Remote Sensing, Institute of Geography and Geology, University of Würzburg

## Supervisors

[Elio Rauth](https://orcid.org/0009-0005-2921-252X), Earth Observation Research Cluster, Department of Remote Sensing, Institute of Geography and Geology, University of Würzburg

[Basil Tufail](https://orcid.org/0009-0004-2884-8799), Earth Observation Research Cluster, Department of Remote Sensing, Institute of Geography and Geology, University of Würzburg
