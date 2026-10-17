# feature_subset_search.py

import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import joblib
from datetime import datetime

from contextlib import contextmanager
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Progress bar wrapper
@contextmanager
def tqdm_joblib(tqdm_object):
    """Patch joblib to report into tqdm progress bar."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


# -------------------------------------------------------------------
#  Load data
# -------------------------------------------------------------------
import sys
root = Path(__file__).resolve().parents[1] #repo root
sys.path.append(str(root)) #crawl up one level to project root
from utils.paths import load_paths
paths=load_paths()
df_train = pd.read_csv(paths["data"]["dataframes"] / "train_fsc.csv")
df_val   = pd.read_csv(paths["data"]["dataframes"] / "val_fsc.csv")

drop_cols = ["FSC_GT", "x", "y", "band", "spatial_ref", "scene_id"]
all_features = [c for c in df_train.columns if c not in drop_cols]

X_train_full = df_train[all_features]
y_train = df_train["FSC_GT"]

X_val_full = df_val[all_features]
y_val = df_val["FSC_GT"]


# -------------------------------------------------------------------
#  Function: evaluate one subset
# -------------------------------------------------------------------
def evaluate_subset(feature_subset):
    """Train RF on one subset, print results, and return them."""
    
    X_train = X_train_full[list(feature_subset)]
    X_val   = X_val_full[list(feature_subset)]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        n_jobs=1,          # MPI-friendly
        random_state=42
    )
    
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae  = mean_absolute_error(y_val, preds)
    r2   = r2_score(y_val, preds)

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"Features ({len(feature_subset)}): {list(feature_subset)}\n"
        f"    RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}\n"
    )

    return {
        "features": list(feature_subset),
        "n_features": len(feature_subset),
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


# -------------------------------------------------------------------
#  Generate all feature subsets
# -------------------------------------------------------------------
MIN_FEATURES = 3
MAX_FEATURES = len(all_features)   # or set to e.g., 12

all_subsets = []
for k in range(MIN_FEATURES, MAX_FEATURES + 1):
    all_subsets.extend(list(combinations(all_features, k)))

print(f"Total subsets to evaluate: {len(all_subsets):,}")


# -------------------------------------------------------------------
#  Run evaluations in parallel
# -------------------------------------------------------------------
N_JOBS = 32   #cores available on this machine

all_subsets = list(all_subsets)  # ensure materialized

with tqdm_joblib(tqdm(desc="Evaluating subsets", total=len(all_subsets))) as progress:
    results = Parallel(n_jobs=N_JOBS)(
        delayed(evaluate_subset)(subset)
        for subset in all_subsets
    )

# Convert to DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="RMSE")

# Save results
out_path = root / "results/feature_subset_results.csv"
out_path.parent.mkdir(exist_ok=True)
results_df.to_csv(out_path, index=False)

print(f"Saved results to: {out_path}")
print(results_df.head(10))
