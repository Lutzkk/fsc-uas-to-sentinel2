# svm_feature_subset_checkpointed.py

import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import joblib
from datetime import datetime
from contextlib import contextmanager
import hashlib

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# -------------------------------------------------------------------
# Progress bar wrapper (joblib copypaste from llm)
# -------------------------------------------------------------------
@contextmanager
def tqdm_joblib(tqdm_object):
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
# Load data
# -------------------------------------------------------------------
import sys
root = Path(__file__).resolve().parents[1] #repo root
sys.path.append(str(root)) #crawl up one level to project root
from utils.paths import load_paths
paths=load_paths()

df_train = pd.read_csv(paths["data"]["dataframes"] / "train_fsc.csv")
df_val   = pd.read_csv(paths["data"]["dataframes"] / "val_fsc.csv")

spectral_features = [
    "B2","B3","B4","B5","B6","B7",
    "B8","B8A","B11","B12",
    "NDSI","NDVI","NDWI","DEM"
]

X_train_full = df_train[spectral_features]
y_train = df_train["FSC_GT"]

X_val_full = df_val[spectral_features]
y_val = df_val["FSC_GT"]


# -------------------------------------------------------------------
# Output path + checkpoint loading
# -------------------------------------------------------------------
out_path = root / "results/svm_brute_force/svm_feature_subset_results.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)

if out_path.exists():
    existing_df = pd.read_csv(out_path)
    completed_hashes = set(existing_df["subset_id"])
    print(f"Resuming run: {len(completed_hashes)} subsets already done.")
else:
    existing_df = pd.DataFrame()
    completed_hashes = set()
    print("Starting fresh run.")


# -------------------------------------------------------------------
# Helper: subset hash (unique ID)
# -------------------------------------------------------------------
def subset_hash(features):
    key = ",".join(sorted(features))
    return hashlib.md5(key.encode()).hexdigest()


# -------------------------------------------------------------------
# Evaluate subset
# -------------------------------------------------------------------
def evaluate_subset(feature_subset):

    subset_id = subset_hash(feature_subset)

    if subset_id in completed_hashes:
        return None

    X_train = X_train_full[list(feature_subset)]
    X_val   = X_val_full[list(feature_subset)]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=50, epsilon=0.01, gamma="scale"))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae  = mean_absolute_error(y_val, preds)
    r2   = r2_score(y_val, preds)

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"SVM ({len(feature_subset)}): {list(feature_subset)}, "
        f"RMSE={rmse:.4f}"
    )

    return {
        "subset_id": subset_id,
        "features": list(feature_subset),
        "n_features": len(feature_subset),
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


# -------------------------------------------------------------------
# Generate subsets
# -------------------------------------------------------------------
MIN_FEATURES = 3
MAX_FEATURES = len(spectral_features)

all_subsets = []
for k in range(MIN_FEATURES, MAX_FEATURES + 1):
    all_subsets.extend(list(combinations(spectral_features, k)))

pending_subsets = [
    s for s in all_subsets
    if subset_hash(s) not in completed_hashes
]

print(f"Pending subsets: {len(pending_subsets):,}")


# -------------------------------------------------------------------
# Run in batches (checkpoint safe)
# -------------------------------------------------------------------
N_JOBS = 32
BATCH_SIZE = 50  # save every 50 results

try:
    with tqdm(total=len(pending_subsets), desc="Evaluating SVM subsets") as pbar:

        for i in range(0, len(pending_subsets), BATCH_SIZE):

            batch = pending_subsets[i:i+BATCH_SIZE]

            batch_results = Parallel(n_jobs=N_JOBS)(
                delayed(evaluate_subset)(subset)
                for subset in batch
            )

            batch_results = [r for r in batch_results if r is not None]

            if batch_results:
                batch_df = pd.DataFrame(batch_results)

                if out_path.exists():
                    batch_df.to_csv(out_path, mode="a", header=False, index=False)
                else:
                    batch_df.to_csv(out_path, index=False)

            pbar.update(len(batch))

except KeyboardInterrupt:
    print("\nInterrupted! progress saved.")


# -------------------------------------------------------------------
# Final sort
# -------------------------------------------------------------------
final_df = pd.read_csv(out_path)
final_df = final_df.sort_values("RMSE")
final_df.to_csv(out_path, index=False)

print("\n Finished. Top results:")
print(final_df.head(10))