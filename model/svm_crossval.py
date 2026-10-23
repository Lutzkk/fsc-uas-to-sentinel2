import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

import sys
root = Path(__file__).resolve().parents[1] #repo root
sys.path.append(str(root)) #crawl up one level to project root
from utils.paths import load_paths
paths=load_paths()
df = pd.concat([pd.read_csv(paths["data"]["dataframes"] / "train_fsc.csv"), pd.read_csv(paths["data"]["dataframes"] / "val_fsc.csv")], ignore_index=True)  # merged train+val (same as old full_fsc.csv)

# Best features
best_features = ['B2', 'B3', 'B12']

X = df[best_features]
y = df["FSC_GT"]
groups = df["scene_id"]   

# Best hyperparameters
best_C = 50
best_epsilon = 0.01
best_gamma = "scale"

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(
        kernel="rbf",
        C=best_C,
        epsilon=best_epsilon,
        gamma=best_gamma
    ))
])

logo = LeaveOneGroupOut()

rmse_scores = []
mae_scores = []
r2_scores = []

print("\nRunning Leave-One-Scene-Out Cross Validation\n")

for train_idx, val_idx in logo.split(X, y, groups):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
    mae_scores.append(mean_absolute_error(y_val, preds))
    r2_scores.append(r2_score(y_val, preds))

print("Cross-Validation Results:")
print("RMSE (per scene):", rmse_scores)
print("MAE  (per scene):", mae_scores)
print("R²   (per scene):", r2_scores)
print("\nAverages:")
print("Mean RMSE:", np.mean(rmse_scores))
print("Mean MAE :", np.mean(mae_scores))
print("Mean R²  :", np.mean(r2_scores))
