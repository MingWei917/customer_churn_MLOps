from pathlib import Path
import json
import yaml
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score
)

# -----------------------
# Path setup
# -----------------------
BASE_DIR = Path(__file__).resolve().parents[2]

FEATURE_PATH = "data/features/feature.parquet"
LABEL_PATH = "data/features/label.parquet"
CONFIG_PATH = BASE_DIR / "configs/model/train.yaml"
#CONFIG_PATH = "configs/model/train.yaml"

Path("models").mkdir(exist_ok=True)
Path("metrics").mkdir(exist_ok=True)

# -----------------------
# Load config
# -----------------------
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)
# -----------------------
# Load data
# -----------------------
X = pd.read_parquet(FEATURE_PATH)
y = pd.read_parquet(LABEL_PATH).iloc[:, 0]  # Series

# -----------------------
# Train / valid split
# -----------------------
stratify = y if cfg["training"]["stratify"] else None

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=cfg["training"]["test_size"],
    random_state=cfg["model"]["random_state"],
    stratify=stratify,
)

# -----------------------
# Model
# -----------------------
model = LogisticRegression(
    random_state=cfg["model"]["random_state"],
    max_iter=cfg["model"]["max_iter"],
    C=cfg["model"]["C"],
    solver=cfg["model"]["solver"],
)

model.fit(X_train, y_train)

# -----------------------
# Evaluation
# -----------------------
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_val, y_pred),
    "roc_auc": roc_auc_score(y_val, y_proba),
    "precision": precision_score(y_val, y_pred),
    "recall": recall_score(y_val, y_pred),
}

# -----------------------
# Save artifacts
# -----------------------
joblib.dump(model, cfg["output"]["model_path"])

with open(cfg["output"]["metrics_path"], "w") as f:
    json.dump(metrics, f, indent=2)

print("Training finished")
print(metrics)
