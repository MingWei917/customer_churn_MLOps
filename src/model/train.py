import json
import yaml
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn
import os
import joblib

# --------------------
# paths
# --------------------
CFG_PATH = "configs/model/train.yaml"
METRICS_PATH = "metrics/metrics.json"
MODEL_PATH = "models/model.pkl"

TRAIN_X = "data/split/train_features.parquet"
TRAIN_Y = "data/split/train_labels.parquet"
VAL_X = "data/split/val_features.parquet"
VAL_Y = "data/split/val_labels.parquet"


def load_config():
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    model_cfg = cfg["model"]

    # --------------------
    # MLflow setup
    # --------------------
    mlflow.set_experiment("customer_churn")

    with mlflow.start_run():
        mlflow.set_tags({
            "git_branch": os.getenv("GITHUB_REF_NAME", "local"),
            "git_sha": os.getenv("GITHUB_SHA", "local"),
            "stage": "train",
        })

        # --------------------
        # Log params
        # --------------------
        mlflow.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": model_cfg["n_estimators"],
            "max_depth": model_cfg["max_depth"],
            "min_samples_split": model_cfg["min_samples_split"],
            "min_samples_leaf": model_cfg["min_samples_leaf"],
            "random_state": model_cfg["random_state"],
        })

        # --------------------
        # Load data
        # --------------------
        X_train = pd.read_parquet(TRAIN_X)
        y_train = pd.read_parquet(TRAIN_Y).iloc[:, 0]

        X_val = pd.read_parquet(VAL_X)
        y_val = pd.read_parquet(VAL_Y).iloc[:, 0]

        # --------------------
        # Model pipeline
        # (No scaler for RF)
        # --------------------
        pipeline = Pipeline(
            steps=[
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=model_cfg["n_estimators"],
                        max_depth=model_cfg["max_depth"],
                        min_samples_split=model_cfg["min_samples_split"],
                        min_samples_leaf=model_cfg["min_samples_leaf"],
                        random_state=model_cfg["random_state"],
                        n_jobs=model_cfg.get("n_jobs", -1),
                    ),
                )
            ]
        )

        # 1️⃣ Train
        pipeline.fit(X_train, y_train)

        # 2️⃣ Validation
        val_preds = pipeline.predict(X_val)

        val_accuracy = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds)

        metrics = {
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
        }

        mlflow.log_metrics(metrics)

        # --------------------
        # MLflow model registry
        # --------------------
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="customer_churn_model"
        )

        # --------------------
        # Save artifacts
        # --------------------
        Path("models").mkdir(exist_ok=True)
        Path("metrics").mkdir(exist_ok=True)

        joblib.dump(pipeline, MODEL_PATH)

        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)

        print("✅ Training completed")
        print(metrics)


if __name__ == "__main__":
    main()
