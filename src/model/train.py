import json
import yaml
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn

# paths
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
        # tags (PR / CI friendly)
        mlflow.set_tags({
            "git_branch": os.getenv("GITHUB_REF_NAME", "local"),
            "git_sha": os.getenv("GITHUB_SHA", "local"),
            "stage": "train",
        })

        # log params
        mlflow.log_params({
            "model": "LogisticRegression",
            "random_state": model_cfg["random_state"],
            "max_iter": model_cfg["max_iter"],
            "C": model_cfg["C"],
            "solver": model_cfg["solver"],
        })

        X_train = pd.read_parquet(TRAIN_X)
        y_train = pd.read_parquet(TRAIN_Y).iloc[:, 0]

        X_val = pd.read_parquet(VAL_X)
        y_val = pd.read_parquet(VAL_Y).iloc[:, 0]


        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        random_state=model_cfg["random_state"],
                        max_iter=model_cfg["max_iter"],
                        C=model_cfg["C"],
                        solver=model_cfg["solver"],
                    ),
                ),
            ]
        )

        # 1️⃣ Train
        pipeline.fit(X_train, y_train)

        # 2️⃣ Validation evaluation
        val_preds = pipeline.predict(X_val)

        val_accuracy = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds)

        metrics = {
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
        }

        # log metrics to MLflow
        mlflow.log_metrics(metrics)

        # save artifacts
        Path("models").mkdir(exist_ok=True)
        Path("metrics").mkdir(exist_ok=True)

        import joblib
        joblib.dump(pipeline, MODEL_PATH)

        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)

        print("✅ Training completed")
        print(metrics)

if __name__ == "__main__":
    main()
