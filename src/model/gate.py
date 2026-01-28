import json, yaml, sys

metrics = json.load(open("metrics/metrics.json"))
gate = yaml.safe_load(open("configs/model/validation.yaml"))["gate"]

if metrics["val_accuracy"] < gate["min_val_accuracy"]:
    sys.exit("❌ Validation accuracy gate failed")

if metrics["val_f1"] < gate["min_val_f1"]:
    sys.exit("❌ Validation F1 gate failed")

print("✅ Validation gate passed")
