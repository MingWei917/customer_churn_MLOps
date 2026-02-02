import json
import sys

with open("baseline_metrics.json") as f:
    base = json.load(f)

with open("metrics.json") as f:
    pr = json.load(f)

THRESHOLD = 0.01

if pr["auc"] < base["auc"] - THRESHOLD:
    print("❌ Regression detected")
    sys.exit(1)

print("✅ No regression")
