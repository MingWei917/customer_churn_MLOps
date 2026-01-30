import json
import sys

with open("base_metrics.json") as f:
    base = json.load(f)

with open("pr_metrics.json") as f:
    pr = json.load(f)

THRESHOLD = 0.01

if pr["auc"] < base["auc"] - THRESHOLD:
    print("❌ Regression detected")
    sys.exit(1)

print("✅ No regression")
