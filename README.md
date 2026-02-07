## Overview

This project implements a **production-grade MLOps pipeline** for customer churn prediction using:

- **GitHub** (PR-based workflow)
- **DVC** (data & model versioning)
- **MLflow** (experiment tracking & model registry)
- **DagsHub** (remote storage & UI)
- **GitHub Actions** (CI/CD)

---

## Pipeline Flow
raw → preprocess → feature → split → train → evaluate → gate

---

Each stage is:

- **Reproducible** (DVC)
- **Traceable** (MLflow)
- **Validated** (CI + validation gate)

---

## Key Guarantees

- No unvalidated model reaches `main`
- Every metric is reproducible
- Every model is traceable to **data + code**
- Rollback is always possible

---

## Final Status

| Area                 | Status |
| -------------------- | ------ |
| GitHub PR Flow       | ✅      |
| DVC Versioning       | ✅      |
| MLflow Tracking      | ✅      |
| DagsHub Remote       | ✅      |
| CI/CD                | ✅      |
| Production Readiness | ✅      |

---

##  Next Steps (Optional)

- Model serving (FastAPI)
- Data drift detection
- Scheduled retraining
- Feature Store integration


