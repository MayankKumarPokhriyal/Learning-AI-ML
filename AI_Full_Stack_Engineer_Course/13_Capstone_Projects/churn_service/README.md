# churn-service — customer churn prediction with a profit-based decision

A complete, small, production-style ML project: it predicts which telecom customers are likely to cancel, decides **who should get a retention offer** by maximizing expected profit, and serves that decision through a tested FastAPI service in a Docker container.

It is the project folder of the course notebook [`01_End_to_End_ML_Project.ipynb`](../01_End_to_End_ML_Project.ipynb), which walks through every step and runs all of this code.

## What it does

| Step | Where |
|---|---|
| Download the data and verify its SHA-256 checksum | `src/churn_service/data.py` |
| Schema and data-quality validation (types, allowed values, ranges, consistency rules, blank `TotalCharges`) | `data.py` |
| Stratified train / validation / test split, saved to `data/splits/` | `data.py` |
| Feature engineering inside the scikit-learn pipeline (no training/serving skew) | `features.py` |
| Logistic regression vs gradient boosting with 5-fold cross-validation, one-standard-error selection rule | `train.py` |
| Decision threshold that maximizes expected profit **on the validation split** | `business.py` |
| One final test evaluation with bootstrap confidence intervals, slices, and a generated model card | `evaluate.py` |
| Model + metadata (version, data hashes, metrics, threshold, assumptions) + drift reference statistics | `predict.py`, `monitoring.py` |
| FastAPI service: lifespan model loading, Pydantic v2 validation, single + batch predictions, health/readiness | `api.py`, `schemas.py` |
| Unit, API, and model-quality tests | `tests/` |
| Multi-stage, non-root Docker image with a health check | `Dockerfile`, `.dockerignore` |
| Example GitHub Actions pipeline (lint → train + test → build + smoke test) | `ci/github-actions.yml` |

## Business assumptions (edit these!)

The threshold depends on assumptions that are **not** in the data. They live in `src/churn_service/config.py`:

| Assumption | Default | Meaning |
|---|---|---|
| `offer_cost` | $60 | cost of one retention offer, paid for every contacted customer |
| `acceptance_rate` | 35% | share of would-be churners who stay because of the offer |
| `months_retained` | 12 | months of revenue kept when a churner is saved |

Contacting a churner is worth `acceptance_rate × MonthlyCharges × months_retained − offer_cost`; contacting a loyal customer costs `offer_cost`. Change the numbers, retrain, and the threshold is re-tuned.

## Quickstart

```bash
# 1. Environment (uv: https://docs.astral.sh/uv/) — locked runtime versions + dev tools
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]" -c requirements.lock

# 2. Train: downloads ~1 MB of data to data/, writes artifacts/ (model.joblib, metadata.json,
#    reference_stats.json, MODEL_CARD.md). Takes well under a minute on a laptop.
python -m churn_service.train

# 3. Test (unit + API + model-quality gate)
pytest

# 4. Serve
uvicorn churn_service.api:app --reload
curl -s localhost:8000/ready
curl -s -X POST localhost:8000/predict -H "content-type: application/json" \
  -d '{"gender":"Female","senior_citizen":0,"partner":"Yes","dependents":"No","tenure":1,"phone_service":"No","multiple_lines":"No phone service","internet_service":"DSL","online_security":"No","online_backup":"Yes","device_protection":"No","tech_support":"No","streaming_tv":"No","streaming_movies":"No","contract":"Month-to-month","paperless_billing":"Yes","payment_method":"Electronic check","monthly_charges":29.85,"total_charges":29.85}'
# Interactive docs: http://localhost:8000/docs

# 5. Container (after training)
docker build -t churn-service:1.0.0 .
docker run --rm -p 8000:8000 churn-service:1.0.0
```

`data/` and `artifacts/` are git-ignored: they are **recreated** by step 2. The model version (`1.0.0+<fingerprint>`) changes whenever the training data, hyperparameters, threshold, or assumptions change.

`requirements.lock` pins exact runtime versions for Docker and CI (the model is a pickle, so train and serve with the same scikit-learn). Regenerate it with:

```bash
uv pip compile pyproject.toml -o requirements.lock --python-version 3.12
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness — the process is up |
| GET | `/ready` | readiness — the model is loaded (503 otherwise) |
| GET | `/model` | version, threshold, assumptions, headline test metrics |
| POST | `/predict` | one customer → churn probability, `contact` decision, threshold, model version |
| POST | `/predict/batch` | up to 1,000 customers in one vectorized call |

Requests use snake_case fields with the dataset's allowed values; invalid categories, out-of-range numbers, unknown fields, and inconsistent records (e.g. streaming TV without internet) return **422**.

## Monitoring plan

- **Log** every request's inputs, probability, decision, and model version (without personal identifiers).
- **Drift:** weekly, compare recent inputs and scores with `artifacts/reference_stats.json` using `churn_service.monitoring.drift_report` (PSI < 0.1 stable, 0.1–0.25 watch, > 0.25 investigate).
- **Outcomes:** churn labels arrive ~1 month later; then recompute ROC-AUC, recall at the threshold, and realized campaign profit.
- **Retrain** when drift or outcome metrics cross their alert levels, or when the business assumptions change; the CI quality gate blocks worse models.

## Data and license

[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) sample data (Apache-2.0): 7,043 fictional customers, 21 columns. Quirk: `TotalCharges` is stored as text and 11 rows (all with `tenure == 0`) contain a blank — the validator reports them and cleaning sets them to 0.
