# Smartwatch Rating Predictor

Predicts a smartwatch's likely customer rating (1–5) from its specs — price discount, battery life, display size, materials, and so on — trained on a dataset of ~450 real smartwatch listings.

**Live demo:** [smartwatch-rating-prediction.onrender.com](https://smartwatch-rating-prediction.onrender.com)

## What's actually in this repo

This project went through many rounds of fixing real bugs — data leakage, overfitting, hardcoded paths, an unstable evaluation metric — before landing on the current pipeline. The numbers below are the honest result, not the best-looking one from an earlier, flawed run.

- **Model:** Lasso Regression
- **Cross-validated R²:** 0.174 (mean over 5 folds, std 0.102)
- **Test R²:** 0.357 · **Adjusted R²:** 0.057 · **MAE:** 0.228 · **RMSE:** 0.358
- **Features:** 22, after cutting from an original 53 to fix a severe feature-to-sample-ratio problem
- **Training rows:** 277 · **Test rows:** 70 (from 347 rows remaining after removing corrupted/placeholder data)

### Read the R² honestly

0.174 means the model explains roughly 17% of the variance in customer ratings. That's a real, modest, non-overfit signal — not a strong one. Every model tried here (Ridge, Lasso, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost) topped out in a similar 0.15–0.28 range on cross-validation. That ceiling is most likely the data itself: customer ratings are subjective, and a spec sheet only weakly determines them. Lasso was chosen over higher-scoring but far more overfit alternatives (see `notebooks/06_final_evaluation.ipynb` for the selection logic).

## Project structure

```
.
├── Dataset/
│   ├── raw/smartwatches.csv          # original data, not committed to git
│   ├── Processed/cleaned_dataset.csv # after 01_data_cleaning
│   └── split/                        # train/test splits, from 03_preprocessing
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   ├── 06_final_evaluation.ipynb
│   └── Model_Evaluation.ipynb        # per-model individual diagnostics
├── src/smartwatch_ml/
│   ├── feature.py                    # single source of truth for raw input -> model input
│   └── predict.py                    # single prediction entry point
├── api/
│   ├── main.py                       # FastAPI app, also serves the frontend
│   ├── schemas.py
│   └── static/index.html             # dynamic frontend, calls the API
├── frontend/
│   └── index_static.html             # standalone version, model weights baked in, no backend needed
├── tests/
│   └── test_predict.py               # 15 tests against real saved artifacts
├── experiments/                      # every non-production model, kept for reference
│   ├── baseline/
│   └── tuned_models/
├── artifacts/                        # the ONE production model and everything it needs to run
│   ├── model.joblib
│   ├── encoder.joblib / scaler.joblib / weight_encoder.joblib
│   ├── feature_order.joblib, categorical/numerical/binary_columns.joblib
│   ├── brand_freq_map.joblib, strap_color_freq_map.joblib (+ defaults)
│   └── metadata.json                 # real training metrics, not cherry-picked
├── results/
│   ├── Hyperparameter_Tuning_Results.csv
│   └── Final_Model_Comparison.csv
├── Requirements.txt
└── .gitignore
```

## How the pipeline works

Run the notebooks in order — each one depends on files the previous one wrote:

```
01_data_cleaning.ipynb        cleans raw data, removes placeholder/corrupted rows, groups rare categories
02_eda.ipynb                  exploratory analysis, no downstream dependency
03_preprocessing.ipynb        train/test split, frequency + ordinal + one-hot encoding, saves all artifacts
04_model_training.ipynb       baseline models, ranked by 5-fold CV
05_hyperparameter_tuning.ipynb  regularized grid search per model
06_final_evaluation.ipynb     selects the production model, saves artifacts/model.joblib + metadata.json
```

### A few specific decisions worth knowing, not just assuming

- **Rows with `Rating == 2.5` or broken price logic (`Current Price > Original Price`) are dropped entirely** in cleaning — these are known placeholder/corrupted values in the raw data, not real signal.
- **`Brand` and `Strap Color` are frequency-encoded, fit on the training split only**, after the train/test split — fitting on the full dataset first was an earlier bug (data leakage) that's since been fixed.
- **`Weight` is ordinal-encoded**, not one-hot — its five bins have a natural light-to-heavy order that one-hot encoding would throw away.
- **Model selection uses cross-validated R² and a CV-based overfitting gap**, not a single train/test split — the ~70-row test set proved too noisy to trust on its own; test R² swung between 0.02 and 0.56 across iterations of the same pipeline with only minor code changes, while CV R² stayed far more stable.
- **`Current Price` and `Original Price` both have a near-zero trained coefficient** (`0.0` and `0.0000075` respectively) and are not asked of the user at all — a fixed placeholder value is used internally for each. Only `Discount Percentage` is collected, since it's the one pricing input with a real, non-negligible effect (`-0.0032`).
- **`Discount Percentage` assumes the user is looking at a real listing**, not inventing a hypothetical one — it's meant to be read directly off a product page (e.g. "43% off"), not guessed. This tool is designed for checking an existing/planned listing, not for a shopper imagining a product that doesn't exist yet.
- **`Number of Ratings` is not collected from users** — it's meaningless for a not-yet-launched product, and defaults to 0 (log1p-transformed) at inference time.

## Running it

### Predict via the static frontend (no backend required)
Open `frontend/index_static.html` directly in a browser. The model's weights are compiled into the page itself. Only works because the production model is linear (Lasso) — if a future retrain picks a tree-based model, this file needs to be regenerated or retired in favor of the dynamic version below.

### Predict via the API + dynamic frontend
```bash
pip install -r Requirements.txt
uvicorn api.main:app --reload --port 8000
```
Open `http://localhost:8000/` — this serves the form and calls `/predict` on the same server.

### Run the tests
```bash
pytest tests/ -v
```

### Retrain and track experiments
Run notebooks `01` through `06`. Tuning and final evaluation log to [DagsHub](https://dagshub.com) via MLflow — set your `repo_owner`/`repo_name` at the top of `05_hyperparameter_tuning.ipynb` and `06_final_evaluation.ipynb` first.

## Deployment

Deployed on [Render](https://render.com) as a single web service:
- **Build command:** `pip install -r Requirements.txt`
- **Start command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

Pushing to `main` triggers an automatic redeploy. Note: `artifacts/*.joblib` files are intentionally committed to git (force-added past `.gitignore`) since Render only deploys what's in the repository — these are the one thing that must ship, unlike everything in `experiments/`.

## Known limitations

- CV R² (~0.17) is modest — treat predictions as a rough estimate, not a precise forecast.
- `Discount Percentage` isn't range-validated (negative or >100% values are accepted without error) — a known, documented gap, see `tests/test_predict.py::TestKnownGaps`.
- The static frontend's baked-in weights go stale if the model is ever retrained — it isn't automatically kept in sync with `artifacts/model.joblib`.