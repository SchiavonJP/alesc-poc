# Data Science Rules

## Immutable Raw Data
- Never write to `data/01_raw/`. Raw CSVs are read-only.
- All transformations must produce artifacts in `data/02_intermediate/` or deeper.
- Processed artifacts go to `data/03_primary/`, `data/04_feature/`, `data/05_model_output/`.

## No Hardcoded Paths
- Never hardcode absolute paths like `/home/jovyan/`, `/content/`, `/Users/`.
- All paths must be relative to the project root or come from `conf/base/parameters.yaml`.

## Reproducibility
- All models must use `random_state=params:random_seed` (default 42).
- Contamination is always estimated from the training set only, never from test or full data.
- `mean_value` feature is computed on training data only (no leakage to test/infer).

## Model Artifacts
- Save models with `joblib.dump` to `outputs/models/{CATEGORY}/`.
- Never pickle raw DataFrames as model artifacts.

## No PyCaret
- PyCaret is dropped. Use PyOD + scikit-learn directly.
- Do not import `pycaret` in any `src/` file.

## Experiment Results
- Every pipeline run appends to `outputs/pipeline.log`.
- Results go to `outputs/results/{CATEGORY}_anomalies.csv` and `{CATEGORY}_metrics.json`.
- Do not overwrite existing results without running the full pipeline.
