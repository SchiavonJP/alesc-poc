# ALESC Anomaly Detection Pipeline

Ensemble anomaly detection on ALESC (Assembleia Legislativa de Santa Catarina)
parliamentary expense data (2011–2026). Applies the methodology from the BPED paper
using IForest + KNN + LOF + GMM + OCSVM with consensus voting.

## Reference

This project is inspired by — and applies the methodology of:

> **An ensemble approach to detect anomalies in public expenditures (BPED).**
> *International Journal of Data Science and Analytics*, 2026.
> <https://link.springer.com/article/10.1007/s41060-026-01079-9>

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Requires Python 3.11+. Raw ALESC CSV files go in `data/01_raw/`.

## Run the full pipeline

```bash
# All categories
python -m alesc_poc.pipeline_runner

# Single category (incremental — skips if model already exists)
python -m alesc_poc.pipeline_runner --category COMBUSTIVEIS

# Specific year range
python -m alesc_poc.pipeline_runner --years 2019-2024

# Force retrain
python -m alesc_poc.pipeline_runner --category COMBUSTIVEIS --force
```

## Serving

```bash
# FastAPI (port 8000)
uvicorn app.api.main:app --reload

# Streamlit dashboard (port 8501)
streamlit run app/dashboard/app.py
```

## Outputs

| Path | Contents |
|------|----------|
| `outputs/models/{CATEGORY}/` | Trained model joblib artifacts |
| `outputs/results/{CATEGORY}_anomalies.csv` | Flagged records + scores |
| `outputs/results/all_metrics.json` | Aggregate metrics per category |
| `outputs/plots/` | Boxplots, scatter, SHAP waterfall, temporal trend |

## Claude Code Skills

```
/eda [--category X] [--year Y]    # Run ydata-profiling EDA
/train [--category X] [--force]   # Train models (incremental)
/report [--section dataset|results|discussion|all]  # Draft article sections
```

## Project structure

```
src/alesc_poc/pipelines/
  ingestion/   # CSV load, parse Valor, tag reversals, UTF-8 export
  features/    # IPCA correction, train/test split, feature engineering
  training/    # 5 PyOD models + ensemble (consensus voting)
  explanation/ # SHAP TreeExplainer, waterfall + summary plots
  reporting/   # Metrics JSON, results CSV, LaTeX tables, trend plots
  reversal/    # Separate EDA for devoluções
```

## Article

BRACIS 2026 — General Applications track.  
Target submission: May 4, 2026.
