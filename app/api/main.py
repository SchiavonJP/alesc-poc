"""FastAPI serving layer: /predict, /predict/batch, /explain/{record_id}."""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from alesc_poc.pipelines.features.nodes import (
    engineer_features,
    load_ipca_factors,
    correct_inflation,
)
from alesc_poc.pipelines.training.nodes import build_ensemble, load_model_bundle
from alesc_poc.pipelines.explanation.nodes import compute_shap_values, extract_top_shap_features

logger = logging.getLogger(__name__)
app = FastAPI(
    title="ALESC Anomaly Detection API",
    description="Ensemble anomaly detection on ALESC parliamentary expenses.",
    version="0.1.0",
)

# ── Model registry: loaded lazily per category ───────────────────────────────
_MODEL_CACHE: dict[str, dict] = {}
_IPCA_FACTORS: dict[int, float] | None = None
_PARAMS: dict | None = None
_SHAP_CACHE: dict[str, Any] = {}


def _load_params() -> dict:
    global _PARAMS
    if _PARAMS is None:
        import yaml
        with open("conf/base/parameters.yaml") as f:
            _PARAMS = yaml.safe_load(f)
    return _PARAMS


def _load_ipca() -> dict[int, float]:
    global _IPCA_FACTORS
    if _IPCA_FACTORS is None:
        p = _load_params()
        _IPCA_FACTORS = load_ipca_factors(
            os.path.join(p["raw_data_dir"], "ipca_series.csv")
        )
    return _IPCA_FACTORS


def _get_bundles(category_key: str) -> dict:
    if category_key not in _MODEL_CACHE:
        bundles = load_model_bundle(category_key)
        if not bundles:
            raise HTTPException(
                status_code=404,
                detail=f"No trained models found for category '{category_key}'. "
                       f"Run the pipeline first: python -m alesc_poc.pipeline_runner --category {category_key}",
            )
        _MODEL_CACHE[category_key] = bundles
    return _MODEL_CACHE[category_key]


def _record_to_df(record: dict) -> pd.DataFrame:
    """Convert a single expense record dict to a DataFrame compatible with the pipeline."""
    return pd.DataFrame([record])


def _preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply minimal preprocessing needed for inference on arbitrary input."""
    p = _load_params()
    ipca = _load_ipca()

    # Parse Valor if it's a string
    if "Valor" in df.columns and df["Valor"].dtype == object:
        df["Valor"] = df["Valor"].str.replace(".", "").str.replace(",", ".").astype(float)

    # Map column names (accept both raw and normalised)
    rename_map = {
        "Verba": "verba", "Descrição": "descricao", "Conta": "conta",
        "Favorecido": "favorecido", "Trecho": "trecho",
        "Vencimento": "vencimento", "Valor": "valor",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "year" not in df.columns and "vencimento" in df.columns:
        df["vencimento"] = pd.to_datetime(df["vencimento"], errors="coerce")
        df["year"] = df["vencimento"].dt.year.fillna(2025).astype(int)

    df["valor"] = pd.to_numeric(df.get("valor", df.get("Valor", 0)), errors="coerce").fillna(0)
    df["is_reversal"] = df["valor"] < 0

    # IPCA correction
    df["ipca_factor"] = df["year"].map(ipca).fillna(1.0)
    df["valor_adjusted"] = df["valor"] * df["ipca_factor"]

    # Feature engineering (use empty dfs for train/test since we only need infer)
    empty = pd.DataFrame(columns=df.columns)
    _, _, infer_f = engineer_features(
        empty, empty, df,
        p["hash_dim_favorecido"], p["mean_value_group"],
    )
    return infer_f


def _predict_df(df: pd.DataFrame) -> pd.DataFrame:
    """Score a preprocessed DataFrame, auto-routing by verba."""
    from unidecode import unidecode

    results = []
    for verba, group in df.groupby("verba"):
        cat_key = unidecode(str(verba)).upper().replace(" ", "_").replace("/", "_")
        try:
            bundles = _get_bundles(cat_key)
        except HTTPException:
            group["ensemble_flag"] = -1
            group["ensemble_score"] = 0.0
            group["error"] = f"No model for category {cat_key}"
            results.append(group)
            continue

        p = _load_params()
        preds = build_ensemble(
            group,
            bundles.get("iforest"), bundles.get("knn"), bundles.get("lof"),
            bundles.get("gmm"), bundles.get("ocsvm"), bundles.get("sod"),
            p["ensemble_strategy"],
        )
        results.append(preds)

    return pd.concat(results, ignore_index=False) if results else df


# ── Request / Response models ─────────────────────────────────────────────────

class ExpenseRecord(BaseModel):
    verba: str
    conta: str
    favorecido: str | None = None
    vencimento: str | None = None
    valor: float
    year: int | None = None

    model_config = {"extra": "allow"}


class PredictionResponse(BaseModel):
    ensemble_flag: int
    ensemble_score: float
    scores: dict[str, float]
    is_anomaly: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(_MODEL_CACHE.keys())}


@app.post("/predict", response_model=PredictionResponse)
def predict_single(record: ExpenseRecord):
    """Score a single expense record."""
    df = _record_to_df(record.model_dump())
    df = _preprocess_df(df)
    preds = _predict_df(df)

    row = preds.iloc[0]
    score_cols = {
        k.replace("score_", ""): float(row[k])
        for k in preds.columns if k.startswith("score_")
    }
    return PredictionResponse(
        ensemble_flag=int(row.get("ensemble_flag", 0)),
        ensemble_score=float(row.get("ensemble_score", 0.0)),
        scores=score_cols,
        is_anomaly=bool(row.get("ensemble_flag", 0) == 1),
    )


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Score a CSV file of expense records. Returns JSON list of predictions."""
    content = await file.read()
    try:
        df_raw = pd.read_csv(io.BytesIO(content), sep=";", encoding="latin1", on_bad_lines="skip")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    df = _preprocess_df(df_raw)
    preds = _predict_df(df)

    # Return key columns only
    out_cols = ["ensemble_flag", "ensemble_score", "verba", "conta", "valor_adjusted"]
    out_cols += [c for c in preds.columns if c.startswith("score_")]
    out_cols = [c for c in out_cols if c in preds.columns]

    return preds[out_cols].to_dict(orient="records")


@app.get("/explain/{record_id}")
def explain_record(record_id: str, category: str):
    """
    Return SHAP top features for a previously scored record.
    record_id: the pandas index value from predict/batch output.
    category: the verba category key (e.g. COMBUSTIVEIS).
    """
    cache_key = f"{category}_{record_id}"
    if cache_key not in _SHAP_CACHE:
        raise HTTPException(
            status_code=404,
            detail="SHAP values not cached for this record. Re-run /predict/batch first.",
        )
    return _SHAP_CACHE[cache_key]


@app.get("/categories")
def list_categories():
    """List categories with trained models available."""
    model_dir = "outputs/models"
    if not os.path.isdir(model_dir):
        return {"categories": []}
    cats = [
        d for d in os.listdir(model_dir)
        if os.path.isfile(os.path.join(model_dir, d, "iforest.joblib"))
    ]
    return {"categories": sorted(cats)}


if __name__ == "__main__":
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
