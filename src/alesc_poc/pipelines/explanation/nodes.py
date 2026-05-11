"""Explanation pipeline: SHAP TreeExplainer on IForest, waterfall + summary plots."""

from __future__ import annotations

import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def compute_shap_values(
    predictions_df: pd.DataFrame,
    iforest_bundle: dict[str, Any],
    shap_max_records: int,
) -> dict[str, Any]:
    """
    Compute SHAP values using TreeExplainer on IForest's sklearn backend.
    No PCA on the IForest path preserves interpretable feature names.

    Returns dict with: shap_values, expected_value, feature_names, X_explained, record_indices.
    """
    model = iforest_bundle["model"]
    scaler = iforest_bundle["scaler"]
    feature_cols = iforest_bundle["feature_cols"]

    # Work only on flagged records (ensemble_flag == 1), capped at shap_max_records
    flagged = predictions_df[predictions_df["ensemble_flag"] == 1].copy()
    if len(flagged) == 0:
        logger.warning("No flagged records — SHAP computation skipped.")
        return {}

    if len(flagged) > shap_max_records:
        flagged = flagged.sample(shap_max_records, random_state=42)
        logger.info("SHAP: capped to %d records", shap_max_records)

    X = flagged[feature_cols].fillna(0).values.astype(np.float32)
    X_scaled = scaler.transform(X)

    # IForest in PyOD exposes the sklearn model via .detector_
    sklearn_iforest = model.detector_
    explainer = shap.TreeExplainer(sklearn_iforest)
    shap_values = explainer.shap_values(X_scaled)

    logger.info(
        "SHAP computed for %d flagged records, %d features",
        len(flagged), len(feature_cols),
    )
    return {
        "shap_values": shap_values,
        "expected_value": explainer.expected_value,
        "feature_names": feature_cols,
        "X_explained": X_scaled,
        "record_indices": flagged.index.tolist(),
        "original_data": flagged,
    }


def plot_waterfall_charts(
    shap_bundle: dict[str, Any],
    category_key: str,
    top_n: int,
    output_dir: str = "outputs/plots",
) -> list[str]:
    """Generate waterfall SHAP plots for top-N most anomalous records."""
    if not shap_bundle:
        return []

    shap_values = shap_bundle["shap_values"]
    expected_value = shap_bundle["expected_value"]
    feature_names = shap_bundle["feature_names"]
    X_explained = shap_bundle["X_explained"]
    original_data = shap_bundle["original_data"]

    cat_dir = os.path.join(output_dir, category_key)
    os.makedirs(cat_dir, exist_ok=True)
    saved_paths = []

    # Sort by ensemble_score descending (most anomalous first)
    scores = original_data["ensemble_score"].values
    top_idx = np.argsort(scores)[::-1][:top_n]

    for rank, i in enumerate(top_idx):
        explanation = shap.Explanation(
            values=shap_values[i],
            base_values=float(expected_value),
            data=X_explained[i],
            feature_names=feature_names,
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False, max_display=15)
        path = os.path.join(cat_dir, f"waterfall_rank{rank+1:03d}.png")
        plt.savefig(path, bbox_inches="tight", dpi=120)
        plt.close()
        saved_paths.append(path)

    logger.info("Saved %d waterfall plots for %s", len(saved_paths), category_key)
    return saved_paths


def plot_shap_summary(
    shap_bundle: dict[str, Any],
    category_key: str,
    output_dir: str = "outputs/plots",
) -> str:
    """Generate beeswarm SHAP summary plot for the category."""
    if not shap_bundle:
        return ""

    shap_values = shap_bundle["shap_values"]
    feature_names = shap_bundle["feature_names"]
    X_explained = shap_bundle["X_explained"]

    cat_dir = os.path.join(output_dir, category_key)
    os.makedirs(cat_dir, exist_ok=True)

    plt.figure(figsize=(12, 7))
    shap.summary_plot(
        shap_values, X_explained,
        feature_names=feature_names,
        show=False,
        max_display=20,
        plot_type="dot",
    )
    path = os.path.join(cat_dir, "shap_summary.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    logger.info("Saved SHAP summary plot: %s", path)
    return path


def extract_top_shap_features(
    shap_bundle: dict[str, Any],
    n_top: int = 5,
) -> pd.DataFrame:
    """
    For each flagged record return the top-N features by absolute SHAP value.
    Output: DataFrame with record_index, rank, feature_name, shap_value, feature_value.
    Used to populate the results CSV and FastAPI /explain endpoint.
    """
    if not shap_bundle:
        return pd.DataFrame()

    shap_values = shap_bundle["shap_values"]
    feature_names = shap_bundle["feature_names"]
    X_explained = shap_bundle["X_explained"]
    record_indices = shap_bundle["record_indices"]

    rows = []
    for i, rec_idx in enumerate(record_indices):
        sv = shap_values[i]
        top = np.argsort(np.abs(sv))[::-1][:n_top]
        for rank, feat_idx in enumerate(top):
            rows.append({
                "record_index": rec_idx,
                "rank": rank + 1,
                "feature_name": feature_names[feat_idx],
                "shap_value": float(sv[feat_idx]),
                "feature_value": float(X_explained[i, feat_idx]),
            })

    return pd.DataFrame(rows)
