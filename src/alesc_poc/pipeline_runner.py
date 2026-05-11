"""
Full pipeline orchestrator.

Usage:
  python -m alesc_poc.pipeline_runner                       # all categories
  python -m alesc_poc.pipeline_runner --category DIARIAS    # single category
  python -m alesc_poc.pipeline_runner --years 2015-2022     # custom year window
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import pandas as pd
import yaml

from alesc_poc.pipelines.ingestion.nodes import (
    export_intermediate,
    load_raw_expenses,
    parse_valor,
    tag_reversal,
)
from alesc_poc.pipelines.features.nodes import (
    correct_inflation,
    engineer_features,
    load_ipca_factors,
    split_by_category,
    split_train_test_infer,
)
from alesc_poc.pipelines.training.nodes import (
    build_ensemble,
    estimate_contamination,
    save_model_bundle,
    train_gmm,
    train_iforest,
    train_knn,
    train_lof,
    train_ocsvm,
    train_sod,
)
from alesc_poc.pipelines.explanation.nodes import (
    compute_shap_values,
    extract_top_shap_features,
    plot_shap_summary,
    plot_waterfall_charts,
)
from alesc_poc.pipelines.reporting.nodes import (
    compute_metrics,
    compute_temporal_metrics,
    export_results_csv,
    generate_latex_tables,
    plot_boxplot,
    plot_category_profile,
    plot_scatter_anomalies,
    plot_temporal_trend,
    save_metrics_json,
)
from alesc_poc.pipelines.reversal.nodes import analyze_reversals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def load_params() -> dict:
    with open("conf/base/parameters.yaml") as f:
        return yaml.safe_load(f)


def run_pipeline(
    category_filter: str | None = None,
    year_range: tuple[int, int] | None = None,
) -> None:
    os.makedirs("outputs", exist_ok=True)
    p = load_params()

    # Override year range if provided via CLI
    if year_range:
        start, end = year_range
        p["train_years"] = [y for y in p["train_years"] if start <= y <= end]
        p["test_years"] = [y for y in p["test_years"] if start <= y <= end]

    logger.info("=== ALESC Pipeline START ===")

    # ── INGESTION ────────────────────────────────────────────────────────────
    logger.info("Step 1/5: Ingestion")
    raw = load_raw_expenses(p["raw_data_dir"])
    parsed = parse_valor(raw)
    tagged = tag_reversal(parsed)
    intermediate = export_intermediate(tagged)

    # ── FEATURES ─────────────────────────────────────────────────────────────
    logger.info("Step 2/5: Features")
    ipca_path = os.path.join(p["raw_data_dir"], "ipca_series.csv")
    ipca_factors = load_ipca_factors(ipca_path)
    adjusted = correct_inflation(intermediate, ipca_factors)
    train_raw, test_raw, infer_raw, reversals = split_train_test_infer(
        adjusted, p["train_years"], p["test_years"], p["inference_years"]
    )
    train_f, test_f, infer_f = engineer_features(
        train_raw, test_raw, infer_raw,
        p["hash_dim_favorecido"], p["mean_value_group"],
    )
    categories = split_by_category(train_f, test_f, infer_f, p["min_rows_train"])

    # Reversal EDA (always runs)
    analyze_reversals(reversals if "valor_adjusted" in reversals.columns
                      else reversals.assign(valor_adjusted=reversals["valor"]))

    if category_filter:
        categories = {k: v for k, v in categories.items() if k == category_filter.upper()}
        if not categories:
            logger.error("Category '%s' not found. Available: %s",
                         category_filter, list(categories.keys()))
            sys.exit(1)

    # ── TRAINING + ENSEMBLE + EXPLANATION + REPORTING (per category) ──────
    all_metrics = []
    all_temporal = []

    for cat_key, splits in categories.items():
        logger.info("=== Category: %s ===", cat_key)
        cat_train = splits["train"]
        cat_test = splits["test"]
        cat_infer = splits["infer"]

        # Check if model already exists (incremental)
        model_dir = os.path.join("outputs/models", cat_key)
        if os.path.isfile(os.path.join(model_dir, "iforest.joblib")):
            logger.info("Models for %s already exist — skipping training.", cat_key)
            from alesc_poc.pipelines.training.nodes import load_model_bundle
            bundles = load_model_bundle(cat_key)
        else:
            # Step 3: Training
            logger.info("Step 3/5: Training [%s]", cat_key)
            contamination = estimate_contamination(
                cat_train,
                p["contamination_iqr_multiplier"],
                p["contamination_min"],
                p["contamination_max"],
            )
            iforest = train_iforest(cat_train, contamination, p["random_seed"])
            knn = train_knn(cat_train, contamination, p["knn_k_candidates"], p["random_seed"])
            lof = train_lof(cat_train, contamination, p["lof_k_candidates"],
                            p["lof_subsample_threshold"], p["lof_subsample_size"], p["random_seed"])
            gmm = train_gmm(cat_train, contamination, p["gmm_max_components"], p["random_seed"])
            ocsvm = train_ocsvm(cat_train, contamination,
                                p["ocsvm_subsample_threshold"], p["ocsvm_subsample_size"],
                                p["ocsvm_nu"], p["random_seed"])
            sod = train_sod(cat_train, contamination, p["random_seed"], p["include_sod"])
            save_model_bundle(cat_key, iforest, knn, lof, gmm, ocsvm, sod)
            bundles = {"iforest": iforest, "knn": knn, "lof": lof,
                       "gmm": gmm, "ocsvm": ocsvm, "sod": sod}

        # Score on test + infer combined for full coverage
        eval_parts = [x for x in (cat_test, cat_infer) if len(x) > 0]
        if not eval_parts:
            logger.warning("Category %s has no test or infer records — skipping scoring.", cat_key)
            continue
        eval_df = pd.concat(eval_parts, ignore_index=False)

        predictions = build_ensemble(
            eval_df,
            bundles["iforest"], bundles["knn"], bundles["lof"],
            bundles["gmm"], bundles["ocsvm"], bundles.get("sod"),
            p["ensemble_strategy"],
        )

        # Step 4: Explanation
        logger.info("Step 4/5: Explanation [%s]", cat_key)
        shap_bundle = compute_shap_values(predictions, bundles["iforest"], p["shap_max_records"])
        plot_waterfall_charts(shap_bundle, cat_key, p["top_anomalies_per_category"])
        plot_shap_summary(shap_bundle, cat_key)
        shap_top = extract_top_shap_features(shap_bundle)

        # Step 5: Reporting
        logger.info("Step 5/5: Reporting [%s]", cat_key)
        contamination_rate = estimate_contamination(
            cat_train,
            p["contamination_iqr_multiplier"], p["contamination_min"], p["contamination_max"],
        )
        metrics = compute_metrics(predictions, cat_key, contamination_rate)
        temporal = compute_temporal_metrics(predictions, cat_key)
        export_results_csv(predictions, shap_top, cat_key)
        save_metrics_json(metrics, cat_key)
        plot_boxplot(predictions, cat_key)
        plot_scatter_anomalies(predictions, cat_key)
        plot_category_profile(predictions, cat_key, train_df=cat_train)

        all_metrics.append(metrics)
        all_temporal.append(temporal)

    # ── GLOBAL REPORTS ────────────────────────────────────────────────────────
    logger.info("Generating global reports...")
    plot_temporal_trend(
        all_temporal,
        p["temporal_trend_start_year"],
        p["temporal_trend_end_year"],
    )
    generate_latex_tables(all_metrics)

    # Save combined metrics summary
    with open("outputs/results/all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    logger.info("=== ALESC Pipeline DONE ===")
    logger.info("Results: outputs/results/ | Plots: outputs/plots/ | Reports: outputs/reports/")


def main():
    parser = argparse.ArgumentParser(description="ALESC Anomaly Detection Pipeline")
    parser.add_argument("--category", type=str, default=None,
                        help="Run only this category (e.g. DIARIAS, COMBUSTIVEIS)")
    parser.add_argument("--years", type=str, default=None,
                        help="Year range for training window, e.g. 2015-2022")
    args = parser.parse_args()

    year_range = None
    if args.years:
        start, end = map(int, args.years.split("-"))
        year_range = (start, end)

    run_pipeline(category_filter=args.category, year_range=year_range)


if __name__ == "__main__":
    main()
