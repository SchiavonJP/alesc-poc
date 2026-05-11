"""Reporting pipeline: metrics JSON, results CSV, LaTeX tables, temporal trend plot."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


# ── Metrics computation ───────────────────────────────────────────────────────

def compute_metrics(
    predictions_df: pd.DataFrame,
    category_key: str,
    contamination_rate: float,
) -> dict[str, Any]:
    """
    Compute per-category metrics:
      - n_records, n_flagged_ensemble, flagged_pct
      - individual model flagged rates
      - ensemble reduction vs IQR baseline (contamination_rate)
      - basic stats on anomalous Valor
    """
    n_total = len(predictions_df)
    n_ensemble = int(predictions_df["ensemble_flag"].sum())

    model_names = ["iforest", "knn", "lof", "gmm", "ocsvm"]
    if "anomaly_sod" in predictions_df.columns:
        model_names.append("sod")

    individual_rates = {}
    for m in model_names:
        col = f"anomaly_{m}"
        if col in predictions_df.columns:
            individual_rates[m] = float(predictions_df[col].mean())

    flagged = predictions_df[predictions_df["ensemble_flag"] == 1]
    anomaly_valor_stats = {}
    if "valor_adjusted" in predictions_df.columns and len(flagged):
        vals = flagged["valor_adjusted"]
        anomaly_valor_stats = {
            "mean": round(float(vals.mean()), 2),
            "median": round(float(vals.median()), 2),
            "min": round(float(vals.min()), 2),
            "max": round(float(vals.max()), 2),
        }

    metrics = {
        "category": category_key,
        "n_records": n_total,
        "n_flagged_ensemble": n_ensemble,
        "flagged_pct": round(100 * n_ensemble / max(n_total, 1), 4),
        "iqr_baseline_pct": round(100 * contamination_rate, 4),
        "reduction_factor": round(contamination_rate / max(n_ensemble / max(n_total, 1), 1e-9), 2),
        "individual_model_rates": individual_rates,
        "anomaly_valor_stats": anomaly_valor_stats,
    }
    logger.info(
        "%s: %d flagged / %d total (%.4f%%, IQR baseline %.4f%%)",
        category_key, n_ensemble, n_total,
        metrics["flagged_pct"], metrics["iqr_baseline_pct"],
    )
    return metrics


def compute_temporal_metrics(
    predictions_df: pd.DataFrame,
    category_key: str,
) -> pd.DataFrame:
    """Anomaly count and rate per year for temporal trend analysis."""
    if "year" not in predictions_df.columns:
        return pd.DataFrame()

    grouped = (
        predictions_df.groupby("year")
        .agg(
            n_records=("ensemble_flag", "count"),
            n_flagged=("ensemble_flag", "sum"),
        )
        .reset_index()
    )
    grouped["flagged_pct"] = 100 * grouped["n_flagged"] / grouped["n_records"].clip(lower=1)
    grouped["category"] = category_key
    return grouped


# ── Results CSV export ────────────────────────────────────────────────────────

def export_results_csv(
    predictions_df: pd.DataFrame,
    shap_top_features: pd.DataFrame,
    category_key: str,
    output_dir: str = "outputs/results",
) -> str:
    """Export flagged records with scores and top SHAP features to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    flagged = predictions_df[predictions_df["ensemble_flag"] == 1].copy()

    if not shap_top_features.empty:
        # Pivot top SHAP features into wide format for readability
        pivot = (
            shap_top_features[shap_top_features["rank"] == 1]
            .set_index("record_index")[["feature_name", "shap_value"]]
            .rename(columns={"feature_name": "top_feature", "shap_value": "top_shap_value"})
        )
        flagged = flagged.join(pivot, how="left")

    # Keep key human-readable columns at the front
    front_cols = [c for c in ("conta", "verba", "valor_adjusted", "year",
                               "ensemble_flag", "ensemble_score",
                               "top_feature", "top_shap_value") if c in flagged.columns]
    score_cols = [c for c in flagged.columns if c.startswith("score_") or c.startswith("anomaly_")]
    other_cols = [c for c in flagged.columns if c not in front_cols + score_cols]
    flagged = flagged[front_cols + score_cols + other_cols]

    path = os.path.join(output_dir, f"{category_key}_anomalies.csv")
    flagged.to_csv(path, index=True)
    logger.info("Saved %d flagged records to %s", len(flagged), path)
    return path


def save_metrics_json(
    metrics: dict[str, Any],
    category_key: str,
    output_dir: str = "outputs/results",
) -> str:
    """Save metrics dict to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{category_key}_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_boxplot(
    df: pd.DataFrame,
    category_key: str,
    output_dir: str = "outputs/plots",
) -> str:
    """Boxplot of valor_adjusted with anomaly overlay."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    normal = df[df["ensemble_flag"] == 0]["valor_adjusted"].dropna()
    flagged = df[df["ensemble_flag"] == 1]["valor_adjusted"].dropna()

    ax.boxplot([normal], positions=[1], widths=0.6, patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.7), showfliers=False)
    if len(flagged):
        ax.scatter([1] * len(flagged), flagged, color="red", zorder=5,
                   alpha=0.7, s=20, label=f"Anomalies (n={len(flagged)})")
    ax.set_title(f"Reimbursement Values — {category_key}")
    ax.set_ylabel("Valor Adjusted (BRL Jan-2026)")
    ax.set_xticks([1])
    ax.set_xticklabels([category_key])
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f"boxplot_{category_key}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_scatter_anomalies(
    df: pd.DataFrame,
    category_key: str,
    output_dir: str = "outputs/plots",
) -> str:
    """Scatter plot of valor_adjusted vs record index, colored by anomaly flag."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    normal = df[df["ensemble_flag"] == 0]
    flagged = df[df["ensemble_flag"] == 1]

    ax.scatter(normal.index, normal["valor_adjusted"], c="#4C72B0",
               alpha=0.3, s=5, label="Normal")
    if len(flagged):
        ax.scatter(flagged.index, flagged["valor_adjusted"], c="red",
                   alpha=0.8, s=20, marker="x", label=f"Anomaly (n={len(flagged)})")
    ax.set_title(f"Anomaly Scatter — {category_key}")
    ax.set_xlabel("Record Index")
    ax.set_ylabel("Valor Adjusted (BRL Jan-2026)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f"scatter_{category_key}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_temporal_trend(
    all_temporal_metrics: list[pd.DataFrame],
    start_year: int,
    end_year: int,
    output_dir: str = "outputs/plots",
) -> str:
    """Year-over-year anomaly rate chart across all categories."""
    if not all_temporal_metrics:
        return ""
    combined = pd.concat(all_temporal_metrics, ignore_index=True)
    combined = combined[combined["year"].between(start_year, end_year)]

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for cat, grp in combined.groupby("category"):
        ax.plot(grp["year"], grp["flagged_pct"], marker="o", label=cat, linewidth=1.5)
    ax.set_title("Anomaly Rate by Year and Category (2011–2025)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Flagged (%)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "temporal_trend.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved temporal trend plot: %s", path)
    return path


def plot_category_profile(
    df: pd.DataFrame,
    category_key: str,
    train_df: pd.DataFrame | None = None,
    output_dir: str = "outputs/plots",
) -> str:
    """
    Two-panel profile comparison:
      Top: KDE of valor_adjusted (train overlay optional, normal vs anomalous)
      Bottom: scatter by year, anomalies sized by ensemble_score
    """
    cat_dir = os.path.join(output_dir, category_key)
    os.makedirs(cat_dir, exist_ok=True)

    normal = df[df["ensemble_flag"] == 0]["valor_adjusted"].dropna()
    flagged_df = df[df["ensemble_flag"] == 1]
    flagged_vals = flagged_df["valor_adjusted"].dropna()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]}
    )

    # ── Top panel: KDE ────────────────────────────────────────────────────────
    if train_df is not None and "valor_adjusted" in train_df.columns:
        train_vals = train_df["valor_adjusted"].dropna()
        if len(train_vals) > 1:
            sns.kdeplot(
                train_vals, ax=ax1, color="#4C72B0", alpha=0.35, fill=True,
                label=f"Treino 2011–2022 (n={len(train_vals):,})",
            )

    if len(normal) > 1:
        sns.kdeplot(
            normal, ax=ax1, color="#2CA02C", alpha=0.5, fill=True,
            label=f"Normal 2023–2025 (n={len(normal):,})",
        )

    if len(flagged_vals) > 0:
        ax1.plot(
            flagged_vals, [ax1.get_ylim()[1] * 0.02] * len(flagged_vals),
            "|", color="red", alpha=0.7, markersize=12,
            label=f"Anômalos (n={len(flagged_vals)})",
        )
        ax1.axvline(
            flagged_vals.median(), color="red", linestyle="--", alpha=0.6,
            label=f"Mediana anômalos: R$ {flagged_vals.median():,.0f}",
        )

    if len(normal) > 0:
        ax1.axvline(
            normal.median(), color="#2CA02C", linestyle="--", alpha=0.6,
            label=f"Mediana normal: R$ {normal.median():,.0f}",
        )

    ax1.set_title(f"Perfil de Gastos — {category_key}", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Valor Corrigido (R$ Jan/2026)")
    ax1.set_ylabel("Densidade")
    ax1.legend(fontsize=9)

    # ── Bottom panel: scatter by year ─────────────────────────────────────────
    if "year" in df.columns:
        rng = np.random.RandomState(42)
        normal_rows = df[df["ensemble_flag"] == 0]
        anom_rows = df[df["ensemble_flag"] == 1]

        jitter_n = rng.uniform(-0.25, 0.25, len(normal_rows))
        ax2.scatter(
            normal_rows["year"] + jitter_n, normal_rows["valor_adjusted"],
            c="#2CA02C", alpha=0.2, s=6, label="Normal",
        )
        if len(anom_rows):
            jitter_a = rng.uniform(-0.25, 0.25, len(anom_rows))
            sizes = (anom_rows["ensemble_score"].clip(lower=0.3) * 150).clip(lower=20)
            ax2.scatter(
                anom_rows["year"] + jitter_a, anom_rows["valor_adjusted"],
                c="red", alpha=0.75, s=sizes, marker="D", label="Anômalo",
            )
        ax2.set_xlabel("Ano")
        ax2.set_ylabel("Valor (R$)")
        ax2.set_title("Dispersão por Ano — tamanho proporcional ao score de anomalia")
        ax2.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(cat_dir, "profile_comparison.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved category profile: %s", path)
    return path


# ── LaTeX table generation ────────────────────────────────────────────────────

def generate_latex_tables(
    all_metrics: list[dict[str, Any]],
    output_dir: str = "outputs/reports",
) -> str:
    """Generate LaTeX table strings from list of per-category metrics dicts."""
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for m in sorted(all_metrics, key=lambda x: x.get("category", "")):
        cat = m.get("category", "—")
        n = m.get("n_records", 0)
        n_flag = m.get("n_flagged_ensemble", 0)
        pct = m.get("flagged_pct", 0.0)
        iqr = m.get("iqr_baseline_pct", 0.0)
        val_mean = m.get("anomaly_valor_stats", {}).get("mean", "—")
        val_max = m.get("anomaly_valor_stats", {}).get("max", "—")
        rows.append(
            f"  {cat} & {n:,} & {n_flag} & {pct:.4f}\\% & {iqr:.4f}\\% "
            f"& R\\$ {val_mean} & R\\$ {val_max} \\\\"
        )

    header = (
        "\\begin{table}[h]\n"
        "\\caption{Anomalias detectadas pelo ensemble por categoria de Verba}\n"
        "\\label{tab:anomalias}\n"
        "\\centering\n"
        "\\begin{tabular}{lrrrrrr}\n"
        "\\hline\n"
        "Categoria & N & Anomalias & \\% Ensemble & \\% IQR base & "
        "Valor Médio & Valor Máximo \\\\\n"
        "\\hline\n"
    )
    footer = "\\hline\n\\end{tabular}\n\\end{table}"
    latex = header + "\n".join(rows) + "\n" + footer

    path = os.path.join(output_dir, "table_anomalias.tex")
    with open(path, "w") as f:
        f.write(latex)
    logger.info("LaTeX table saved: %s", path)
    return path
