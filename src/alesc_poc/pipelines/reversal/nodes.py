"""Reversal EDA: separate analysis of devolution/cancellation records."""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_reversals(
    reversals: pd.DataFrame,
    output_dir: str = "outputs/results",
) -> dict:
    """
    EDA on reversal records:
      - Count and rate per Verba
      - Distribution of absolute reversal values
      - Top parliamentarians with most reversals
    """
    os.makedirs(output_dir, exist_ok=True)
    if reversals.empty:
        logger.warning("No reversal records found.")
        return {}

    verba_counts = reversals["verba"].value_counts().to_dict()
    top_conta = reversals["conta"].value_counts().head(10).to_dict()
    valor_abs = reversals["valor_adjusted"].abs() if "valor_adjusted" in reversals.columns \
                else reversals["valor"].abs()

    summary = {
        "total_reversals": len(reversals),
        "total_reversal_value_brl": round(float(valor_abs.sum()), 2),
        "mean_reversal_value_brl": round(float(valor_abs.mean()), 2),
        "max_reversal_value_brl": round(float(valor_abs.max()), 2),
        "by_verba": verba_counts,
        "top_parliamentarians": top_conta,
    }

    # Bar chart: reversals by Verba
    fig, ax = plt.subplots(figsize=(10, 5))
    cats = list(verba_counts.keys())[:12]
    vals = [verba_counts[c] for c in cats]
    ax.barh(cats, vals, color="#E07B54")
    ax.set_title("Reversals (Devoluções) by Verba Category")
    ax.set_xlabel("Count")
    plt.tight_layout()
    path = os.path.join(output_dir, "reversals_by_verba.png")
    plt.savefig(path, dpi=120)
    plt.close()
    summary["plot_path"] = path

    logger.info(
        "Reversal EDA: %d records, total R$ %.2f",
        len(reversals), summary["total_reversal_value_brl"],
    )
    return summary
