"""
Generate top anomaly rankings from all category anomaly CSVs.

Produces outputs/results/top_anomalies.json with:
  - global_top50: top 50 anomalies by audit_score across all categories
  - per_category_top5: top 5 per category by audit_score

── Audit Score: Materiality-Weighted Ranking ────────────────────────────────

The raw ensemble_score measures *statistical deviation* from historical patterns.
It gives equal weight to a R$200 anomaly and a R$50,000 anomaly with the same
deviation profile. For public spending audit this is wrong: an auditor's time
is better spent on high-value irregular expenses.

    audit_score = ensemble_score × log1p(valor_adjusted)

Why log1p (= log(1 + x))?
  - Compresses the value range so a single very large expense does not
    completely dominate the ranking (avoids the "R$1M outlier" problem).
  - Defined for x ≥ 0 (no issues with zero-value records).
  - Preserves ranking monotonicity: higher value → higher weight.
  - Practical ranges (ensemble_score ≈ 0.3–0.7):
      valor R$200   → log1p ≈ 5.3  → audit_score ≈ 1.6–3.7
      valor R$5000  → log1p ≈ 8.5  → audit_score ≈ 2.6–5.9
      valor R$50000 → log1p ≈ 10.8 → audit_score ≈ 3.2–7.6

This approach is consistent with materiality-weighted risk scoring used in
audit analytics (ACFE fraud examination guidelines; ISA 320 on materiality).

IMPORTANT DISTINCTION:
  - ensemble_score  → primary statistical metric for the BRACIS article.
                      Reports how anomalous a record is, model-wise.
  - audit_score     → prioritization tool for auditors in the dashboard.
                      Does NOT replace or modify the model output.

The two metrics serve different purposes and both are reported.

──────────────────────────────────────────────────────────────────────────────

Usage:
  python scripts/generate_rankings.py [--top-n 50] [--per-cat 5]
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


RESULTS_DIR = "outputs/results"


def fmt_brl(value: float) -> str:
    try:
        s = f"{value:,.2f}"
        return "R$ " + s.replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return "R$ —"


def compute_audit_score(ensemble_score: pd.Series, valor_adjusted: pd.Series) -> pd.Series:
    """
    Materiality-weighted anomaly score for auditor prioritization.

    audit_score = ensemble_score × log1p(valor_adjusted)

    Higher values indicate both strong statistical deviation AND high monetary
    impact — the cases most worth an auditor's attention.
    """
    return ensemble_score * np.log1p(valor_adjusted.clip(lower=0))


def load_all_anomalies() -> pd.DataFrame:
    frames = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith("_anomalies.csv"):
            continue
        category = fname.replace("_anomalies.csv", "")
        path = os.path.join(RESULTS_DIR, fname)
        df = pd.read_csv(path)
        flagged = df[df["ensemble_flag"] == True].copy()
        if flagged.empty:
            continue
        flagged["category"] = category
        frames.append(flagged)
    if not frames:
        raise RuntimeError("No anomaly CSVs found in outputs/results/")

    all_df = pd.concat(frames, ignore_index=True)

    # Compute materiality-weighted audit score
    all_df["audit_score"] = compute_audit_score(
        all_df["ensemble_score"], all_df["valor_adjusted"]
    )

    # Category rank by audit_score (determines which records get waterfall plots
    # from the pipeline's top_anomalies_per_category setting)
    all_df["category_rank"] = (
        all_df.groupby("category")["audit_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    # Keep ensemble-based rank separately for reference
    all_df["category_rank_by_score"] = (
        all_df.groupby("category")["ensemble_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    return all_df


def build_record(row: pd.Series, rank: int) -> dict:
    return {
        "rank": rank,
        "category": row.get("category", row.get("verba", "")),
        "conta": row.get("conta", ""),
        "favorecido": row.get("favorecido", ""),
        "valor_adjusted": round(float(row.get("valor_adjusted", 0)), 2),
        "valor_brl": fmt_brl(row.get("valor_adjusted", 0)),
        "year": int(row.get("year", 0)),
        "month": int(row.get("month", 0)),
        "ensemble_score": round(float(row.get("ensemble_score", 0)), 6),
        "audit_score": round(float(row.get("audit_score", 0)), 6),
        "top_feature": row.get("top_feature", ""),
        "top_shap_value": round(float(row.get("top_shap_value", 0)), 6),
        "mean_value": round(float(row.get("mean_value", 0)), 2),
        "mean_value_brl": fmt_brl(row.get("mean_value", 0)),
        "llm_explanation": row.get("llm_explanation", ""),
        "category_rank": int(row.get("category_rank", 1)),
        "category_rank_by_score": int(row.get("category_rank_by_score", 1)),
    }


def main(top_n: int = 50, per_cat: int = 5) -> None:
    print("Loading anomaly CSVs...")
    all_df = load_all_anomalies()
    n_total = len(all_df)
    print(f"  Total flagged records: {n_total}")

    audit_max = all_df["audit_score"].max()
    ensemble_max = all_df["ensemble_score"].max()
    print(f"  audit_score  range: {all_df['audit_score'].min():.3f} – {audit_max:.3f}")
    print(f"  ensemble_score range: {all_df['ensemble_score'].min():.3f} – {ensemble_max:.3f}")

    # Global top-N ranked by audit_score
    global_top = (
        all_df.sort_values("audit_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    global_records = [build_record(row, i + 1) for i, row in global_top.iterrows()]

    # Per-category top-5 ranked by audit_score
    per_category: dict[str, list] = {}
    for cat, grp in all_df.groupby("category"):
        top5 = (
            grp.sort_values("audit_score", ascending=False)
            .head(per_cat)
            .reset_index(drop=True)
        )
        per_category[cat] = [build_record(row, i + 1) for i, row in top5.iterrows()]

    output = {
        "generated_at": datetime.now().isoformat(),
        "total_flagged": n_total,
        "ranking_method": "audit_score = ensemble_score × log1p(valor_adjusted)",
        "global_top_n": top_n,
        "per_category_top_n": per_cat,
        f"global_top{top_n}": global_records,
        "per_category_top5": per_category,
    }

    out_path = os.path.join(RESULTS_DIR, "top_anomalies.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {out_path}")
    print(f"  Global top {top_n}: {len(global_records)} records (by audit_score)")
    print(f"  Per-category top {per_cat}: {sum(len(v) for v in per_category.values())} records")
    print("\nTop 5 audit_score:")
    for rec in global_records[:5]:
        print(f"  #{rec['rank']:2d}  {rec['conta'][:30]:30s}  {rec['category']:25s}  "
              f"{rec['valor_brl']:>14s}  audit={rec['audit_score']:.3f}  score={rec['ensemble_score']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate materiality-weighted anomaly rankings (audit_score)"
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--per-cat", type=int, default=5)
    args = parser.parse_args()
    main(top_n=args.top_n, per_cat=args.per_cat)
