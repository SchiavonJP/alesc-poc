"""
EDA profiling script: load all ALESC raw CSVs, apply ingestion + IPCA correction,
then run ydata-profiling (minimal mode) for each verba category.

Outputs:
  outputs/reports/eda_{CATEGORY_KEY}_2011-2026.html  (one per category)
  outputs/results/eda_{CATEGORY_KEY}_summary.json    (one per category)
  outputs/results/eda_all_summary.json               (cross-category)
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from unidecode import unidecode
from ydata_profiling import ProfileReport

from alesc_poc.pipelines.ingestion.nodes import (
    export_intermediate,
    load_raw_expenses,
    parse_valor,
    tag_reversal,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "01_raw")
INTERMEDIATE_PARQUET = os.path.join(PROJECT_ROOT, "data", "02_intermediate", "expenses.parquet")
IPCA_PATH = os.path.join(RAW_DIR, "ipca_series.csv")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
YEAR_RANGE = "2011-2026"

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "data", "02_intermediate"), exist_ok=True)


def safe_key(name: str) -> str:
    return unidecode(name).upper().replace(" ", "_").replace("/", "_")


def load_or_build_dataframe() -> pd.DataFrame:
    """Load parquet if exists, otherwise build from raw CSVs and save."""
    if os.path.exists(INTERMEDIATE_PARQUET):
        print(f"Loading existing parquet: {INTERMEDIATE_PARQUET}")
        df = pd.read_parquet(INTERMEDIATE_PARQUET)
    else:
        print("Building intermediate DataFrame from raw CSVs...")
        df = load_raw_expenses(RAW_DIR)
        df = parse_valor(df)
        df = tag_reversal(df)
        df = export_intermediate(df)

        # Apply IPCA correction if available
        if os.path.exists(IPCA_PATH):
            ipca = pd.read_csv(IPCA_PATH)
            factors = dict(
                zip(ipca["year"].astype(int), ipca["ipca_correction_factor"].astype(float))
            )
            df["ipca_factor"] = df["year"].map(factors).fillna(1.0)
            df["valor_adjusted"] = df["valor"] * df["ipca_factor"]
            print(f"IPCA correction applied for years: {sorted(factors.keys())}")
        else:
            df["valor_adjusted"] = df["valor"]
            print("IPCA series not found; valor_adjusted = valor")

        df.to_parquet(INTERMEDIATE_PARQUET, index=False)
        print(f"Saved parquet to {INTERMEDIATE_PARQUET}")

    print(f"DataFrame loaded: {df.shape[0]:,} rows, columns: {df.columns.tolist()}")
    return df


def compute_category_summary(cat_df: pd.DataFrame, category_name: str) -> dict:
    valor = cat_df["valor"]
    n_reversals = int(cat_df["is_reversal"].sum()) if "is_reversal" in cat_df.columns else 0
    reversal_rate = float(n_reversals / len(cat_df)) if len(cat_df) > 0 else 0.0

    # null favorecido pct
    null_fav = 0.0
    if "favorecido" in cat_df.columns:
        null_fav = float(
            (cat_df["favorecido"].isna() | (cat_df["favorecido"] == "nan")).sum()
            / len(cat_df)
        )

    # top 5 parlamentarians by count
    top_conta: list = []
    if "conta" in cat_df.columns:
        top_conta = (
            cat_df["conta"]
            .value_counts()
            .head(5)
            .rename_axis("conta")
            .reset_index(name="count")
            .to_dict(orient="records")
        )

    return {
        "category": category_name,
        "n_records": len(cat_df),
        "n_years": int(cat_df["year"].nunique()) if "year" in cat_df.columns else None,
        "year_min": int(cat_df["year"].min()) if "year" in cat_df.columns else None,
        "year_max": int(cat_df["year"].max()) if "year" in cat_df.columns else None,
        "valor_mean": round(float(valor.mean()), 2),
        "valor_median": round(float(valor.median()), 2),
        "valor_std": round(float(valor.std()), 2),
        "valor_min": round(float(valor.min()), 2),
        "valor_max": round(float(valor.max()), 2),
        "top_conta": top_conta,
        "null_favorecido_pct": round(null_fav * 100, 4),
        "n_reversals": n_reversals,
        "reversal_rate_pct": round(reversal_rate * 100, 4),
    }


def check_cross_year_consistency(cat_df: pd.DataFrame) -> list[dict]:
    """Flag years where mean valor deviates more than 2 std from inter-year mean."""
    if "year" not in cat_df.columns:
        return []
    year_stats = (
        cat_df.groupby("year")["valor"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    overall_mean = year_stats["mean"].mean()
    overall_std = year_stats["mean"].std()
    if overall_std == 0 or pd.isna(overall_std):
        return []
    year_stats["z_score"] = (year_stats["mean"] - overall_mean) / overall_std
    flagged = year_stats[year_stats["z_score"].abs() > 2.0]
    return flagged[["year", "mean", "std", "count", "z_score"]].to_dict(orient="records")


def profile_category(cat_df: pd.DataFrame, category_name: str, key: str) -> dict:
    """Run ydata-profiling for one category. Returns the summary dict."""
    print(f"\n--- {category_name} ({len(cat_df):,} rows) ---")

    summary = compute_category_summary(cat_df, category_name)
    flagged_years = check_cross_year_consistency(cat_df)
    summary["cross_year_flagged"] = flagged_years

    # Drop high-cardinality text columns that slow profiling significantly
    cols_to_drop = [c for c in ["descricao", "trecho", "favorecido", "valor_raw"] if c in cat_df.columns]
    profile_df = cat_df.drop(columns=cols_to_drop)

    profile = ProfileReport(
        profile_df,
        title=f"ALESC EDA — {category_name} ({YEAR_RANGE})",
        minimal=True,
        progress_bar=False,
    )

    html_path = os.path.join(REPORTS_DIR, f"eda_{key}_{YEAR_RANGE}.html")
    profile.to_file(html_path)
    print(f"  HTML report saved: {html_path}")

    json_path = os.path.join(RESULTS_DIR, f"eda_{key}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Summary JSON saved: {json_path}")

    print(
        f"  valor: mean={summary['valor_mean']:.2f}, median={summary['valor_median']:.2f}, "
        f"std={summary['valor_std']:.2f}, min={summary['valor_min']:.2f}, max={summary['valor_max']:.2f}"
    )
    print(f"  reversals: {summary['n_reversals']} ({summary['reversal_rate_pct']:.2f}%)")
    print(f"  null_favorecido: {summary['null_favorecido_pct']:.2f}%")
    if flagged_years:
        print(f"  CROSS-YEAR FLAGS: {flagged_years}")

    return summary


def main():
    df = load_or_build_dataframe()

    print(f"\n=== Unique verba categories ===")
    verba_counts = df["verba"].value_counts()
    print(verba_counts.to_string())

    categories = sorted(df["verba"].dropna().unique())
    print(f"\nTotal categories: {len(categories)}")

    all_summaries = []
    profiled_keys = []

    for cat_name in categories:
        cat_df = df[df["verba"] == cat_name].copy()
        key = safe_key(cat_name)
        summary = profile_category(cat_df, cat_name, key)
        all_summaries.append(summary)
        profiled_keys.append(key)

    # Global cross-category summary
    global_summary = {
        "total_records": len(df),
        "n_categories": len(all_summaries),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "n_years": int(df["year"].nunique()),
        "overall_reversal_pct": round(float(df["is_reversal"].mean() * 100), 4)
        if "is_reversal" in df.columns
        else None,
        "overall_null_favorecido_pct": round(
            float(
                (df["favorecido"].isna() | (df["favorecido"] == "nan")).sum()
                / len(df)
                * 100
            ),
            4,
        )
        if "favorecido" in df.columns
        else None,
        "categories": all_summaries,
    }

    global_json_path = os.path.join(RESULTS_DIR, "eda_all_summary.json")
    with open(global_json_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== DONE ===")
    print(f"Categories profiled: {len(profiled_keys)}")
    print(f"Category keys: {profiled_keys}")
    print(f"Global summary: {global_json_path}")
    for s in all_summaries:
        key = safe_key(s["category"])
        print(
            f"  {key}: n={s['n_records']:>7,}  "
            f"mean={s['valor_mean']:>10.2f}  "
            f"reversals={s['reversal_rate_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
