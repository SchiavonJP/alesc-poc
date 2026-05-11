"""Features pipeline: IPCA correction → filter → feature engineering → split."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import murmurhash3_32

logger = logging.getLogger(__name__)


# ── IPCA correction ──────────────────────────────────────────────────────────

def load_ipca_factors(ipca_series_path: str) -> dict[int, float]:
    """Load IPCA correction factors from static CSV."""
    df = pd.read_csv(ipca_series_path)
    factors = dict(zip(df["year"].astype(int), df["ipca_correction_factor"].astype(float)))
    logger.info("IPCA factors loaded for years: %s", sorted(factors.keys()))
    return factors


def correct_inflation(
    df: pd.DataFrame,
    ipca_factors: dict[int, float],
) -> pd.DataFrame:
    """Multiply each row's valor by the IPCA correction factor for its year."""
    df = df.copy()
    df["ipca_factor"] = df["year"].map(ipca_factors).fillna(1.0)
    df["valor_adjusted"] = df["valor"] * df["ipca_factor"]
    logger.info(
        "Inflation corrected. Valor range: [%.2f, %.2f] → [%.2f, %.2f]",
        df["valor"].min(), df["valor"].max(),
        df["valor_adjusted"].min(), df["valor_adjusted"].max(),
    )
    return df


# ── Train/test/inference split ────────────────────────────────────────────────

def split_train_test_infer(
    df: pd.DataFrame,
    train_years: list[int],
    test_years: list[int],
    inference_years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into four subsets:
      - train: for model fitting
      - test: held-out for evaluation
      - infer: current-year deployment
      - reversals: is_reversal=True records (kept separate for EDA)
    Years not in any list (excluded_years) are dropped.
    """
    all_included = set(train_years) | set(test_years) | set(inference_years)
    df_included = df[df["year"].isin(all_included)].copy()

    dropped_years = set(df["year"].unique()) - all_included
    if dropped_years:
        logger.info("Excluded years from training: %s", sorted(dropped_years))

    reversals = df_included[df_included["is_reversal"]].copy()
    positive = df_included[~df_included["is_reversal"]].copy()

    train = positive[positive["year"].isin(train_years)].copy()
    test = positive[positive["year"].isin(test_years)].copy()
    infer = positive[positive["year"].isin(inference_years)].copy()

    logger.info(
        "Split sizes — train: %d, test: %d, infer: %d, reversals: %d",
        len(train), len(test), len(infer), len(reversals),
    )
    return train, test, infer, reversals


# ── Feature engineering ───────────────────────────────────────────────────────

def _extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract month, quarter, day_of_week, year from vencimento."""
    df = df.copy()
    dt = df["vencimento"]
    df["month"] = dt.dt.month.fillna(0).astype(int)
    df["quarter"] = dt.dt.quarter.fillna(0).astype(int)
    df["day_of_week"] = dt.dt.dayofweek.fillna(0).astype(int)
    # year already present as int column
    return df


def _hash_encode(series: pd.Series, n_features: int, col_prefix: str) -> pd.DataFrame:
    """
    Hash-trick encoding: map each string to n_features binary columns.
    Uses MurmurHash3 mod n_features.
    """
    values = series.fillna("__MISSING__").astype(str)
    matrix = np.zeros((len(values), n_features), dtype=np.float32)
    for i, val in enumerate(values):
        idx = abs(murmurhash3_32(val, seed=0)) % n_features
        matrix[i, idx] = 1.0
    cols = [f"{col_prefix}_{j}" for j in range(n_features)]
    return pd.DataFrame(matrix, columns=cols, index=series.index)


def _compute_mean_value(train_df: pd.DataFrame, group_keys: list[str]) -> pd.Series:
    """
    Compute mean adjusted valor grouped by group_keys on training data only.
    Decision: use Verba+Conta combination (documented in QUESTIONS.md Q2.1a).
    """
    return train_df.groupby(group_keys)["valor_adjusted"].transform("mean")


def engineer_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    infer: pd.DataFrame,
    hash_dim_favorecido: int,
    mean_value_group: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply all feature transformations.
    mean_value is fitted on train only to prevent leakage.
    Favorecido null → fallback to Conta (documented: QUESTIONS.md Q1.1b).
    Trecho is dropped.
    """
    # Fallback: Favorecido null → use Conta
    for df in (train, test, infer):
        mask = df["favorecido"].isna() | (df["favorecido"] == "nan")
        df.loc[mask, "favorecido"] = df.loc[mask, "conta"]

    # Compute mean_value on train, then map to test/infer via group keys
    group_means = (
        train.groupby(mean_value_group)["valor_adjusted"]
        .mean()
        .rename("mean_value")
    )

    def _apply_features(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        df = df.copy()
        df = _extract_date_features(df)

        # mean_value: join from train-computed means (no leakage)
        df = df.join(group_means, on=mean_value_group, how="left")
        global_mean = group_means.mean()
        df["mean_value"] = df["mean_value"].fillna(global_mean)

        # Hash encoding for favorecido and conta
        hash_fav = _hash_encode(df["favorecido"], hash_dim_favorecido, "fav")
        hash_conta = _hash_encode(df["conta"], hash_dim_favorecido // 2, "cta")
        df = pd.concat([df, hash_fav, hash_conta], axis=1)

        # Drop columns not used as model features
        df = df.drop(columns=["trecho", "vencimento", "valor_raw",
                               "ipca_factor", "valor", "descricao"],
                     errors="ignore")
        return df

    train_f = _apply_features(train, is_train=True)
    test_f = _apply_features(test, is_train=False)
    infer_f = _apply_features(infer, is_train=False)

    logger.info(
        "Features engineered. Shape — train: %s, test: %s, infer: %s",
        train_f.shape, test_f.shape, infer_f.shape,
    )
    return train_f, test_f, infer_f


# ── Category split with minimum-rows guard ────────────────────────────────────

def split_by_category(
    train: pd.DataFrame,
    test: pd.DataFrame,
    infer: pd.DataFrame,
    min_rows_train: int,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Group by `verba`. Skip categories with fewer than min_rows_train training rows.
    Returns {category_key: {"train": df, "test": df, "infer": df}}.
    """
    categories = train["verba"].dropna().unique()
    result: dict[str, dict[str, pd.DataFrame]] = {}
    skipped = []

    for cat in sorted(categories):
        tr = train[train["verba"] == cat].copy()
        te = test[test["verba"] == cat].copy()
        inf = infer[infer["verba"] == cat].copy()

        if len(tr) < min_rows_train:
            skipped.append((cat, len(tr)))
            continue

        # Safe key for filenames
        key = _safe_key(cat)
        result[key] = {"train": tr, "test": te, "infer": inf}
        logger.info(
            "Category '%s' → key '%s': train=%d, test=%d, infer=%d",
            cat, key, len(tr), len(te), len(inf),
        )

    if skipped:
        logger.warning(
            "Skipped categories (< %d train rows): %s",
            min_rows_train,
            [(c, n) for c, n in skipped],
        )

    logger.info("Total categories to model: %d", len(result))
    return result


def _safe_key(name: str) -> str:
    """Convert a Verba name to a filesystem-safe key."""
    from unidecode import unidecode
    return unidecode(name).upper().replace(" ", "_").replace("/", "_")
