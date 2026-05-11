"""Ingestion pipeline: load raw CSVs → parse → tag reversals → UTF-8 parquet."""

from __future__ import annotations

import glob
import logging
import os

import pandas as pd
from unidecode import unidecode

logger = logging.getLogger(__name__)

_REVERSAL_KEYWORDS = ("devolu", "estorno", "cancelamento")


def load_raw_expenses(raw_data_dir: str) -> pd.DataFrame:
    """Load all ALESC CSVs (Latin-1, semicolon-delimited), add `year` column."""
    pattern = os.path.join(raw_data_dir, "alesc_gabinetes_parlamentares_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No CSV files found at {pattern}")

    frames = []
    for path in files:
        year = int(os.path.basename(path).split("_")[-1].replace(".csv", ""))
        df = pd.read_csv(
            path,
            sep=";",
            encoding="latin1",
            on_bad_lines="skip",  # drops rows with semicolons inside Trecho field
            dtype=str,
        )
        df["year"] = year
        frames.append(df)
        logger.info("Loaded %d rows from %s", len(df), os.path.basename(path))

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Total raw rows: %d", len(combined))
    return combined


def parse_valor(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Brazilian number format to float; drop unparseable rows."""
    before = len(df)

    def _parse(s: str) -> float | None:
        if pd.isna(s) or str(s).strip() == "":
            return None
        try:
            return float(str(s).replace(".", "").replace(",", "."))
        except ValueError:
            return None

    df = df.copy()
    df["valor_raw"] = df["Valor"].copy()
    df["Valor"] = df["Valor"].map(_parse)
    dropped = before - len(df.dropna(subset=["Valor"]))
    if dropped:
        logger.warning("Dropped %d rows with unparseable Valor", dropped)
    df = df.dropna(subset=["Valor"])
    df["Valor"] = df["Valor"].astype(float)
    return df


def tag_reversal(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean `is_reversal` derived from Descrição containing reversal keywords."""
    df = df.copy()
    descricao_lower = df["Descrição"].fillna("").str.lower().map(unidecode)
    df["is_reversal"] = descricao_lower.str.contains(
        "|".join(_REVERSAL_KEYWORDS), regex=True
    )
    # Also flag any negative Valor as reversal (belt-and-suspenders)
    df["is_reversal"] = df["is_reversal"] | (df["Valor"] < 0)
    logger.info(
        "Tagged %d reversals (%.2f%% of total)",
        df["is_reversal"].sum(),
        100 * df["is_reversal"].mean(),
    )
    return df


def export_intermediate(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to snake_case, normalise string fields, return clean DataFrame."""
    df = df.copy()
    df = df.rename(
        columns={
            "Verba": "verba",
            "Descrição": "descricao",
            "Conta": "conta",
            "Favorecido": "favorecido",
            "Trecho": "trecho",      # kept for now, dropped later in features
            "Vencimento": "vencimento",
            "Valor": "valor",
            "valor_raw": "valor_raw",
        }
    )
    # Normalise string columns: strip, UTF-8 safe
    for col in ("verba", "descricao", "conta", "favorecido", "trecho", "vencimento"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("nan", pd.NA)

    # Parse vencimento as date
    df["vencimento"] = pd.to_datetime(df["vencimento"], errors="coerce")
    df["year"] = df["year"].astype(int)

    total = len(df)
    reversal_count = int(df["is_reversal"].sum())
    logger.info(
        "Intermediate export: %d rows, %d reversals, %d years",
        total,
        reversal_count,
        df["year"].nunique(),
    )
    return df
