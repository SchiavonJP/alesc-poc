"""Training pipeline: per-category model fitting + ensemble construction."""

from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

# Feature columns that go into the model (all numeric after engineering)
_META_COLS = {"verba", "conta", "favorecido", "is_reversal", "year",
              "valor_adjusted", "mean_value"}


def _get_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Extract numeric feature matrix from engineered DataFrame."""
    # All columns except meta-level string/identity columns
    exclude = {"verba", "conta", "favorecido", "is_reversal"}
    feature_cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype in (np.float32, np.float64, np.int32, np.int64, float, int)
    ]
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    return X, feature_cols


# ── Contamination estimation ──────────────────────────────────────────────────

def estimate_contamination(
    train_df: pd.DataFrame,
    iqr_multiplier: float,
    contamination_min: float,
    contamination_max: float,
) -> float:
    """
    Estimate contamination fraction using IQR on valor_adjusted (train set only).
    Fixes B3 from example: uses a single consistent DataFrame.
    """
    vals = train_df["valor_adjusted"].dropna()
    q1 = vals.quantile(0.25)
    q3 = vals.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    n_outliers = ((vals < lower) | (vals > upper)).sum()
    contamination = n_outliers / len(vals)
    contamination = float(np.clip(contamination, contamination_min, contamination_max))
    logger.info(
        "Contamination estimate: %.4f (IQR=[%.2f, %.2f], outliers=%d/%d)",
        contamination, lower, upper, n_outliers, len(vals),
    )
    return contamination


# ── Individual model training ─────────────────────────────────────────────────

def train_iforest(
    train_df: pd.DataFrame,
    contamination: float,
    random_seed: int,
) -> dict[str, Any]:
    """Train Isolation Forest. Returns model + scaler + feature names."""
    from pyod.models.iforest import IForest

    X, feature_cols = _get_feature_matrix(train_df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IForest(
        contamination=contamination,
        n_estimators=100,
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        random_state=random_seed,
    )
    model.fit(X_scaled)
    logger.info("IForest trained on %d samples, %d features", *X_scaled.shape)
    return {"model": model, "scaler": scaler, "feature_cols": feature_cols}


def train_knn(
    train_df: pd.DataFrame,
    contamination: float,
    k_candidates: list[int],
    random_seed: int,
) -> dict[str, Any]:
    """Train KNN detector; select k via silhouette score."""
    from pyod.models.knn import KNN

    X, feature_cols = _get_feature_matrix(train_df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, best_score = k_candidates[0], -1.0
    for k in k_candidates:
        try:
            m = KNN(n_neighbors=k, contamination=contamination)
            m.fit(X_scaled)
            labels = m.labels_
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
            logger.debug("KNN k=%d silhouette=%.4f", k, score)
            if score > best_score:
                best_score, best_k = score, k
        except Exception as e:
            logger.warning("KNN k=%d failed: %s", k, e)

    model = KNN(n_neighbors=best_k, contamination=contamination)
    model.fit(X_scaled)
    logger.info("KNN trained with k=%d (silhouette=%.4f)", best_k, best_score)
    return {"model": model, "scaler": scaler, "feature_cols": feature_cols, "best_k": best_k}


def train_lof(
    train_df: pd.DataFrame,
    contamination: float,
    k_candidates: list[int],
    subsample_threshold: int,
    subsample_size: int,
    random_seed: int,
) -> dict[str, Any]:
    """Train LOF detector; subsample if dataset is large."""
    from pyod.models.lof import LOF

    X, feature_cols = _get_feature_matrix(train_df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if len(X_scaled) > subsample_threshold:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(X_scaled), subsample_size, replace=False)
        X_fit = X_scaled[idx]
        logger.info("LOF: subsampling %d → %d rows", len(X_scaled), subsample_size)
    else:
        X_fit = X_scaled

    best_k, best_score = k_candidates[0], -1.0
    for k in k_candidates:
        try:
            m = LOF(n_neighbors=k, contamination=contamination)
            m.fit(X_fit)
            labels = m.labels_
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(X_fit, labels, sample_size=min(5000, len(X_fit)))
            logger.debug("LOF k=%d silhouette=%.4f", k, score)
            if score > best_score:
                best_score, best_k = score, k
        except Exception as e:
            logger.warning("LOF k=%d failed: %s", k, e)

    model = LOF(n_neighbors=best_k, contamination=contamination)
    model.fit(X_fit)
    logger.info("LOF trained with k=%d on %d samples", best_k, len(X_fit))
    return {
        "model": model, "scaler": scaler, "feature_cols": feature_cols,
        "best_k": best_k, "fit_on_subsample": len(X_scaled) > subsample_threshold,
    }


def train_gmm(
    train_df: pd.DataFrame,
    contamination: float,
    max_components: int,
    random_seed: int,
) -> dict[str, Any]:
    """Train GMM detector; select n_components via BIC."""
    from pyod.models.gmm import GMM

    X, feature_cols = _get_feature_matrix(train_df)
    scaler = StandardScaler()
    # float64 required: GMM covariance decomposition fails on float32 for sparse hashes
    X_scaled = scaler.fit_transform(X).astype(np.float64)

    # BIC-based component selection on a subsample to keep it fast
    bic_sample = X_scaled[:min(5000, len(X_scaled))]
    best_n, best_bic = 1, np.inf
    for n in range(1, max_components + 1):
        try:
            gm = GaussianMixture(
                n_components=n, random_state=random_seed, max_iter=100,
                reg_covar=1e-4,  # regularize against near-singular covariance matrices
            )
            gm.fit(bic_sample)
            bic = gm.bic(bic_sample)
            logger.debug("GMM n_components=%d BIC=%.2f", n, bic)
            if bic < best_bic:
                best_bic, best_n = bic, n
        except Exception as e:
            logger.warning("GMM n=%d BIC failed: %s", n, e)
            break

    model = GMM(n_components=best_n, contamination=contamination, reg_covar=1e-4)
    model.fit(X_scaled)
    logger.info("GMM trained with n_components=%d (BIC=%.2f)", best_n, best_bic)
    return {"model": model, "scaler": scaler, "feature_cols": feature_cols, "best_n": best_n}


def train_ocsvm(
    train_df: pd.DataFrame,
    contamination: float,
    subsample_threshold: int,
    subsample_size: int,
    nu: float,
    random_seed: int,
) -> dict[str, Any]:
    """Train One-Class SVM; subsample if dataset is large (speed budget)."""
    from pyod.models.ocsvm import OCSVM

    X, feature_cols = _get_feature_matrix(train_df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if len(X_scaled) > subsample_threshold:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(X_scaled), subsample_size, replace=False)
        X_fit = X_scaled[idx]
        logger.info("OCSVM: subsampling %d → %d rows", len(X_scaled), subsample_size)
    else:
        X_fit = X_scaled

    model = OCSVM(kernel="rbf", nu=nu, contamination=contamination)
    model.fit(X_fit)
    logger.info("OCSVM trained on %d samples", len(X_fit))
    return {
        "model": model, "scaler": scaler, "feature_cols": feature_cols,
        "fit_on_subsample": len(X_scaled) > subsample_threshold,
    }


def train_sod(
    train_df: pd.DataFrame,
    contamination: float,
    random_seed: int,
    include_sod: bool,
) -> dict[str, Any] | None:
    """Train SOD (optional). Returns None if include_sod=False."""
    if not include_sod:
        logger.info("SOD skipped (include_sod=False)")
        return None

    from pyod.models.sod import SOD

    X, feature_cols = _get_feature_matrix(train_df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SOD is expensive — subsample if large
    if len(X_scaled) > 5000:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(X_scaled), 5000, replace=False)
        X_fit = X_scaled[idx]
        logger.info("SOD: subsampling to 5000 rows")
    else:
        X_fit = X_scaled

    try:
        model = SOD(contamination=contamination)
        model.fit(X_fit)
        logger.info("SOD trained on %d samples", len(X_fit))
        return {"model": model, "scaler": scaler, "feature_cols": feature_cols}
    except Exception as e:
        logger.warning("SOD training failed: %s. Skipping.", e)
        return None


# ── Ensemble construction ─────────────────────────────────────────────────────

def build_ensemble(
    df: pd.DataFrame,
    iforest_bundle: dict[str, Any],
    knn_bundle: dict[str, Any],
    lof_bundle: dict[str, Any],
    gmm_bundle: dict[str, Any],
    ocsvm_bundle: dict[str, Any],
    sod_bundle: dict[str, Any] | None,
    ensemble_strategy: str,
) -> pd.DataFrame:
    """
    Score df with each model, normalize scores to [0,1], apply consensus voting.
    Returns df with columns: anomaly_{model}, score_{model}, ensemble_flag, ensemble_score.
    Decision: min-max normalize scores before aggregation (QUESTIONS.md Q6.3a).
    Consensus: all active models must flag (QUESTIONS.md Q6.2a).
    """
    results = df.copy()
    model_flags: list[pd.Series] = []
    model_scores: dict[str, np.ndarray] = {}

    bundles = {
        "iforest": iforest_bundle,
        "knn": knn_bundle,
        "lof": lof_bundle,
        "gmm": gmm_bundle,
        "ocsvm": ocsvm_bundle,
    }
    if sod_bundle is not None:
        bundles["sod"] = sod_bundle

    for name, bundle in bundles.items():
        model = bundle["model"]
        scaler = bundle["scaler"]
        feature_cols = bundle["feature_cols"]

        X = df[feature_cols].fillna(0).values.astype(np.float32)
        X_scaled = scaler.transform(X)

        try:
            labels = model.predict(X_scaled)          # 0=normal, 1=anomaly
            scores = model.decision_function(X_scaled) # higher = more anomalous
        except Exception as e:
            logger.warning("Model %s prediction failed: %s. Marking all normal.", name, e)
            labels = np.zeros(len(X), dtype=int)
            scores = np.zeros(len(X), dtype=float)

        # Min-max normalize scores to [0, 1]
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.zeros_like(scores)

        results[f"score_{name}"] = norm_scores
        results[f"anomaly_{name}"] = labels
        model_flags.append(pd.Series(labels, index=df.index))
        model_scores[name] = norm_scores

        logger.info(
            "Model %s: flagged %d anomalies (%.3f%%)",
            name, labels.sum(), 100 * labels.mean(),
        )

    # Consensus voting: all models must agree
    flags_df = pd.concat(model_flags, axis=1)
    results["ensemble_flag"] = (flags_df.sum(axis=1) == len(bundles)).astype(int)

    # Ensemble score = mean of normalized individual scores
    score_matrix = np.column_stack(list(model_scores.values()))
    results["ensemble_score"] = score_matrix.mean(axis=1)

    n_flagged = int(results["ensemble_flag"].sum())
    logger.info(
        "Ensemble (%s): flagged %d / %d (%.4f%%)",
        ensemble_strategy, n_flagged, len(results), 100 * n_flagged / max(len(results), 1),
    )
    return results


# ── Save model bundle ─────────────────────────────────────────────────────────

def save_model_bundle(
    category_key: str,
    iforest_bundle: dict,
    knn_bundle: dict,
    lof_bundle: dict,
    gmm_bundle: dict,
    ocsvm_bundle: dict,
    sod_bundle: dict | None,
    output_dir: str = "outputs/models",
) -> str:
    """Persist all model artifacts to disk via joblib."""
    import os
    cat_dir = os.path.join(output_dir, category_key)
    os.makedirs(cat_dir, exist_ok=True)

    bundles = {
        "iforest": iforest_bundle,
        "knn": knn_bundle,
        "lof": lof_bundle,
        "gmm": gmm_bundle,
        "ocsvm": ocsvm_bundle,
    }
    if sod_bundle is not None:
        bundles["sod"] = sod_bundle

    for name, bundle in bundles.items():
        path = os.path.join(cat_dir, f"{name}.joblib")
        joblib.dump(bundle, path)
        logger.info("Saved %s bundle to %s", name, path)

    return cat_dir


def load_model_bundle(category_key: str, output_dir: str = "outputs/models") -> dict:
    """Load all model artifacts for a category. Returns dict of bundles."""
    import os
    cat_dir = os.path.join(output_dir, category_key)
    bundles = {}
    for name in ("iforest", "knn", "lof", "gmm", "ocsvm", "sod"):
        path = os.path.join(cat_dir, f"{name}.joblib")
        if os.path.exists(path):
            bundles[name] = joblib.load(path)
    return bundles
