#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ml_common.py - shared definitions of the pipeline (imported by the other scripts).

- Repeated nested CV: the outer CV is run N_REPEATS times with seed SEED + repeat index
  (repeat 0, seed 42, is the single-run result).
- Optional covariate adjustment (e.g. age, sex): the linear effect of the covariates is
  removed from every feature. Coefficients are estimated on the training fold only,
  also within the inner CV, so there is no leakage.
- Feature selection frequencies are recorded for every outer fold.
- The permutation test calls nested_cv_once() of this module, so observed and null
  results come from exactly the same pipeline.

Unchanged: Welch t-test ranking (training data only), K_PERCENTAGES / K_CAP, 5x5 nested
CV, classifiers and hyperparameters, smaller k on ties, metrics.

All paths are built from the SNET_BASE environment variable (default ./snet_data)."""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"   # one core per job; parallelism comes from the Pool

import json
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- SETTINGS
BASE = Path(os.environ.get("SNET_BASE", "snet_data"))
GLOBAL_CACHE = BASE / "feature_cache"
FINAL_ML = BASE / "final_ml"
DEFAULT_MEASURE = "sift_invnodevol_radius2_count"

RESOLUTIONS = [156, 256, 356, 456, 556, 656, 756, 856, 956, 1056]
if os.environ.get("SNET_RESOLUTIONS"):          # for testing only
    RESOLUTIONS = [int(x) for x in os.environ["SNET_RESOLUTIONS"].split(",")]
THRESHOLDS = [0.05, 0.10, 0.15, 0.20]
NODE_METRICS = ["strength", "clustering", "betweenness", "efficiency", "closeness", "eigenvector"]

SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 5
N_REPEATS = 10
K_PERCENTAGES = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
K_CAP = 400

# (class 0, class 1); labels: 0=CN, 1=MCI, 2=AD
PAIRS = {"CN_AD": (0, 2), "CN_MCI": (0, 1), "MCI_AD": (1, 2)}
MODEL_NAMES = ["LogReg", "RandomForest", "SVM-RBF"]


def make_model(name):
    if name == "LogReg":
        return LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                      random_state=SEED, n_jobs=1)
    if name == "SVM-RBF":
        return SVC(kernel="rbf", class_weight="balanced", C=1.0, gamma="scale",
                   random_state=SEED)
    raise ValueError(name)


# ---------------------------------------------------------------- PATHS
def thresh_tag(pct):
    return f"thresh{int(round(pct * 100)):02d}"


def node_cache_dir(pct, measure):
    return BASE / f"feature_cache_node_{thresh_tag(pct)}" / measure


def result_path(pair, pct, measure, covariates):
    tag = "_cov" if covariates else ""
    return FINAL_ML / f"results_node_{thresh_tag(pct)}" / f"{pair}__{measure}{tag}.json"


def perm_dir(pair, measure, covariates):
    tag = "_cov" if covariates else ""
    return FINAL_ML / "permutations_node" / f"{pair}__{measure}{tag}"


# ---------------------------------------------------------------- DATA
def load_covariates(csv_path, subjects):
    """CSV with a 'subject' column plus covariate columns (e.g. age, sex).
        Non-numeric columns (e.g. sex = M/F) are converted to 0/1 dummy variables."""
    import pandas as pd
    df = pd.read_csv(csv_path, dtype={"subject": str})
    if "subject" not in df.columns:
        raise SystemExit("Covariate CSV must contain a 'subject' column")
    df = df.set_index("subject")
    missing = [s for s in subjects if s not in df.index]
    if missing:
        raise SystemExit(f"{len(missing)} participants missing from covariate CSV, e.g.: {missing[:5]}")
    df = df.loc[subjects]
    df = pd.get_dummies(df, drop_first=True, dtype=float)
    if df.isna().any().any():
        raise SystemExit("Covariates contain missing values; impute them or drop the participant")
    return df.to_numpy(dtype=np.float64), list(df.columns)


def flatten_metric_major(X3d):
    """(subject, node, metric) -> (subject, metric*node): all strengths first, then clustering, ..."""
    return X3d.transpose(0, 2, 1).reshape(X3d.shape[0], -1)


def metric_major_names(n_nodes, metric_names):
    width = len(str(n_nodes))
    return [f"{m}_node{node + 1:0{width}d}" for m in metric_names for node in range(n_nodes)]


def load_pair(pair, pct, measure, covariates_csv=None):
    cache = node_cache_dir(pct, measure)
    meta = json.load(open(cache / "meta.json"))
    if abs(meta["threshold_pct"] - pct) > 1e-9:
        raise SystemExit(f"Cache threshold {meta['threshold_pct']} != requested {pct}")
    labels3 = np.array(meta["labels3"])
    c0, c1 = PAIRS[pair]
    mask = np.isin(labels3, [c0, c1])
    y = (labels3[mask] == c1).astype(int)
    subjects = [s for s, m in zip(meta["subjects"], mask) if m]
    xs, names = {}, {}
    for r in RESOLUTIONS:
        X3d = np.load(cache / f"X_{r}.npy")[mask]
        xs[r] = flatten_metric_major(X3d)
        names[r] = metric_major_names(X3d.shape[1], meta["metric_names"])
    C, cov_names = (None, [])
    if covariates_csv:
        C, cov_names = load_covariates(covariates_csv, subjects)
    return y, xs, names, subjects, C, cov_names


def k_grid_for(n_features):
    ks = sorted(set(min(int(round(p * n_features)), K_CAP) for p in K_PERCENTAGES))
    return [k for k in ks if k >= 1]


# ---------------------------------------------------------------- PIPELINE
def rank_features(X, y):
    """Rank features by Welch t-test p-value (smallest first). Constant feature -> p = 1."""
    _, p = stats.ttest_ind(X[y == 0], X[y == 1], equal_var=False)
    p = np.nan_to_num(p, nan=1.0)
    return np.argsort(p, kind="stable")


def residualize(X_tr, X_te, C_tr, C_te):
    """Remove the covariate effect; coefficients are estimated on the training data only."""
    if C_tr is None:
        return X_tr, X_te
    D_tr = np.column_stack([np.ones(len(C_tr)), C_tr])
    D_te = np.column_stack([np.ones(len(C_te)), C_te])
    B, *_ = np.linalg.lstsq(D_tr, X_tr, rcond=None)
    return X_tr - D_tr @ B, X_te - D_te @ B


def _fit_predict(X_tr, y_tr, X_te, cols, model_name):
    scaler = StandardScaler().fit(X_tr[:, cols])
    model = make_model(model_name).fit(scaler.transform(X_tr[:, cols]), y_tr)
    Z = scaler.transform(X_te[:, cols])
    pred = model.predict(Z)
    s = model.decision_function(Z) if isinstance(model, SVC) else model.predict_proba(Z)[:, 1]
    return pred, s


def _sub(C, idx):
    return None if C is None else C[idx]


def select_k(X, y, C, model_name, k_grid, seed):
    """Inner CV; uses only the data it is given (the outer training set)."""
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    ba = np.zeros((INNER_FOLDS, len(k_grid)))
    for i, (tr, va) in enumerate(inner.split(X, y)):
        X_tr, X_va = residualize(X[tr], X[va], _sub(C, tr), _sub(C, va))
        order = rank_features(X_tr, y[tr])
        for j, k in enumerate(k_grid):
            pred, _ = _fit_predict(X_tr, y[tr], X_va, order[:k], model_name)
            ba[i, j] = balanced_accuracy_score(y[va], pred)
    mean_ba = ba.mean(axis=0)
    return k_grid[int(np.argmax(mean_ba))], mean_ba        # smaller k on ties


def safe_auc(y, s):
    try:
        return roc_auc_score(y, s)
    except ValueError:
        return 0.5


def nested_cv_once(X, y, C, model_name, k_grid, seed):
    """One repeat: 5 outer folds. Returns fold metrics, selected k values and selected columns."""
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
    out = {"acc": [], "ba": [], "auc": [], "f1": [], "mcc": [], "k": [], "cols": []}
    for tr, te in outer.split(X, y):
        k, _ = select_k(X[tr], y[tr], _sub(C, tr), model_name, k_grid, seed)
        X_tr, X_te = residualize(X[tr], X[te], _sub(C, tr), _sub(C, te))
        cols = rank_features(X_tr, y[tr])[:k]
        pred, s = _fit_predict(X_tr, y[tr], X_te, cols, model_name)
        out["acc"].append(accuracy_score(y[te], pred))
        out["ba"].append(balanced_accuracy_score(y[te], pred))
        out["auc"].append(safe_auc(y[te], s))
        out["f1"].append(f1_score(y[te], pred, average="macro", zero_division=0))
        out["mcc"].append(matthews_corrcoef(y[te], pred))
        out["k"].append(int(k))
        out["cols"].append(cols)
    return out


def run_repeated(X, y, C, model_name, k_grid, feature_names, n_repeats, top_n=50):
    """Nested CV repeated n_repeats times; repeat r uses seed SEED + r."""
    metrics = ("acc", "ba", "auc", "f1", "mcc")
    per_rep = {m: [] for m in metrics}
    fold_ba, ks = [], []
    counts = np.zeros(X.shape[1], dtype=int)
    n_folds_total = 0
    for r in range(n_repeats):
        o = nested_cv_once(X, y, C, model_name, k_grid, SEED + r)
        for m in metrics:
            per_rep[m].append(float(np.mean(o[m])))
        fold_ba.extend(float(v) for v in o["ba"])
        ks.extend(o["k"])
        for cols in o["cols"]:
            counts[cols] += 1
            n_folds_total += 1

    res = {}
    for m in metrics:
        v = np.array(per_rep[m])
        res[m] = float(v.mean())
        res[f"{m}_sd_repeats"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        res[f"{m}_per_repeat"] = [float(x) for x in v]
    res["ba_sd_folds"] = float(np.std(fold_ba, ddof=1))
    res["ba_rep0"] = per_rep["ba"][0]          # compared with the permutation test
    res["auc_rep0"] = per_rep["auc"][0]
    res["chosen_k"] = ks
    res["k_median"] = float(np.median(ks))
    top = np.argsort(-counts, kind="stable")[:top_n]
    res["selection_freq_top"] = [{"feature": feature_names[j], "freq": counts[j] / n_folds_total}
                                 for j in top if counts[j] > 0]
    res["n_outer_folds_total"] = n_folds_total
    return res
