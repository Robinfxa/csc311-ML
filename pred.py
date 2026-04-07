"""
CSC311 ML Challenge - Final Prediction Script
=============================================
Ensemble of three models (majority vote):
  1. Logistic Regression  — pure numpy (weights + bias)
  2. Random Forest        — pure numpy (tree structure arrays)
  3. Gaussian Naive Bayes — pure numpy (theta + var + class_prior)

Allowed imports: numpy, pandas, sys, csv, random (no sklearn/joblib).

Usage:
    python pred.py                       # self-test on training CSV
    from pred import predict_all
    predict_all("test.csv")
"""

import os
import re
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "cleaned_data")
LR_DIR    = os.path.join(BASE_DIR, "lr_results")
RF_DIR    = os.path.join(BASE_DIR, "rf_results")
NB_DIR    = os.path.join(BASE_DIR, "nb_results")

# ── label names (alphabetical) ─────────────────────────────────────────
LABEL_NAMES = [
    "The Persistence of Memory",
    "The Starry Night",
    "The Water Lily Pond",
]

# ── column aliases ─────────────────────────────────────────────────────
COL_ALIASES = {
    "On a scale of 1\u201310, how intense is the emotion conveyed by the artwork?": "emotion_intensity",
    "Describe how this painting makes you feel.": "describe_feeling",
    "This art piece makes me feel sombre.": "feel_sombre",
    "This art piece makes me feel content.": "feel_content",
    "This art piece makes me feel calm.": "feel_calm",
    "This art piece makes me feel uneasy.": "feel_uneasy",
    "How many prominent colours do you notice in this painting?": "num_colours",
    "How many objects caught your eye in the painting?": "num_objects",
    "How much (in Canadian dollars) would you be willing to pay for this painting?": "willing_to_pay",
    "If you could purchase this painting, which room would you put that painting in?": "room",
    "If you could view this art in person, who would you want to view it with?": "view_with",
    "What season does this art piece remind you of?": "season",
    "If this painting was a food, what would be?": "food",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.": "soundtrack",
}

ORDINAL_COLS = ["feel_sombre", "feel_content", "feel_calm", "feel_uneasy"]
NUMERIC_COLS = ["emotion_intensity", "num_colours", "num_objects"]
MULTISELECT_COLS = {
    "room":      ["Bedroom", "Bathroom", "Office", "Living room", "Dining room"],
    "view_with": ["Friends", "Family members", "Coworkers/Classmates", "Strangers", "By yourself"],
    "season":    ["Spring", "Summer", "Fall", "Winter"],
}


# ═══════════════════════════════════════════════════════════════════════
#  LOAD MODEL PARAMETERS (once at import time)
# ═══════════════════════════════════════════════════════════════════════

# --- Scaler ---
_scaler_mean = np.load(os.path.join(DATA_DIR, "scaler_mean.npy"))
_scaler_std  = np.load(os.path.join(DATA_DIR, "scaler_std.npy"))

# --- Logistic Regression ---
_lr_W = np.load(os.path.join(LR_DIR, "lr_weights.npy"))
_lr_b = np.load(os.path.join(LR_DIR, "lr_bias.npy"))

# --- Random Forest tree structure ---
_rf_children_left  = np.load(os.path.join(RF_DIR, "rf_children_left.npy"))
_rf_children_right = np.load(os.path.join(RF_DIR, "rf_children_right.npy"))
_rf_feature        = np.load(os.path.join(RF_DIR, "rf_feature.npy"))
_rf_threshold      = np.load(os.path.join(RF_DIR, "rf_threshold.npy"))
_rf_value          = np.load(os.path.join(RF_DIR, "rf_value.npy"))
_rf_n_trees, _, _rf_n_classes = _rf_value.shape

# --- Gaussian Naive Bayes ---
_nb_theta       = np.load(os.path.join(NB_DIR, "theta.npy"))        # (n_classes, n_features)
_nb_var         = np.load(os.path.join(NB_DIR, "var.npy"))          # (n_classes, n_features)
_nb_class_prior = np.load(os.path.join(NB_DIR, "class_prior.npy"))  # (n_classes,)


# ═══════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

def _extract_ordinal(val):
    if pd.isna(val):
        return np.nan
    m = re.match(r"^(\d)", str(val).strip())
    return float(m.group(1)) if m else np.nan


def _extract_price(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    s = s.replace(",", "").replace("$", "").replace("cad", "").replace("dollars", "").replace("dollar", "")
    m = re.search(r"(\d+\.?\d*)", s)
    return float(m.group(1)) if m else np.nan


def preprocess(df):
    df = df.copy()
    df.rename(columns=COL_ALIASES, inplace=True)

    for col in ORDINAL_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_extract_ordinal)

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in NUMERIC_COLS + ORDINAL_COLS:
        if col in df.columns:
            med = df[col].median()
            df[col] = df[col].fillna(med if not pd.isna(med) else 3.0)

    if "willing_to_pay" in df.columns:
        df["willing_to_pay"] = df["willing_to_pay"].apply(_extract_price).clip(lower=0)
        df["willing_to_pay_log"] = np.log1p(df["willing_to_pay"])
        med = df["willing_to_pay_log"].median()
        df["willing_to_pay_log"] = df["willing_to_pay_log"].fillna(med if not pd.isna(med) else 5.0)
    else:
        df["willing_to_pay_log"] = 5.0

    for col, categories in MULTISELECT_COLS.items():
        for cat in categories:
            col_name = f"{col}_{cat}"
            if col in df.columns:
                df[col_name] = df[col].astype(str).apply(lambda x, c=cat: 1 if c in x else 0)
                mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
                df.loc[mask, col_name] = 0
            else:
                df[col_name] = 0

    feature_cols = NUMERIC_COLS + ORDINAL_COLS + ["willing_to_pay_log"]
    for col, categories in MULTISELECT_COLS.items():
        feature_cols += [f"{col}_{cat}" for cat in categories]

    return df[feature_cols].values.astype(float)


# ═══════════════════════════════════════════════════════════════════════
#  MODEL INFERENCE
# ═══════════════════════════════════════════════════════════════════════

def _predict_lr(X_scaled):
    """Logistic Regression: linear forward pass, argmax."""
    logits = X_scaled @ _lr_W.T + _lr_b
    return np.argmax(logits, axis=1)


def _predict_rf(X_scaled):
    """
    Random Forest: vectorized pure-numpy tree traversal.
    For each tree, propagate all samples simultaneously down the tree.
    Accumulate leaf class counts, then argmax.
    """
    n_samples = X_scaled.shape[0]
    votes = np.zeros((n_samples, _rf_n_classes), dtype=np.float64)

    for t in range(_rf_n_trees):
        nodes = np.zeros(n_samples, dtype=np.int32)
        while True:
            left = _rf_children_left[t, nodes]
            is_leaf = left == -1
            if is_leaf.all():
                break
            feat   = _rf_feature[t, nodes]
            thresh = _rf_threshold[t, nodes]
            # clip feat to valid range (padded nodes have feat=-2)
            feat_safe = np.clip(feat, 0, X_scaled.shape[1] - 1)
            go_right  = X_scaled[np.arange(n_samples), feat_safe] > thresh
            right     = _rf_children_right[t, nodes]
            new_nodes = np.where(go_right, right, left)
            nodes     = np.where(is_leaf, nodes, new_nodes)
        votes += _rf_value[t, nodes]   # (n_samples, n_classes)

    return np.argmax(votes, axis=1)


def _predict_nb(X_scaled):
    """
    Gaussian Naive Bayes: vectorized joint log-likelihood.
    Matches pred_nb.py implementation by Boyu Song.
    """
    var = np.maximum(_nb_var, 1e-9)
    log_prior = np.log(_nb_class_prior)
    log_prob  = (-0.5 * np.sum(np.log(2.0 * np.pi * var), axis=1)).reshape(1, -1)
    squared   = -0.5 * np.sum(
        ((X_scaled[:, None, :] - _nb_theta[None, :, :]) ** 2) / var[None, :, :],
        axis=2,
    )
    jll = squared + log_prob + log_prior.reshape(1, -1)
    return np.argmax(jll, axis=1)


# ═══════════════════════════════════════════════════════════════════════
#  ENSEMBLE: MAJORITY VOTE
# ═══════════════════════════════════════════════════════════════════════

def _majority_vote(pred_lr, pred_rf, pred_nb):
    """
    For each sample, take the majority vote of three classifiers.
    Ties broken by preferring RF > LR > NB (RF is highest CV accuracy).
    """
    n = len(pred_lr)
    result = np.empty(n, dtype=np.int32)
    for i in range(n):
        votes = [pred_lr[i], pred_rf[i], pred_nb[i]]
        counts = np.bincount(votes, minlength=len(LABEL_NAMES))
        max_count = counts.max()
        if max_count >= 2:
            result[i] = np.argmax(counts)
        else:
            # All three disagree — fall back to RF
            result[i] = pred_rf[i]
    return result


# ═══════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

def predict_all(filename):
    """
    Load a CSV, preprocess, run ensemble, return list of painting names.
    This is the function signature required by the ML Challenge.
    """
    df       = pd.read_csv(filename)
    X        = preprocess(df)
    X_scaled = (X - _scaler_mean) / _scaler_std

    pred_lr = _predict_lr(X_scaled)
    pred_rf = _predict_rf(X_scaled)
    pred_nb = _predict_nb(X_scaled)

    final   = _majority_vote(pred_lr, pred_rf, pred_nb)
    return [LABEL_NAMES[i] for i in final]


# ═══════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    csv_path = os.path.join(BASE_DIR, "ml_challenge_dataset.csv")
    print(f"Testing pred.py (ensemble) on: {csv_path}")

    preds = predict_all(csv_path)
    print(f"Total predictions: {len(preds)}")
    print(f"Sample: {preds[:5]}")

    df = pd.read_csv(csv_path)
    if "Painting" in df.columns:
        labels = df["Painting"].values
        acc = np.mean(np.array(preds) == labels)

        # Also report individual model accuracy for comparison
        X        = preprocess(df)
        X_scaled = (X - _scaler_mean) / _scaler_std
        acc_lr = np.mean(np.array([LABEL_NAMES[i] for i in _predict_lr(X_scaled)]) == labels)
        acc_rf = np.mean(np.array([LABEL_NAMES[i] for i in _predict_rf(X_scaled)]) == labels)
        acc_nb = np.mean(np.array([LABEL_NAMES[i] for i in _predict_nb(X_scaled)]) == labels)

        print(f"\n  LR  accuracy: {acc_lr:.4f}")
        print(f"  RF  accuracy: {acc_rf:.4f}")
        print(f"  NB  accuracy: {acc_nb:.4f}")
        print(f"  Ensemble:     {acc:.4f}")
