"""
CSC311 ML Challenge - Gaussian Naive Bayes Inference (Pure Numpy)
=================================================================
Loads exported GaussianNB parameters and performs prediction without sklearn.

Usage:
    python pred_nb.py
    from pred_nb import predict_all
    predict_all("test.csv")
"""

import os
import re
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(BASE_DIR, "nb_results")
DATA_DIR = os.path.join(BASE_DIR, "cleaned_data")

COL_ALIASES = {
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?": "emotion_intensity",
    "On a scale of 1鈥?0, how intense is the emotion conveyed by the artwork?": "emotion_intensity",
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
    "room": ["Bedroom", "Bathroom", "Office", "Living room", "Dining room"],
    "view_with": ["Friends", "Family members", "Coworkers/Classmates", "Strangers", "By yourself"],
    "season": ["Spring", "Summer", "Fall", "Winter"],
}


def _load_params():
    class_prior = np.load(os.path.join(NB_DIR, "class_prior.npy"))
    theta = np.load(os.path.join(NB_DIR, "theta.npy"))
    var = np.load(os.path.join(NB_DIR, "var.npy"))
    scaler_mean = np.load(os.path.join(DATA_DIR, "scaler_mean.npy"))
    scaler_std = np.load(os.path.join(DATA_DIR, "scaler_std.npy"))

    label_map = {}
    with open(os.path.join(NB_DIR, "label_map.txt"), "r", encoding="utf-8") as f:
        for line in f:
            idx, name = line.strip().split(",", 1)
            label_map[int(idx)] = name

    return class_prior, theta, var, scaler_mean, scaler_std, label_map


_class_prior, _theta, _var, _scaler_mean, _scaler_std, _label_map = _load_params()


def _extract_ordinal(val):
    """Extract leading integer from Likert string like '4 - Agree'."""
    if pd.isna(val):
        return np.nan
    m = re.match(r"^(\d)", str(val).strip())
    return float(m.group(1)) if m else np.nan


def _extract_price(val):
    """Extract numeric dollar amount from free text."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    s = s.replace(",", "").replace("$", "").replace("cad", "").replace("dollars", "").replace("dollar", "")
    m = re.search(r"(\d+\.?\d*)", s)
    return float(m.group(1)) if m else np.nan


def preprocess(df):
    """Apply the same preprocessing pipeline used by the numeric models."""
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
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 3.0
            df[col] = df[col].fillna(median_val)

    if "willing_to_pay" in df.columns:
        df["willing_to_pay"] = df["willing_to_pay"].apply(_extract_price)
        df["willing_to_pay"] = df["willing_to_pay"].clip(lower=0)
        df["willing_to_pay_log"] = np.log1p(df["willing_to_pay"])
        median_wtp = df["willing_to_pay_log"].median()
        if pd.isna(median_wtp):
            median_wtp = 5.0
        df["willing_to_pay_log"] = df["willing_to_pay_log"].fillna(median_wtp)
    else:
        df["willing_to_pay_log"] = 5.0

    for col, categories in MULTISELECT_COLS.items():
        for cat in categories:
            col_name = f"{col}_{cat}"
            if col in df.columns:
                text_values = df[col].fillna("").map(str)
                df[col_name] = text_values.apply(lambda x, c=cat: 1 if c in x else 0)
                mask = text_values.str.strip() == ""
                df.loc[mask, col_name] = 0
            else:
                df[col_name] = 0

    feature_cols = NUMERIC_COLS + ORDINAL_COLS + ["willing_to_pay_log"]
    for col, categories in MULTISELECT_COLS.items():
        feature_cols += [f"{col}_{cat}" for cat in categories]

    return df[feature_cols].values.astype(float)


def _joint_log_likelihood(X):
    """Compute Gaussian NB joint log likelihood for each class."""
    var = np.maximum(_var, 1e-9)
    log_prior = np.log(_class_prior)
    log_prob = -0.5 * np.sum(np.log(2.0 * np.pi * var), axis=1)
    log_prob = log_prob.reshape(1, -1)
    squared_term = -0.5 * np.sum(((X[:, None, :] - _theta[None, :, :]) ** 2) / var[None, :, :], axis=2)
    return squared_term + log_prob + log_prior.reshape(1, -1)


def predict_all(filename):
    """
    Load a CSV, preprocess it, and return predictions as painting names.
    """
    df = pd.read_csv(filename)
    X = preprocess(df)
    X_scaled = (X - _scaler_mean) / _scaler_std
    jll = _joint_log_likelihood(X_scaled)
    pred_indices = np.argmax(jll, axis=1)
    return [_label_map[i] for i in pred_indices]


if __name__ == "__main__":
    csv_path = os.path.join(BASE_DIR, "ml_challenge_dataset.csv")
    print(f"Testing pred_nb on: {csv_path}")

    preds = predict_all(csv_path)
    print(f"Total predictions: {len(preds)}")
    print(f"Sample: {preds[:5]}")

    df = pd.read_csv(csv_path)
    if "Painting" in df.columns:
        labels = df["Painting"].values
        acc = np.mean(np.array(preds) == labels)
        print(f"Accuracy on training CSV: {acc:.4f}")
