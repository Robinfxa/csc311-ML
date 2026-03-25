"""
CSC311 ML Challenge - Data Exploration & Cleaning Pipeline
==========================================================
This script loads, cleans, and explores the ML Challenge dataset.
It produces:
  - Cleaned train/validation splits (CSV + npy)
  - EDA visualizations (feature distributions, correlation heatmap, class counts)
  - Saved scaler parameters for later use in pred.py

Usage:
    python explore_data.py
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── paths ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "ml_challenge_dataset.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "cleaned_data")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "eda_plots")

# Shorter column aliases for convenience
COL_ALIASES = {
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?": "emotion_intensity",
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

# Feature groups
ORDINAL_COLS = ["feel_sombre", "feel_content", "feel_calm", "feel_uneasy"]
NUMERIC_COLS = ["emotion_intensity", "num_colours", "num_objects"]
MULTISELECT_COLS = {
    "room": ["Bedroom", "Bathroom", "Office", "Living room", "Dining room"],
    "view_with": ["Friends", "Family members", "Coworkers/Classmates", "Strangers", "By yourself"],
    "season": ["Spring", "Summer", "Fall", "Winter"],
}
TEXT_COLS = ["describe_feeling", "food", "soundtrack"]


# ═══════════════════════════════════════════════════════════════════════
#  1. LOAD & INSPECT
# ═══════════════════════════════════════════════════════════════════════

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the CSV and rename columns to short aliases."""
    df = pd.read_csv(path)
    df.rename(columns=COL_ALIASES, inplace=True)
    print("=" * 60)
    print("DATASET LOADED")
    print("=" * 60)
    print(f"  Rows:    {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Label distribution:")
    print(df["Painting"].value_counts().to_string())
    print()
    return df


def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Print and return per-column missing-value stats."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    report = report[report.missing_count > 0].sort_values("missing_pct", ascending=False)
    print("MISSING VALUE REPORT")
    print("-" * 40)
    if report.empty:
        print("  No missing values found.")
    else:
        print(report.to_string())
    print()
    return report


# ═══════════════════════════════════════════════════════════════════════
#  2. CLEANING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def clean_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the leading integer from Likert-scale strings like '4 - Agree'."""
    for col in ORDINAL_COLS:
        df[col] = df[col].astype(str).str.extract(r"^(\d)")[0]
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns to float and fill NaN with median."""
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Fill NaN with median for all core numeric + ordinal cols
    all_num = NUMERIC_COLS + ORDINAL_COLS
    for col in all_num:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    return df


def clean_willing_to_pay(df: pd.DataFrame) -> pd.DataFrame:
    """Extract a numeric dollar amount from free-text willing_to_pay column."""

    def _parse_price(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        # Remove common words, currency symbols, commas
        s = s.replace(",", "").replace("$", "").replace("cad", "").replace("dollars", "").replace("dollar", "")
        # Try to find a number (int or float)
        match = re.search(r"(\d+\.?\d*)", s)
        if match:
            return float(match.group(1))
        return np.nan

    df["willing_to_pay"] = df["willing_to_pay"].apply(_parse_price)
    # Log-transform to handle huge range (0 to 100_000_000+)
    df["willing_to_pay"] = df["willing_to_pay"].clip(lower=0)
    df["willing_to_pay_log"] = np.log1p(df["willing_to_pay"])
    # Fill remaining NaN with median
    median_val = df["willing_to_pay_log"].median()
    df["willing_to_pay_log"] = df["willing_to_pay_log"].fillna(median_val)
    parse_rate = df["willing_to_pay"].notna().mean() * 100
    print(f"  willing_to_pay parse success rate: {parse_rate:.1f}%")
    return df


def encode_multiselect(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode multi-select columns (comma-separated values)."""
    for col, categories in MULTISELECT_COLS.items():
        for cat in categories:
            col_name = f"{col}_{cat}"
            df[col_name] = df[col].astype(str).apply(
                lambda x, c=cat: 1 if c in x else 0
            )
        # Rows where original was NaN → set all to 0
        mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        for cat in categories:
            df.loc[mask, f"{col}_{cat}"] = 0
    return df


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where all feature columns are empty."""
    feature_cols = [c for c in df.columns if c not in ("unique_id", "Painting")]
    before = len(df)
    df = df.dropna(subset=feature_cols, how="all").reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} completely empty rows.")
    return df


# ═══════════════════════════════════════════════════════════════════════
#  3. SCALING & SPLITTING
# ═══════════════════════════════════════════════════════════════════════

def get_feature_columns() -> list:
    """Return the list of feature column names used for modeling."""
    features = NUMERIC_COLS + ORDINAL_COLS + ["willing_to_pay_log"]
    for col, categories in MULTISELECT_COLS.items():
        features += [f"{col}_{cat}" for cat in categories]
    return features


def scale_features(X_train: np.ndarray, X_val: np.ndarray):
    """StandardScaler fit on train, transform both. Returns scaled arrays + scaler params."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled, scaler


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/val split. Returns X_train, X_val, y_train, y_val, feature_names."""
    feature_cols = get_feature_columns()
    X = df[feature_cols].values.astype(float)
    y = df["Painting"].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\n  Train set: {X_train.shape[0]} samples")
    print(f"  Val set:   {X_val.shape[0]} samples")
    print(f"  Features:  {X_train.shape[1]}")
    return X_train, X_val, y_train, y_val, feature_cols


def save_data(X_train, X_val, y_train, y_val, feature_cols, scaler):
    """Save cleaned data and scaler parameters."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(OUTPUT_DIR, "scaler_std.npy"), scaler.scale_)
    with open(os.path.join(OUTPUT_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_cols))
    print(f"\n  Saved cleaned data to {OUTPUT_DIR}/")


# ═══════════════════════════════════════════════════════════════════════
#  4. EDA VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════

def plot_class_distribution(df: pd.DataFrame):
    """Bar chart of painting class counts."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    counts = df["Painting"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax, color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="black")
    ax.set_title("Class Distribution (Painting Labels)", fontsize=14)
    ax.set_ylabel("Count")
    ax.set_xlabel("Painting")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "class_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_feature_distributions(df: pd.DataFrame):
    """Grouped histograms for each numeric feature, colored by painting."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    num_features = NUMERIC_COLS + ORDINAL_COLS + ["willing_to_pay_log"]
    paintings = df["Painting"].unique()
    n = len(num_features)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()
    for i, feat in enumerate(num_features):
        ax = axes[i]
        for painting in paintings:
            subset = df[df["Painting"] == painting][feat].dropna()
            ax.hist(subset, bins=20, alpha=0.5, label=painting)
        ax.set_title(feat, fontsize=11)
        ax.legend(fontsize=7)
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Feature Distributions by Painting", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "feature_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_correlation_matrix(df: pd.DataFrame):
    """Heatmap of feature correlations."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    num_features = NUMERIC_COLS + ORDINAL_COLS + ["willing_to_pay_log"]
    corr = df[num_features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_features)))
    ax.set_yticks(range(len(num_features)))
    ax.set_xticklabels(num_features, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(num_features, fontsize=9)
    # Annotate cells
    for i in range(len(num_features)):
        for j in range(len(num_features)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "correlation_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  5. MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    # 1. Load
    df = load_data()
    report_missing(df)

    # 2. Clean
    print("CLEANING DATA")
    print("-" * 40)
    df = drop_empty_rows(df)
    df = clean_ordinal(df)
    df = clean_numeric(df)
    df = clean_willing_to_pay(df)
    df = encode_multiselect(df)

    # Post-cleaning report
    print("\nPOST-CLEANING STATS")
    print("-" * 40)
    print(f"  Rows remaining: {len(df)}")
    feature_cols = get_feature_columns()
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")
    missing_in_features = df[feature_cols].isnull().sum().sum()
    print(f"  NaN in features: {missing_in_features}")

    # 3. Split & Scale
    print("\nSPLITTING & SCALING")
    print("-" * 40)
    X_train, X_val, y_train, y_val, feat_names = split_data(df)
    X_train_s, X_val_s, scaler = scale_features(X_train, X_val)

    # 4. Save
    save_data(X_train_s, X_val_s, y_train, y_val, feat_names, scaler)

    # 5. Visualize
    print("\nGENERATING EDA PLOTS")
    print("-" * 40)
    plot_class_distribution(df)
    plot_feature_distributions(df)
    plot_correlation_matrix(df)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
