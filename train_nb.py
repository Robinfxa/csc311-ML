"""
CSC311 ML Challenge - Gaussian Naive Bayes Training
===================================================
Loads cleaned numeric data from cleaned_data/, tunes GaussianNB with
cross-validation, evaluates on the validation split, and exports the
learned class statistics for later pure-numpy inference if needed.

Usage:
    python train_nb.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "cleaned_data")
OUT_DIR = os.path.join(BASE_DIR, "nb_results")

LABEL_NAMES = [
    "The Persistence of Memory",
    "The Starry Night",
    "The Water Lily Pond",
]


def load_data():
    """Load standardized train/validation arrays and encode labels as ints."""
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_train_str = np.load(os.path.join(DATA_DIR, "y_train.npy"), allow_pickle=True)
    y_val_str = np.load(os.path.join(DATA_DIR, "y_val.npy"), allow_pickle=True)

    label_to_int = {name: i for i, name in enumerate(LABEL_NAMES)}
    y_train = np.array([label_to_int[s] for s in y_train_str])
    y_val = np.array([label_to_int[s] for s in y_val_str])

    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}")
    print(f"  Classes: {LABEL_NAMES}")
    print(f"  Train distribution: {np.bincount(y_train)}")
    print(f"  Val   distribution: {np.bincount(y_val)}")
    return X_train, X_val, y_train, y_val


def save_label_map():
    """Save integer-to-label mapping."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "label_map.txt")
    with open(path, "w", encoding="utf-8") as f:
        for i, name in enumerate(LABEL_NAMES):
            f.write(f"{i},{name}\n")
    print(f"  Saved label map: {path}")


def grid_search(X_train, y_train):
    """Tune GaussianNB via var_smoothing using 5-fold stratified CV."""
    smoothing_values = np.logspace(-12, -6, 7)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    print("\n  Grid Search Progress:")
    print(f"  {'var_smoothing':>14}  {'mean_cv':>8}  {'std_cv':>7}  {'train_acc':>9}")
    print("  " + "-" * 48)

    for smoothing in smoothing_values:
        model = GaussianNB(var_smoothing=smoothing)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)

        row = {
            "var_smoothing": smoothing,
            "mean_cv_accuracy": round(cv_scores.mean(), 4),
            "std_cv_accuracy": round(cv_scores.std(), 4),
            "train_accuracy": round(train_acc, 4),
        }
        results.append(row)
        print(
            f"  {smoothing:>14.1e}  {row['mean_cv_accuracy']:>8.4f}  "
            f"{row['std_cv_accuracy']:>7.4f}  {row['train_accuracy']:>9.4f}"
        )

    df = pd.DataFrame(results)
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "hyperparameter_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    return df


def plot_tuning_curve(results_df):
    """Plot CV and train accuracy against var_smoothing."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        results_df["var_smoothing"],
        results_df["mean_cv_accuracy"],
        yerr=results_df["std_cv_accuracy"],
        marker="o",
        capsize=4,
        label="CV accuracy",
    )
    ax.plot(
        results_df["var_smoothing"],
        results_df["train_accuracy"],
        "--",
        alpha=0.7,
        label="Train accuracy",
    )
    ax.set_xscale("log")
    ax.set_xlabel("var_smoothing (log scale)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Gaussian Naive Bayes: Hyperparameter Tuning", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "tuning_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def train_best_model(X_train, y_train, X_val, y_val, results_df):
    """Train the best GaussianNB model and evaluate it on the validation split."""
    best_row = results_df.loc[results_df["mean_cv_accuracy"].idxmax()]
    best_smoothing = float(best_row["var_smoothing"])

    print(f"\n  Best hyperparameter: var_smoothing={best_smoothing:.1e}")
    print(f"  Best CV accuracy: {best_row['mean_cv_accuracy']:.4f} +/- {best_row['std_cv_accuracy']:.4f}")

    model = GaussianNB(var_smoothing=best_smoothing)
    model.fit(X_train, y_train)

    val_acc = model.score(X_val, y_val)
    y_pred = model.predict(X_val)

    print(f"\n  Validation accuracy: {val_acc:.4f}")
    print("\n  Classification Report:")
    report = classification_report(y_val, y_pred, target_names=LABEL_NAMES)
    print(report)

    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Gaussian Naive Bayes Confusion Matrix", fontsize=14)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    return model


def export_params(model):
    """Export class priors, means, and variances for later reuse."""
    np.save(os.path.join(OUT_DIR, "class_prior.npy"), model.class_prior_)
    np.save(os.path.join(OUT_DIR, "theta.npy"), model.theta_)
    np.save(os.path.join(OUT_DIR, "var.npy"), model.var_)
    print(f"\n  Exported: class_prior.npy shape={model.class_prior_.shape}")
    print(f"  Exported: theta.npy shape={model.theta_.shape}")
    print(f"  Exported: var.npy shape={model.var_.shape}")


def main():
    print("=" * 60)
    print("GAUSSIAN NAIVE BAYES TRAINING")
    print("=" * 60)

    print("\n1. LOADING DATA")
    print("-" * 40)
    X_train, X_val, y_train, y_val = load_data()
    save_label_map()

    print("\n2. HYPERPARAMETER GRID SEARCH")
    print("-" * 40)
    results_df = grid_search(X_train, y_train)
    plot_tuning_curve(results_df)

    print("\n3. BEST MODEL TRAINING & EVALUATION")
    print("-" * 40)
    model = train_best_model(X_train, y_train, X_val, y_val, results_df)

    print("\n4. PARAMETER EXPORT")
    print("-" * 40)
    export_params(model)

    print("\nDone.")


if __name__ == "__main__":
    main()
