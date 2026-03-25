"""
CSC311 ML Challenge - Logistic Regression Training & Hyperparameter Tuning
==========================================================================
Loads cleaned data from cleaned_data/, trains LR with grid search over
C × penalty using 5-fold stratified CV, and exports model parameters.

Usage:
    python train_lr.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "cleaned_data")
OUT_DIR = os.path.join(BASE_DIR, "lr_results")

# ── label mapping (alphabetical) ──────────────────────────────────────
LABEL_NAMES = [
    "The Persistence of Memory",
    "The Starry Night",
    "The Water Lily Pond",
]


# ═══════════════════════════════════════════════════════════════════════
#  1. DATA LOADING & LABEL ENCODING
# ═══════════════════════════════════════════════════════════════════════

def load_data():
    """Load standardized data and encode string labels to ints."""
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_train_str = np.load(os.path.join(DATA_DIR, "y_train.npy"), allow_pickle=True)
    y_val_str = np.load(os.path.join(DATA_DIR, "y_val.npy"), allow_pickle=True)

    # encode
    label_to_int = {name: i for i, name in enumerate(LABEL_NAMES)}
    y_train = np.array([label_to_int[s] for s in y_train_str])
    y_val = np.array([label_to_int[s] for s in y_val_str])

    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}")
    print(f"  Classes: {LABEL_NAMES}")
    print(f"  Train distribution: {np.bincount(y_train)}")
    print(f"  Val   distribution: {np.bincount(y_val)}")
    return X_train, X_val, y_train, y_val


def save_label_map():
    """Save label→int mapping to file."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "label_map.txt")
    with open(path, "w") as f:
        for i, name in enumerate(LABEL_NAMES):
            f.write(f"{i},{name}\n")
    print(f"  Saved label map: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  2. GRID SEARCH
# ═══════════════════════════════════════════════════════════════════════

def grid_search(X_train, y_train):
    """Grid search over C × penalty with 5-fold stratified CV."""
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    l1_ratios = {"l1": 1.0, "l2": 0.0}  # l1_ratio=1 → L1, l1_ratio=0 → L2
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    print("\n  Grid Search Progress:")
    print(f"  {'C':>8}  {'penalty':>7}  {'mean_cv':>8}  {'std_cv':>7}  {'train_acc':>9}")
    print("  " + "-" * 50)

    for penalty_name, l1_ratio in l1_ratios.items():
        solver = "saga"  # saga supports both L1 and L2
        for C in C_values:
            model = LogisticRegression(
                C=C, l1_ratio=l1_ratio, solver=solver,
                max_iter=5000, random_state=42,
            )
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
            # also get train accuracy
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)

            row = {
                "C": C,
                "penalty": penalty_name,
                "mean_cv_accuracy": round(cv_scores.mean(), 4),
                "std_cv_accuracy": round(cv_scores.std(), 4),
                "train_accuracy": round(train_acc, 4),
            }
            results.append(row)
            print(f"  {C:>8}  {penalty_name:>7}  {row['mean_cv_accuracy']:>8.4f}  "
                  f"{row['std_cv_accuracy']:>7.4f}  {row['train_accuracy']:>9.4f}")

    df = pd.DataFrame(results)
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "hyperparameter_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    return df


def plot_tuning_curve(results_df):
    """Line plot: C vs accuracy for each penalty."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for penalty in ["l1", "l2"]:
        subset = results_df[results_df["penalty"] == penalty]
        ax.errorbar(
            subset["C"], subset["mean_cv_accuracy"], yerr=subset["std_cv_accuracy"],
            marker="o", capsize=4, label=f"{penalty.upper()} (CV)",
        )
        ax.plot(subset["C"], subset["train_accuracy"], "--", alpha=0.5,
                label=f"{penalty.upper()} (Train)")
    ax.set_xscale("log")
    ax.set_xlabel("C (regularization strength, log scale)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Logistic Regression: Hyperparameter Tuning", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "tuning_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  3. BEST MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def train_best_model(X_train, y_train, X_val, y_val, results_df):
    """Train final model with best hyperparameters, evaluate on val set."""
    best_row = results_df.loc[results_df["mean_cv_accuracy"].idxmax()]
    best_C = best_row["C"]
    best_penalty = best_row["penalty"]
    l1_ratio = 1.0 if best_penalty == "l1" else 0.0

    print(f"\n  Best hyperparameters: C={best_C}, penalty={best_penalty}")
    print(f"  Best CV accuracy: {best_row['mean_cv_accuracy']:.4f} ± {best_row['std_cv_accuracy']:.4f}")

    model = LogisticRegression(
        C=best_C, l1_ratio=l1_ratio, solver="saga",
        max_iter=5000, random_state=42,
    )
    model.fit(X_train, y_train)

    # Validation performance
    val_acc = model.score(X_val, y_val)
    y_pred = model.predict(X_val)

    print(f"\n  Validation accuracy: {val_acc:.4f}")
    print(f"\n  Classification Report:")
    report = classification_report(y_val, y_pred, target_names=LABEL_NAMES)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix (Validation Set)", fontsize=14)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    return model, y_pred


# ═══════════════════════════════════════════════════════════════════════
#  4. PARAMETER EXPORT
# ═══════════════════════════════════════════════════════════════════════

def export_params(model):
    """Export weights and bias as npy files."""
    W = model.coef_          # shape: (n_classes, n_features)
    b = model.intercept_     # shape: (n_classes,)

    np.save(os.path.join(OUT_DIR, "lr_weights.npy"), W)
    np.save(os.path.join(OUT_DIR, "lr_bias.npy"), b)
    print(f"\n  Exported: lr_weights.npy shape={W.shape}")
    print(f"  Exported: lr_bias.npy shape={b.shape}")


# ═══════════════════════════════════════════════════════════════════════
#  5. NUMPY INFERENCE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

def verify_numpy_inference(model, X_val, y_val):
    """Verify that manual numpy softmax matches sklearn predictions."""
    W = model.coef_
    b = model.intercept_

    # Manual forward pass
    logits = X_val @ W.T + b
    numpy_preds = np.argmax(logits, axis=1)

    # sklearn predictions
    sklearn_preds = model.predict(X_val)

    match = np.all(numpy_preds == sklearn_preds)
    accuracy = np.mean(numpy_preds == sklearn_preds)

    print(f"\n  Numpy vs sklearn match: {accuracy*100:.1f}% ({np.sum(numpy_preds == sklearn_preds)}/{len(sklearn_preds)})")
    if match:
        print("  ✓ 100% match — numpy inference verified!")
    else:
        mismatches = np.where(numpy_preds != sklearn_preds)[0]
        print(f"  ⚠ {len(mismatches)} mismatches at indices: {mismatches[:10]}...")
    return match


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("LOGISTIC REGRESSION TRAINING")
    print("=" * 60)

    # 1. Load
    print("\n1. LOADING DATA")
    print("-" * 40)
    X_train, X_val, y_train, y_val = load_data()
    save_label_map()

    # 2. Grid Search
    print("\n2. HYPERPARAMETER GRID SEARCH")
    print("-" * 40)
    results_df = grid_search(X_train, y_train)
    plot_tuning_curve(results_df)

    # 3. Best Model
    print("\n3. BEST MODEL TRAINING & EVALUATION")
    print("-" * 40)
    model, y_pred = train_best_model(X_train, y_train, X_val, y_val, results_df)

    # 4. Export
    print("\n4. PARAMETER EXPORT")
    print("-" * 40)
    export_params(model)

    # 5. Verify numpy inference
    print("\n5. NUMPY INFERENCE VERIFICATION")
    print("-" * 40)
    verify_numpy_inference(model, X_val, y_val)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
