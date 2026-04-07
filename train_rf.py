"""
CSC311 ML Challenge - Random Forest Training & Hyperparameter Tuning
=====================================================================
Loads cleaned data from cleaned_data/, trains Random Forest with grid search
over n_estimators × max_depth × max_features using 5-fold stratified CV,
and exports the best model via joblib.

Usage:
    python train_rf.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "cleaned_data")
OUT_DIR  = os.path.join(BASE_DIR, "rf_results")

# ── label mapping (alphabetical, same as LR) ──────────────────────────
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
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_train_str = np.load(os.path.join(DATA_DIR, "y_train.npy"), allow_pickle=True)
    y_val_str   = np.load(os.path.join(DATA_DIR, "y_val.npy"),   allow_pickle=True)

    label_to_int = {name: i for i, name in enumerate(LABEL_NAMES)}
    y_train = np.array([label_to_int[s] for s in y_train_str])
    y_val   = np.array([label_to_int[s] for s in y_val_str])

    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}")
    print(f"  Classes: {LABEL_NAMES}")
    print(f"  Train distribution: {np.bincount(y_train)}")
    print(f"  Val   distribution: {np.bincount(y_val)}")
    return X_train, X_val, y_train, y_val


def load_feature_names():
    """Load feature names saved by explore_data.py."""
    path = os.path.join(DATA_DIR, "feature_names.txt")
    if os.path.exists(path):
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    # fallback: reconstruct order used in pred_lr.py
    numeric  = ["emotion_intensity", "num_colours", "num_objects"]
    ordinal  = ["feel_sombre", "feel_content", "feel_calm", "feel_uneasy"]
    wtp      = ["willing_to_pay_log"]
    room     = [f"room_{c}" for c in ["Bedroom", "Bathroom", "Office", "Living room", "Dining room"]]
    view     = [f"view_with_{c}" for c in ["Friends", "Family members", "Coworkers/Classmates", "Strangers", "By yourself"]]
    season   = [f"season_{c}" for c in ["Spring", "Summer", "Fall", "Winter"]]
    return numeric + ordinal + wtp + room + view + season


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
    """Grid search over RF hyperparameters with 5-fold stratified CV."""
    param_grid = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth":    [None, 5, 8, 20],
        "max_features": ["sqrt", "log2"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    print("\n  Grid Search Progress:")
    print(f"  {'n_est':>5}  {'max_d':>6}  {'max_f':>6}  {'mean_cv':>8}  {'std_cv':>7}  {'train_acc':>9}")
    print("  " + "-" * 58)

    for n_est in param_grid["n_estimators"]:
        for max_d in param_grid["max_depth"]:
            for max_f in param_grid["max_features"]:
                model = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    max_features=max_f,
                    n_jobs=-1,
                    random_state=42,
                )
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
                model.fit(X_train, y_train)
                train_acc = model.score(X_train, y_train)

                depth_label = str(max_d) if max_d is not None else "None"
                row = {
                    "n_estimators":   n_est,
                    "max_depth":      depth_label,
                    "max_features":   max_f,
                    "mean_cv_accuracy": round(cv_scores.mean(), 4),
                    "std_cv_accuracy":  round(cv_scores.std(),  4),
                    "train_accuracy":   round(train_acc,        4),
                }
                results.append(row)
                print(f"  {n_est:>5}  {depth_label:>6}  {max_f:>6}  "
                      f"{row['mean_cv_accuracy']:>8.4f}  {row['std_cv_accuracy']:>7.4f}  "
                      f"{row['train_accuracy']:>9.4f}")

    df = pd.DataFrame(results)
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "hyperparameter_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    return df


def plot_tuning_curve(results_df):
    """Grouped bar chart: mean CV accuracy for each hyperparameter combo."""
    fig, ax = plt.subplots(figsize=(14, 5))
    labels = [
        f"n={r['n_estimators']} d={r['max_depth']} f={r['max_features']}"
        for _, r in results_df.iterrows()
    ]
    x = np.arange(len(labels))
    ax.bar(x, results_df["mean_cv_accuracy"], yerr=results_df["std_cv_accuracy"],
           capsize=4, color="steelblue", alpha=0.8, label="CV Accuracy")
    ax.plot(x, results_df["train_accuracy"], "ro--", alpha=0.6, label="Train Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Random Forest: Hyperparameter Tuning", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "tuning_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  3. BEST MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def train_best_model(X_train, y_train, X_val, y_val, results_df):
    """Let user pick a config from the grid search results, then train it."""
    # Print full results table for user to choose
    print("\n  Full Grid Search Results:")
    print(f"  {'#':>3}  {'n_est':>5}  {'max_d':>6}  {'max_f':>6}  {'cv_acc':>7}  {'std':>6}  {'train_acc':>9}")
    print("  " + "-" * 56)
    for i, row in results_df.iterrows():
        print(f"  {i:>3}  {int(row['n_estimators']):>5}  {str(row['max_depth']):>6}  "
              f"{row['max_features']:>6}  {row['mean_cv_accuracy']:>7.4f}  "
              f"{row['std_cv_accuracy']:>6.4f}  {row['train_accuracy']:>9.4f}")

    best_idx = results_df["mean_cv_accuracy"].idxmax()
    while True:
        choice = input(f"\n  Enter row # to use (default={best_idx}, highest CV): ").strip()
        if choice == "":
            chosen = results_df.loc[best_idx]
            break
        try:
            idx = int(choice)
            if idx in results_df.index:
                chosen = results_df.loc[idx]
                break
            print(f"  Invalid index, pick from 0-{len(results_df)-1}")
        except ValueError:
            print("  Please enter a number.")

    best_n_est  = int(chosen["n_estimators"])
    best_max_d  = None if chosen["max_depth"] == "None" else int(chosen["max_depth"])
    best_max_f  = chosen["max_features"]

    print(f"\n  Selected: n_estimators={best_n_est}, "
          f"max_depth={best_max_d}, max_features={best_max_f}")
    print(f"  CV accuracy: {chosen['mean_cv_accuracy']:.4f} ± {chosen['std_cv_accuracy']:.4f}  "
          f"train: {chosen['train_accuracy']:.4f}")

    model = RandomForestClassifier(
        n_estimators=best_n_est,
        max_depth=best_max_d,
        max_features=best_max_f,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    val_acc = model.score(X_val, y_val)
    y_pred  = model.predict(X_val)

    print(f"\n  Validation accuracy: {val_acc:.4f}")
    print(f"\n  Classification Report:")
    report = classification_report(y_val, y_pred, target_names=LABEL_NAMES)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, cmap="Greens", values_format="d")
    ax.set_title("Confusion Matrix — Random Forest (Validation)", fontsize=14)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    return model


# ═══════════════════════════════════════════════════════════════════════
#  4. FEATURE IMPORTANCE PLOT
# ═══════════════════════════════════════════════════════════════════════

def plot_feature_importance(model, feature_names):
    """Horizontal bar chart of mean decrease in impurity per feature."""
    importances = model.feature_importances_
    idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(range(len(idx)), importances[idx], align="center", color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=9)
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=12)
    ax.set_title("Random Forest Feature Importances", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  5. MODEL EXPORT
# ═══════════════════════════════════════════════════════════════════════

def export_model(model):
    """Serialize trained model with joblib."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "rf_model.joblib")
    joblib.dump(model, path)
    print(f"\n  Exported model: {path}")
    print(f"  n_estimators={model.n_estimators}  "
          f"max_depth={model.max_depth}  "
          f"max_features={model.max_features}")


def export_tree_structure(model):
    """
    Export the full forest as stacked numpy arrays for pure-numpy inference.

    Per tree arrays (shape: n_estimators x max_nodes):
        rf_children_left.npy   — left child index  (-1 = leaf)
        rf_children_right.npy  — right child index (-1 = leaf)
        rf_feature.npy         — split feature index (-2 = leaf)
        rf_threshold.npy       — split threshold
    Per tree leaf values (shape: n_estimators x max_nodes x n_classes):
        rf_value.npy           — class counts at each node
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    trees     = model.estimators_
    n_trees   = len(trees)
    n_classes = model.n_classes_
    max_nodes = max(t.tree_.node_count for t in trees)

    children_left  = np.full((n_trees, max_nodes), -1, dtype=np.int32)
    children_right = np.full((n_trees, max_nodes), -1, dtype=np.int32)
    feature        = np.full((n_trees, max_nodes), -2, dtype=np.int32)
    threshold      = np.full((n_trees, max_nodes), -2.0, dtype=np.float64)
    value          = np.zeros((n_trees, max_nodes, n_classes), dtype=np.float64)

    for i, tree in enumerate(trees):
        t = tree.tree_
        n = t.node_count
        children_left[i,  :n] = t.children_left
        children_right[i, :n] = t.children_right
        feature[i,        :n] = t.feature
        threshold[i,      :n] = t.threshold
        value[i,          :n] = t.value[:, 0, :]   # shape: (n_nodes, n_classes)

    np.save(os.path.join(OUT_DIR, "rf_children_left.npy"),  children_left)
    np.save(os.path.join(OUT_DIR, "rf_children_right.npy"), children_right)
    np.save(os.path.join(OUT_DIR, "rf_feature.npy"),        feature)
    np.save(os.path.join(OUT_DIR, "rf_threshold.npy"),      threshold)
    np.save(os.path.join(OUT_DIR, "rf_value.npy"),          value)

    print(f"\n  Exported tree structure:")
    print(f"    n_estimators={n_trees}  max_nodes={max_nodes}  n_classes={n_classes}")
    print(f"    children_left:  {children_left.shape}")
    print(f"    children_right: {children_right.shape}")
    print(f"    feature:        {feature.shape}")
    print(f"    threshold:      {threshold.shape}")
    print(f"    value:          {value.shape}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("RANDOM FOREST TRAINING")
    print("=" * 60)

    # 1. Load
    print("\n1. LOADING DATA")
    print("-" * 40)
    X_train, X_val, y_train, y_val = load_data()
    feature_names = load_feature_names()
    save_label_map()

    # 2. Grid Search
    print("\n2. HYPERPARAMETER GRID SEARCH")
    print("-" * 40)
    results_df = grid_search(X_train, y_train)
    plot_tuning_curve(results_df)

    # 3. Best Model
    print("\n3. BEST MODEL TRAINING & EVALUATION")
    print("-" * 40)
    model = train_best_model(X_train, y_train, X_val, y_val, results_df)

    # 4. Feature Importance
    print("\n4. FEATURE IMPORTANCE")
    print("-" * 40)
    plot_feature_importance(model, feature_names)

    # 5. Export
    print("\n5. MODEL EXPORT")
    print("-" * 40)
    export_model(model)
    export_tree_structure(model)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE [OK]")
    print("=" * 60)


if __name__ == "__main__":
    main()
