"""
CSC311 ML Challenge - Text Bernoulli Naive Bayes Training
=========================================================
Uses binary word-presence features from the food and soundtrack columns,
then trains a Bernoulli Naive Bayes classifier.

Usage:
    python train_text_bnb.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import BernoulliNB

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "ml_challenge_dataset.csv")
OUT_DIR = os.path.join(BASE_DIR, "text_bnb_results")

LABEL_NAMES = [
    "The Persistence of Memory",
    "The Starry Night",
    "The Water Lily Pond",
]

TEXT_COLS = [
    "If this painting was a food, what would be?",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.",
]


def load_data():
    """Load the raw CSV and combine food + soundtrack into a single text field."""
    df = pd.read_csv(DATA_PATH)
    for col in TEXT_COLS:
        df[col] = df[col].fillna("").astype(str)

    df["text_input"] = (df[TEXT_COLS[0]] + " " + df[TEXT_COLS[1]]).str.strip()
    X_text = df["text_input"].values
    y = df["Painting"].values

    print(f"  Samples: {len(df)}")
    print(f"  Classes: {LABEL_NAMES}")
    print(f"  Label distribution:\n{df['Painting'].value_counts().to_string()}")
    return X_text, y


def save_label_map():
    """Save integer-to-label mapping."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "label_map.txt")
    with open(path, "w", encoding="utf-8") as f:
        for i, name in enumerate(LABEL_NAMES):
            f.write(f"{i},{name}\n")
    print(f"  Saved label map: {path}")


def build_features(X_train_text, X_val_text):
    """Vectorize text into binary word-presence features."""
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        binary=True,
        min_df=2,
        ngram_range=(1, 2),
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    vocab = vectorizer.get_feature_names_out()
    vocab_path = os.path.join(OUT_DIR, "vocabulary.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab))

    print(f"  Train matrix: {X_train.shape}")
    print(f"  Val matrix:   {X_val.shape}")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Saved vocabulary: {vocab_path}")
    return X_train, X_val, vectorizer


def grid_search(X_train, y_train):
    """Tune BernoulliNB smoothing with 5-fold stratified CV."""
    alpha_values = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    print("\n  Grid Search Progress:")
    print(f"  {'alpha':>8}  {'mean_cv':>8}  {'std_cv':>7}  {'train_acc':>9}")
    print("  " + "-" * 42)

    for alpha in alpha_values:
        model = BernoulliNB(alpha=alpha)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)

        row = {
            "alpha": alpha,
            "mean_cv_accuracy": round(cv_scores.mean(), 4),
            "std_cv_accuracy": round(cv_scores.std(), 4),
            "train_accuracy": round(train_acc, 4),
        }
        results.append(row)
        print(
            f"  {alpha:>8.1f}  {row['mean_cv_accuracy']:>8.4f}  "
            f"{row['std_cv_accuracy']:>7.4f}  {row['train_accuracy']:>9.4f}"
        )

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, "hyperparameter_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    return df


def plot_tuning_curve(results_df):
    """Plot CV and train accuracy against alpha."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        results_df["alpha"],
        results_df["mean_cv_accuracy"],
        yerr=results_df["std_cv_accuracy"],
        marker="o",
        capsize=4,
        label="CV accuracy",
    )
    ax.plot(
        results_df["alpha"],
        results_df["train_accuracy"],
        "--",
        alpha=0.7,
        label="Train accuracy",
    )
    ax.set_xlabel("alpha", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Text Bernoulli Naive Bayes: Hyperparameter Tuning", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "tuning_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def train_best_model(X_train, y_train, X_val, y_val, results_df):
    """Train the best BernoulliNB model and evaluate on the validation split."""
    best_row = results_df.loc[results_df["mean_cv_accuracy"].idxmax()]
    best_alpha = float(best_row["alpha"])

    print(f"\n  Best hyperparameter: alpha={best_alpha:.1f}")
    print(f"  Best CV accuracy: {best_row['mean_cv_accuracy']:.4f} +/- {best_row['std_cv_accuracy']:.4f}")

    model = BernoulliNB(alpha=best_alpha)
    model.fit(X_train, y_train)

    val_acc = model.score(X_val, y_val)
    y_pred = model.predict(X_val)

    print(f"\n  Validation accuracy: {val_acc:.4f}")
    print("\n  Classification Report:")
    report = classification_report(y_val, y_pred, target_names=LABEL_NAMES)
    print(report)

    cm = confusion_matrix(y_val, y_pred, labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Text Bernoulli Naive Bayes Confusion Matrix", fontsize=14)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    return model


def export_params(model):
    """Export learned BernoulliNB parameters."""
    np.save(os.path.join(OUT_DIR, "class_log_prior.npy"), model.class_log_prior_)
    np.save(os.path.join(OUT_DIR, "feature_log_prob.npy"), model.feature_log_prob_)
    print(f"\n  Exported: class_log_prior.npy shape={model.class_log_prior_.shape}")
    print(f"  Exported: feature_log_prob.npy shape={model.feature_log_prob_.shape}")


def main():
    print("=" * 60)
    print("TEXT BERNOULLI NAIVE BAYES TRAINING")
    print("=" * 60)

    print("\n1. LOADING RAW TEXT DATA")
    print("-" * 40)
    X_text, y = load_data()
    save_label_map()

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train set: {len(X_train_text)} samples")
    print(f"  Val set:   {len(X_val_text)} samples")

    print("\n2. BINARY TEXT FEATURE EXTRACTION")
    print("-" * 40)
    os.makedirs(OUT_DIR, exist_ok=True)
    X_train, X_val, _ = build_features(X_train_text, X_val_text)

    print("\n3. HYPERPARAMETER GRID SEARCH")
    print("-" * 40)
    results_df = grid_search(X_train, y_train)
    plot_tuning_curve(results_df)

    print("\n4. BEST MODEL TRAINING & EVALUATION")
    print("-" * 40)
    model = train_best_model(X_train, y_train, X_val, y_val, results_df)

    print("\n5. PARAMETER EXPORT")
    print("-" * 40)
    export_params(model)

    print("\nDone.")


if __name__ == "__main__":
    main()
