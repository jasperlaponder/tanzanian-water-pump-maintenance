import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from .features import build_preprocessor, prepare_features


def _metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def evaluate_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: str | Path,
    random_state: int = 42,
):
    """Compare a majority-class baseline with a Random Forest on held-out data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train = prepare_features(train)
    y_train = (train["status_group"] != "functional").astype(int)

    X_val = prepare_features(validation)
    y_val = (validation["status_group"] != "functional").astype(int)

    # Fit the baseline and Random Forest on X_train, y_train.
    # Evaluate both on X_val, y_val.
    results = {}
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_val)
    results["majority_class_baseline"] = _metrics(y_val, baseline_pred)

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    valid_pred = model.predict(X_val)
    results["random_forest"] = _metrics(y_val, valid_pred)

    report = classification_report(y_val, valid_pred, output_dict=True, zero_division=0)
    with open(output_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(output_dir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    labels = sorted(y_val.unique())
    cm = confusion_matrix(y_val, valid_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Random Forest: validation confusion matrix")
    ax.set_xlabel("Predicted status")
    ax.set_ylabel("Observed status")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print("\nValidation metrics:")
    print(pd.DataFrame(results).T.round(4).to_string())
    print("\nPer-class validation report:")
    print(classification_report(y_val, valid_pred, zero_division=0))
    print(f"Model evaluation outputs saved to: {output_dir}")

    return model, results
