import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from .features import (
    build_preprocessor,
    prepare_catboost_features,
    prepare_features,
)

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


RANDOM_STATE = 42
INTERVENTION_CLASSES = {
    "functional needs repair",
    "non functional",
}


def make_binary_target(y: pd.Series) -> pd.Series:
    """Encode pumps requiring intervention as 1; functional pumps as 0."""
    return y.isin(INTERVENTION_CLASSES).astype(int)


def make_random_forest(
    X_train: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build a Random Forest pipeline with training-fitted preprocessing."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def calculate_metrics(
    y_true,
    y_pred,
    prefix: str = "",
) -> dict:
    """Calculate classification metrics."""
    return {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


def calculate_screening_metrics(
    y_true_binary: pd.Series,
    y_pred_binary,
) -> dict:
    """Evaluate whether pumps needing intervention are identified."""
    return {
        "screening_accuracy": accuracy_score(y_true_binary, y_pred_binary),
        "screening_balanced_accuracy": balanced_accuracy_score(
            y_true_binary, y_pred_binary
        ),
        "screening_precision": precision_score(
            y_true_binary,
            y_pred_binary,
            pos_label=1,
            zero_division=0,
        ),
        "screening_recall": recall_score(
            y_true_binary,
            y_pred_binary,
            pos_label=1,
            zero_division=0,
        ),
        "screening_f1": f1_score(
            y_true_binary,
            y_pred_binary,
            pos_label=1,
            zero_division=0,
        ),
    }


def save_evaluation(
    y_true,
    y_pred,
    labels: list,
    target_names: list[str],
    output_dir: Path,
    task_name: str,
    model_name: str,
) -> None:
    """Save a classification report and confusion-matrix plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{task_name}_{model_name}"

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    with (output_dir / f"{prefix}_classification_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=target_names,
    )
    display.plot(values_format="d", xticks_rotation=30)
    plt.title(f"{model_name.replace('_', ' ').title()} — {task_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close()


def evaluate_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: str | Path,
    random_state: int = RANDOM_STATE,
    include_catboost: bool = False,
) -> tuple[dict, pd.DataFrame]:
    """
    Train and compare models for three-class classification and
    binary intervention screening.

    Returns
    -------
    models:
        Fitted models keyed by (task_name, model_name).
    results:
        Validation metrics for baselines and fitted models.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Shared engineered features; preprocessing is model-specific.
    X_train = prepare_features(train)
    X_val = prepare_features(validation)

    y_train_three = train["status_group"]
    y_val_three = validation["status_group"]

    y_train_binary = make_binary_target(y_train_three)
    y_val_binary = make_binary_target(y_val_three)

    tasks = {
        "three_class": (y_train_three, y_val_three),
        "binary_screening": (y_train_binary, y_val_binary),
    }

    models = {}
    result_rows = []

    for task_name, (y_train, y_val) in tasks.items():
        print(f"\nTraining task: {task_name}")
        # The baseline predicts the most common training class.
        baseline_class = y_train.mode().iloc[0]
        baseline_pred = np.full(len(y_val), baseline_class)

        if task_name == "three_class":
            labels = sorted(y_train.unique().tolist())
            target_names = [str(label) for label in labels]
        else:
            labels = [0, 1]
            target_names = ["functional", "intervention_needed"]

        # Keep natural task metrics and also assess intervention screening.
        predictions = {
            "majority_baseline": baseline_pred,
        }

        # Random Forest: imputation and one-hot encoding in a pipeline.
        print("  Fitting Random Forest...")
        rf_model = make_random_forest(X_train, random_state)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_val)

        models[(task_name, "random_forest")] = rf_model
        predictions["random_forest"] = rf_pred

        # CatBoost: pass categorical features as strings, without one-hot
        # encoding. Numeric missing values can remain NaN.
        if CatBoostClassifier is not None and include_catboost:
            X_train_cb, categorical_columns = prepare_catboost_features(X_train)
            X_val_cb, _ = prepare_catboost_features(X_val)

            loss_function = "MultiClass" if task_name == "three_class" else "Logloss"

            catboost_model = CatBoostClassifier(
                iterations=500,
                depth=8,
                learning_rate=0.05,
                loss_function=loss_function,
                auto_class_weights="Balanced",
                random_seed=random_state,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            )
            print("  Fitting CatBoost...")
            catboost_model.fit(
                X_train_cb,
                y_train,
                cat_features=categorical_columns,
                eval_set=(X_val_cb, y_val),
                early_stopping_rounds=50,
                verbose=False,
            )

            catboost_pred = np.asarray(catboost_model.predict(X_val_cb)).ravel()

            models[(task_name, "catboost")] = catboost_model
            predictions["catboost"] = catboost_pred
        elif CatBoostClassifier is None and include_catboost:
            warnings.warn(
                "CatBoost is not installed; skipping CatBoost models. "
                "Install it with `pip install catboost` to enable them."
            )

        # Score the baseline and every fitted model.
        for model_name, y_pred in predictions.items():
            row = {
                "task": task_name,
                "model": model_name,
                **calculate_metrics(y_val, y_pred),
            }

            # Also compare each model as a binary intervention screen.
            if task_name == "three_class":
                screening_pred = (
                    pd.Series(y_pred, index=y_val.index)
                    .ne("functional")
                    .astype(int)
                    .to_numpy()
                )
                screening_true = y_val_binary.to_numpy()
            else:
                screening_pred = np.asarray(y_pred, dtype=int)
                screening_true = y_val.to_numpy()

            row.update(calculate_screening_metrics(screening_true, screening_pred))
            result_rows.append(row)

            save_evaluation(
                y_true=y_val,
                y_pred=y_pred,
                labels=labels,
                target_names=target_names,
                output_dir=reports_dir,
                task_name=task_name,
                model_name=model_name,
            )

    results = pd.DataFrame(result_rows)
    results.to_csv(output_dir / "metrics" / "model_comparison.csv", index=False)

    print("\nValidation model comparison:")
    print(
        results.sort_values(
            ["task", "screening_balanced_accuracy"],
            ascending=[True, False],
        ).to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    return models, results
