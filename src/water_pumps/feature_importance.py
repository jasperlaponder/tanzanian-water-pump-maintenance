from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance


def evaluate_feature_importance(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    output_dir: str | Path,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calculate and save validation-set permutation importance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    importance = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="balanced_accuracy",
        n_repeats=n_repeats,
        random_state=random_state,
    )

    importance_df = (
        pd.DataFrame(
            {
                "feature": X_val.columns,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    importance_df.to_csv(
        output_dir / "permutation_importance.csv",
        index=False,
    )

    print("\nTop 15 features by permutation importance:")
    print(importance_df.head(15).to_string(index=False))

    return importance_df
