import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DROP_COLUMNS = ["id", "date_recorded"]

CATBOOST_CATEGORICAL_COLUMNS = [
    "region_code",
    "district_code",
]


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create modest, explainable features without using the target."""
    X = frame.drop(columns=["status_group"], errors="ignore").copy()

    if "date_recorded" in X.columns:
        dates = pd.to_datetime(X["date_recorded"], errors="coerce")
        X["recorded_year"] = dates.dt.year
        X["recorded_month"] = dates.dt.month
        X = X.drop(columns="date_recorded")

    if "construction_year" in X.columns and "recorded_year" in X.columns:
        year_built = pd.to_numeric(X["construction_year"], errors="coerce")
        X["pump_age_at_recording"] = (X["recorded_year"] - year_built).where(
            year_built > 0
        )
        X.loc[X["pump_age_at_recording"] < 0, "pump_age_at_recording"] = pd.NA

    # IDs identify individual rows, not reusable pump characteristics.
    X = X.drop(columns=["id"], errors="ignore")

    # Normalize categorical values to strings while preserving missing values.
    for column in X.select_dtypes(include=["object"]).columns:
        X[column] = X[column].replace("", np.nan)
        X[column] = X[column].map(
            lambda value: str(value) if pd.notna(value) else np.nan
        )
    return X


def prepare_catboost_features(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare features for CatBoost's native categorical handling."""
    X_cb = X.copy()

    categorical_columns = X_cb.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    for column in CATBOOST_CATEGORICAL_COLUMNS:
        if column in X_cb.columns and column not in categorical_columns:
            categorical_columns.append(column)

    for column in categorical_columns:
        X_cb[column] = X_cb[column].map(
            lambda value: str(value) if pd.notna(value) else "Missing"
        )

    return X_cb, categorical_columns


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
