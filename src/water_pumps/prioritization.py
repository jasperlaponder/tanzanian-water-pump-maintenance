from pathlib import Path

import numpy as np
import pandas as pd

from .features import prepare_features


def create_priority_lists(model, test: pd.DataFrame, output_dir: str | Path) -> None:
    """Create risk-first and illustrative risk×population rankings for unlabelled test rows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = test.drop(columns=["status_group"], errors="ignore")

    X = prepare_features(X)
    probabilities = model.predict_proba(X)
    classes = list(model.named_steps["classifier"].classes_)
    probability_frame = pd.DataFrame(probabilities, columns=classes, index=test.index)

    risk_columns = [
        c
        for c in ["non functional", "functional needs repair"]
        if c in probability_frame
    ]
    if not risk_columns:
        raise ValueError(f"Could not find expected risk classes in model: {classes}")

    output = pd.DataFrame(index=test.index)
    if "id" in test.columns:
        output["id"] = test["id"].values
    output["predicted_status"] = model.predict(X)
    output["probability_non_functional"] = probability_frame.get(
        "non functional", pd.Series(0.0, index=test.index)
    ).values
    output["probability_needs_repair"] = probability_frame.get(
        "functional needs repair", pd.Series(0.0, index=test.index)
    ).values
    output["risk_not_fully_functional"] = (
        probability_frame[risk_columns].sum(axis=1).values
    )

    # Population is an imperfect impact proxy, not a measure of actual dependence.
    if "population" in test.columns:
        population = pd.to_numeric(test["population"], errors="coerce").clip(lower=0)
        output["population_proxy"] = population.fillna(0).values
        output["risk_x_log_population_proxy"] = output[
            "risk_not_fully_functional"
        ] * np.log1p(output["population_proxy"])
    if "region" in test.columns:
        output["region"] = test["region"].values
    if "district_code" in test.columns:
        output["district_code"] = test["district_code"].values
    if "latitude" in test.columns:
        output["latitude"] = test["latitude"].values
    if "longitude" in test.columns:
        output["longitude"] = test["longitude"].values

    output.sort_values("risk_not_fully_functional", ascending=False).to_csv(
        output_dir / "test_pumps_ranked_by_risk.csv", index=False
    )
    if "risk_x_log_population_proxy" in output.columns:
        output.sort_values("risk_x_log_population_proxy", ascending=False).to_csv(
            output_dir / "test_pumps_ranked_by_risk_and_population_proxy.csv",
            index=False,
        )

    print(f"\nCreated risk rankings for {len(output):,} unlabelled test pumps.")
    print("These are model predictions, not verified current pump conditions.")
    print(f"Prediction outputs saved to: {output_dir}")
