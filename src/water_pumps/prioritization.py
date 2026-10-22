
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances

from .features import prepare_features

EARTH_RADIUS_KM = 6371.0
DEFAULT_NEIGHBOUR_RADIUS_KM = 2.0


def _get_probabilities(model, X: pd.DataFrame) -> pd.DataFrame:
    """Return class probabilities using the model's actual class labels."""
    probabilities = model.predict_proba(X)

    if hasattr(model, "named_steps"):
        classes = model.named_steps["classifier"].classes_
    else:
        classes = model.classes_

    return pd.DataFrame(probabilities, columns=classes, index=X.index)


def _add_geographical_scarcity(
    output: pd.DataFrame,
    radius_km: float,
) -> pd.DataFrame:
    """Estimate nearby functioning alternatives from predicted probabilities."""
    output = output.copy()
    output["nearby_pump_count"] = np.nan
    output["nearby_functional_alternatives"] = np.nan
    output["geographical_scarcity"] = np.nan

    lat = pd.to_numeric(output["latitude"], errors="coerce")
    lon = pd.to_numeric(output["longitude"], errors="coerce")

    valid = (
        lat.between(-90, 90)
        & lon.between(-180, 180)
        & lat.notna()
        & lon.notna()
    )
    indices = output.index[valid]

    if len(indices) < 2:
        return output

    coordinates = np.radians(
        np.column_stack([lat.loc[indices], lon.loc[indices]])
    )
    distances_km = (
        haversine_distances(coordinates, coordinates) * EARTH_RADIUS_KM
    )

    nearby = distances_km <= radius_km
    np.fill_diagonal(nearby, False)

    functional_probabilities = output.loc[
        indices, "probability_functional"
    ].to_numpy()

    neighbour_counts = nearby.sum(axis=1)
    alternatives = nearby @ functional_probabilities

    output.loc[indices, "nearby_pump_count"] = neighbour_counts
    output.loc[indices, "nearby_functional_alternatives"] = alternatives
    output.loc[indices, "geographical_scarcity"] = 1 / (1 + alternatives)

    return output


def create_priority_lists(
    model,
    data: pd.DataFrame,
    output_dir: str | Path,
    neighbour_radius_km: float = DEFAULT_NEIGHBOUR_RADIUS_KM,
) -> pd.DataFrame:
    """
    Create risk-first, risk × scarcity, and exploratory population rankings.

    Binary target encoding: 0 = functional; 1 = intervention needed.
    """
    if neighbour_radius_km <= 0:
        raise ValueError("neighbour_radius_km must be greater than zero.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = prepare_features(data)
    probabilities = _get_probabilities(model, X)

    if not {0, 1}.issubset(probabilities.columns):
        raise ValueError(
            "Expected a binary model with classes 0 (functional) "
            "and 1 (intervention needed)."
        )

    output = pd.DataFrame(index=data.index)
    if "id" in data.columns:
        output["id"] = data["id"]

    output["probability_functional"] = probabilities[0]
    output["risk_not_fully_functional"] = probabilities[1]
    output["predicted_status"] = pd.Series(
        model.predict(X), index=data.index
    ).map({0: "functional", 1: "intervention needed"})

    for column in ["region", "district_code", "latitude", "longitude"]:
        if column in data.columns:
            output[column] = data[column]

    if {"latitude", "longitude"}.issubset(output.columns):
        output = _add_geographical_scarcity(
            output, radius_km=neighbour_radius_km
        )
        output["risk_x_geographical_scarcity"] = (
            output["risk_not_fully_functional"]
            * output["geographical_scarcity"]
        )

    # Population is a questionable impact proxy: zero does not necessarily
    # mean that nobody depends on the pump.
    if "population" in data.columns:
        population = pd.to_numeric(data["population"], errors="coerce")
        valid_population = population.notna() & (population > 0)

        output["population_proxy"] = population
        output["population_missing_or_nonpositive"] = ~valid_population

        population_score = np.log1p(population.where(valid_population))
        if "geographical_scarcity" in output.columns:
            output["risk_x_population"] = (
                output["risk_not_fully_functional"] * population_score
            )
            output["risk_x_population_x_scarcity"] = (
                output["risk_x_population"]
                * output["geographical_scarcity"]
            )
        else:
            output["risk_x_population"] = (
                output["risk_not_fully_functional"] * population_score
            )

    # Risk-first ranking is the simplest, least assumption-heavy output.
    output.sort_values(
        "risk_not_fully_functional",
        ascending=False,
    ).to_csv(output_dir / "pumps_ranked_by_risk.csv", index=False)

    if "risk_x_geographical_scarcity" in output.columns:
        output.sort_values(
            "risk_x_geographical_scarcity",
            ascending=False,
            na_position="last",
        ).to_csv(
            output_dir / "pumps_ranked_by_risk_and_scarcity.csv",
            index=False,
        )

    if "risk_x_population" in output.columns:
        population_rank = (
            "risk_x_population_x_scarcity"
            if "risk_x_population_x_scarcity" in output.columns
            else "risk_x_population"
        )
        output.sort_values(
            population_rank,
            ascending=False,
            na_position="last",
        ).to_csv(
            output_dir / "pumps_ranked_by_risk_and_population.csv",
            index=False,
        )

    print(f"\nCreated priority rankings for {len(output):,} pumps.")
    print(f"Neighbour radius: {neighbour_radius_km:g} km")
    print("Population-adjusted rankings are exploratory.")
    print("Predictions describe recorded-condition risk, not verified "
          "current pump condition.")
    print(f"Outputs saved to: {output_dir}")

    return output
