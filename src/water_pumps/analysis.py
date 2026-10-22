from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .features import prepare_features


def plot_population_distribution(
    train: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Plot the distribution of recorded population served."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    population = pd.to_numeric(train["population"], errors="coerce")
    valid = population[population >= 0]

    summary = pd.Series(
        {
            "total_pumps": len(population),
            "missing_population": population.isna().sum(),
            "zero_population": (population == 0).sum(),
            "negative_population": (population < 0).sum(),
            "median_population_including_zeros": population.median(),
            "positive_population_median": population[population > 0].median(),
        }
    )
    summary.to_csv(output_dir / "population_summary.csv", header=["value"])

    # log1p keeps zero values while making a long right tail easier to see.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.log1p(valid), bins=40)
    ax.set_title("Distribution of recorded population served")
    ax.set_xlabel("log(1 + recorded population)")
    ax.set_ylabel("Number of pumps")
    fig.tight_layout()
    fig.savefig(output_dir / "population_distribution.png", dpi=160)
    plt.close(fig)


def plot_pump_age_distribution(
    train: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Plot pump age at the time the record was collected."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = prepare_features(train)
    age = pd.to_numeric(features["pump_age_at_recording"], errors="coerce")
    valid = age[age >= 0]

    summary = pd.Series(
        {
            "total_pumps": len(age),
            "missing_or_invalid_age": age.isna().sum(),
            "median_age_years": valid.median(),
            "p90_age_years": valid.quantile(0.90),
            "maximum_age_years": valid.max(),
        }
    )
    summary.to_csv(output_dir / "pump_age_summary.csv", header=["value"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid, bins=30)
    ax.set_title("Pump age at the time of recording")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of pumps")
    fig.tight_layout()
    fig.savefig(output_dir / "pump_age_distribution.png", dpi=160)
    plt.close(fig)


def run_eda(train: pd.DataFrame, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    summary = {
        "rows": int(len(train)),
        "columns_including_target": int(train.shape[1]),
        "duplicate_rows": int(train.duplicated().sum()),
        "duplicate_ids": int(train["id"].duplicated().sum()) if "id" in train else None,
        "target_missing": int(train["status_group"].isna().sum()),
        "missing_cells": int(train.isna().sum().sum()),
        "missing_cell_fraction": float(train.isna().sum().sum() / max(train.size, 1)),
    }

    target_counts = train["status_group"].value_counts(dropna=False)
    target_counts.to_csv(output_dir / "class_distribution.csv", header=["count"])

    fig, ax = plt.subplots(figsize=(9, 5))
    target_counts.plot(kind="bar", ax=ax)
    ax.set_title("Pump status distribution in labelled training data")
    ax.set_xlabel("Observed status")
    ax.set_ylabel("Number of pumps")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=160)
    plt.close(fig)

    if "region" in train.columns:
        rates = pd.crosstab(train["region"], train["status_group"], normalize="index")
        rates = rates.reindex(rates.sum(axis=1).sort_values(ascending=False).index)
        rates.to_csv(output_dir / "status_by_region.csv")
        fig, ax = plt.subplots(figsize=(12, 7))
        rates.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Observed pump status by region (row-normalized)")
        ax.set_xlabel("Region")
        ax.set_ylabel("Share of pumps")
        ax.legend(title="Observed status", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(output_dir / "status_by_region.png", dpi=160)
        plt.close(fig)

    for column in [
        "quantity",
        "quantity_group",
        "waterpoint_type",
        "source",
        "management",
    ]:
        if column in train.columns:
            rates = pd.crosstab(
                train[column].fillna("Missing"),
                train["status_group"],
                normalize="index",
            )
            rates["number_of_pumps"] = train[column].fillna("Missing").value_counts()
            rates = rates.sort_values("number_of_pumps", ascending=False)
            rates.to_csv(output_dir / f"status_by_{column}.csv")

    missing = train.isna().mean().sort_values(ascending=False)
    missing.head(20).to_csv(
        output_dir / "top_missingness.csv", header=["missing_fraction"]
    )

    plot_population_distribution(train, output_dir)
    plot_pump_age_distribution(train, output_dir)

    print("\nTarget distribution:")
    print(target_counts.to_string())
    print("\nData quality summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nEDA outputs saved to: {output_dir}")
    return summary
