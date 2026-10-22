from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw")
VALUES_FILE = "training_set_values.csv"
LABELS_FILE = "training_set_labels.csv"


from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def split_data(data: pd.DataFrame):

    # Reserve 40% for validation and final testing.
    train, remaining = train_test_split(
        data,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=data["status_group"],
    )

    # Split the remaining 40% equally into validation and final test sets.
    val, test = train_test_split(
        remaining,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=remaining["status_group"],
    )

    return train, val, test


def load_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load pump features and labels into one DataFrame."""
    values = pd.read_csv(data_dir / VALUES_FILE)
    labels = pd.read_csv(data_dir / LABELS_FILE)

    if values["id"].duplicated().any() or labels["id"].duplicated().any():
        raise ValueError("Duplicate pump IDs found.")

    data = values.merge(
        labels[["id", "status_group"]],
        on="id",
        how="inner",
        validate="one_to_one",
    )

    if len(data) != len(values) or len(data) != len(labels):
        raise ValueError("Feature rows and labels do not match one-to-one.")

    return data


if __name__ == "__main__":
    data = load_data()
    print(f"Loaded {len(data):,} pumps with {len(data.columns):,} columns.")
    print("\nTarget distribution:")
    print(data["status_group"].value_counts())
