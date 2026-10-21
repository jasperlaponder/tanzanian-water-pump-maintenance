from .analysis import run_eda
from .dataloader import load_data, split_data
from .model import evaluate_models
from .prioritization import create_priority_lists


def main():
    data = load_data()
    train, validation, test = split_data(data)

    print(
        f"Train: {len(train):,}, "
        f"validation: {len(validation):,}, "
        f"holdout: {len(test):,}"
    )

    run_eda(train, "outputs/figures")
    model, _ = evaluate_models(train, validation, "outputs/metrics")
    create_priority_lists(
        model=model,
        test=test,
        output_dir="outputs/predictions",
    )


if __name__ == "__main__":
    main()