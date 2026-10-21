from .analysis import run_eda
from .dataloader import load_data, split_data
from .feature_importance import evaluate_feature_importance
from .features import prepare_features
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

    run_eda(data, "outputs/figures")
    model, _ = evaluate_models(train, validation, "outputs/metrics")
    evaluate_feature_importance(model, prepare_features(validation), (validation["status_group"] != "functional").astype(int), "outputs/figures")
    # create_priority_lists(
    #     model=model,
    #     test=test,
    #     output_dir="outputs/predictions",
    # )


if __name__ == "__main__":
    main()