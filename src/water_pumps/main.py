from .analysis import run_eda
from .dataloader import load_data, split_data
from .feature_importance import evaluate_feature_importance
from .features import prepare_features
from .model import evaluate_models, make_binary_target
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
    models, results = evaluate_models(
        train=train,
        validation=validation,
        output_dir="outputs",
    )

    X_val = prepare_features(validation)
    y_val_binary = make_binary_target(validation["status_group"])

    evaluate_feature_importance(
        models[("binary_screening", "random_forest")],
        X_val,
        y_val_binary,
        output_dir="outputs/figures",
    )


if __name__ == "__main__":
    main()
