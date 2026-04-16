"""
Main script to easily run different models for various configs
"""

import wandb
import torch

from data import load_data
from models import LinearRegressionModel, RandomForestModel, XGBoostModel
from train import train_linear_regression, train_random_forest, train_xgboost
from tune import tune_random_forest, tune_xgboost


def main():
    run = wandb.init(
        entity="CaseStudyAssessment",
        project="Job Model Risk Assessment",
        config={
            "architecture": "XGBoost",
            "learning_rate": 0.03,
            "epochs": 100,
            "batch_size": 32,
            "n_estimators": 1000,
            "max_depth": 2,
        },
    )

    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, input_dim = load_data(
        file_path="data/cleaned_data.xlsx",
        target_column="prospective_risk",
        batch_size=config.batch_size,
        test_size=0.2,
        random_state=1802
    )

    if config.architecture == "Linear Regression":
        model = LinearRegressionModel(input_dim).to(device)
        train_linear_regression(model, train_loader, val_loader, config, device)

    elif config.architecture == "Random Forest Tune":
        best_model, best_config, results = tune_random_forest(train_loader, val_loader)

    elif config.architecture == "XGBoost Tune":
        best_model, best_config, results = tune_xgboost(train_loader, val_loader)

    elif config.architecture == "Random Forest":
        model = RandomForestModel(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=1802
        )
        train_random_forest(model, train_loader, val_loader)
    
    elif config.architecture == "XGBoost":
        model = XGBoostModel(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            random_state=1802
        )
        train_xgboost(model, train_loader, val_loader)

    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")

    # print("Selected best config:", best_config)
    # wandb.config.update(best_config, allow_val_change=True)

    wandb.finish()


if __name__ == "__main__":
    main()