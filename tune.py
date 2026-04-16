
import wandb
import torch

from data import load_data
from models import LinearRegressionModel, RandomForestModel, XGBoostModel
from train import train_linear_regression, train_random_forest, train_xgboost


def tune_random_forest(train_loader, val_loader):
    configs = [
        # Shallow + strong regularization
        {"n_estimators": 200, "max_depth": 3, "min_samples_leaf": 10, "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": 3, "min_samples_leaf": 10, "max_features": 0.5},

        # Moderate depth
        {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 5, "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": 5, "min_samples_leaf": 5, "max_features": 0.8},
        {"n_estimators": 800, "max_depth": 5, "min_samples_leaf": 2, "max_features": 0.8},

        # Balanced configs (likely sweet spot)
        {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 5, "max_features": "sqrt"},
        {"n_estimators": 800, "max_depth": 8, "min_samples_leaf": 5, "max_features": 0.8},
        {"n_estimators": 1000, "max_depth": 8, "min_samples_leaf": 2, "max_features": 0.8},

        # Deeper trees (risk of overfit)
        {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"n_estimators": 800, "max_depth": 12, "min_samples_leaf": 1, "max_features": 0.8},
        {"n_estimators": 1000, "max_depth": None, "min_samples_leaf": 1, "max_features": 1.0},

        # High randomness configs
        {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 10, "max_features": 0.5},
        {"n_estimators": 800, "max_depth": 10, "min_samples_leaf": 5, "max_features": 0.5},

        # High capacity (test limits)
        {"n_estimators": 1200, "max_depth": None, "min_samples_leaf": 1, "max_features": 0.8},
        {"n_estimators": 1500, "max_depth": None, "min_samples_leaf": 1, "max_features": 1.0},
    ]

    best_model = None
    best_config = None
    best_val_rmse = float("inf")
    results = []

    for i, cfg in enumerate(configs, 1):
        print(f"\nRunning config {i}/{len(configs)}: {cfg}")

        model = RandomForestModel(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            random_state=1802
        )

        trained_model, metrics = train_random_forest(model, train_loader, val_loader)

        results.append({
            "config": cfg,
            "val_rmse": metrics["val_rmse"],
            "val_mae": metrics["val_mae"],
            "val_rmse_normalized": metrics["val_rmse_normalized"],
        })

        if metrics["val_rmse"] < best_val_rmse:
            best_val_rmse = metrics["val_rmse"]
            best_config = cfg
            best_model = trained_model

    print("\nBest config:")
    print(best_config)
    print(f"Best val RMSE: {best_val_rmse:.4f}")

    print("\nAll results:")
    for r in results:
        print(r)

    return best_model, best_config, results



def tune_xgboost(train_loader, val_loader):
    configs = [
        # Conservative / anti-overfit
        {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 2},
        {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 2},
        {"n_estimators": 800, "learning_rate": 0.02, "max_depth": 2},

        # Balanced
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 800, "learning_rate": 0.03, "max_depth": 3},

        # More capacity
        {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 4},
        {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 4},
        {"n_estimators": 800, "learning_rate": 0.03, "max_depth": 4},

        # Aggressive / likely overfit
        {"n_estimators": 500, "learning_rate": 0.1, "max_depth": 5},
        {"n_estimators": 800, "learning_rate": 0.05, "max_depth": 5},
        {"n_estimators": 1000, "learning_rate": 0.03, "max_depth": 6},
    ]

    best_model = None
    best_config = None
    best_val_rmse = float("inf")
    results = []

    for i, cfg in enumerate(configs, 1):
        print(f"\nRunning config {i}/{len(configs)}: {cfg}")

        model = XGBoostModel(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            random_state=1802
        )

        trained_model, metrics = train_xgboost(model, train_loader, val_loader)

        results.append({
            "config": cfg,
            "val_rmse": metrics["val_rmse"],
            "val_mae": metrics["val_mae"],
            "val_rmse_normalized": metrics["val_rmse_normalized"],
        })

        if metrics["val_rmse"] < best_val_rmse:
            best_val_rmse = metrics["val_rmse"]
            best_config = cfg
            best_model = trained_model

    print("\nBest config:")
    print(best_config)
    print(f"Best val RMSE: {best_val_rmse:.4f}")

    print("\nAll results:")
    for r in results:
        print(r)

    return best_model, best_config, results

