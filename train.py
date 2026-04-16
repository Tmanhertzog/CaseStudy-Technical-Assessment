"""
Training utilities for the models defined in models.py.
This file exposes functions to train:
- Linear Regression (PyTorch)
- Random Forest Regressor
- XGBoost Regressor
"""
 
import numpy as np
import torch
import torch.nn as nn
import wandb
import joblib
 
from sklearn.metrics import mean_absolute_error, mean_squared_error
 
 
def loaders_to_numpy(train_loader, val_loader):
    X_train_list, y_train_list = [], []
    for X_batch, y_batch in train_loader:
        X_train_list.append(X_batch.numpy())
        y_train_list.append(y_batch.numpy())
 
    X_val_list, y_val_list = [], []
    for X_batch, y_batch in val_loader:
        X_val_list.append(X_batch.numpy())
        y_val_list.append(y_batch.numpy())
 
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
 
    X_val = np.vstack(X_val_list)
    y_val = np.concatenate(y_val_list)
 
    return X_train, X_val, y_train, y_val
 
 
# ======================
# Metric Functions
# ======================
 
def get_target_mean(train_loader):
    all_targets = []
    for _, y_batch in train_loader:
        all_targets.append(y_batch)
    return torch.cat(all_targets).float().mean().item()
 
 
def compute_regression_metrics(y_true, y_pred, target_mean):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    rmse_normalized = rmse / target_mean
    return {
        "mae": mae,
        "rmse": rmse,
        "rmse_normalized": rmse_normalized,
    }
 
 
# ======================
# Model Training Functions
# ======================
 
 
def train_linear_regression(model, train_loader, val_loader, config, device):
    # FIX: Use reduction='sum' so we can accurately accumulate loss across
    # variable-sized batches (the last batch is often smaller). With the
    # default reduction='mean' you would need to weight each batch by its
    # size, which the original code attempted but got subtly wrong.
    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    target_mean = get_target_mean(train_loader)
 
    history = []
    best_val_rmse = float("inf")
 
    for epoch in range(config.epochs):
        model.train()
 
        total_train_loss = 0.0
        total_train_mae = 0.0
        total_train_samples = 0
 
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.float().unsqueeze(1).to(device)
 
            optimizer.zero_grad()
 
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
 
            loss.backward()
            optimizer.step()
 
            batch_size = y_batch.size(0)
            # FIX: loss.item() is now the *sum* of squared errors for the
            # batch (not the mean), so accumulating it directly and dividing
            # by total_train_samples at the end gives the true MSE across
            # all samples regardless of batch size variation.
            total_train_loss += loss.item()
            total_train_mae += torch.abs(preds - y_batch).sum().item()
            total_train_samples += batch_size
 
        avg_train_loss = total_train_loss / total_train_samples
        train_mae = total_train_mae / total_train_samples
        train_rmse = avg_train_loss ** 0.5
 
        model.eval()
        total_val_loss = 0.0
        total_val_mae = 0.0
        total_val_samples = 0
 
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.float().unsqueeze(1).to(device)
 
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
 
                batch_size = y_batch.size(0)
                # FIX: Same correction applied to validation loss accumulation.
                total_val_loss += loss.item()
                total_val_mae += torch.abs(preds - y_batch).sum().item()
                total_val_samples += batch_size
 
        avg_val_loss = total_val_loss / total_val_samples
        val_mae = total_val_mae / total_val_samples
        val_rmse = avg_val_loss ** 0.5
        val_rmse_normalized = val_rmse / target_mean
 
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), "best_linear_model.pth")
            wandb.save("best_linear_model.pth")
 
        metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "val_loss": avg_val_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_rmse_normalized": val_rmse_normalized,
            "best_val_rmse": best_val_rmse,
        }
        history.append(metrics)
 
        print(
            f"Epoch [{epoch+1}/{config.epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | Train MAE: {train_mae:.4f} | Train RMSE: {train_rmse:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | "
            f"Val RMSE/Mean: {val_rmse_normalized:.4f} | Best Val RMSE: {best_val_rmse:.4f}"
        )
 
        wandb.log(metrics)
 
    return model, history
 
 
def train_random_forest(model, train_loader, val_loader):
    X_train, X_val, y_train, y_val = loaders_to_numpy(train_loader, val_loader)
    target_mean = float(np.mean(y_train))
 
    model.fit(X_train, y_train)
 
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
 
    train_metrics = compute_regression_metrics(y_train, train_preds, target_mean)
    val_metrics = compute_regression_metrics(y_val, val_preds, target_mean)
 
    metrics = {
        "train_mae": train_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_rmse_normalized": val_metrics["rmse_normalized"],
    }
 
    joblib.dump(model.model, "best_random_forest_model.pkl")
    wandb.save("best_random_forest_model.pkl")
 
    print(
        f"Random Forest | "
        f"Train MAE: {metrics['train_mae']:.4f} | Train RMSE: {metrics['train_rmse']:.4f} | "
        f"Val MAE: {metrics['val_mae']:.4f} | Val RMSE: {metrics['val_rmse']:.4f} | "
        f"Val RMSE/Mean: {metrics['val_rmse_normalized']:.4f}"
    )
 
    wandb.log(metrics)
    return model, metrics
 
 
def train_xgboost(model, train_loader, val_loader):
    X_train, X_val, y_train, y_val = loaders_to_numpy(train_loader, val_loader)
    target_mean = float(np.mean(y_train))
 
    model.fit(X_train, y_train)
 
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
 
    train_metrics = compute_regression_metrics(y_train, train_preds, target_mean)
    val_metrics = compute_regression_metrics(y_val, val_preds, target_mean)
 
    metrics = {
        "train_mae": train_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_rmse_normalized": val_metrics["rmse_normalized"],
    }
 
    joblib.dump(model.model, "best_xgboost_model.pkl")
    wandb.save("best_xgboost_model.pkl")
 
    print(
        f"XGBoost | "
        f"Train MAE: {metrics['train_mae']:.4f} | Train RMSE: {metrics['train_rmse']:.4f} | "
        f"Val MAE: {metrics['val_mae']:.4f} | Val RMSE: {metrics['val_rmse']:.4f} | "
        f"Val RMSE/Mean: {metrics['val_rmse_normalized']:.4f}"
    )
 
    wandb.log(metrics)
    return model, metrics