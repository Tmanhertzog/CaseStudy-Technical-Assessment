"""
File that contains all the model definitions for the project. This includes:
- LogisticRegression: A simple binary logistic regression implemented in PyTorch.
- RandomForestModel: A wrapper around sklearn's RandomForestClassifier.
- XGBoostModel: A wrapper around xgboost's XGBClassifier.
"""

import torch
import torch.nn as nn

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor



class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)



class RandomForestModel:
    def __init__(
        self,
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=1802
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class XGBoostModel:
    def __init__(
        self,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=1802,
        gamma=1.0,
        min_child_weight=5,
    ):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            objective="reg:squarederror",
            gamma=gamma,
            min_child_weight=min_child_weight
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
