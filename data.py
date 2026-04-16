"""
File that contains all the data loading and preprocessing logic for the project
Mostly focused on loading Excel files and creating PyTorch Datasets and DataLoaders.
"""
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ExcelDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(file_path, target_column, batch_size=32, test_size=0.2, random_state=1802, scale_features=True,):
    """
    Load numeric Excel data into PyTorch DataLoaders.

    Assumes the dataset is already cleaned, but some columns may be stored
    as object/string and need conversion to numeric types.
    """

    df = pd.read_excel(file_path)

    # Convert every column to numeric if possible
    df = df.apply(pd.to_numeric, errors="raise")

    # Split features and target
    X = df.drop(columns=[target_column]).to_numpy(dtype=np.float32)
    y = df[target_column].to_numpy(dtype=np.float32)

    # Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # Scale feature columns 
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)

    # Build datasets
    train_dataset = ExcelDataset(X_train, y_train)
    val_dataset = ExcelDataset(X_val, y_val)

    # Build dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]

    return train_loader, val_loader, input_dim