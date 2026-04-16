# Model Cost Assessment Project

## Overview

This project explores multiple machine learning models to predict a target variable from structured Excel data. It includes data preprocessing, training, hyperparameter tuning, and experiment tracking.

The goal is to simulate a realistic ML workflow with multiple modeling approaches and evaluation strategies.

---

## Idea Behind Design

I wanted to be able to find high risk, high cost patients from the data given and split patients into groups of high-risk + high-cost, high-risk + low-cost, low-risk + high-cost, and low-risk + low-cost.

Once patients are categorized, you could find which patients could require attention and the most care. I would also use an exisitng LLM to what find improvements could be made for them or what their sentiment was of the treatment they recieved based on their reviews.

This design would allow you to determine what were the key drivers for risk, the key drivers for amount needed for payment, as well as ways to improve help for patients.

Afterwards I wanted to create a dashboard to view statistics, visuals, and a patient table list prioritizing those who are most at risk.


## Data Pipeline

- Loads Excel data using pandas  
- Converts all columns to numeric  
- Splits into train/validation sets   
- Wraps data in PyTorch Dataset and DataLoader  

---

## Models

The project implements:

- Linear Regression (PyTorch)
- Random Forest Regressor (sklearn)
- XGBoost Regressor

Each model supports training and prediction.

My reason for choosing regressor models over other types, was the tabular data strucutre and the small amount of data.

Since models like XGBoost and decision trees work best on tabular data I decided to develop my models using those algorithms.

---

## Training

- PyTorch training loop for linear regression  
- NumPy conversion for tree-based models  
- Metrics used:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - Normalized RMSE  

Models are saved after training:
- `.pkl` for tree-based models  

---

## Hyperparameter Tuning

Includes manual grid-style tuning for:

- Random Forest
- XGBoost

Tracks validation performance and selects best configuration.

---

## Experiment Tracking

Uses Weights & Biases (wandb) for:

- Logging metrics
- Tracking experiments
- Comparing model runs

---

## How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Login to wandb

wandb login

### 3. Run the project

python run.py

---

## Configuration

Modify settings in `run.py`:

- Model type (architecture)
- Learning rate
- Epochs
- Batch size
- Tree parameters (n_estimators, max_depth)

---

## Output

- Training and validation metrics printed to console  
- Metrics logged to wandb  
- Best model saved locally  

---

## Limitations

- Size of data 
- Basic hyperparameter tuning  
- Performance depends heavily on dataset quality
- Time restrictions
- Spent too much time on trying to improve model performance

---

## Future Improvements

- Improve model performance
- Look to reformat data and implement temporal models such as an RNN
- 

---

## Summary

This project demonstrates a full ML pipeline including:

- Data preprocessing  
- Model training  
- Hyperparameter tuning  
- Experiment tracking  

It is designed to be easily extendable and adaptable for real-world ML workflows.
