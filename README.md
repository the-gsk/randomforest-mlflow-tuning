# Random Forest Hyperparameter Tuning with MLflow

## Overview
This repository contains a **Random Forest Classifier** hyperparameter tuning experiment using **GridSearchCV** with **MLflow** for tracking. The goal is to optimize the model and log the best hyperparameters, accuracy scores, and trained models for future reference.

## Features
- **Hyperparameter Tuning** using `GridSearchCV`
- **Experiment Tracking** with `MLflow`
- **Best Model Logging** for reproducibility
- **Performance Evaluation** on a test dataset
- **Visualization in MLflow UI**

## Requirements
Install the required libraries before running the project:
```bash
pip install -r requirements.txt
```

## Getting Started
### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/randomforest-mlflow-tuning.git
cd randomforest-mlflow-tuning
```

### 2. Run Hyperparameter Tuning
```bash
python hyperparameter_tuning.py
```

### 3. Start MLflow UI to View Results
```bash
mlflow ui
```
Then, open **http://127.0.0.1:5000** in your browser to explore experiment logs.

## Directory Structure
```
randomforest-mlflow-tuning/
│── hyperparameter_tuning.py  # Main script
│── requirements.txt          # Dependencies
│── README.md                 # Project documentation
```

## Results & Screenshots
Include screenshots of **MLflow UI**, showing:
✅ Hyperparameters tested  
✅ Accuracy scores  
✅ Logged best model  

## License
This project is open-source and available under the MIT License.

