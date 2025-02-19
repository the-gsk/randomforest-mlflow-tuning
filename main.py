import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Set MLflow experiment
mlflow.set_experiment("RandomForest_Hyperparameter_Tuning")

with mlflow.start_run():
    # Log parameter grid
    mlflow.log_params(param_grid)
    
    # Initialize RandomForest and perform GridSearchCV
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    # Get best parameters and best accuracy
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    
    # Log best parameters and accuracy
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_accuracy", best_accuracy)
    
    # Get best model and log it
    best_model = grid_search.best_estimator_
    mlflow.sklearn.log_model(best_model, "random_forest_model")
    
    print("Best Hyperparameters:", best_params)
    print("Best Cross-Validated Accuracy:", best_accuracy)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", test_accuracy)
    
    print("Test Accuracy:", test_accuracy)
