# Experiment Tracking Skill

## Skill Overview Table

| Tool/Skill | Description |
|---|---|
| Experiment Tracking | Records, manages, and compares multiple ML experiment runs with their parameters, metrics, and artifacts |

## Overview

Experiment Tracking is the core feature of MLflow that allows you to log and monitor machine learning experiments. It creates a comprehensive record of each training run, including hyperparameters, performance metrics, and generated artifacts. This skill enables reproducibility, comparison of different model approaches, and systematic hyperparameter tuning.

## When to Use This Skill

Use experiment tracking when:

- **Running multiple model training iterations** - You need to compare different hyperparameters or algorithms
- **Hyperparameter tuning** - Testing various parameter combinations and tracking their performance
- **Debugging models** - Need to understand which parameters led to good/poor performance
- **Reproducibility** - Must document exactly what parameters and code produced a specific result
- **Team collaboration** - Multiple team members need to compare and share experiment results
- **Model selection** - Deciding between different models or approaches based on metrics
- **A/B testing** - Comparing model variants in a structured way
- **Tracking training history** - Logging metrics over time during long training runs

## Key Concepts

- **Experiment**: A logical grouping of related runs (e.g., "XGBoost Hyperparameter Tuning")
- **Run**: A single execution of model training with specific parameters
- **MLflow UI**: Web interface to view, search, and compare experiments and runs

## Common Operations

- Creating and naming experiments
- Starting and ending runs
- Recording parameters and metrics
- Accessing run history and comparing performance
- Using the MLflow Tracking Server

## Python Code Example

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set experiment name
mlflow.set_experiment("iris_classification")

# Load and prepare data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Start tracking run
with mlflow.start_run(run_name="random_forest_v1"):
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test)))

    # Log model
    mlflow.sklearn.log_model(model, "model")

    print(f"Run ID: {mlflow.active_run().info.run_id}")
```
