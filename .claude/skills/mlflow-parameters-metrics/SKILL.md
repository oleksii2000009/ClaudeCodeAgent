---
name: mlflow-parameters-metrics
description: Logs hyperparameters and performance metrics for ML runs. Use when recording model hyperparameters, logging training metrics like accuracy or loss, tracking model performance, or documenting experiment configurations.
---

# Parameter and Metrics Logging Skill

## Skill Overview Table

| Tool/Skill | Description |
|---|---|
| Parameter and Metrics Logging | Logs hyperparameters and performance metrics during model training for tracking and comparison |

## Overview

Parameter and Metrics Logging is the fundamental practice of recording training configurations and model performance data. Parameters are the hyperparameters you set before training (learning rate, batch size, etc.), while metrics are the performance measurements computed during training (accuracy, loss, F1-score, etc.). This skill ensures all experiments are properly documented and comparable.

## When to Use This Skill

Use parameter and metrics logging when:

- **Starting any model training** - Every run should log its configuration
- **Tuning hyperparameters** - Need to track which parameter values produced which results
- **Evaluating model performance** - Recording accuracy, loss, precision, recall, etc.
- **Tracking training progress** - Logging metrics at different epochs or iterations
- **Benchmarking models** - Comparing performance across different algorithms or datasets
- **Auditing ML pipelines** - Creating an audit trail of what was trained and how well it performed
- **Automated workflows** - CI/CD pipelines that need structured logging
- **Cross-validation** - Logging metrics for each fold or split

## Common Parameters to Log

- Learning rate, batch size, epochs
- Model architecture details (hidden layers, dropout, etc.)
- Data preprocessing parameters
- Random seeds for reproducibility

## Common Metrics to Log

- Training loss and validation loss
- Accuracy, precision, recall, F1-score
- ROC-AUC, confusion matrix
- Custom domain-specific metrics
- Training time and resource usage

## Python Code Example

```python
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Prepare data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run():
    # Train model with specific hyperparameters
    model = LogisticRegression(C=1.0, max_iter=100, solver='lbfgs')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Log parameters (hyperparameters used)
    mlflow.log_param("C", 1.0)
    mlflow.log_param("max_iter", 100)
    mlflow.log_param("solver", "lbfgs")

    # Log metrics (performance results)
    mlflow.log_metric("accuracy", accuracy_score(y_test, predictions))
    mlflow.log_metric("precision", precision_score(y_test, predictions))
    mlflow.log_metric("f1_score", f1_score(y_test, predictions))

    # Log multiple params/metrics at once
    mlflow.log_params({"random_state": 42, "test_size": 0.2})
    mlflow.log_metrics({"train_samples": len(X_train), "test_samples": len(X_test)})
```
