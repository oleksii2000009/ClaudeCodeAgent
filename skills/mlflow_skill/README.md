# MLflow Overview

## What is MLflow?

MLflow is an open-source platform designed to manage the end-to-end machine learning lifecycle. It provides a set of tools and APIs to help data scientists and ML engineers track experiments, package models, manage versions, and deploy them to production. MLflow is library-agnostic and works with any ML framework (scikit-learn, TensorFlow, PyTorch, XGBoost, etc.).

## Core Components

### 1. **Tracking**
- Records and compares parameters, metrics, and artifacts from ML experiments
- Logs training runs with their associated data
- Creates an experiment history for reproducibility and comparison
- Stores metadata like duration, source code, and custom tags

### 2. **Projects**
- Packages ML code in a reproducible format
- Uses a `MLproject` file to define entry points and dependencies
- Ensures consistent execution across different environments
- Facilitates collaboration and code sharing

### 3. **Models**
- Provides a standard format for packaging ML models
- Supports multiple "flavors" (scikit-learn, TensorFlow, PyTorch, etc.)
- Includes model metadata and dependencies
- Enables easy serialization and deserialization

### 4. **Registry**
- Central repository for model versions and lifecycle management
- Tracks model stages (Staging, Production, Archived)
- Manages model transitions between stages
- Provides collaboration and audit trails

## Key Concepts

- **Experiment**: A collection of related runs
- **Run**: A single execution of ML code with logged parameters, metrics, and artifacts
- **Parameter**: A configuration value (hyperparameter) used in the run
- **Metric**: A measurement of model performance (accuracy, loss, etc.)
- **Artifact**: Output files from a run (models, plots, data files, etc.)
- **Model**: A packaged ML model with metadata, signature, and dependencies
- **Stage**: A model's lifecycle state (None, Staging, Production, Archived)

## Basic Workflow

1. **Log Experiments**: Use MLflow Tracking to log parameters, metrics, and artifacts during model training
2. **Compare Runs**: View and compare different experiment runs in the UI
3. **Package Model**: Save trained models using MLflow Models format
4. **Register Model**: Register models in the Model Registry for centralized management
5. **Deploy**: Load and serve models from the registry in production

## Common Use Cases

- Hyperparameter tuning and experiment comparison
- Model versioning and reproducibility
- Collaborative ML development
- Production model deployment and monitoring
- Automated retraining pipelines

## Key APIs

- `mlflow.start_run()`: Initialize a new run
- `mlflow.log_param()`: Log a parameter
- `mlflow.log_metric()`: Log a metric
- `mlflow.log_artifact()`: Log a file
- `mlflow.sklearn.log_model()`: Log a scikit-learn model
- `mlflow.models.load_model()`: Load a model
- `mlflow.register_model()`: Register a model in the registry
