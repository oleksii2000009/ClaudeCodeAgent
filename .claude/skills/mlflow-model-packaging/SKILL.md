---
name: mlflow-model-packaging
description: Packages ML models in MLflow format for deployment. Use when preparing models for production deployment, creating portable model artifacts, adding model signatures, or packaging models with dependencies.
---

# Model Packaging Skill

## Skill Overview Table

| Tool/Skill | Description |
|---|---|
| Model Packaging | Packages trained ML models in a standardized MLflow format for portability and deployment |

## Overview

Model Packaging converts your trained models into a standard MLflow Models format that includes the model artifacts, metadata, and dependencies. This standardized format allows models to be easily shared, deployed to different environments, and served through various MLflow tools. It supports multiple ML frameworks (scikit-learn, TensorFlow, PyTorch, XGBoost, etc.).

## When to Use This Skill

Use model packaging when:

- **Preparing for deployment** - Need to move models from development to production
- **Sharing models with team members** - Ensuring consistent model format and dependencies
- **Model serving** - Planning to serve the model via REST API or batch predictions
- **Model archival** - Storing models in a reproducible, portable format
- **Framework agnostic deployment** - Need to deploy across different systems or frameworks
- **Documenting model requirements** - Recording Python dependencies and environment details
- **Production readiness** - Creating models that can be easily loaded and used elsewhere
- **Version control** - Storing multiple versions of the same model in a consistent format

## Key Components of Packaged Models

- **Model artifacts** - The actual trained model file(s)
- **Model signature** - Input/output schema for the model
- **Environment** - Python dependencies (Conda or pip)
- **Flavor** - Framework-specific format (sklearn, tensorflow, pytorch, etc.)
- **Metadata** - Creation date, description, custom tags

## Common Packaging Scenarios

- Packaging scikit-learn models
- Packaging deep learning models (TensorFlow, PyTorch)
- Packaging custom models with custom flavors
- Adding model metadata and signatures
- Creating reproducible deployment packages

## Python Code Example

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from mlflow.models.signature import infer_signature

# Train a model
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Create model signature (defines input/output schema)
predictions = model.predict(X[:5])
signature = infer_signature(X[:5], predictions)

# Package and save model
with mlflow.start_run():
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name="iris_classifier"
    )

# Load packaged model later
run_id = mlflow.active_run().info.run_id
loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
```
