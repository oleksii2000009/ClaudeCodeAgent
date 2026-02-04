# Model Registry Skill

## Skill Overview Table

| Tool/Skill | Description |
|---|---|
| Model Registry | Central repository for managing model versions, lifecycle stages, and production deployments |

## Overview

The Model Registry is MLflow's centralized hub for model lifecycle management. It allows you to register, version, stage, and track models through their lifecycle (Development → Staging → Production → Archived). The registry provides collaborative features, audit trails, and enables smooth transitions between environments. It acts as a single source of truth for production models.

## When to Use This Skill

Use model registry when:

- **Production deployments** - Need a central place to manage which models are in production
- **Model governance** - Tracking who deployed what model and when
- **Promoting models** - Moving models through stages (Dev → Staging → Production)
- **Version management** - Tracking multiple versions of the same logical model
- **Team collaboration** - Multiple data scientists need to access and use the same models
- **Rollback capability** - Ability to revert to previous model versions if issues occur
- **Audit compliance** - Creating audit trails for regulated industries
- **Model monitoring** - Tracking model performance in production
- **Automated pipelines** - CI/CD workflows that pull and deploy models programmatically

## Model Stages

- **None**: New models, not yet assigned to a stage
- **Staging**: Models under testing/evaluation
- **Production**: Models actively serving predictions
- **Archived**: Retired models, kept for historical reference

## Common Workflows

- Registering a new model version
- Transitioning models between stages
- Adding metadata and descriptions
- Comparing model performance across versions
- Viewing audit history of model transitions
- Loading production models for serving

## Python Code Example

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register a new model version
with mlflow.start_run():
    # Train and log your model
    mlflow.sklearn.log_model(model, "model", registered_model_name="MyModel")

# Transition model to different stages
client.transition_model_version_stage(
    name="MyModel",
    version=1,
    stage="Staging"  # Options: None, Staging, Production, Archived
)

# Add description and tags
client.update_model_version(
    name="MyModel",
    version=1,
    description="Random Forest classifier trained on iris dataset"
)

client.set_model_version_tag("MyModel", "1", "validation_status", "approved")

# Load a specific version or stage
model_staging = mlflow.pyfunc.load_model("models:/MyModel/Staging")
model_production = mlflow.pyfunc.load_model("models:/MyModel/Production")
model_version_2 = mlflow.pyfunc.load_model("models:/MyModel/2")

# Get model version details
versions = client.search_model_versions("name='MyModel'")
for v in versions:
    print(f"Version {v.version}: Stage={v.current_stage}, Run ID={v.run_id}")
```
