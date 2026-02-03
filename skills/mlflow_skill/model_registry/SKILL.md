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
