# Project Guidelines for Claude

## MLflow Skills Usage

When working on machine learning tasks that involve MLflow, **always use the available MLflow skills** before writing custom code. These skills provide standardized patterns and best practices.

### Available Skills

| Skill | When to Use |
|-------|-------------|
| `mlflow-experiment-tracking` | Tracking ML experiments, comparing runs, hyperparameter tuning |
| `mlflow-parameters-metrics` | Logging hyperparameters and performance metrics (accuracy, F1, loss, etc.) |
| `mlflow-artifacts` | Saving visualizations (confusion matrices, ROC curves), data files, reports |
| `mlflow-model-packaging` | Packaging models for deployment with signatures and dependencies |
| `mlflow-model-registry` | Registering models, managing versions, promoting to staging/production |

### Workflow

1. **Before writing MLflow code**, invoke the relevant skill to get the recommended patterns
2. **Use skill examples** as templates for the specific task
3. **Combine multiple skills** when the task spans different MLflow features (e.g., experiment tracking + artifacts + model registry)

### Example Mapping

| Task | Skills to Use |
|------|---------------|
| Train and compare models | `mlflow-experiment-tracking`, `mlflow-parameters-metrics` |
| Save confusion matrix/ROC curve | `mlflow-artifacts` |
| Register best model | `mlflow-model-registry` |
| Package model for deployment | `mlflow-model-packaging` |
| Full ML pipeline | All relevant skills |

## Project Structure

```
tests/           # Test datasets and ML experiments
.claude/skills/  # MLflow skill definitions
```

## General Preferences

- Use existing skills and tools before writing custom solutions
- Track all ML experiments with MLflow
- Save all visualizations as artifacts for reproducibility
