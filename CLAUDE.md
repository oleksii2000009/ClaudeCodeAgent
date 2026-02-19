# Project Guidelines for Claude

## MLflow Skill Usage

When working on machine learning tasks that involve MLflow, **always use the `mlflow` skill** before writing custom code. This skill provides standardized patterns and best practices for beginners and experts alike.

### Available Tools

The `mlflow` skill contains 6 tools, each with detailed step-by-step explanations:

| Tool | What It Does | When to Use |
|------|--------------|-------------|
| **Experiment Tracking** | Organizes experiments into projects | Starting a new ML project, comparing different approaches |
| **Parameters & Metrics** | Records settings and results | Every time you train a model |
| **Artifacts** | Saves files (plots, data, reports) | When you create visualizations or output files |
| **Model Packaging** | Prepares models for sharing/deployment | When your model is ready to use elsewhere |
| **Model Registry** | Manages model versions in production | When deploying models to production |
| **Autologging** | Automatically logs everything | Quick experiments, comparing many models with minimal code |

### Workflow

1. **Before writing MLflow code**, invoke the `mlflow` skill
2. **Read the relevant tool documentation** - each tool has beginner-friendly explanations with analogies and examples
3. **Use the code examples** as templates for your specific task
4. **Combine multiple tools** when needed (e.g., experiment tracking + artifacts + model registry)

### Example Mapping

| Task | Tools to Use |
|------|--------------|
| Train and compare models | Experiment Tracking, Parameters & Metrics |
| Quick prototyping | Autologging |
| Save confusion matrix/ROC curve | Artifacts |
| Register best model | Model Registry |
| Package model for deployment | Model Packaging |
| Full ML pipeline | All tools |

## Project Structure

```
tests/                    # Test datasets and ML experiments
.claude/skills/mlflow/    # MLflow skill with tool documentation
├── SKILL.md              # Main skill overview
└── tools/                # Individual tool guides
    ├── experiment-tracking.md
    ├── parameters-metrics.md
    ├── artifacts.md
    ├── model-packaging.md
    ├── model-registry.md
    └── autologging.md
```

## General Preferences

- Use existing skills and tools before writing custom solutions
- Track all ML experiments with MLflow
- Save all visualizations as artifacts for reproducibility
