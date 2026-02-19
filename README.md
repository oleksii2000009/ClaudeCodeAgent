# MLflow Skills for Claude Code

A Claude Code skill that provides comprehensive MLflow guidance for machine learning experiment tracking, model management, and deployment.

## What This Project Does

This project contains a custom skill for Claude Code that teaches Claude how to help you with MLflow. When you invoke the skill, Claude gains access to detailed documentation about MLflow's capabilities and can write proper MLflow code for your ML projects.

## Project Structure

```
work/
├── README.md                          # This file
├── CLAUDE.md                          # Instructions for Claude
└── .claude/skills/mlflow/             # MLflow skill
    ├── SKILL.md                       # Skill overview and quick start
    ├── experiment-tracking.md         # Creating experiments and runs
    ├── parameters-metrics.md          # Logging params and metrics
    ├── artifacts.md                   # Saving files and plots
    ├── model-packaging.md             # Packaging models for deployment
    ├── model-registry.md              # Version control and staging
    ├── autologging.md                 # Automatic logging setup
    ├── reference.md                   # API reference summary
    └── examples.md                    # Code snippets for common tasks
```

## MLflow Capabilities

| Capability | Description |
|------------|-------------|
| **Experiment Tracking** | Organize ML experiments into projects with runs |
| **Parameters & Metrics** | Record model settings and evaluation results |
| **Artifacts** | Save files (plots, data, reports) |
| **Model Packaging** | Save trained models for reuse and deployment |
| **Model Registry** | Manage model versions and production stages |
| **Autologging** | Automatically log everything with one line |

## Prerequisites

```bash
pip install mlflow
```

## How to Use

Simply ask Claude to help with MLflow tasks:

```
"Help me track my model training with MLflow"
"Log parameters and metrics for this experiment"
"Save my confusion matrix as an artifact"
"Register my model to production"
```

Or invoke the skill directly with `/mlflow`.

## Viewing Results

After logging experiments, view them in the MLflow UI:

```bash
mlflow ui
```

Open http://localhost:5000 in your browser.

## Resources

- [MLflow Official Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow GitHub Repository](https://github.com/mlflow/mlflow)
