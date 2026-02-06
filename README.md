# MLflow Skills - Project Overview

## Overview

This project contains a collection of MLflow skills designed to help data scientists and ML engineers efficiently manage their machine learning workflows. Each skill focuses on a specific aspect of MLflow functionality, providing:

- **Clear documentation** of when and how to use each skill
- **Concise Python code examples** demonstrating core functionality
- **Practical guidance** for real-world ML operations

All skills have been structured following a consistent format with:
- Skill overview and description
- Use cases and scenarios
- Key concepts and terminology
- Python code examples with best practices

## Folder Structure

```
work/
├── README.md                           # This file - detailed overview and skill reference
├── CLAUDE.md                           # Project instructions for Claude
├── .claude/
│   └── skills/
│       ├── mlflow-experiment-tracking/
│       │   └── SKILL.md                # Experiment tracking skill documentation
│       ├── mlflow-parameters-metrics/
│       │   └── SKILL.md                # Parameter and metrics logging documentation
│       ├── mlflow-model-packaging/
│       │   └── SKILL.md                # Model packaging skill documentation
│       ├── mlflow-model-registry/
│       │   └── SKILL.md                # Model registry skill documentation
│       └── mlflow-artifacts/
│           └── SKILL.md                # Artifacts management skill documentation
└── tests/
    ├── test1/                          # Asthma prediction pipeline
    │   ├── asthma_prediction_pipeline.py
    │   └── synthetic_asthma_dataset.csv
    ├── test2/                          # Historical popularity ML
    │   ├── historical_popularity_ml.py
    │   └── database.csv
    ├── test3/                          # Parkinson's disease prediction
    │   ├── parkinsons_prediction.py
    │   ├── parkinsons_prediction_enhanced.py
    │   └── Parkinsons-Telemonitoring-ucirvine.csv
    └── test4/                          # Penguin classifier
        ├── penguin_classifier.py
        └── penguins_size.csv
```

## Skills Reference Table

| Skill Name | Description | Documentation |
|------------|-------------|---------------|
| **Experiment Tracking** | Records, manages, and compares multiple ML experiment runs with their parameters, metrics, and artifacts | [SKILL.md](.claude/skills/mlflow-experiment-tracking/SKILL.md) |
| **Parameter and Metrics Logging** | Logs hyperparameters and performance metrics during model training for tracking and comparison | [SKILL.md](.claude/skills/mlflow-parameters-metrics/SKILL.md) |
| **Model Packaging** | Packages trained ML models in a standardized MLflow format for portability and deployment | [SKILL.md](.claude/skills/mlflow-model-packaging/SKILL.md) |
| **Model Registry** | Central repository for managing model versions, lifecycle stages, and production deployments | [SKILL.md](.claude/skills/mlflow-model-registry/SKILL.md) |
| **Artifacts Management** | Stores and retrieves training outputs like models, plots, data files, and other experiment artifacts | [SKILL.md](.claude/skills/mlflow-artifacts/SKILL.md) |

## Quick Start

Each skill includes a Python code example showing the essential usage pattern. To get started:

1. Choose the skill relevant to your current task from the table above
2. Review the "When to Use This Skill" section to confirm it matches your needs
3. Follow the Python code example to implement the functionality
4. Refer to the "Key Concepts" section for deeper understanding

## Skill Categories

### Tracking & Logging
- **Experiment Tracking**: Organize and compare multiple training runs
- **Parameter and Metrics Logging**: Record configuration and performance data

### Model Management
- **Model Packaging**: Prepare models for deployment
- **Model Registry**: Manage model lifecycle and versions

### Data & Artifacts
- **Artifacts Management**: Store and retrieve experiment outputs

## Prerequisites

To use these skills, you need:
- Python 3.8 or higher
- MLflow installed (`pip install mlflow`)
- scikit-learn (for examples): `pip install scikit-learn`
- Additional dependencies based on your specific use case (PyTorch, TensorFlow, etc.)

## Installation

```bash
# Install MLflow
pip install mlflow

# Install common dependencies
pip install scikit-learn numpy pandas matplotlib

# Optional: For deep learning examples
pip install torch tensorflow
```

## Running MLflow UI

To visualize your experiments, start the MLflow UI:

```bash
mlflow ui
```

Then open your browser to `http://localhost:5000`

## Contributing

Each skill follows a consistent documentation structure:
1. Skill Overview Table
2. Overview description
3. When to Use This Skill
4. Key Concepts
5. Common Operations/Workflows
6. Python Code Example

When adding new skills, please follow this format for consistency.

## Next Steps

- Explore each skill's detailed documentation
- Try the code examples with your own datasets
- Combine multiple skills in your ML workflows
- Set up MLflow tracking server for team collaboration

## Resources

- [MLflow Official Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow GitHub Repository](https://github.com/mlflow/mlflow)
- [MLflow Tutorials](https://mlflow.org/docs/latest/tutorials-and-examples/index.html)
