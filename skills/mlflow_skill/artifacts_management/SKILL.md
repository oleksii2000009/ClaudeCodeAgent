# Artifacts Management Skill

## Skill Overview Table

| Tool/Skill | Description |
|---|---|
| Artifacts Management | Stores and retrieves training outputs like models, plots, data files, and other experiment artifacts |

## Overview

Artifacts Management enables you to store and organize output files from your ML experiments beyond just the model. This includes visualizations, performance plots, preprocessed data, feature importance charts, training logs, and any other files relevant to the experiment. MLflow's artifact storage is flexible, supporting local storage or cloud services (S3, Azure Blob, GCS).

## When to Use This Skill

Use artifacts management when:

- **Saving plots and visualizations** - Store confusion matrices, ROC curves, feature importance plots
- **Logging training outputs** - Save detailed training logs, error analysis, or debugging information
- **Storing preprocessed data** - Archive feature-engineered datasets used in experiments
- **Saving model interpretability data** - SHAP values, LIME explanations, feature importance
- **Storing experiment reports** - HTML reports, Jupyter notebooks, or analysis results
- **Archiving raw outputs** - Any file that provides context about the experiment
- **Reproducibility** - Keeping all experiment outputs together for later review
- **Data versioning** - Tracking which data version was used for training
- **Cloud storage** - Integrating with S3, GCS, or Azure for scalable artifact storage

## Common Artifact Types

- Trained model files (pickle, joblib, etc.)
- Visualizations (PNG, SVG plots)
- CSV/JSON data files
- Training logs and reports
- Configuration files
- Dependency requirements (requirements.txt)
- Documentation files
- Performance metrics plots

## Storage Options

- Local file system
- AWS S3
- Google Cloud Storage (GCS)
- Azure Blob Storage
- HDFS
- HTTP-based storage

## Python Code Example

```python
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

with mlflow.start_run():
    # Log a single file
    with open("config.txt", "w") as f:
        f.write("batch_size=32\nepochs=100")
    mlflow.log_artifact("config.txt")

    # Log a plot/figure
    plt.figure()
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.savefig("training_plot.png")
    mlflow.log_artifact("training_plot.png")

    # Log a DataFrame as CSV
    df = pd.DataFrame({"accuracy": [0.85, 0.89, 0.92], "loss": [0.45, 0.32, 0.21]})
    df.to_csv("metrics.csv", index=False)
    mlflow.log_artifact("metrics.csv")

    # Log an entire directory
    mlflow.log_artifacts("output_folder/", artifact_path="results")

    # Log a dictionary as JSON
    config = {"learning_rate": 0.001, "batch_size": 32}
    mlflow.log_dict(config, "config.json")

    # Log text directly
    mlflow.log_text("Model training completed successfully", "status.txt")

# Download artifacts later
run_id = mlflow.active_run().info.run_id
local_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/training_plot.png")
print(f"Downloaded to: {local_path}")
```
