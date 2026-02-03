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
