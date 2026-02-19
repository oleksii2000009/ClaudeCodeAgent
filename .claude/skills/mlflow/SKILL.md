---
name: mlflow
description: Complete MLflow toolkit for ML experiment lifecycle. Use when tracking experiments, logging parameters/metrics, saving artifacts, packaging models for deployment, or managing model versions in registry.
---

# MLflow Skill

## What is MLflow?

MLflow is a library that helps you track your machine learning experiments. Instead of losing track of which model settings worked best, MLflow saves everything automatically.

**The problem it solves:**
```
Without MLflow:
"I trained a model last week with 95% accuracy...
but I forgot what parameters I used and can't reproduce it"

With MLflow:
Everything is saved automatically - you can always go back and see exactly what you did
```

---

## Installation

```bash
pip install mlflow
```

**Verify installation:**
```bash
python -c "import mlflow; print(mlflow.__version__)"
```

---

## Capabilities Overview

| Capability | What It Does |
|------------|--------------|
| **Experiment Tracking** | Organize experiments into projects with runs |
| **Parameters & Metrics** | Record model settings and evaluation results |
| **Artifacts** | Save files (plots, data, reports) |
| **Model Packaging** | Save trained models for reuse and deployment |
| **Model Registry** | Manage model versions and production stages |
| **Autologging** | Automatically log everything with one line |

---

## When to Use Each Capability

### Experiment Tracking
**When:** At the start of any ML project or when comparing different approaches.

**Why:** Without tracking, you lose history. You train 10 models, close your notebook, and can't remember which configuration worked best. Experiment tracking saves every run automatically.

**Use it when you:**
- Start a new ML project
- Want to compare multiple models or configurations
- Need to reproduce results from days/weeks ago
- Work in a team and need shared experiment history

---

### Parameters & Metrics
**When:** Every time you train a model.

**Why:** Parameters (inputs) and metrics (outputs) are the core of reproducibility. If you can't remember what settings produced your best accuracy, you can't reproduce or improve it.

**Use it when you:**
- Train any model (always log hyperparameters)
- Evaluate model performance (log accuracy, F1, loss, etc.)
- Run hyperparameter tuning (compare metrics across runs)
- Need to find the best model later (query by metrics)

---

### Artifacts
**When:** You create any output file you want to keep.

**Why:** Plots, predictions, and reports are evidence of your model's behavior. Losing them means losing insights and audit trails.

**Use it when you:**
- Generate plots (confusion matrix, ROC curve, feature importance)
- Save predictions to CSV
- Create reports or documentation
- Want stakeholders to access visualizations

---

### Model Packaging
**When:** Your model is ready to be used outside your training script.

**Why:** A trained model is useless if you can't load it later. Packaging saves the model with its dependencies, so anyone can use it without "it works on my machine" problems.

**Use it when you:**
- Want to reuse a model without retraining
- Need to share a model with teammates
- Plan to deploy a model to production
- Want to document what inputs the model expects

---

### Model Registry
**When:** You're deploying models to production or managing multiple versions.

**Why:** Production systems need stability. The registry lets you promote models through stages (Staging → Production) and roll back instantly if something breaks.

**Use it when you:**
- Deploy models to production systems
- Need to track which model version is live
- Want to test models before deploying (Staging)
- Need to roll back to a previous version quickly

---

### Autologging
**When:** You want fast experimentation with minimal code.

**Why:** Writing logging code slows you down during prototyping. Autologging captures everything automatically so you can focus on experimentation.

**Use it when you:**
- Prototype and compare many models quickly
- Don't want to write manual logging code
- Want all parameters logged (including ones you forgot)
- Run quick experiments in notebooks

---

## Quick Start

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load sample data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Start tracking
mlflow.set_experiment("my_first_experiment")

with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, model.predict(X_test))

    # Log everything
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)

    print(f"Accuracy: {accuracy:.2%}")
```

**View results:**
```bash
mlflow ui
# Open http://127.0.0.1:5000
```

---

## Task Mapping

| Task | Capabilities to Use |
|------|---------------------|
| Train and compare models | Experiment Tracking, Parameters & Metrics |
| Quick prototyping | Autologging |
| Save confusion matrix/ROC curve | Artifacts |
| Register best model | Model Registry |
| Package model for deployment | Model Packaging |
| Full ML pipeline | All capabilities |

---

## Detailed Guides

Each capability has a comprehensive guide with step-by-step instructions:

| Guide | Description |
|-------|-------------|
| [experiment-tracking.md](experiment-tracking.md) | Creating experiments, runs, and comparing models |
| [parameters-metrics.md](parameters-metrics.md) | Logging hyperparameters and evaluation metrics |
| [artifacts.md](artifacts.md) | Saving plots, CSV files, and other outputs |
| [model-packaging.md](model-packaging.md) | Packaging models with signatures and examples |
| [model-registry.md](model-registry.md) | Version control and production deployment |
| [autologging.md](autologging.md) | Automatic logging with minimal code |

Additional quick references:
- [reference.md](reference.md) - API reference summary
- [examples.md](examples.md) - Code snippets for common tasks

---

## Understanding the Folder Structure

After running MLflow, you'll see a new folder called `mlruns/`:

```
your_project/
├── my_experiment.py       # Your script
├── mlruns/                # Created by MLflow
│   ├── 0/                 # Default experiment
│   └── 123456789/         # Your experiment (ID varies)
│       └── abc123def/     # Your run (ID varies)
│           ├── params/    # Stored parameters
│           ├── metrics/   # Stored metrics
│           ├── artifacts/ # Stored files
│           └── meta.yaml  # Run metadata
└── mlartifacts/           # Artifact storage (may appear)
```

**Important:** Don't delete `mlruns/` - that's where all your experiment data is stored!

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'mlflow'"
```bash
pip install mlflow
```

### "Address already in use" when running `mlflow ui`
```bash
mlflow ui --port 5001
```

### Can't find my experiments
Make sure you're running `mlflow ui` from the same directory where you ran your Python script.

### Experiments show in "Default" instead of my experiment name
Make sure `mlflow.set_experiment("name")` is called BEFORE `mlflow.start_run()`.
