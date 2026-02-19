# MLflow API Reference

Complete documentation for all MLflow capabilities.

---

## Table of Contents

1. [Experiment Tracking](#experiment-tracking)
2. [Parameters & Metrics](#parameters--metrics)
3. [Artifacts](#artifacts)
4. [Model Packaging](#model-packaging)
5. [Model Registry](#model-registry)
6. [Autologging](#autologging)

---

## Experiment Tracking

Organizes your ML work into **experiments** and **runs**:

- **Experiment** = A project/folder (e.g., "fraud_detection")
- **Run** = One training attempt within that project

```
Experiment: "fraud_detection"
├── Run 1: Trained with RandomForest
├── Run 2: Trained with XGBoost
└── Run 3: Trained with Neural Network
```

### Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `mlflow.set_experiment("name")` | Create or select an experiment | `mlflow.set_experiment("my_project")` |
| `mlflow.start_run()` | Start tracking a run | `with mlflow.start_run():` |
| `mlflow.start_run(run_name="name")` | Start a named run | `with mlflow.start_run(run_name="v1"):` |
| `mlflow.active_run()` | Get current run info | `run_id = mlflow.active_run().info.run_id` |
| `mlflow.end_run()` | End the current run | Usually automatic with `with` |
| `MlflowClient().search_runs()` | Query past runs | See examples |

### Common Errors

**"Experiment does not exist"**
```python
# Wrong: experiment must exist or be created first
with mlflow.start_run(experiment_id="nonexistent"):  # Error!

# Right: use set_experiment to create if needed
mlflow.set_experiment("my_experiment")
with mlflow.start_run():
    pass
```

**"Run already active"**
```python
# Wrong: nested runs without ending the first
mlflow.start_run()
mlflow.start_run()  # Error!

# Right: use context manager or end run
with mlflow.start_run():
    pass  # Automatically ends
```

---

## Parameters & Metrics

Two types of information:

1. **Parameters** = Settings you choose BEFORE training (inputs)
2. **Metrics** = Results you measure AFTER training (outputs)

```
PARAMETERS (your choices):          METRICS (results):
├── learning_rate: 0.01            ├── accuracy: 0.95
├── n_estimators: 100              ├── precision: 0.93
└── max_depth: 5                   └── loss: 0.12
```

### Parameter Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `mlflow.log_param(key, value)` | Log one parameter | `mlflow.log_param("lr", 0.01)` |
| `mlflow.log_params(dict)` | Log multiple parameters | `mlflow.log_params({"lr": 0.01, "epochs": 100})` |

### Metric Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `mlflow.log_metric(key, value)` | Log one metric | `mlflow.log_metric("accuracy", 0.95)` |
| `mlflow.log_metric(key, value, step)` | Log metric with step | `mlflow.log_metric("loss", 0.5, step=10)` |
| `mlflow.log_metrics(dict)` | Log multiple metrics | `mlflow.log_metrics({"acc": 0.95, "loss": 0.1})` |

### Common Mistakes

**Logging params outside of a run**
```python
# WRONG - no active run
mlflow.log_param("lr", 0.01)  # Error!

# RIGHT - inside a run
with mlflow.start_run():
    mlflow.log_param("lr", 0.01)  # OK
```

**Logging the same parameter twice**
```python
# WRONG - can't update params
with mlflow.start_run():
    mlflow.log_param("lr", 0.01)
    mlflow.log_param("lr", 0.02)  # Error! Already logged
```

---

## Artifacts

**Files** you want to save with your experiment:
- Plots (confusion matrix, ROC curve, etc.)
- CSV files (predictions, data samples)
- Text files (reports, logs)
- Any other file

### Artifact Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `mlflow.log_artifact(path)` | Log a single file | `mlflow.log_artifact("plot.png")` |
| `mlflow.log_artifact(path, artifact_path)` | Log file to subfolder | `mlflow.log_artifact("plot.png", "plots")` |
| `mlflow.log_artifacts(dir)` | Log entire folder | `mlflow.log_artifacts("output/")` |
| `mlflow.log_figure(fig, name)` | Log matplotlib figure directly | `mlflow.log_figure(fig, "plot.png")` |
| `mlflow.log_text(text, name)` | Log text directly | `mlflow.log_text("Hello", "note.txt")` |
| `mlflow.log_dict(dict, name)` | Log dict as JSON | `mlflow.log_dict(config, "config.json")` |
| `mlflow.artifacts.download_artifacts()` | Download artifacts | See examples |

### Common Mistakes

**Forgetting to close plots**
```python
# BAD - memory leak
plt.figure()
plt.plot([1,2,3])
plt.savefig("plot.png")
mlflow.log_artifact("plot.png")
# Plot still in memory!

# GOOD - close when done
plt.savefig("plot.png")
mlflow.log_artifact("plot.png")
plt.close()  # Free memory
```

---

## Model Packaging

Saves your trained model so you can:
1. Use it later without retraining
2. Share it with others
3. Deploy it to production

When you package a model, MLflow saves:
- The model itself
- All dependencies (what packages it needs)
- Information about expected inputs/outputs

### Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `mlflow.sklearn.log_model()` | Save sklearn model | `mlflow.sklearn.log_model(model, "model")` |
| `mlflow.sklearn.load_model()` | Load sklearn model | `mlflow.sklearn.load_model("runs:/{id}/model")` |
| `mlflow.pyfunc.load_model()` | Load any model type | `mlflow.pyfunc.load_model("runs:/{id}/model")` |
| `infer_signature()` | Create signature | `infer_signature(X, predictions)` |

### Model URI Formats

| Format | Meaning | Example |
|--------|---------|---------|
| `runs:/run_id/model` | Model from a run | `runs:/abc123/model` |
| `models:/Name/Production` | Current production model | `models:/MyModel/Production` |
| `models:/Name/1` | Specific version | `models:/MyModel/1` |

### Framework-Specific Logging

```python
# Scikit-learn
mlflow.sklearn.log_model(model, "model")

# XGBoost
mlflow.xgboost.log_model(model, "model")

# LightGBM
mlflow.lightgbm.log_model(model, "model")

# TensorFlow/Keras
mlflow.tensorflow.log_model(model, "model")

# PyTorch
mlflow.pytorch.log_model(model, "model")
```

---

## Model Registry

A **catalog** of your production-ready models. Helps you:
1. Keep track of which model version is in production
2. Move models through stages (Testing → Staging → Production)
3. Roll back to previous versions if needed

### Stages

| Stage | Meaning | Use Case |
|-------|---------|----------|
| **None** | Just registered | New model, not evaluated yet |
| **Staging** | Being tested | Model under evaluation |
| **Production** | Live | Model serving real users |
| **Archived** | Retired | Old version, kept for history |

### Key Functions

| Function | Purpose |
|----------|---------|
| `log_model(..., registered_model_name="X")` | Register while logging |
| `client.transition_model_version_stage()` | Change model stage |
| `client.update_model_version()` | Add description |
| `client.set_model_version_tag()` | Add tags |
| `client.search_model_versions()` | List all versions |
| `client.get_latest_versions()` | Get by stage |
| `mlflow.pyfunc.load_model("models:/X/Production")` | Load production model |

### Loading by Stage

```python
# Load whatever is currently in Production
model = mlflow.pyfunc.load_model("models:/IrisClassifier/Production")

# Load Staging model for testing
staging_model = mlflow.pyfunc.load_model("models:/IrisClassifier/Staging")

# Load specific version
model_v1 = mlflow.pyfunc.load_model("models:/IrisClassifier/1")
```

---

## Autologging

Automatically tracks your ML experiments **without writing any logging code**.

```python
# WITHOUT Autologging (manual):
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 10)
mlflow.log_metric("accuracy", 0.95)
mlflow.sklearn.log_model(model, "model")

# WITH Autologging (automatic):
mlflow.autolog()  # Just this one line!
```

### Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `mlflow.autolog()` | Enable autologging for all frameworks | `mlflow.autolog()` |
| `mlflow.sklearn.autolog()` | Enable for sklearn only | `mlflow.sklearn.autolog()` |
| `mlflow.autolog(disable=True)` | Disable autologging | `mlflow.autolog(disable=True)` |
| `mlflow.autolog(log_models=False)` | Don't save models | `mlflow.autolog(log_models=False)` |

### Configuration Options

```python
mlflow.autolog(
    log_models=True,              # Save the model (default: True)
    log_input_examples=False,     # Don't save input examples (default: False)
    log_model_signatures=True,    # Save model signature (default: True)
    log_datasets=False,           # Don't log dataset info (default: True)
    disable=False,                # Enable autologging (default: False)
    exclusive=False,              # Allow manual logging too (default: False)
    silent=False                  # Show autolog messages (default: False)
)
```

### Supported Frameworks

| Framework | Autolog Function |
|-----------|------------------|
| Scikit-learn | `mlflow.sklearn.autolog()` |
| XGBoost | `mlflow.xgboost.autolog()` |
| LightGBM | `mlflow.lightgbm.autolog()` |
| TensorFlow/Keras | `mlflow.tensorflow.autolog()` |
| PyTorch Lightning | `mlflow.pytorch.autolog()` |
| Spark | `mlflow.spark.autolog()` |
| Fastai | `mlflow.fastai.autolog()` |
| Statsmodels | `mlflow.statsmodels.autolog()` |

### When to Use Autolog vs Manual Logging

| Use Autolog When... | Use Manual Logging When... |
|---------------------|---------------------------|
| You want quick experiment tracking | You need custom metrics |
| You're comparing many models | You want specific artifact format |
| You're prototyping | You need precise control |
| You want all framework parameters | You only need specific params |

**Best practice:** Use both! Enable autolog for automatic tracking, then add manual logging for custom needs.
