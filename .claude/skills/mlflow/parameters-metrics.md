# Parameters & Metrics Logging - Detailed Guide

## What This Capability Does

This tool lets you save two types of information:

1. **Parameters** = Settings you choose BEFORE training (inputs)
2. **Metrics** = Results you measure AFTER training (outputs)

```
PARAMETERS (your choices):          METRICS (results):
├── learning_rate: 0.01            ├── accuracy: 0.95
├── n_estimators: 100              ├── precision: 0.93
└── max_depth: 5                   └── loss: 0.12
```

---

## Key Functions - Parameters

### `mlflow.log_param(key, value)`
Logs a single parameter. Parameters are immutable - you cannot overwrite them in the same run.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | str | Parameter name |
| `value` | any | Parameter value (converted to string) |

```python
mlflow.log_param("n_estimators", 100)
mlflow.log_param("model_type", "RandomForest")
```

---

### `mlflow.log_params(params)`
Logs multiple parameters at once from a dictionary.

| Parameter | Type | Description |
|-----------|------|-------------|
| `params` | dict | Dictionary of parameter names and values |

```python
mlflow.log_params({
    "n_estimators": 100,
    "max_depth": 10,
    "learning_rate": 0.01
})
```

---

## Key Functions - Metrics

### `mlflow.log_metric(key, value, step=None)`
Logs a single metric. Unlike parameters, metrics can be logged multiple times (for tracking over epochs).

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | str | Metric name |
| `value` | float | Metric value |
| `step` | int (optional) | Step number (for training curves) |

```python
mlflow.log_metric("accuracy", 0.95)
mlflow.log_metric("loss", 0.5, step=10)  # At epoch 10
```

---

### `mlflow.log_metrics(metrics, step=None)`
Logs multiple metrics at once from a dictionary.

| Parameter | Type | Description |
|-----------|------|-------------|
| `metrics` | dict | Dictionary of metric names and values |
| `step` | int (optional) | Step number for all metrics |

```python
mlflow.log_metrics({
    "accuracy": 0.95,
    "precision": 0.93,
    "recall": 0.91,
    "f1_score": 0.92
})
```

---

## Combining Functions

**Typical training workflow:**
```python
with mlflow.start_run():
    # 1. Log parameters BEFORE training
    mlflow.log_params({"n_estimators": 100, "max_depth": 10})

    # 2. Train model
    model.fit(X_train, y_train)

    # 3. Log metrics AFTER training
    mlflow.log_metrics({"accuracy": acc, "f1": f1})
```

**Training curves (loss over epochs):**
```python
for epoch in range(100):
    loss = train_one_epoch()
    mlflow.log_metric("loss", loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
```

---

## Library Compatibility

Parameters and metrics work with **any ML library** - just extract the values you want to log:

| Library | Common Parameters | Common Metrics |
|---------|------------------|----------------|
| scikit-learn | `n_estimators`, `max_depth`, `C` | `accuracy`, `f1`, `roc_auc` |
| XGBoost | `learning_rate`, `n_estimators`, `max_depth` | `rmse`, `mae`, `auc` |
| TensorFlow/Keras | `epochs`, `batch_size`, `optimizer` | `loss`, `val_loss`, `accuracy` |
| PyTorch | `lr`, `epochs`, `hidden_size` | `train_loss`, `val_accuracy` |

---

## Why Use This

| Problem | How Parameters & Metrics Solves It |
|---------|-----------------------------------|
| "What hyperparameters did I use for my best model?" | All parameters are saved with each run |
| "I can't remember which settings gave 95% accuracy" | Parameters and metrics are linked together |
| "I need to reproduce my results" | Exact configuration is stored permanently |
| "How did my loss change during training?" | Step-based metrics create training curves |
| "I want to find the best model programmatically" | Query runs by metrics (e.g., highest accuracy) |

---

## Prerequisites

```bash
pip install mlflow scikit-learn
```

---

## Part 1: Logging Parameters

### What are parameters?

Parameters are settings you decide BEFORE training:
- Number of trees in a forest
- Learning rate
- Batch size
- Any configuration value

### Step 1.1: Basic parameter logging

Create `params_demo.py`:

```python
import mlflow

mlflow.set_experiment("params_demo")

with mlflow.start_run(run_name="basic_params"):

    # Log a single parameter
    mlflow.log_param("learning_rate", 0.01)

    print("Parameter logged!")
```

Run it:

```bash
python params_demo.py
```

**What happened:**

A file was created at `mlruns/<exp_id>/<run_id>/params/learning_rate` containing:
```
0.01
```

### Step 1.2: Log multiple parameters at once

```python
import mlflow

mlflow.set_experiment("params_demo")

with mlflow.start_run(run_name="multiple_params"):

    # Log multiple parameters in one call
    mlflow.log_params({
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 100,
        "optimizer": "adam"
    })

    print("All parameters logged!")
```

**Folder structure after running:**

```
mlruns/<exp_id>/<run_id>/params/
├── learning_rate    # Contains: 0.01
├── batch_size       # Contains: 32
├── epochs           # Contains: 100
└── optimizer        # Contains: adam
```

### Step 1.3: Log parameters from a dictionary

This is useful when you have a config dict:

```python
import mlflow

mlflow.set_experiment("params_demo")

# Your configuration
config = {
    "model_type": "RandomForest",
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "random_state": 42
}

with mlflow.start_run(run_name="from_config"):

    # Log the entire config
    mlflow.log_params(config)

    # Verify what was logged
    run = mlflow.active_run()
    print(f"Run ID: {run.info.run_id}")
    print(f"Parameters logged: {list(config.keys())}")
```

---

## Part 2: Logging Metrics

### What are metrics?

Metrics are measurements you calculate AFTER training:
- Accuracy
- Loss
- Precision, Recall, F1-score
- Any numeric result

### Step 2.1: Basic metric logging

Create `metrics_demo.py`:

```python
import mlflow

mlflow.set_experiment("metrics_demo")

with mlflow.start_run(run_name="basic_metrics"):

    # Simulate a training result
    accuracy = 0.95

    # Log the metric
    mlflow.log_metric("accuracy", accuracy)

    print(f"Logged accuracy: {accuracy}")
```

Run it:

```bash
python metrics_demo.py
```

**What happened:**

A file was created at `mlruns/<exp_id>/<run_id>/metrics/accuracy` containing:
```
1705312800000 0.95 0
```

Format: `timestamp value step`

### Step 2.2: Log multiple metrics at once

```python
import mlflow

mlflow.set_experiment("metrics_demo")

with mlflow.start_run(run_name="multiple_metrics"):

    # Log multiple metrics in one call
    mlflow.log_metrics({
        "accuracy": 0.95,
        "precision": 0.93,
        "recall": 0.91,
        "f1_score": 0.92,
        "loss": 0.12
    })

    print("All metrics logged!")
```

### Step 2.3: Log metrics over time (training progress)

This is powerful - you can track how metrics change during training:

```python
import mlflow
import time

mlflow.set_experiment("metrics_demo")

with mlflow.start_run(run_name="training_progress"):

    print("Simulating training...")

    # Simulate 10 epochs of training
    for epoch in range(10):

        # Simulate decreasing loss
        loss = 1.0 - (epoch * 0.08)

        # Simulate increasing accuracy
        accuracy = 0.5 + (epoch * 0.05)

        # Log with step number
        mlflow.log_metric("loss", loss, step=epoch)
        mlflow.log_metric("accuracy", accuracy, step=epoch)

        print(f"Epoch {epoch}: loss={loss:.3f}, accuracy={accuracy:.3f}")

        time.sleep(0.1)  # Small delay for realism

    print("\nTraining complete! Check MLflow UI for charts.")
```

Run it:

```bash
python metrics_demo.py
```

**Output:**

```
Simulating training...
Epoch 0: loss=1.000, accuracy=0.500
Epoch 1: loss=0.920, accuracy=0.550
Epoch 2: loss=0.840, accuracy=0.600
Epoch 3: loss=0.760, accuracy=0.650
Epoch 4: loss=0.680, accuracy=0.700
Epoch 5: loss=0.600, accuracy=0.750
Epoch 6: loss=0.520, accuracy=0.800
Epoch 7: loss=0.440, accuracy=0.850
Epoch 8: loss=0.360, accuracy=0.900
Epoch 9: loss=0.280, accuracy=0.950

Training complete! Check MLflow UI for charts.
```

### Step 2.4: View the charts

```bash
mlflow ui
```

1. Open http://127.0.0.1:5000
2. Click on "metrics_demo" experiment
3. Click on "training_progress" run
4. Scroll down to see the charts showing loss decreasing and accuracy increasing!

---

## Part 3: Complete Real-World Example

### Step 3.1: Full training script with all logging

Create `full_training.py`:

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =========================================
# STEP 1: Load and prepare data
# =========================================
print("Loading data...")
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Features: {len(feature_names)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# =========================================
# STEP 2: Define hyperparameters
# =========================================
hyperparameters = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42
}

# =========================================
# STEP 3: Set up MLflow
# =========================================
mlflow.set_experiment("breast_cancer_classification")

# =========================================
# STEP 4: Train and log everything
# =========================================
print("\nStarting MLflow run...")

with mlflow.start_run(run_name="random_forest_full"):

    # ----- LOG PARAMETERS -----
    print("\n--- Logging Parameters ---")

    # Log model hyperparameters
    mlflow.log_params(hyperparameters)
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")

    # Log data parameters
    data_params = {
        "n_features": len(feature_names),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "test_size": 0.2
    }
    mlflow.log_params(data_params)
    print(f"  n_features: {len(feature_names)}")
    print(f"  n_train_samples: {len(X_train)}")
    print(f"  n_test_samples: {len(X_test)}")

    # ----- TRAIN MODEL -----
    print("\n--- Training Model ---")
    model = RandomForestClassifier(**hyperparameters)
    model.fit(X_train, y_train)
    print("  Training complete!")

    # ----- MAKE PREDICTIONS -----
    print("\n--- Making Predictions ---")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print(f"  Predictions made for {len(y_pred)} samples")

    # ----- CALCULATE METRICS -----
    print("\n--- Calculating Metrics ---")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")

    # ----- LOG METRICS -----
    print("\n--- Logging Metrics ---")

    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    })
    print("  All metrics logged!")

    # ----- LOG ADDITIONAL INFO -----
    # Log feature importance as params (optional)
    top_features = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    print("\n--- Top 5 Most Important Features ---")
    for i, (name, importance) in enumerate(top_features):
        mlflow.log_metric(f"feature_importance_{i+1}", importance)
        print(f"  {i+1}. {name}: {importance:.4f}")

    # Get run info
    run_id = mlflow.active_run().info.run_id

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print(f"\nRun ID: {run_id}")
print(f"\nTo view results:")
print(f"  1. Run: mlflow ui")
print(f"  2. Open: http://127.0.0.1:5000")
print(f"  3. Click on 'breast_cancer_classification'")
print(f"  4. Click on 'random_forest_full'")
```

### Step 3.2: Run it

```bash
python full_training.py
```

### Step 3.3: View in MLflow UI

```bash
mlflow ui
```

Open http://127.0.0.1:5000 and explore:

- **Parameters tab**: All hyperparameters and data info
- **Metrics tab**: All performance metrics
- Click any metric to see its value

---

## Part 4: Comparing Runs by Metrics

### Step 4.1: Train multiple models

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("model_comparison")

# =========================================
# Model 1: Logistic Regression
# =========================================
print("Training Logistic Regression...")
with mlflow.start_run(run_name="logistic_regression"):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    mlflow.log_params({
        "model_type": "LogisticRegression",
        "max_iter": 1000
    })

    accuracy = accuracy_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test))

    mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})
    print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# =========================================
# Model 2: Random Forest
# =========================================
print("Training Random Forest...")
with mlflow.start_run(run_name="random_forest"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    mlflow.log_params({
        "model_type": "RandomForest",
        "n_estimators": 100
    })

    accuracy = accuracy_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test))

    mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})
    print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# =========================================
# Model 3: Gradient Boosting
# =========================================
print("Training Gradient Boosting...")
with mlflow.start_run(run_name="gradient_boosting"):
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    mlflow.log_params({
        "model_type": "GradientBoosting",
        "n_estimators": 100
    })

    accuracy = accuracy_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test))

    mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})
    print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

print("\nAll models trained! Compare them in MLflow UI.")
```

### Step 4.2: Compare in UI

1. Run `mlflow ui`
2. Click on "model_comparison" experiment
3. Check all three runs
4. Click "Compare" button
5. See a table comparing all metrics side by side

---

## Part 5: Finding the Best Run Programmatically

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("model_comparison")

# Find the run with highest accuracy
best_run = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1
)[0]

print("="*50)
print("BEST MODEL FOUND")
print("="*50)
print(f"Run Name: {best_run.info.run_name}")
print(f"Run ID: {best_run.info.run_id}")
print(f"\nParameters:")
for key, value in best_run.data.params.items():
    print(f"  {key}: {value}")
print(f"\nMetrics:")
for key, value in best_run.data.metrics.items():
    print(f"  {key}: {value:.4f}")
```

**Output:**

```
==================================================
BEST MODEL FOUND
==================================================
Run Name: gradient_boosting
Run ID: abc123...

Parameters:
  model_type: GradientBoosting
  n_estimators: 100

Metrics:
  accuracy: 0.9649
  f1_score: 0.9714
```

---

## Summary: All Functions

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

---

## Common Mistakes

### Mistake 1: Logging params outside of a run

```python
# WRONG - no active run
mlflow.log_param("lr", 0.01)  # Error!

# RIGHT - inside a run
with mlflow.start_run():
    mlflow.log_param("lr", 0.01)  # OK
```

### Mistake 2: Logging the same parameter twice

```python
# WRONG - can't update params
with mlflow.start_run():
    mlflow.log_param("lr", 0.01)
    mlflow.log_param("lr", 0.02)  # Error! Already logged

# RIGHT - decide the value first
with mlflow.start_run():
    final_lr = 0.02
    mlflow.log_param("lr", final_lr)  # Log once
```

### Mistake 3: Forgetting to log important params

```python
# BAD - missing important info
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    # What about random_state? max_depth? test_size?

# GOOD - log everything needed to reproduce
with mlflow.start_run():
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "test_size": 0.2
    })
```
