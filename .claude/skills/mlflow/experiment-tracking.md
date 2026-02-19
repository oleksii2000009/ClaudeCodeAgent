# Experiment Tracking - Detailed Guide

## What This Capability Does

Experiment Tracking organizes your ML work into **experiments** and **runs**:

- **Experiment** = A project/folder (e.g., "fraud_detection")
- **Run** = One training attempt within that project

```
Experiment: "fraud_detection"
├── Run 1: Trained with RandomForest
├── Run 2: Trained with XGBoost
└── Run 3: Trained with Neural Network
```

---

## Key Functions

### `mlflow.set_experiment(name)`
Creates a new experiment or selects an existing one. All subsequent runs will be logged under this experiment.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Name of the experiment |

```python
mlflow.set_experiment("fraud_detection")
```

---

### `mlflow.start_run()`
Starts a new run within the current experiment. Use as a context manager (`with` statement) to automatically end the run.

| Parameter | Type | Description |
|-----------|------|-------------|
| `run_name` | str (optional) | Human-readable name for the run |
| `run_id` | str (optional) | Resume an existing run by ID |
| `nested` | bool (optional) | Allow nested runs (default: False) |

```python
with mlflow.start_run(run_name="random_forest_v1"):
    # Your training code here
```

---

### `mlflow.active_run()`
Returns the currently active run object. Useful for getting the run ID.

```python
run = mlflow.active_run()
run_id = run.info.run_id
```

---

### `mlflow.end_run()`
Ends the current run. Called automatically when using `with mlflow.start_run()`.

---

### `MlflowClient.search_runs()`
Query past runs programmatically. Useful for finding the best model or comparing results.

| Parameter | Type | Description |
|-----------|------|-------------|
| `experiment_ids` | list | Experiment IDs to search |
| `order_by` | list | Sort criteria (e.g., `["metrics.accuracy DESC"]`) |
| `max_results` | int | Limit number of results |

---

## Combining Functions

**Basic workflow:**
```python
mlflow.set_experiment("my_project")      # 1. Select experiment
with mlflow.start_run(run_name="v1"):    # 2. Start run
    # ... train model ...                 # 3. Your code
    run_id = mlflow.active_run().info.run_id  # 4. Get run ID
# Run automatically ends
```

**Finding the best run:**
```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
best_run = client.search_runs(
    experiment_ids=["123"],
    order_by=["metrics.accuracy DESC"],
    max_results=1
)[0]
```

---

## Library Compatibility

Experiment tracking works with **any ML library**. The experiment/run structure is independent of your model:

| Library | Works? | Notes |
|---------|--------|-------|
| scikit-learn | ✅ | Full support |
| XGBoost | ✅ | Full support |
| LightGBM | ✅ | Full support |
| TensorFlow/Keras | ✅ | Full support |
| PyTorch | ✅ | Full support |
| Custom models | ✅ | Just wrap your training code |

---

## Why Use This

| Problem | How Experiment Tracking Solves It |
|---------|-----------------------------------|
| "I forgot which model configuration worked best" | All runs are saved and searchable |
| "I can't compare my different attempts" | MLflow UI lets you compare runs side-by-side |
| "My experiment results are scattered in notebooks" | Everything is organized in one place |
| "I need to reproduce last week's results" | Each run stores all parameters and metrics |
| "Team members can't see my experiment history" | MLflow provides a shared tracking server |

---

## Prerequisites

Make sure you have MLflow installed:

```bash
pip install mlflow scikit-learn
```

---

## Part 1: Creating Your First Experiment

### Step 1.1: Create a new Python file

Create a file called `experiment_demo.py` and add this code:

```python
import mlflow

# Create an experiment called "my_project"
mlflow.set_experiment("my_project")

print("Experiment created!")
```

### Step 1.2: Run the file

```bash
python experiment_demo.py
```

**What you will see:**

```
2024/01/15 10:00:00 INFO mlflow.tracking.fluent: Experiment with name 'my_project' does not exist. Creating a new experiment.
Experiment created!
```

### Step 1.3: Check what was created

A new folder appeared:

```
your_directory/
└── mlruns/
    ├── 0/                    # "Default" experiment (always exists)
    └── 123456789012345678/   # Your "my_project" experiment
        └── meta.yaml         # Experiment metadata
```

**What's in meta.yaml:**

```yaml
artifact_location: file:///path/to/mlruns/123456789012345678
experiment_id: '123456789012345678'
lifecycle_stage: active
name: my_project
```

---

## Part 2: Creating Runs Inside an Experiment

### Step 2.1: Update your code

```python
import mlflow

# Select the experiment
mlflow.set_experiment("my_project")

# Create a run inside this experiment
with mlflow.start_run():
    print("Run started!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print("Run finished!")
```

### Step 2.2: Run it

```bash
python experiment_demo.py
```

**What you will see:**

```
Run started!
Run ID: a1b2c3d4e5f6g7h8i9j0
Run finished!
```

### Step 2.3: Check the folder structure now

```
mlruns/
└── 123456789012345678/          # Your experiment
    ├── meta.yaml
    └── a1b2c3d4e5f6g7h8i9j0/    # Your run
        ├── meta.yaml            # Run metadata
        ├── params/              # Empty (we didn't log any)
        ├── metrics/             # Empty (we didn't log any)
        ├── tags/                # Auto-generated tags
        │   ├── mlflow.user
        │   └── mlflow.source.name
        └── artifacts/           # Empty (we didn't save any)
```

---

## Part 3: Naming Your Runs

By default, runs have random IDs. You can give them readable names:

### Step 3.1: Add run_name parameter

```python
import mlflow

mlflow.set_experiment("my_project")

# Give the run a descriptive name
with mlflow.start_run(run_name="first_attempt"):
    print("This run is named 'first_attempt'")

with mlflow.start_run(run_name="second_attempt"):
    print("This run is named 'second_attempt'")
```

### Step 3.2: Run it

```bash
python experiment_demo.py
```

### Step 3.3: View in MLflow UI

```bash
mlflow ui
```

Open http://127.0.0.1:5000 and you'll see:

```
┌─────────────────────────────────────────────────────────────┐
│ Experiments        │ Runs                                   │
├────────────────────┼────────────────────────────────────────┤
│ ▼ my_project       │ Run Name          │ Start Time        │
│   Default          │ second_attempt    │ 2024-01-15 10:05  │
│                    │ first_attempt     │ 2024-01-15 10:04  │
└────────────────────┴────────────────────────────────────────┘
```

---

## Part 4: Complete Example with Real ML

### Step 4.1: Full working example

Create `train_model.py`:

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================================
# STEP 1: Load data
# =========================================
print("Loading data...")
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# =========================================
# STEP 2: Set up MLflow experiment
# =========================================
print("\nSetting up MLflow experiment...")
mlflow.set_experiment("iris_classification")
print("Experiment 'iris_classification' is ready")

# =========================================
# STEP 3: Train and track the model
# =========================================
print("\nStarting training run...")

with mlflow.start_run(run_name="random_forest_v1"):

    # Get the run ID for reference
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")

    # Define hyperparameters
    n_estimators = 100
    max_depth = 5
    random_state = 42

    # Log hyperparameters to MLflow
    print("\nLogging parameters...")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  random_state: {random_state}")

    # Train the model
    print("\nTraining model...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    print("Training complete!")

    # Evaluate the model
    print("\nEvaluating model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log metrics to MLflow
    print("\nLogging metrics...")
    mlflow.log_metric("accuracy", accuracy)
    print(f"  accuracy: {accuracy:.4f} ({accuracy:.2%})")

    print("\n" + "="*50)
    print("RUN COMPLETE!")
    print("="*50)
    print(f"\nTo view results, run: mlflow ui")
    print(f"Then open: http://127.0.0.1:5000")
    print(f"\nLook for experiment 'iris_classification'")
    print(f"and run 'random_forest_v1'")
```

### Step 4.2: Run it

```bash
python train_model.py
```

**Full output:**

```
Loading data...
Training samples: 120
Test samples: 30

Setting up MLflow experiment...
Experiment 'iris_classification' is ready

Starting training run...
Run ID: a1b2c3d4e5f6789012345678

Logging parameters...
  n_estimators: 100
  max_depth: 5
  random_state: 42

Training model...
Training complete!

Evaluating model...

Logging metrics...
  accuracy: 0.9667 (96.67%)

==================================================
RUN COMPLETE!
==================================================

To view results, run: mlflow ui
Then open: http://127.0.0.1:5000

Look for experiment 'iris_classification'
and run 'random_forest_v1'
```

### Step 4.3: View in MLflow UI

```bash
mlflow ui
```

**What you'll see in the browser:**

1. Left sidebar: Click "iris_classification"
2. Main area: You'll see "random_forest_v1" run
3. Click on it to see:
   - **Parameters**: n_estimators=100, max_depth=5, random_state=42
   - **Metrics**: accuracy=0.9667

---

## Part 5: Comparing Multiple Runs

### Step 5.1: Train multiple models with different settings

Create `compare_models.py`:

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set experiment
mlflow.set_experiment("model_comparison")

# =========================================
# RUN 1: Random Forest with 50 trees
# =========================================
print("Training Random Forest (50 trees)...")
with mlflow.start_run(run_name="rf_50_trees"):
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 50)
    mlflow.log_metric("accuracy", accuracy)

    print(f"  Accuracy: {accuracy:.2%}")

# =========================================
# RUN 2: Random Forest with 200 trees
# =========================================
print("Training Random Forest (200 trees)...")
with mlflow.start_run(run_name="rf_200_trees"):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("accuracy", accuracy)

    print(f"  Accuracy: {accuracy:.2%}")

# =========================================
# RUN 3: Logistic Regression
# =========================================
print("Training Logistic Regression...")
with mlflow.start_run(run_name="logistic_regression"):
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 200)
    mlflow.log_metric("accuracy", accuracy)

    print(f"  Accuracy: {accuracy:.2%}")

print("\nDone! Run 'mlflow ui' to compare all runs.")
```

### Step 5.2: Run it

```bash
python compare_models.py
```

**Output:**

```
Training Random Forest (50 trees)...
  Accuracy: 96.67%
Training Random Forest (200 trees)...
  Accuracy: 96.67%
Training Logistic Regression...
  Accuracy: 100.00%

Done! Run 'mlflow ui' to compare all runs.
```

### Step 5.3: Compare in UI

```bash
mlflow ui
```

In the browser:
1. Select "model_comparison" experiment
2. Check the boxes next to multiple runs
3. Click "Compare" button
4. You'll see a comparison table with all parameters and metrics

---

## Part 6: Accessing Run Data Programmatically

### Step 6.1: Get information about past runs

```python
import mlflow
from mlflow.tracking import MlflowClient

# Create a client to query MLflow
client = MlflowClient()

# Get the experiment by name
experiment = client.get_experiment_by_name("model_comparison")
print(f"Experiment ID: {experiment.experiment_id}")
print(f"Experiment Name: {experiment.name}")
print(f"Artifact Location: {experiment.artifact_location}")

# List all runs in this experiment
print("\nAll runs in this experiment:")
runs = client.search_runs(experiment_ids=[experiment.experiment_id])

for run in runs:
    print(f"\n  Run: {run.info.run_name}")
    print(f"    Run ID: {run.info.run_id}")
    print(f"    Status: {run.info.status}")
    print(f"    Parameters: {run.data.params}")
    print(f"    Metrics: {run.data.metrics}")
```

### Step 6.2: Run it

```bash
python query_runs.py
```

**Output:**

```
Experiment ID: 123456789012345678
Experiment Name: model_comparison
Artifact Location: file:///path/to/mlruns/123456789012345678

All runs in this experiment:

  Run: logistic_regression
    Run ID: abc123...
    Status: FINISHED
    Parameters: {'model_type': 'LogisticRegression', 'max_iter': '200'}
    Metrics: {'accuracy': 1.0}

  Run: rf_200_trees
    Run ID: def456...
    Status: FINISHED
    Parameters: {'model_type': 'RandomForest', 'n_estimators': '200'}
    Metrics: {'accuracy': 0.9666666666666667}

  Run: rf_50_trees
    Run ID: ghi789...
    Status: FINISHED
    Parameters: {'model_type': 'RandomForest', 'n_estimators': '50'}
    Metrics: {'accuracy': 0.9666666666666667}
```

---

## Part 7: Finding the Best Run

### Step 7.1: Query for best accuracy

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("model_comparison")

# Search runs, ordered by accuracy (descending)
best_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],  # Sort by accuracy, highest first
    max_results=1                          # Only get the best one
)

if best_runs:
    best_run = best_runs[0]
    print("BEST RUN:")
    print(f"  Name: {best_run.info.run_name}")
    print(f"  Run ID: {best_run.info.run_id}")
    print(f"  Accuracy: {best_run.data.metrics['accuracy']:.2%}")
    print(f"  Parameters: {best_run.data.params}")
```

**Output:**

```
BEST RUN:
  Name: logistic_regression
  Run ID: abc123...
  Accuracy: 100.00%
  Parameters: {'model_type': 'LogisticRegression', 'max_iter': '200'}
```

---

## Summary: Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `mlflow.set_experiment("name")` | Create or select an experiment | `mlflow.set_experiment("my_project")` |
| `mlflow.start_run()` | Start tracking a run | `with mlflow.start_run():` |
| `mlflow.start_run(run_name="name")` | Start a named run | `with mlflow.start_run(run_name="v1"):` |
| `mlflow.active_run()` | Get current run info | `run_id = mlflow.active_run().info.run_id` |
| `mlflow.end_run()` | End the current run | Usually automatic with `with` |
| `MlflowClient().search_runs()` | Query past runs | See examples above |

---

## Common Errors and Fixes

### Error: "Experiment does not exist"

```python
# Wrong: experiment must exist or be created first
with mlflow.start_run(experiment_id="nonexistent"):  # Error!

# Right: use set_experiment to create if needed
mlflow.set_experiment("my_experiment")  # Creates if doesn't exist
with mlflow.start_run():
    pass
```

### Error: "Run already active"

```python
# Wrong: nested runs without ending the first
mlflow.start_run()
mlflow.start_run()  # Error! Previous run still active

# Right: use context manager or end run
with mlflow.start_run():
    pass  # Automatically ends

# Or manually:
mlflow.start_run()
# ... do stuff ...
mlflow.end_run()
mlflow.start_run()  # Now OK
```

### Error: Can't see experiments in UI

Make sure you:
1. Run `mlflow ui` from the same directory as your Python script
2. The `mlruns/` folder exists in that directory
3. You actually ran some experiments (check for folders in `mlruns/`)
