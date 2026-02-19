# Artifacts Management - Detailed Guide

## What This Capability Does

Artifacts are **files** you want to save with your experiment:
- Plots (confusion matrix, ROC curve, etc.)
- CSV files (predictions, data samples)
- Text files (reports, logs)
- Any other file

---

## Key Functions

### `mlflow.log_artifact(local_path, artifact_path=None)`
Logs a local file to MLflow. The file must exist on disk.

| Parameter | Type | Description |
|-----------|------|-------------|
| `local_path` | str | Path to the file on your system |
| `artifact_path` | str (optional) | Subfolder in MLflow artifacts |

```python
mlflow.log_artifact("confusion_matrix.png")                    # Root
mlflow.log_artifact("confusion_matrix.png", artifact_path="plots")  # In "plots" folder
```

---

### `mlflow.log_artifacts(local_dir, artifact_path=None)`
Logs all files in a directory. Note the plural: `log_artifacts`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `local_dir` | str | Path to directory |
| `artifact_path` | str (optional) | Subfolder in MLflow artifacts |

```python
mlflow.log_artifacts("./output/")  # Logs all files in output folder
```

---

### `mlflow.log_figure(figure, artifact_file)`
Logs a matplotlib/plotly figure directly without saving to disk first.

| Parameter | Type | Description |
|-----------|------|-------------|
| `figure` | figure object | Matplotlib or Plotly figure |
| `artifact_file` | str | Filename in artifacts |

```python
fig, ax = plt.subplots()
ax.plot(data)
mlflow.log_figure(fig, "my_plot.png")
```

---

### `mlflow.log_text(text, artifact_file)`
Logs text content directly without creating a file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | str | Text content |
| `artifact_file` | str | Filename in artifacts |

```python
mlflow.log_text("Model accuracy: 95%", "notes.txt")
```

---

### `mlflow.log_dict(dictionary, artifact_file)`
Logs a dictionary as JSON or YAML file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dictionary` | dict | Dictionary to save |
| `artifact_file` | str | Filename (use .json or .yaml extension) |

```python
mlflow.log_dict({"threshold": 0.5, "classes": ["cat", "dog"]}, "config.json")
```

---

## Combining Functions

**Typical workflow - plots and data:**
```python
with mlflow.start_run():
    # Log plots to "plots" folder
    mlflow.log_artifact("confusion_matrix.png", artifact_path="plots")
    mlflow.log_artifact("roc_curve.png", artifact_path="plots")

    # Log data to "data" folder
    mlflow.log_artifact("predictions.csv", artifact_path="data")

    # Log config directly
    mlflow.log_dict(config, "config.json")
```

**Result in MLflow UI:**
```
Artifacts/
├── plots/
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── data/
│   └── predictions.csv
└── config.json
```

---

## Library Compatibility

Artifacts work with **any file type** from **any library**:

| Library | Common Artifacts |
|---------|-----------------|
| matplotlib | `.png`, `.pdf` plots |
| seaborn | Heatmaps, pair plots |
| plotly | Interactive `.html` charts |
| pandas | `.csv`, `.parquet` data files |
| Any | `.txt` reports, `.json` configs |

---

## Why Use This

| Problem | How Artifacts Solves It |
|---------|------------------------|
| "I lost the confusion matrix from my best model" | All plots are saved with the run |
| "Which predictions file goes with which model?" | Artifacts are linked to their run |
| "I need to share my model's visualizations" | Team can download from MLflow UI |
| "My reports are scattered across folders" | Everything is organized per experiment |
| "I want to audit what my model produced" | Complete output history is preserved |

---

## Prerequisites

```bash
pip install mlflow scikit-learn matplotlib seaborn pandas
```

---

## Part 1: Saving a Simple File

### Step 1.1: Save a text file

Create `artifacts_demo.py`:

```python
import mlflow

mlflow.set_experiment("artifacts_demo")

with mlflow.start_run(run_name="text_file"):

    # Step 1: Create a file on disk
    with open("notes.txt", "w") as f:
        f.write("This is my experiment note.\n")
        f.write("Model performed well on test data.")

    # Step 2: Log the file to MLflow
    mlflow.log_artifact("notes.txt")

    print("File saved to MLflow!")
```

Run it:

```bash
python artifacts_demo.py
```

**What happened:**

```
your_directory/
├── notes.txt              # The file you created
└── mlruns/
    └── <exp_id>/
        └── <run_id>/
            └── artifacts/
                └── notes.txt   # Copy saved in MLflow
```

### Step 1.2: View in MLflow UI

```bash
mlflow ui
```

1. Open http://127.0.0.1:5000
2. Click "artifacts_demo" experiment
3. Click on the run
4. Click "Artifacts" tab on the right
5. You'll see `notes.txt` listed
6. Click on it to view the contents

---

## Part 2: Saving Plots

### Step 2.1: Save a matplotlib plot

```python
import mlflow
import matplotlib.pyplot as plt
import numpy as np

mlflow.set_experiment("artifacts_demo")

with mlflow.start_run(run_name="simple_plot"):

    # Create some data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title("Sine Wave")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Save to disk first
    plt.savefig("sine_wave.png", dpi=100, bbox_inches='tight')
    print("Plot saved to disk: sine_wave.png")

    # Then log to MLflow
    mlflow.log_artifact("sine_wave.png")
    print("Plot logged to MLflow!")

    # Close the plot to free memory
    plt.close()
```

Run it:

```bash
python artifacts_demo.py
```

**View in MLflow UI:**

1. Go to http://127.0.0.1:5000
2. Click the run
3. Click "Artifacts" tab
4. Click on `sine_wave.png`
5. You'll see the plot displayed!

### Step 2.2: Use log_figure (direct method)

This skips saving to disk:

```python
import mlflow
import matplotlib.pyplot as plt
import numpy as np

mlflow.set_experiment("artifacts_demo")

with mlflow.start_run(run_name="direct_plot"):

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.cos(x))
    ax.set_title("Cosine Wave")

    # Log directly - no need to save to disk first!
    mlflow.log_figure(fig, "cosine_wave.png")
    print("Plot logged directly to MLflow!")

    plt.close()
```

---

## Part 3: Saving a Confusion Matrix

### Step 3.1: Complete example

```python
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

mlflow.set_experiment("confusion_matrix_demo")

# Load and split data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
class_names = ['setosa', 'versicolor', 'virginica']

with mlflow.start_run(run_name="with_confusion_matrix"):

    # Train model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy:.2%}")

    # Create confusion matrix
    print("Creating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)

    # Plot it
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,           # Show numbers in cells
        fmt='d',              # Format as integers
        cmap='Blues',         # Color scheme
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save and log
    plt.savefig("confusion_matrix.png", dpi=100, bbox_inches='tight')
    mlflow.log_artifact("confusion_matrix.png")
    print("Confusion matrix saved!")

    plt.close()

print("\nDone! Check 'Artifacts' tab in MLflow UI.")
```

Run it:

```bash
python confusion_matrix_demo.py
```

---

## Part 4: Organizing Artifacts in Folders

### Step 4.1: Use artifact_path parameter

```python
import mlflow
import matplotlib.pyplot as plt
import numpy as np

mlflow.set_experiment("organized_artifacts")

with mlflow.start_run(run_name="organized"):

    # Create multiple plots
    for i, func in enumerate([np.sin, np.cos, np.tan]):
        plt.figure()
        x = np.linspace(0, 2*np.pi, 100)
        plt.plot(x, func(x))
        plt.title(f"Plot {i+1}")

        filename = f"plot_{i+1}.png"
        plt.savefig(filename)

        # Save to a subfolder called "plots"
        mlflow.log_artifact(filename, artifact_path="plots")

        plt.close()
        print(f"Saved {filename} to 'plots' folder")

    # Create a data file
    with open("data.csv", "w") as f:
        f.write("x,y\n1,2\n3,4\n5,6\n")

    # Save to a subfolder called "data"
    mlflow.log_artifact("data.csv", artifact_path="data")
    print("Saved data.csv to 'data' folder")
```

**Result in MLflow UI:**

```
Artifacts/
├── plots/
│   ├── plot_1.png
│   ├── plot_2.png
│   └── plot_3.png
└── data/
    └── data.csv
```

---

## Part 5: Saving an Entire Folder

### Step 5.1: Use log_artifacts (plural)

```python
import mlflow
import os
import matplotlib.pyplot as plt
import numpy as np

mlflow.set_experiment("folder_artifact")

# Create a folder with multiple files
os.makedirs("output", exist_ok=True)

# Create multiple files in the folder
for i in range(3):
    plt.figure()
    plt.plot(np.random.randn(100))
    plt.title(f"Random Data {i+1}")
    plt.savefig(f"output/chart_{i+1}.png")
    plt.close()

with open("output/summary.txt", "w") as f:
    f.write("This folder contains 3 charts with random data.")

print("Created output folder with files:")
print("  output/chart_1.png")
print("  output/chart_2.png")
print("  output/chart_3.png")
print("  output/summary.txt")

with mlflow.start_run(run_name="folder_upload"):

    # Log the entire folder
    mlflow.log_artifacts("output")  # Note: log_artifacts (plural)

    print("\nEntire folder logged to MLflow!")
```

**Result in MLflow UI:**

```
Artifacts/
├── chart_1.png
├── chart_2.png
├── chart_3.png
└── summary.txt
```

### Step 5.2: Log folder to a specific path

```python
# Log folder contents to a subfolder called "results"
mlflow.log_artifacts("output", artifact_path="results")
```

**Result:**

```
Artifacts/
└── results/
    ├── chart_1.png
    ├── chart_2.png
    ├── chart_3.png
    └── summary.txt
```

---

## Part 6: Saving Data Without Files

### Step 6.1: log_text - Save text directly

```python
import mlflow

mlflow.set_experiment("direct_logging")

with mlflow.start_run(run_name="text_demo"):

    # Save text directly (no need to create a file first)
    mlflow.log_text(
        "Model trained successfully!\nAccuracy: 95%",
        "status.txt"
    )

    print("Text saved directly to MLflow!")
```

### Step 6.2: log_dict - Save dictionary as JSON

```python
import mlflow

mlflow.set_experiment("direct_logging")

with mlflow.start_run(run_name="dict_demo"):

    # Save a dictionary as JSON
    config = {
        "model": "RandomForest",
        "version": "1.0",
        "features": ["age", "income", "score"],
        "thresholds": {
            "low": 0.3,
            "high": 0.7
        }
    }

    mlflow.log_dict(config, "config.json")

    print("Dictionary saved as JSON!")
```

**Result in MLflow UI:**

`config.json` will contain:
```json
{
    "model": "RandomForest",
    "version": "1.0",
    "features": ["age", "income", "score"],
    "thresholds": {
        "low": 0.3,
        "high": 0.7
    }
}
```

---

## Part 7: Complete Example - Full Training Pipeline

```python
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, roc_curve, auc
)

mlflow.set_experiment("full_pipeline_with_artifacts")

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="complete_training"):

    print("="*50)
    print("TRAINING PIPELINE")
    print("="*50)

    # --- Train model ---
    print("\n[1/6] Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- Log parameters ---
    print("[2/6] Logging parameters...")
    mlflow.log_params({
        "n_estimators": 100,
        "random_state": 42,
        "n_features": len(feature_names)
    })

    # --- Evaluate model ---
    print("[3/6] Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    print(f"      Accuracy: {accuracy:.2%}")

    # --- ARTIFACT 1: Confusion Matrix ---
    print("[4/6] Creating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig("confusion_matrix.png", bbox_inches='tight')
    mlflow.log_artifact("confusion_matrix.png", artifact_path="plots")
    plt.close()
    print("      Saved: plots/confusion_matrix.png")

    # --- ARTIFACT 2: ROC Curve ---
    print("[5/6] Creating ROC curve...")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    mlflow.log_metric("roc_auc", roc_auc)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("roc_curve.png", bbox_inches='tight')
    mlflow.log_artifact("roc_curve.png", artifact_path="plots")
    plt.close()
    print("      Saved: plots/roc_curve.png")

    # --- ARTIFACT 3: Feature Importance ---
    print("[6/6] Creating feature importance plot...")
    importance = model.feature_importances_
    indices = np.argsort(importance)[-10:]  # Top 10

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance')
    plt.savefig("feature_importance.png", bbox_inches='tight')
    mlflow.log_artifact("feature_importance.png", artifact_path="plots")
    plt.close()
    print("      Saved: plots/feature_importance.png")

    # --- ARTIFACT 4: Classification Report ---
    report = classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'])
    mlflow.log_text(report, "classification_report.txt")
    print("      Saved: classification_report.txt")

    # --- ARTIFACT 5: Predictions CSV ---
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'probability': y_proba,
        'correct': y_test == y_pred
    })
    predictions_df.to_csv("predictions.csv", index=False)
    mlflow.log_artifact("predictions.csv", artifact_path="data")
    print("      Saved: data/predictions.csv")

    # --- ARTIFACT 6: Config JSON ---
    config = {
        "model_type": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": 100,
            "random_state": 42
        },
        "features": list(feature_names),
        "target_classes": ["Benign", "Malignant"]
    }
    mlflow.log_dict(config, "model_config.json")
    print("      Saved: model_config.json")

    run_id = mlflow.active_run().info.run_id

print("\n" + "="*50)
print("PIPELINE COMPLETE!")
print("="*50)
print(f"\nRun ID: {run_id}")
print("\nArtifacts saved:")
print("  plots/confusion_matrix.png")
print("  plots/roc_curve.png")
print("  plots/feature_importance.png")
print("  classification_report.txt")
print("  data/predictions.csv")
print("  model_config.json")
print("\nRun 'mlflow ui' to view everything!")
```

---

## Part 8: Downloading Artifacts

### Step 8.1: Download from a previous run

```python
import mlflow

# Get the run ID from MLflow UI or from your script
run_id = "your_run_id_here"  # Replace with actual run ID

# Download a specific file
local_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="plots/confusion_matrix.png"
)
print(f"Downloaded to: {local_path}")

# Download all artifacts
all_artifacts = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path=""  # Empty string = all artifacts
)
print(f"All artifacts downloaded to: {all_artifacts}")
```

---

## Summary: All Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `mlflow.log_artifact(path)` | Log a single file | `mlflow.log_artifact("plot.png")` |
| `mlflow.log_artifact(path, artifact_path)` | Log file to subfolder | `mlflow.log_artifact("plot.png", "plots")` |
| `mlflow.log_artifacts(dir)` | Log entire folder | `mlflow.log_artifacts("output/")` |
| `mlflow.log_figure(fig, name)` | Log matplotlib figure directly | `mlflow.log_figure(fig, "plot.png")` |
| `mlflow.log_text(text, name)` | Log text directly | `mlflow.log_text("Hello", "note.txt")` |
| `mlflow.log_dict(dict, name)` | Log dict as JSON | `mlflow.log_dict(config, "config.json")` |
| `mlflow.artifacts.download_artifacts()` | Download artifacts | See example above |

---

## Common Mistakes

### Mistake 1: Forgetting to close plots

```python
# BAD - memory leak
plt.figure()
plt.plot([1,2,3])
plt.savefig("plot.png")
mlflow.log_artifact("plot.png")
# Plot still in memory!

# GOOD - close when done
plt.figure()
plt.plot([1,2,3])
plt.savefig("plot.png")
mlflow.log_artifact("plot.png")
plt.close()  # Free memory
```

### Mistake 2: Logging outside a run

```python
# BAD - no active run
mlflow.log_artifact("file.txt")  # Error!

# GOOD - inside a run
with mlflow.start_run():
    mlflow.log_artifact("file.txt")  # OK
```

### Mistake 3: File doesn't exist

```python
# BAD - file must exist
mlflow.log_artifact("nonexistent.png")  # Error!

# GOOD - create file first
plt.savefig("plot.png")  # Create file
mlflow.log_artifact("plot.png")  # Then log it
```
