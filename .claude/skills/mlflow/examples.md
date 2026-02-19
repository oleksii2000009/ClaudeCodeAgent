# MLflow Code Examples

Practical examples for all MLflow capabilities.

---

## Table of Contents

1. [Experiment Tracking](#experiment-tracking)
2. [Parameters & Metrics](#parameters--metrics)
3. [Artifacts](#artifacts)
4. [Model Packaging](#model-packaging)
5. [Model Registry](#model-registry)
6. [Autologging](#autologging)
7. [Complete Pipelines](#complete-pipelines)

---

## Experiment Tracking

### Basic Experiment Setup

```python
import mlflow

# Create an experiment
mlflow.set_experiment("my_project")

# Create a run inside this experiment
with mlflow.start_run():
    print("Run started!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

### Named Runs

```python
import mlflow

mlflow.set_experiment("my_project")

with mlflow.start_run(run_name="first_attempt"):
    print("This run is named 'first_attempt'")

with mlflow.start_run(run_name="second_attempt"):
    print("This run is named 'second_attempt'")
```

### Compare Multiple Models

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("model_comparison")

# Run 1: Random Forest with 50 trees
with mlflow.start_run(run_name="rf_50_trees"):
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 50)
    mlflow.log_metric("accuracy", accuracy)

# Run 2: Logistic Regression
with mlflow.start_run(run_name="logistic_regression"):
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)
```

### Find Best Run Programmatically

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("model_comparison")

best_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1
)

if best_runs:
    best_run = best_runs[0]
    print(f"Best Run: {best_run.info.run_name}")
    print(f"Accuracy: {best_run.data.metrics['accuracy']:.2%}")
```

---

## Parameters & Metrics

### Log Multiple Parameters

```python
import mlflow

mlflow.set_experiment("params_demo")

with mlflow.start_run(run_name="multiple_params"):
    mlflow.log_params({
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 100,
        "optimizer": "adam"
    })
```

### Log Metrics Over Time (Training Progress)

```python
import mlflow

mlflow.set_experiment("metrics_demo")

with mlflow.start_run(run_name="training_progress"):
    for epoch in range(10):
        loss = 1.0 - (epoch * 0.08)
        accuracy = 0.5 + (epoch * 0.05)

        mlflow.log_metric("loss", loss, step=epoch)
        mlflow.log_metric("accuracy", accuracy, step=epoch)

        print(f"Epoch {epoch}: loss={loss:.3f}, accuracy={accuracy:.3f}")
```

### Full Training with All Metrics

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("breast_cancer_classification")

with mlflow.start_run(run_name="random_forest_full"):
    # Log parameters
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 10,
        "n_features": len(data.feature_names)
    })

    # Train
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    # Log metrics
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    })
```

---

## Artifacts

### Save a Plot

```python
import mlflow
import matplotlib.pyplot as plt
import numpy as np

mlflow.set_experiment("artifacts_demo")

with mlflow.start_run(run_name="simple_plot"):
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title("Sine Wave")
    plt.savefig("sine_wave.png", dpi=100, bbox_inches='tight')

    mlflow.log_artifact("sine_wave.png")
    plt.close()
```

### Save Confusion Matrix

```python
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
class_names = ['setosa', 'versicolor', 'virginica']

mlflow.set_experiment("confusion_matrix_demo")

with mlflow.start_run(run_name="with_confusion_matrix"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig("confusion_matrix.png", dpi=100, bbox_inches='tight')

    mlflow.log_artifact("confusion_matrix.png")
    plt.close()
```

### Organize Artifacts in Folders

```python
import mlflow
import matplotlib.pyplot as plt
import numpy as np

mlflow.set_experiment("organized_artifacts")

with mlflow.start_run(run_name="organized"):
    # Save plots to "plots" subfolder
    for i, func in enumerate([np.sin, np.cos]):
        plt.figure()
        x = np.linspace(0, 2*np.pi, 100)
        plt.plot(x, func(x))
        filename = f"plot_{i+1}.png"
        plt.savefig(filename)
        mlflow.log_artifact(filename, artifact_path="plots")
        plt.close()

    # Save data to "data" subfolder
    with open("data.csv", "w") as f:
        f.write("x,y\n1,2\n3,4\n")
    mlflow.log_artifact("data.csv", artifact_path="data")
```

### Save Text and JSON Directly

```python
import mlflow

mlflow.set_experiment("direct_logging")

with mlflow.start_run(run_name="text_and_json"):
    # Save text directly
    mlflow.log_text("Model trained successfully!\nAccuracy: 95%", "status.txt")

    # Save dictionary as JSON
    config = {
        "model": "RandomForest",
        "features": ["age", "income", "score"],
        "thresholds": {"low": 0.3, "high": 0.7}
    }
    mlflow.log_dict(config, "config.json")
```

---

## Model Packaging

### Basic Model Packaging

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

mlflow.set_experiment("model_packaging")

with mlflow.start_run(run_name="my_first_model"):
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )
    run_id = mlflow.active_run().info.run_id
    print(f"Model URI: runs:/{run_id}/model")
```

### Load and Use a Saved Model

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris

run_id = "your_run_id_here"  # Replace with actual run_id

model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)

X, y = load_iris(return_X_y=True)
predictions = loaded_model.predict(X[:5])
print(f"Predictions: {predictions}")
```

### Model with Signature and Input Example

```python
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Create signature and input example
predictions = model.predict(X_test[:5])
signature = infer_signature(X_test[:5], predictions)
input_example = X_test.head(3)

mlflow.set_experiment("model_with_signature")

with mlflow.start_run(run_name="model_v1"):
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )
```

---

## Model Registry

### Register Model While Logging

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

mlflow.set_experiment("model_registry_demo")

with mlflow.start_run(run_name="v1"):
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="IrisClassifier"  # This registers it!
    )
```

### Transition Model Between Stages

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Move to Staging
client.transition_model_version_stage(
    name="IrisClassifier",
    version=1,
    stage="Staging"
)

# Move to Production
client.transition_model_version_stage(
    name="IrisClassifier",
    version=2,
    stage="Production"
)

# Archive old version
client.transition_model_version_stage(
    name="IrisClassifier",
    version=1,
    stage="Archived"
)
```

### Load Production Model

```python
import mlflow

# Load whatever is currently in Production
model = mlflow.pyfunc.load_model("models:/IrisClassifier/Production")

# Make predictions
predictions = model.predict(your_data)
```

### Add Model Documentation

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Add description to model
client.update_registered_model(
    name="IrisClassifier",
    description="Classifies iris flowers into 3 species."
)

# Add description to version
client.update_model_version(
    name="IrisClassifier",
    version=2,
    description="Random Forest with 200 trees. Accuracy: 96.67%."
)

# Add tags
client.set_model_version_tag("IrisClassifier", "2", "approved_by", "data_team")
client.set_model_version_tag("IrisClassifier", "2", "environment", "production")
```

---

## Autologging

### Basic Autologging

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Enable autologging
mlflow.autolog()

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow automatically logs everything!
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
```

### Autolog with Custom Additions

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.autolog()
mlflow.set_experiment("autolog_plus_manual")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="my_custom_run"):
    # Autolog captures this
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Add custom metrics manually
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_param("dataset", "iris")
    mlflow.set_tag("notes", "Best model so far")
```

### Compare Models with Autolog

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

mlflow.autolog()
mlflow.set_experiment("model_comparison_autolog")

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# All models automatically logged!
with mlflow.start_run(run_name="random_forest"):
    RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

with mlflow.start_run(run_name="gradient_boosting"):
    GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

with mlflow.start_run(run_name="logistic_regression"):
    LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
```

---

## Complete Pipelines

### Full Training Pipeline with Artifacts

```python
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc

mlflow.set_experiment("full_pipeline_with_artifacts")

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="complete_training"):
    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Log parameters
    mlflow.log_params({"n_estimators": 100, "random_state": 42})

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png", bbox_inches='tight')
    mlflow.log_artifact("confusion_matrix.png", artifact_path="plots")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    mlflow.log_metric("roc_auc", roc_auc)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("roc_curve.png", bbox_inches='tight')
    mlflow.log_artifact("roc_curve.png", artifact_path="plots")
    plt.close()

    # Classification Report
    report = classification_report(y_test, y_pred)
    mlflow.log_text(report, "classification_report.txt")

    # Predictions CSV
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'probability': y_proba
    })
    predictions_df.to_csv("predictions.csv", index=False)
    mlflow.log_artifact("predictions.csv", artifact_path="data")
```

### Production Deployment Workflow

```python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

client = MlflowClient()
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("production_workflow")
MODEL_NAME = "ProductionIrisModel"

# Step 1: Train and register
with mlflow.start_run(run_name="new_model"):
    model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)

    signature = infer_signature(X_test, model.predict(X_test))
    mlflow.sklearn.log_model(model, "model", signature=signature,
                             registered_model_name=MODEL_NAME)

# Step 2: Get latest version
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max([int(v.version) for v in versions])

# Step 3: Move to Staging for testing
client.transition_model_version_stage(MODEL_NAME, latest_version, "Staging")

# Step 4: Test staging model
staging_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Staging")
staging_accuracy = accuracy_score(y_test, staging_model.predict(X_test))

# Step 5: Promote to Production if tests pass
if staging_accuracy >= 0.90:
    # Archive existing production
    try:
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        for v in prod_versions:
            client.transition_model_version_stage(MODEL_NAME, v.version, "Archived")
    except:
        pass

    # Promote new model
    client.transition_model_version_stage(MODEL_NAME, latest_version, "Production")
    print(f"Version {latest_version} promoted to Production!")
```
