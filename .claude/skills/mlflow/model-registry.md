# Model Registry - Detailed Guide

## What This Capability Does

The Model Registry is a **catalog** of your production-ready models. It helps you:
1. Keep track of which model version is in production
2. Move models through stages (Testing → Staging → Production)
3. Roll back to previous versions if needed

---

## Model Stages

| Stage | Description | Use Case |
|-------|-------------|----------|
| `None` | Just registered | New model, not yet evaluated |
| `Staging` | Being tested | Model under evaluation before production |
| `Production` | Live | Model serving real users |
| `Archived` | Retired | Old version, kept for history |

---

## Key Functions - Registration

### Register via `log_model()`
The simplest way to register - add `registered_model_name` parameter:

```python
mlflow.sklearn.log_model(
    model, "model",
    registered_model_name="MyModel"  # Creates version 1, 2, 3...
)
```

---

### `MlflowClient.create_registered_model(name)`
Creates an empty registered model (without a version).

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.create_registered_model("MyModel")
```

---

## Key Functions - Stage Management

### `client.transition_model_version_stage(name, version, stage)`
Moves a model version to a different stage.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Registered model name |
| `version` | int | Version number |
| `stage` | str | `"None"`, `"Staging"`, `"Production"`, `"Archived"` |

```python
client.transition_model_version_stage("MyModel", version=1, stage="Production")
```

---

### `client.get_latest_versions(name, stages)`
Gets the latest versions for specified stages.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Registered model name |
| `stages` | list | List of stages to query |

```python
prod_versions = client.get_latest_versions("MyModel", stages=["Production"])
```

---

## Key Functions - Metadata

### `client.update_model_version(name, version, description)`
Adds a description to a model version.

```python
client.update_model_version(
    "MyModel", version=1,
    description="Random Forest, accuracy=96.7%, approved 2024-01-15"
)
```

---

### `client.set_model_version_tag(name, version, key, value)`
Adds a tag to a model version.

```python
client.set_model_version_tag("MyModel", "1", "approved_by", "data_team")
client.set_model_version_tag("MyModel", "1", "environment", "production")
```

---

## Key Functions - Loading

### `mlflow.pyfunc.load_model()` with registry URI
Load models by stage or version:

```python
# By stage (recommended for production code)
model = mlflow.pyfunc.load_model("models:/MyModel/Production")

# By version
model = mlflow.pyfunc.load_model("models:/MyModel/1")
```

---

## Combining Functions

**Typical deployment workflow:**
```python
# 1. Register new model
mlflow.sklearn.log_model(model, "model", registered_model_name="MyModel")

# 2. Move to Staging for testing
client.transition_model_version_stage("MyModel", version=2, stage="Staging")

# 3. Test it
staging_model = mlflow.pyfunc.load_model("models:/MyModel/Staging")
# ... run tests ...

# 4. Promote to Production
client.transition_model_version_stage("MyModel", version=2, stage="Production")

# 5. Archive old version
client.transition_model_version_stage("MyModel", version=1, stage="Archived")
```

**Rollback:**
```python
client.transition_model_version_stage("MyModel", version=2, stage="Archived")    # Bad version
client.transition_model_version_stage("MyModel", version=1, stage="Production")  # Restore
```

---

## Library Compatibility

The Model Registry works with **all MLflow-supported libraries**. The registry tracks model versions regardless of the underlying framework:

| Library | Can Register? | Notes |
|---------|--------------|-------|
| scikit-learn | ✅ | Full support |
| XGBoost | ✅ | Full support |
| LightGBM | ✅ | Full support |
| TensorFlow | ✅ | Full support |
| PyTorch | ✅ | Full support |
| Any pyfunc | ✅ | Custom models work too |

---

## Why Use This

| Problem | How Model Registry Solves It |
|---------|------------------------------|
| "Which model version is currently in production?" | Registry shows stage for each version |
| "I need to roll back to yesterday's model" | One command switches production version |
| "How do I test a model before deploying?" | Use Staging stage for testing |
| "Who approved this model for production?" | Add tags and descriptions for audit trail |
| "My production code breaks when I retrain" | `models:/Name/Production` always gets the right version |

---

## Prerequisites

```bash
pip install mlflow scikit-learn
```

---

## Part 1: Understanding Stages

Every model version has a **stage**:

| Stage | Meaning | Use Case |
|-------|---------|----------|
| **None** | Just registered | New model, not evaluated yet |
| **Staging** | Being tested | Model under evaluation |
| **Production** | Live | Model serving real users |
| **Archived** | Retired | Old version, kept for history |

**Typical lifecycle:**

```
Train → Register (None) → Test (Staging) → Deploy (Production) → Replace (Archived)
```

---

## Part 2: Registering Your First Model

### Step 2.1: Train and register in one step

Create `register_model.py`:

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Train a model
print("Training model...")
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {accuracy:.2%}")

# Register the model
mlflow.set_experiment("model_registry_demo")

with mlflow.start_run(run_name="v1"):

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)

    # This line registers the model!
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="IrisClassifier"  # <-- This registers it
    )

    print("\nModel registered as 'IrisClassifier' version 1")
```

### Step 2.2: Run it

```bash
python register_model.py
```

**Output:**

```
Training model...
Accuracy: 96.67%

Model registered as 'IrisClassifier' version 1
Registered model 'IrisClassifier' already exists. Creating a new version of this model...
Successfully registered model 'IrisClassifier'.
Created version '1' of model 'IrisClassifier'.
```

### Step 2.3: View in MLflow UI

```bash
mlflow ui
```

1. Open http://127.0.0.1:5000
2. Click on "Models" tab at the top
3. You'll see "IrisClassifier"
4. Click on it to see version 1

---

## Part 3: Creating Multiple Versions

Each time you register with the same name, a new version is created.

### Step 3.1: Register more versions

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("model_registry_demo")

# ===============================
# VERSION 1: Random Forest (100 trees)
# ===============================
print("Creating Version 1...")
with mlflow.start_run(run_name="v1_rf_100"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_params({"model": "RandomForest", "n_estimators": 100})
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        model, "model",
        registered_model_name="IrisClassifier"
    )
    print(f"  Accuracy: {accuracy:.2%}")

# ===============================
# VERSION 2: Random Forest (200 trees)
# ===============================
print("\nCreating Version 2...")
with mlflow.start_run(run_name="v2_rf_200"):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_params({"model": "RandomForest", "n_estimators": 200})
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        model, "model",
        registered_model_name="IrisClassifier"
    )
    print(f"  Accuracy: {accuracy:.2%}")

# ===============================
# VERSION 3: Logistic Regression
# ===============================
print("\nCreating Version 3...")
with mlflow.start_run(run_name="v3_logistic"):
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_params({"model": "LogisticRegression", "max_iter": 200})
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        model, "model",
        registered_model_name="IrisClassifier"
    )
    print(f"  Accuracy: {accuracy:.2%}")

print("\n3 versions created! Check MLflow UI → Models → IrisClassifier")
```

### Step 3.2: View in MLflow UI

Click on "Models" → "IrisClassifier" and you'll see:

```
IrisClassifier
├── Version 3 (latest)  Stage: None
├── Version 2           Stage: None
└── Version 1           Stage: None
```

---

## Part 4: Moving Models Between Stages

### Step 4.1: Using MlflowClient

```python
from mlflow.tracking import MlflowClient

# Create a client
client = MlflowClient()

# Move Version 1 to Staging
print("Moving Version 1 to Staging...")
client.transition_model_version_stage(
    name="IrisClassifier",
    version=1,
    stage="Staging"
)
print("Done!")

# Move Version 2 to Production
print("Moving Version 2 to Production...")
client.transition_model_version_stage(
    name="IrisClassifier",
    version=2,
    stage="Production"
)
print("Done!")

# Archive Version 3 (we don't need it)
print("Archiving Version 3...")
client.transition_model_version_stage(
    name="IrisClassifier",
    version=3,
    stage="Archived"
)
print("Done!")

print("\nCurrent stages:")
print("  Version 1: Staging")
print("  Version 2: Production")
print("  Version 3: Archived")
```

### Step 4.2: Run it

```bash
python manage_stages.py
```

### Step 4.3: View in MLflow UI

Click on "Models" → "IrisClassifier":

```
IrisClassifier
├── Version 3           Stage: Archived
├── Version 2           Stage: Production  ← Live model
└── Version 1           Stage: Staging     ← Testing
```

---

## Part 5: Loading Models by Stage

This is the key benefit - load models by their stage, not version number.

### Step 5.1: Load production model

```python
import mlflow

# Load whatever is currently in Production
print("Loading Production model...")
model = mlflow.pyfunc.load_model("models:/IrisClassifier/Production")
print("Production model loaded!")

# Load Staging model for testing
print("Loading Staging model...")
staging_model = mlflow.pyfunc.load_model("models:/IrisClassifier/Staging")
print("Staging model loaded!")

# Make predictions
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
sample = X[:3]

print("\nPredictions from Production model:")
print(model.predict(sample))

print("\nPredictions from Staging model:")
print(staging_model.predict(sample))
```

### Step 5.2: Load by version number

```python
import mlflow

# Load specific version
model_v1 = mlflow.pyfunc.load_model("models:/IrisClassifier/1")
model_v2 = mlflow.pyfunc.load_model("models:/IrisClassifier/2")

print("Loaded version 1 and version 2")
```

---

## Part 6: Adding Descriptions and Tags

### Step 6.1: Add description to model

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Add description to the registered model
client.update_registered_model(
    name="IrisClassifier",
    description="Classifies iris flowers into 3 species based on sepal and petal measurements."
)

print("Description added!")
```

### Step 6.2: Add description to specific version

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Add description to version 2
client.update_model_version(
    name="IrisClassifier",
    version=2,
    description="Random Forest with 200 trees. Best accuracy: 96.67%. Approved for production on 2024-01-15."
)

print("Version description added!")
```

### Step 6.3: Add tags for organization

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Add tags to version 2
client.set_model_version_tag("IrisClassifier", "2", "approved_by", "data_team")
client.set_model_version_tag("IrisClassifier", "2", "environment", "production")
client.set_model_version_tag("IrisClassifier", "2", "tested", "true")

print("Tags added!")
```

---

## Part 7: Complete Production Workflow

```python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature

print("="*60)
print("PRODUCTION MODEL DEPLOYMENT WORKFLOW")
print("="*60)

# Setup
client = MlflowClient()
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("production_workflow")

MODEL_NAME = "ProductionIrisModel"

# =============================================
# STEP 1: Train and register new model
# =============================================
print("\n[1/6] Training new model...")
with mlflow.start_run(run_name="new_model"):
    model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"      Accuracy: {accuracy:.2%}")

    mlflow.log_params({"n_estimators": 150, "max_depth": 8})
    mlflow.log_metric("accuracy", accuracy)

    signature = infer_signature(X_test, model.predict(X_test))

    model_info = mlflow.sklearn.log_model(
        model, "model",
        signature=signature,
        registered_model_name=MODEL_NAME
    )

# Get the latest version number
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max([int(v.version) for v in versions])
print(f"      Registered as version {latest_version}")

# =============================================
# STEP 2: Move to Staging for testing
# =============================================
print(f"\n[2/6] Moving version {latest_version} to Staging...")
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_version,
    stage="Staging"
)
print("      Done!")

# =============================================
# STEP 3: Test the staging model
# =============================================
print(f"\n[3/6] Testing Staging model...")
staging_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Staging")
staging_predictions = staging_model.predict(X_test)
staging_accuracy = accuracy_score(y_test, staging_predictions)
print(f"      Staging model accuracy: {staging_accuracy:.2%}")

# =============================================
# STEP 4: If tests pass, promote to Production
# =============================================
ACCURACY_THRESHOLD = 0.90

if staging_accuracy >= ACCURACY_THRESHOLD:
    print(f"\n[4/6] Tests passed! Promoting to Production...")

    # First, archive any existing production model
    try:
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        for v in prod_versions:
            print(f"      Archiving old production version {v.version}...")
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=v.version,
                stage="Archived"
            )
    except:
        print("      No existing production model to archive")

    # Promote new model to production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version,
        stage="Production"
    )
    print(f"      Version {latest_version} is now in Production!")
else:
    print(f"\n[4/6] Tests FAILED! Model stays in Staging.")
    print(f"      Required accuracy: {ACCURACY_THRESHOLD:.0%}")
    print(f"      Actual accuracy: {staging_accuracy:.2%}")

# =============================================
# STEP 5: Add documentation
# =============================================
print(f"\n[5/6] Adding documentation...")
client.update_model_version(
    name=MODEL_NAME,
    version=latest_version,
    description=f"Random Forest (150 trees, depth 8). Accuracy: {accuracy:.2%}. Deployed to production."
)
client.set_model_version_tag(MODEL_NAME, str(latest_version), "accuracy", f"{accuracy:.4f}")
client.set_model_version_tag(MODEL_NAME, str(latest_version), "deployed_date", "2024-01-15")
print("      Done!")

# =============================================
# STEP 6: Show how to use production model
# =============================================
print(f"\n[6/6] Production model ready!")
print("\n" + "="*60)
print("HOW TO USE THE PRODUCTION MODEL")
print("="*60)
print(f"""
# In your application code:

import mlflow

# Load the production model (always gets the latest production version)
model = mlflow.pyfunc.load_model("models:/{MODEL_NAME}/Production")

# Make predictions
predictions = model.predict(your_data)
""")

# Verify it works
print("--- Verifying production model ---")
prod_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
test_pred = prod_model.predict(X_test[:1])
print(f"Test prediction: {test_pred}")
print("\nWorkflow complete!")
```

---

## Part 8: Querying the Registry

### Step 8.1: List all registered models

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

print("All registered models:")
print("-" * 40)

for model in client.search_registered_models():
    print(f"\nModel: {model.name}")
    print(f"  Description: {model.description or 'None'}")
    print(f"  Latest versions:")

    for version in model.latest_versions:
        print(f"    - Version {version.version}: {version.current_stage}")
```

### Step 8.2: Get all versions of a model

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

MODEL_NAME = "IrisClassifier"

print(f"All versions of {MODEL_NAME}:")
print("-" * 50)

versions = client.search_model_versions(f"name='{MODEL_NAME}'")

for v in versions:
    print(f"\nVersion {v.version}:")
    print(f"  Stage: {v.current_stage}")
    print(f"  Description: {v.description or 'None'}")
    print(f"  Run ID: {v.run_id}")
    print(f"  Created: {v.creation_timestamp}")
```

### Step 8.3: Find the production model

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

MODEL_NAME = "IrisClassifier"

# Get models in Production stage
prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])

if prod_versions:
    prod = prod_versions[0]
    print(f"Production Model:")
    print(f"  Name: {MODEL_NAME}")
    print(f"  Version: {prod.version}")
    print(f"  Run ID: {prod.run_id}")
    print(f"\nTo load: mlflow.pyfunc.load_model('models:/{MODEL_NAME}/Production')")
else:
    print("No model in Production stage")
```

---

## Part 9: Rolling Back

If a production model has issues, roll back to a previous version:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

MODEL_NAME = "IrisClassifier"

# Current production is version 3, but it has bugs
# We want to roll back to version 2

print("Rolling back to version 2...")

# 1. Archive the broken version
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=3,
    stage="Archived"
)
print("  Version 3 → Archived")

# 2. Promote the good version back to production
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=2,
    stage="Production"
)
print("  Version 2 → Production")

print("\nRollback complete!")
```

---

## Summary: Model URI Formats

| Format | Meaning | Example |
|--------|---------|---------|
| `models:/Name/Production` | Current production model | `models:/MyModel/Production` |
| `models:/Name/Staging` | Current staging model | `models:/MyModel/Staging` |
| `models:/Name/1` | Specific version | `models:/MyModel/1` |
| `runs:/run_id/model` | Model from a run | `runs:/abc123/model` |

---

## Summary: Key Functions

| Function | Purpose |
|----------|---------|
| `log_model(..., registered_model_name="X")` | Register while logging |
| `client.transition_model_version_stage()` | Change model stage |
| `client.update_model_version()` | Add description |
| `client.set_model_version_tag()` | Add tags |
| `client.search_model_versions()` | List all versions |
| `client.get_latest_versions()` | Get by stage |
| `mlflow.pyfunc.load_model("models:/X/Production")` | Load production model |

---

## Common Mistakes

### Mistake 1: Wrong stage name

```python
# BAD - "Prod" is not a valid stage
client.transition_model_version_stage(name="M", version=1, stage="Prod")

# GOOD - Use exact stage names
client.transition_model_version_stage(name="M", version=1, stage="Production")
# Valid stages: None, Staging, Production, Archived
```

### Mistake 2: Forgetting to archive old production

```python
# BAD - Now you have 2 models in Production!
client.transition_model_version_stage(name="M", version=2, stage="Production")

# GOOD - Archive the old one first
client.transition_model_version_stage(name="M", version=1, stage="Archived")
client.transition_model_version_stage(name="M", version=2, stage="Production")
```

### Mistake 3: Wrong URI format

```python
# BAD - missing "models:/" prefix
model = mlflow.pyfunc.load_model("MyModel/Production")

# GOOD - correct format
model = mlflow.pyfunc.load_model("models:/MyModel/Production")
```
