# Model Packaging - Detailed Guide

## What This Capability Does

Model Packaging saves your trained model so you can:
1. Use it later without retraining
2. Share it with others
3. Deploy it to production

When you package a model, MLflow saves:
- The model itself
- All dependencies (what packages it needs)
- Information about expected inputs/outputs

---

## Key Functions - Saving Models

### `mlflow.<flavor>.log_model()`
Saves a model to MLflow. Each ML library has its own "flavor" (module).

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | model object | Your trained model |
| `artifact_path` | str | Folder name in artifacts |
| `signature` | Signature (optional) | Input/output schema |
| `input_example` | array (optional) | Sample input data |
| `registered_model_name` | str (optional) | Register to Model Registry |

```python
mlflow.sklearn.log_model(model, artifact_path="model")
mlflow.sklearn.log_model(model, "model", signature=sig, input_example=X[:3])
```

---

### `infer_signature(input, output)`
Automatically creates a model signature from sample data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | array/DataFrame | Sample input data |
| `output` | array (optional) | Sample predictions |

```python
from mlflow.models.signature import infer_signature
signature = infer_signature(X_test, model.predict(X_test))
```

---

## Key Functions - Loading Models

### `mlflow.<flavor>.load_model(model_uri)`
Loads a model using the library-specific loader.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_uri` | str | Model location (see URI formats below) |

```python
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
```

---

### `mlflow.pyfunc.load_model(model_uri)`
Generic loader that works with **any** ML library. Returns a unified interface.

```python
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
predictions = model.predict(data)  # Same interface for all libraries
```

---

## Model URI Formats

| Format | Description | Example |
|--------|-------------|---------|
| `runs:/<run_id>/<path>` | Load from a specific run | `runs:/abc123/model` |
| `models:/<name>/<version>` | Load from registry by version | `models:/MyModel/1` |
| `models:/<name>/<stage>` | Load from registry by stage | `models:/MyModel/Production` |
| `file://<path>` | Load from local path | `file://./my_model` |

---

## Library Compatibility

MLflow has native support for many ML libraries ("flavors"):

| Library | Module | Save Function |
|---------|--------|--------------|
| scikit-learn | `mlflow.sklearn` | `mlflow.sklearn.log_model()` |
| XGBoost | `mlflow.xgboost` | `mlflow.xgboost.log_model()` |
| LightGBM | `mlflow.lightgbm` | `mlflow.lightgbm.log_model()` |
| TensorFlow/Keras | `mlflow.tensorflow` | `mlflow.tensorflow.log_model()` |
| PyTorch | `mlflow.pytorch` | `mlflow.pytorch.log_model()` |
| Spark MLlib | `mlflow.spark` | `mlflow.spark.log_model()` |
| statsmodels | `mlflow.statsmodels` | `mlflow.statsmodels.log_model()` |
| Prophet | `mlflow.prophet` | `mlflow.prophet.log_model()` |

**Tip:** Use `mlflow.pyfunc.load_model()` to load any model type with a unified interface.

---

## Combining Functions

**Save model with signature:**
```python
signature = infer_signature(X_test, model.predict(X_test))
mlflow.sklearn.log_model(model, "model", signature=signature)
```

**Save and register in one step:**
```python
mlflow.sklearn.log_model(
    model, "model",
    registered_model_name="MyProductionModel"  # Also registers it!
)
```

**Load and predict:**
```python
model = mlflow.pyfunc.load_model("models:/MyModel/Production")
predictions = model.predict(new_data)
```

---

## Why Use This

| Problem | How Model Packaging Solves It |
|---------|------------------------------|
| "I need to retrain my model every time I use it" | Model is saved and ready to load instantly |
| "My colleague can't run my model - missing packages" | Dependencies are saved with the model |
| "What input format does this model expect?" | Signature documents input/output schema |
| "I want to deploy my model to production" | Packaged models are deployment-ready |
| "Which scikit-learn version was this model trained with?" | Requirements.txt is auto-generated |

---

## Prerequisites

```bash
pip install mlflow scikit-learn
```

---

## Part 1: Basic Model Packaging

### Step 1.1: Train and save a model

Create `package_model.py`:

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Train a model
print("Training model...")
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print("Model trained!")

# Save the model with MLflow
mlflow.set_experiment("model_packaging")

with mlflow.start_run(run_name="my_first_model"):

    # This is the key function: log_model
    mlflow.sklearn.log_model(
        sk_model=model,           # Your trained model
        artifact_path="model"     # Folder name to save it in
    )

    run_id = mlflow.active_run().info.run_id
    print(f"\nModel saved!")
    print(f"Run ID: {run_id}")
    print(f"Model URI: runs:/{run_id}/model")
```

### Step 1.2: Run it

```bash
python package_model.py
```

**Output:**

```
Training model...
Model trained!

Model saved!
Run ID: a1b2c3d4e5f6...
Model URI: runs:/a1b2c3d4e5f6.../model
```

### Step 1.3: What was created

Look in `mlruns/<exp_id>/<run_id>/artifacts/model/`:

```
model/
├── MLmodel              # Describes how to load the model
├── model.pkl            # The actual model (serialized)
├── conda.yaml           # Conda environment
├── python_env.yaml      # Python environment
├── requirements.txt     # Pip requirements
└── input_example.json   # (if you provided an example)
```

**Look at MLmodel:**

```yaml
artifact_path: model
flavors:
  python_function:
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
    python_version: 3.9.0
  sklearn:
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 1.3.0
```

**Look at requirements.txt:**

```
mlflow==2.9.2
scikit-learn==1.3.0
cloudpickle==2.2.1
```

---

## Part 2: Loading a Saved Model

### Step 2.1: Load and use the model

Create `load_model.py`:

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris

# The run_id from when you saved the model
run_id = "a1b2c3d4e5f6..."  # Replace with your actual run_id

# Load the model
print(f"Loading model from run: {run_id}")
model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)
print("Model loaded!")

# Use the model
X, y = load_iris(return_X_y=True)
sample = X[:5]  # Take 5 samples

print("\nMaking predictions...")
predictions = loaded_model.predict(sample)
print(f"Input shape: {sample.shape}")
print(f"Predictions: {predictions}")
```

### Step 2.2: Run it

```bash
python load_model.py
```

**Output:**

```
Loading model from run: a1b2c3d4e5f6...
Model loaded!

Making predictions...
Input shape: (5, 4)
Predictions: [0 0 0 0 0]
```

---

## Part 3: Adding Model Signature

A **signature** tells users what data format the model expects.

### Step 3.1: Why use signatures?

Without signature:
```
User: "What data should I pass to this model?"
You: "Uh... 4 numbers I think?"
```

With signature:
```
Model signature shows:
  Input: float64[4] (4 float numbers)
  Output: int64[1] (1 integer)
```

### Step 3.2: Add a signature

```python
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Train model
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Make sample predictions for signature inference
sample_input = X_test[:5]
sample_output = model.predict(sample_input)

# Create signature automatically
signature = infer_signature(sample_input, sample_output)
print("Signature created:")
print(signature)

# Save model with signature
mlflow.set_experiment("model_with_signature")

with mlflow.start_run(run_name="model_v1"):

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature  # Add the signature!
    )

    print("\nModel saved with signature!")
```

**Output:**

```
Signature created:
inputs:
  ['input': float64 (4,)]
outputs:
  ['output': int64 (1,)]

Model saved with signature!
```

### Step 3.3: View signature in MLflow UI

1. Run `mlflow ui`
2. Click on the run
3. Click "Artifacts" → "model" → "MLmodel"
4. You'll see the signature in the file

---

## Part 4: Adding Input Example

An **input example** shows users exactly what data format to use.

### Step 4.1: Add input example

```python
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data with feature names
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Create signature
predictions = model.predict(X_test[:5])
signature = infer_signature(X_test[:5], predictions)

# Create input example (first 3 rows)
input_example = X_test.head(3)
print("Input example:")
print(input_example)

# Save model with signature AND input example
mlflow.set_experiment("model_with_example")

with mlflow.start_run(run_name="model_v1"):

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example  # Add example!
    )

    print("\nModel saved with input example!")
```

**Output:**

```
Input example:
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
73                 6.1               2.8                4.7               1.2
18                 5.7               3.8                1.7               0.3
118                7.7               2.6                6.9               2.3

Model saved with input example!
```

### Step 4.2: View in MLflow UI

The input example is saved as `input_example.json`:

```json
[
  {"sepal length (cm)": 6.1, "sepal width (cm)": 2.8, "petal length (cm)": 4.7, "petal width (cm)": 1.2},
  {"sepal length (cm)": 5.7, "sepal width (cm)": 3.8, "petal length (cm)": 1.7, "petal width (cm)": 0.3},
  {"sepal length (cm)": 7.7, "sepal width (cm)": 2.6, "petal length (cm)": 6.9, "petal width (cm)": 2.3}
]
```

---

## Part 5: Complete Example

```python
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("="*50)
print("MODEL PACKAGING PIPELINE")
print("="*50)

# =============================================
# STEP 1: Load and prepare data
# =============================================
print("\n[1/5] Loading data...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"      Training samples: {len(X_train)}")
print(f"      Test samples: {len(X_test)}")
print(f"      Features: {len(data.feature_names)}")

# =============================================
# STEP 2: Train model
# =============================================
print("\n[2/5] Training model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"      Accuracy: {accuracy:.2%}")

# =============================================
# STEP 3: Create signature and input example
# =============================================
print("\n[3/5] Creating signature and input example...")
signature = infer_signature(X_test, predictions)
input_example = X_test.head(3)
print("      Signature created")
print("      Input example created (3 samples)")

# =============================================
# STEP 4: Package the model
# =============================================
print("\n[4/5] Packaging model...")
mlflow.set_experiment("breast_cancer_model")

with mlflow.start_run(run_name="random_forest_v1"):

    # Log parameters
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    })

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Package and log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="breast_cancer_classifier",
        signature=signature,
        input_example=input_example
    )

    run_id = mlflow.active_run().info.run_id

print("      Model packaged!")

# =============================================
# STEP 5: Show how to load
# =============================================
print("\n[5/5] Model information:")
print(f"      Run ID: {run_id}")
print(f"      Model URI: runs:/{run_id}/breast_cancer_classifier")

print("\n" + "="*50)
print("HOW TO USE THIS MODEL")
print("="*50)
print("""
# Load the model:
import mlflow
model = mlflow.sklearn.load_model("runs:/{run_id}/breast_cancer_classifier")

# Make predictions:
predictions = model.predict(your_data)
""".format(run_id=run_id))

# Actually demonstrate loading
print("\n--- Demonstrating model loading ---")
loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/breast_cancer_classifier")
test_prediction = loaded_model.predict(X_test[:1])
print(f"Loaded model prediction: {test_prediction}")
print("(0 = Malignant, 1 = Benign)")
```

---

## Part 6: Different Ways to Load Models

### Method 1: By Run ID (most common)

```python
import mlflow

run_id = "your_run_id"
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
```

### Method 2: By Model Registry (for production)

```python
import mlflow

# Load specific version
model = mlflow.pyfunc.load_model("models:/MyModel/1")

# Load by stage
model = mlflow.pyfunc.load_model("models:/MyModel/Production")
```

### Method 3: From local path

```python
import mlflow

# If you downloaded the model folder
model = mlflow.sklearn.load_model("./downloaded_model/")
```

### Method 4: Generic pyfunc loader (works with any framework)

```python
import mlflow

# Works with sklearn, tensorflow, pytorch, etc.
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
predictions = model.predict(data)
```

---

## Part 7: Packaging Different Model Types

### Scikit-learn

```python
import mlflow.sklearn

mlflow.sklearn.log_model(model, "model")
model = mlflow.sklearn.load_model("runs:/{run_id}/model")
```

### XGBoost

```python
import mlflow.xgboost

mlflow.xgboost.log_model(model, "model")
model = mlflow.xgboost.load_model("runs:/{run_id}/model")
```

### LightGBM

```python
import mlflow.lightgbm

mlflow.lightgbm.log_model(model, "model")
model = mlflow.lightgbm.load_model("runs:/{run_id}/model")
```

### TensorFlow/Keras

```python
import mlflow.tensorflow

mlflow.tensorflow.log_model(model, "model")
model = mlflow.tensorflow.load_model("runs:/{run_id}/model")
```

### PyTorch

```python
import mlflow.pytorch

mlflow.pytorch.log_model(model, "model")
model = mlflow.pytorch.load_model("runs:/{run_id}/model")
```

---

## Part 8: Register Model While Packaging

You can package AND register in one step:

```python
import mlflow
import mlflow.sklearn

mlflow.set_experiment("model_registration")

with mlflow.start_run():

    # Package AND register in one call
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name="MyProductionModel"  # This registers it!
    )

    print("Model packaged AND registered!")
```

Now the model appears in the Model Registry (see model-registry.md for details).

---

## Summary: Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `mlflow.sklearn.log_model()` | Save sklearn model | `mlflow.sklearn.log_model(model, "model")` |
| `mlflow.sklearn.load_model()` | Load sklearn model | `mlflow.sklearn.load_model("runs:/{id}/model")` |
| `mlflow.pyfunc.load_model()` | Load any model type | `mlflow.pyfunc.load_model("runs:/{id}/model")` |
| `infer_signature()` | Create signature | `infer_signature(X, predictions)` |

---

## Common Mistakes

### Mistake 1: Forgetting the artifact_path

```python
# BAD - where does the model go?
mlflow.sklearn.log_model(model)  # Error!

# GOOD - specify where to save
mlflow.sklearn.log_model(model, "model")  # OK
mlflow.sklearn.log_model(model, artifact_path="model")  # Also OK
```

### Mistake 2: Wrong model URI format

```python
# BAD - missing "runs:/" prefix
model = mlflow.sklearn.load_model("abc123/model")  # Error!

# GOOD - correct format
model = mlflow.sklearn.load_model("runs:/abc123/model")  # OK
```

### Mistake 3: Using wrong loader for model type

```python
# BAD - using sklearn loader for xgboost model
model = mlflow.sklearn.load_model(...)  # May fail!

# GOOD - use matching loader or pyfunc
model = mlflow.xgboost.load_model(...)  # For xgboost
model = mlflow.pyfunc.load_model(...)   # Works for all
```
