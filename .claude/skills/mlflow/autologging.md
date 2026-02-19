# Autologging - Detailed Guide

## What This Capability Does

Autologging automatically tracks your ML experiments **without writing any logging code**. MLflow watches your training and saves everything for you.

**The difference:**

```
WITHOUT Autologging (manual):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")

WITH Autologging (automatic):
    mlflow.autolog()  # Just this one line!
    # Everything above is logged automatically
```

---

## Key Functions

### `mlflow.autolog()`
Enables automatic logging for **all supported libraries** at once.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_models` | bool | True | Save the trained model |
| `log_input_examples` | bool | False | Save sample input data |
| `log_model_signatures` | bool | True | Save input/output schema |
| `disable` | bool | False | Turn off autologging |
| `silent` | bool | False | Suppress autolog messages |

```python
mlflow.autolog()                        # Enable everything
mlflow.autolog(log_models=False)        # Metrics only, no model
mlflow.autolog(disable=True)            # Turn off
```

---

### `mlflow.<flavor>.autolog()`
Enables autologging for a **specific library only**.

```python
mlflow.sklearn.autolog()      # Only sklearn
mlflow.xgboost.autolog()      # Only XGBoost
mlflow.tensorflow.autolog()   # Only TensorFlow
```

---

## What Gets Logged Automatically

| Library | Parameters | Metrics | Model | Extra |
|---------|------------|---------|-------|-------|
| sklearn | All hyperparameters | accuracy, f1, precision, recall | ✅ | Feature importance |
| XGBoost | All params | Training metrics per iteration | ✅ | Feature importance |
| LightGBM | All params | Training metrics per iteration | ✅ | Feature importance |
| TensorFlow | Optimizer, loss, layers | Loss/accuracy per epoch | ✅ | TensorBoard logs |
| PyTorch Lightning | All hparams | Metrics per epoch | ✅ | - |

---

## Library Compatibility

### Supported Libraries

| Library | Autolog Module | Call |
|---------|---------------|------|
| scikit-learn | `mlflow.sklearn` | `mlflow.sklearn.autolog()` |
| XGBoost | `mlflow.xgboost` | `mlflow.xgboost.autolog()` |
| LightGBM | `mlflow.lightgbm` | `mlflow.lightgbm.autolog()` |
| TensorFlow/Keras | `mlflow.tensorflow` | `mlflow.tensorflow.autolog()` |
| PyTorch Lightning | `mlflow.pytorch` | `mlflow.pytorch.autolog()` |
| Spark MLlib | `mlflow.spark` | `mlflow.spark.autolog()` |
| statsmodels | `mlflow.statsmodels` | `mlflow.statsmodels.autolog()` |
| Fastai | `mlflow.fastai` | `mlflow.fastai.autolog()` |

**Tip:** Use `mlflow.autolog()` to enable for ALL supported libraries at once.

---

## Combining Functions

**Autolog + manual logging (add custom metrics):**
```python
mlflow.autolog()  # Captures everything automatically

with mlflow.start_run():
    model.fit(X_train, y_train)  # Autologged

    # Add your own custom metrics
    test_acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("test_accuracy", test_acc)
```

**Enable for specific experiment:**
```python
mlflow.set_experiment("my_project")
mlflow.autolog()
model.fit(X_train, y_train)  # Logged to "my_project"
```

**Temporarily disable:**
```python
mlflow.autolog()
model1.fit(X, y)  # Logged

mlflow.autolog(disable=True)
model2.fit(X, y)  # NOT logged

mlflow.autolog(disable=False)
model3.fit(X, y)  # Logged again
```

---

## When to Use Autolog vs Manual

| Scenario | Recommendation |
|----------|---------------|
| Quick prototyping | ✅ Use autolog |
| Comparing many models | ✅ Use autolog |
| Production pipeline | Consider manual for control |
| Need custom metrics only | ✅ Autolog + manual |
| Memory-constrained | `autolog(log_models=False)` |

---

## Why Use This

| Problem | How Autologging Solves It |
|---------|--------------------------|
| "I forgot to log important parameters" | All parameters are captured automatically |
| "Logging code clutters my training script" | One line replaces dozens of log calls |
| "I want to quickly compare many models" | Just train models, MLflow logs everything |
| "I missed logging the model itself" | Model is automatically saved as an artifact |
| "I'm prototyping and don't want to set up logging" | Enable autolog and focus on experimentation |

---

## Prerequisites

```bash
pip install mlflow scikit-learn
```

---

## Part 1: Basic Autologging

### Step 1.1: Enable autologging

Create `autolog_demo.py`:

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Enable autologging - this is the magic line!
mlflow.autolog()

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model - MLflow automatically logs everything!
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

print("Training complete!")
print("Check MLflow UI to see all the automatically logged data.")
```

### Step 1.2: Run it

```bash
python autolog_demo.py
```

**Output:**

```
Training complete!
Check MLflow UI to see all the automatically logged data.
```

### Step 1.3: View in MLflow UI

```bash
mlflow ui
```

Open http://127.0.0.1:5000 and you'll see:

**Automatically logged parameters:**
- n_estimators: 100
- max_depth: 5
- random_state: 42
- min_samples_split: 2
- min_samples_leaf: 1
- ... (all RandomForest parameters!)

**Automatically logged metrics:**
- training_accuracy_score
- training_f1_score
- training_precision_score
- training_recall_score
- training_log_loss
- ... (many more!)

**Automatically logged artifacts:**
- model/ (the trained model, ready to load)
- model signature
- requirements.txt

---

## Part 2: Autologging with Experiment Name

### Step 2.1: Combine with set_experiment

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# First, set your experiment name
mlflow.set_experiment("breast_cancer_autolog")

# Then enable autologging
mlflow.autolog()

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model = RandomForestClassifier(n_estimators=150, max_depth=10)
model.fit(X_train, y_train)

print("Done! Check experiment 'breast_cancer_autolog' in MLflow UI.")
```

### Step 2.2: Run and check

```bash
python autolog_demo.py
mlflow ui
```

Now your automatically logged run appears in the "breast_cancer_autolog" experiment.

---

## Part 3: Autologging for Different Frameworks

### Step 3.1: Scikit-learn (sklearn)

```python
import mlflow
mlflow.sklearn.autolog()  # Or just mlflow.autolog()

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)  # Automatically logged!
```

**What gets logged automatically:**
- All model hyperparameters
- Training metrics (accuracy, f1, precision, recall, etc.)
- The model itself
- Model signature

### Step 3.2: XGBoost

```python
import mlflow
mlflow.xgboost.autolog()

import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)  # Automatically logged!
```

**What gets logged automatically:**
- All XGBoost parameters
- Training metrics at each boosting round
- Feature importance
- The model

### Step 3.3: LightGBM

```python
import mlflow
mlflow.lightgbm.autolog()

import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=100)
model.fit(X_train, y_train)  # Automatically logged!
```

### Step 3.4: TensorFlow/Keras

```python
import mlflow
mlflow.tensorflow.autolog()

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)  # All epochs logged automatically!
```

**What gets logged automatically:**
- All model parameters
- Metrics at each epoch (loss, accuracy, val_loss, val_accuracy)
- The model
- TensorBoard logs

### Step 3.5: PyTorch Lightning

```python
import mlflow
mlflow.pytorch.autolog()

import pytorch_lightning as pl

trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_dataloader)  # Automatically logged!
```

---

## Part 4: Controlling What Gets Logged

### Step 4.1: Disable model logging

Sometimes you don't want to save the model (to save space):

```python
import mlflow

# Log parameters and metrics, but NOT the model
mlflow.autolog(log_models=False)

model = RandomForestClassifier()
model.fit(X_train, y_train)
# Parameters and metrics logged, but no model artifact
```

### Step 4.2: Disable input examples

```python
import mlflow

# Don't save input examples (saves space)
mlflow.autolog(log_input_examples=False)
```

### Step 4.3: Disable model signatures

```python
import mlflow

# Don't infer model signature
mlflow.autolog(log_model_signatures=False)
```

### Step 4.4: All options combined

```python
import mlflow

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

---

## Part 5: Combining Autolog with Manual Logging

### Step 5.1: Add custom metrics

Autologging doesn't prevent you from adding your own logs:

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Enable autologging
mlflow.autolog()

# Set experiment
mlflow.set_experiment("autolog_plus_manual")

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start a run explicitly to add custom logging
with mlflow.start_run(run_name="my_custom_run"):

    # Train model - autolog captures this
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Add your own custom metrics
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("test_accuracy", test_accuracy)  # Manual addition!

    # Add a custom parameter
    mlflow.log_param("dataset", "iris")  # Manual addition!

    # Add a note
    mlflow.set_tag("notes", "This is my best model so far")

    print(f"Test accuracy: {test_accuracy:.2%}")
```

**Result in MLflow UI:**
- Automatic: All RandomForest parameters, training metrics, model
- Manual: test_accuracy, dataset parameter, notes tag

### Step 5.2: Save additional artifacts

```python
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

mlflow.autolog()
mlflow.set_experiment("autolog_with_artifacts")

with mlflow.start_run():

    # Train model (autologged)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Create and save confusion matrix (manual)
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")  # Manual!

    print("Model trained and confusion matrix saved!")
```

---

## Part 6: Comparing Multiple Models with Autolog

### Step 6.1: Train multiple models automatically

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Enable autologging
mlflow.autolog()

# Set experiment
mlflow.set_experiment("model_comparison_autolog")

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================
# Model 1: Random Forest
# =========================================
print("Training Random Forest...")
with mlflow.start_run(run_name="random_forest"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

# =========================================
# Model 2: Gradient Boosting
# =========================================
print("Training Gradient Boosting...")
with mlflow.start_run(run_name="gradient_boosting"):
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

# =========================================
# Model 3: Logistic Regression
# =========================================
print("Training Logistic Regression...")
with mlflow.start_run(run_name="logistic_regression"):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

print("\nAll models trained!")
print("Open MLflow UI and compare the three runs.")
print("Each run has all parameters and metrics logged automatically!")
```

### Step 6.2: Run and compare

```bash
python model_comparison.py
mlflow ui
```

In MLflow UI:
1. Go to "model_comparison_autolog" experiment
2. Check all three runs
3. Click "Compare"
4. See side-by-side comparison of all automatically logged metrics!

---

## Part 7: Disable Autologging

### Step 7.1: Turn off autologging

```python
import mlflow

# Disable all autologging
mlflow.autolog(disable=True)

# Or disable for specific framework
mlflow.sklearn.autolog(disable=True)
```

### Step 7.2: Temporarily disable

```python
import mlflow

# Enable autologging
mlflow.autolog()

# Train first model (logged)
model1 = RandomForestClassifier()
model1.fit(X_train, y_train)

# Disable autologging
mlflow.autolog(disable=True)

# Train second model (NOT logged)
model2 = RandomForestClassifier()
model2.fit(X_train, y_train)

# Re-enable autologging
mlflow.autolog(disable=False)

# Train third model (logged again)
model3 = RandomForestClassifier()
model3.fit(X_train, y_train)
```

---

## Part 8: Complete Example - Full Pipeline with Autolog

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("AUTOLOGGING DEMO - FULL PIPELINE")
print("="*60)

# =============================================
# STEP 1: Enable autologging
# =============================================
print("\n[1/6] Enabling autologging...")
mlflow.autolog()
print("      Autologging enabled!")

# =============================================
# STEP 2: Set experiment
# =============================================
print("\n[2/6] Setting up experiment...")
mlflow.set_experiment("autolog_full_demo")
print("      Experiment 'autolog_full_demo' ready")

# =============================================
# STEP 3: Load and prepare data
# =============================================
print("\n[3/6] Loading data...")
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"      Features: {X.shape[1]}")
print(f"      Training samples: {len(X_train)}")
print(f"      Test samples: {len(X_test)}")

# =============================================
# STEP 4: Train model (autologging captures this!)
# =============================================
print("\n[4/6] Training model (autologging active)...")

with mlflow.start_run(run_name="breast_cancer_rf"):

    # Define and train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("      Model trained!")

    # =============================================
    # STEP 5: Add custom metrics (manual additions)
    # =============================================
    print("\n[5/6] Adding custom metrics...")

    # Test set evaluation
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    mlflow.log_metric("test_accuracy", test_accuracy)
    print(f"      Test accuracy: {test_accuracy:.2%}")

    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5)
    mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
    mlflow.log_metric("cv_std", cv_scores.std())
    print(f"      CV accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

    # Custom tag
    mlflow.set_tag("model_quality", "production_ready" if test_accuracy > 0.95 else "needs_improvement")

    # =============================================
    # STEP 6: Save additional artifacts (manual)
    # =============================================
    print("\n[6/6] Saving additional artifacts...")

    # Feature importance plot
    importance = model.feature_importances_
    indices = np.argsort(importance)[-10:]  # Top 10 features

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [data.feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title("Top 10 Most Important Features")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()
    print("      Feature importance plot saved!")

    # Classification report
    report = classification_report(y_test, test_predictions)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")
    print("      Classification report saved!")

    # Get run ID
    run_id = mlflow.active_run().info.run_id

print("\n" + "="*60)
print("PIPELINE COMPLETE!")
print("="*60)

print(f"""
WHAT WAS LOGGED AUTOMATICALLY:
  - All RandomForest parameters (n_estimators, max_depth, etc.)
  - Training metrics (training_accuracy, training_f1, etc.)
  - The model artifact (ready to load and use)
  - Model signature

WHAT WE ADDED MANUALLY:
  - test_accuracy metric
  - cv_mean_accuracy, cv_std metrics
  - model_quality tag
  - feature_importance.png artifact
  - classification_report.txt artifact

TO VIEW RESULTS:
  Run: mlflow ui
  Open: http://127.0.0.1:5000
  Look for: experiment 'autolog_full_demo', run 'breast_cancer_rf'

Run ID: {run_id}
""")
```

---

## Summary: Key Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `mlflow.autolog()` | Enable autologging for all frameworks | `mlflow.autolog()` |
| `mlflow.sklearn.autolog()` | Enable for sklearn only | `mlflow.sklearn.autolog()` |
| `mlflow.autolog(disable=True)` | Disable autologging | `mlflow.autolog(disable=True)` |
| `mlflow.autolog(log_models=False)` | Don't save models | `mlflow.autolog(log_models=False)` |

---

## Supported Frameworks

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

Or just use `mlflow.autolog()` to enable for ALL frameworks at once!

---

## Common Mistakes

### Mistake 1: Calling autolog() after training

```python
# WRONG - autolog must be called BEFORE training
model.fit(X_train, y_train)
mlflow.autolog()  # Too late! Nothing was logged

# RIGHT - call autolog first
mlflow.autolog()
model.fit(X_train, y_train)  # Now it's logged
```

### Mistake 2: Forgetting to set experiment

```python
# Works, but goes to "Default" experiment
mlflow.autolog()
model.fit(X_train, y_train)

# Better - set your experiment first
mlflow.set_experiment("my_project")
mlflow.autolog()
model.fit(X_train, y_train)
```

### Mistake 3: Expecting test metrics to be logged

```python
mlflow.autolog()
model.fit(X_train, y_train)
# Autolog only captures TRAINING metrics!

# If you want test metrics, add them manually:
with mlflow.start_run():
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("test_accuracy", test_acc)  # Manual!
```

---

## When to Use Autolog vs Manual Logging

| Use Autolog When... | Use Manual Logging When... |
|---------------------|---------------------------|
| You want quick experiment tracking | You need custom metrics |
| You're comparing many models | You want specific artifact format |
| You're prototyping | You need precise control |
| You want all framework parameters | You only need specific params |
| Training = what you want to track | You have complex pipelines |

**Best practice:** Use both! Enable autolog for automatic tracking, then add manual logging for custom needs.
