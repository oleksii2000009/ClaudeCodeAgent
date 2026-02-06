"""
Asthma Prediction ML Pipeline with MLflow Tracking
===================================================
This script:
1. Explores and visualizes the asthma dataset
2. Trains multiple models (Random Forest, Logistic Regression, XGBoost)
3. Compares model performance with confusion matrices and ROC curves
4. Tracks all experiments with MLflow
5. Registers the best model in MLflow Model Registry
6. Analyzes feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import warnings
import os

warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "synthetic_asthma_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Will skip XGBoost model.")


def load_and_explore_data(data_path):
    """Load and perform initial data exploration."""
    print("=" * 60)
    print("1. DATA EXPLORATION")
    print("=" * 60)

    df = pd.read_csv(data_path)

    print(f"\nDataset Shape: {df.shape[0]} patients, {df.shape[1]} features")
    print(f"\nTarget Distribution (Has_Asthma):")
    print(df['Has_Asthma'].value_counts())
    print(f"\nClass Balance:")
    print(df['Has_Asthma'].value_counts(normalize=True).round(3))

    print(f"\nNumerical Features Statistics:")
    print(df.describe().round(2))

    return df


def create_visualizations(df, output_dir):
    """Create and save exploratory visualizations."""
    print("\n" + "=" * 60)
    print("2. DATA VISUALIZATIONS")
    print("=" * 60)

    # 1. Target Distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#3498db', '#e74c3c']
    counts = df['Has_Asthma'].value_counts()
    ax.bar(['No Asthma', 'Has Asthma'], counts.values, color=colors)
    ax.set_title('Distribution of Asthma Cases', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count')
    for i, v in enumerate(counts.values):
        ax.text(i, v + 100, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'), dpi=150)
    plt.close()
    print("[OK] Saved: target_distribution.png")

    # 2. Age Distribution by Asthma Status
    fig, ax = plt.subplots(figsize=(10, 6))
    for asthma_status, color, label in zip([0, 1], colors, ['No Asthma', 'Has Asthma']):
        subset = df[df['Has_Asthma'] == asthma_status]['Age']
        ax.hist(subset, bins=30, alpha=0.6, label=label, color=color)
    ax.set_title('Age Distribution by Asthma Status', fontsize=14, fontweight='bold')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'), dpi=150)
    plt.close()
    print("[OK] Saved: age_distribution.png")

    # 3. BMI vs Age colored by Asthma
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['Age'], df['BMI'], c=df['Has_Asthma'],
                        cmap='coolwarm', alpha=0.5, s=30)
    ax.set_title('BMI vs Age by Asthma Status', fontsize=14, fontweight='bold')
    ax.set_xlabel('Age')
    ax.set_ylabel('BMI')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Has Asthma')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bmi_vs_age.png'), dpi=150)
    plt.close()
    print("[OK] Saved: bmi_vs_age.png")

    # 4. Categorical Features Analysis
    cat_features = ['Gender', 'Smoking_Status', 'Allergies', 'Air_Pollution_Level',
                    'Physical_Activity_Level', 'Occupation_Type', 'Comorbidities']

    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()

    for idx, col in enumerate(cat_features):
        if idx < len(axes):
            asthma_rate = df.groupby(col)['Has_Asthma'].mean().sort_values(ascending=False)
            axes[idx].bar(range(len(asthma_rate)), asthma_rate.values, color='#3498db')
            axes[idx].set_title(f'Asthma Rate by {col}', fontsize=10, fontweight='bold')
            axes[idx].set_ylabel('Asthma Rate')
            axes[idx].set_xticks(range(len(asthma_rate)))
            axes[idx].set_xticklabels(asthma_rate.index, rotation=45, ha='right')
            axes[idx].set_ylim(0, 1)

    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'categorical_features.png'), dpi=150)
    plt.close()
    print("[OK] Saved: categorical_features.png")

    # 5. Correlation Heatmap (numeric features only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'Patient_ID']

    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = df[numeric_cols].corr()
    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(numeric_cols)))
    ax.set_yticks(np.arange(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
    ax.set_yticklabels(numeric_cols)

    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha='center', va='center', fontsize=8)

    plt.colorbar(im, ax=ax)
    ax.set_title('Correlation Heatmap of Numeric Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150)
    plt.close()
    print("[OK] Saved: correlation_heatmap.png")

    # 6. Clinical Measurements Distribution
    clinical_cols = ['Peak_Expiratory_Flow', 'FeNO_Level']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, col in enumerate(clinical_cols):
        for asthma_status, color, label in zip([0, 1], colors, ['No Asthma', 'Has Asthma']):
            subset = df[df['Has_Asthma'] == asthma_status][col]
            axes[idx].hist(subset, bins=30, alpha=0.6, label=label, color=color)
        axes[idx].set_title(f'{col} Distribution by Asthma Status', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clinical_measurements.png'), dpi=150)
    plt.close()
    print("[OK] Saved: clinical_measurements.png")

    print(f"\nAll visualizations saved to: {output_dir}")


def preprocess_data(df):
    """Preprocess data for ML models."""
    print("\n" + "=" * 60)
    print("3. DATA PREPROCESSING")
    print("=" * 60)

    df_processed = df.copy()

    cols_to_drop = ['Patient_ID', 'Asthma_Control_Level']
    df_processed = df_processed.drop(columns=cols_to_drop)

    cat_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns to encode: {cat_cols}")

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le

    X = df_processed.drop('Has_Asthma', axis=1)
    y = df_processed['Has_Asthma']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, X.columns.tolist()


def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Asthma', 'Has Asthma'])
    ax.set_yticklabels(['No Asthma', 'Has Asthma'])

    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha='center', va='center',
                          fontsize=20, fontweight='bold',
                          color='white' if cm[i, j] > cm.max()/2 else 'black')

    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    filepath = os.path.join(output_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath


def plot_roc_curve(y_true, y_prob, model_name, output_dir):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    filepath = os.path.join(output_dir, f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath, auc


def plot_all_roc_curves(results, y_test, output_dir):
    """Plot all ROC curves on a single plot for comparison."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    for idx, (model_name, result) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
        auc = result['metrics']['roc_auc']
        ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                label=f'{model_name} (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison - All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'roc_curves_comparison.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"[OK] Saved: roc_curves_comparison.png")
    return filepath


def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test,
                             feature_names, output_dir):
    """Train a model and evaluate its performance."""
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    metrics['cv_roc_auc_mean'] = cv_scores.mean()
    metrics['cv_roc_auc_std'] = cv_scores.std()

    cm_path = plot_confusion_matrix(y_test, y_pred, model_name, output_dir)
    roc_path, _ = plot_roc_curve(y_test, y_prob, model_name, output_dir)

    return {
        'model': model,
        'metrics': metrics,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'cm_path': cm_path,
        'roc_path': roc_path
    }


def plot_feature_importance(model, feature_names, model_name, output_dir):
    """Plot and save feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(importance_df)))
    ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    filepath = os.path.join(output_dir, f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(filepath, dpi=150)
    plt.close()

    return filepath, importance_df


def run_mlflow_experiment(X_train, X_test, y_train, y_test, X_train_scaled,
                          X_test_scaled, feature_names, output_dir):
    """Run the full ML experiment with MLflow tracking."""
    print("\n" + "=" * 60)
    print("4. MODEL TRAINING WITH MLFLOW TRACKING")
    print("=" * 60)

    mlflow.set_experiment("Asthma_Prediction")

    models = {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, random_state=42, C=1.0, solver='lbfgs'),
            {'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs'},
            X_train_scaled, X_test_scaled
        ),
        'Random Forest': (
            RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5,
                                   random_state=42, n_jobs=-1),
            {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
            X_train, X_test
        )
    }

    if XGBOOST_AVAILABLE:
        models['XGBoost'] = (
            XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                         random_state=42, use_label_encoder=False, eval_metric='logloss'),
            {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            X_train, X_test
        )

    results = {}
    run_ids = {}

    for model_name, (model, params, X_tr, X_te) in models.items():
        print(f"\n--- Training {model_name} ---")

        with mlflow.start_run(run_name=model_name) as run:
            run_ids[model_name] = run.info.run_id

            mlflow.log_params(params)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("train_samples", len(X_tr))
            mlflow.log_param("test_samples", len(X_te))
            mlflow.log_param("n_features", X_tr.shape[1] if hasattr(X_tr, 'shape') else len(X_tr[0]))

            result = train_and_evaluate_model(
                model, model_name, X_tr, X_te, y_train, y_test,
                feature_names, output_dir
            )
            results[model_name] = result

            mlflow.log_metrics(result['metrics'])

            mlflow.log_artifact(result['cm_path'])
            mlflow.log_artifact(result['roc_path'])

            fi_result = plot_feature_importance(model, feature_names, model_name, output_dir)
            if fi_result:
                fi_path, fi_df = fi_result
                mlflow.log_artifact(fi_path)
                fi_csv_path = os.path.join(output_dir, f'feature_importance_{model_name.lower().replace(" ", "_")}.csv')
                fi_df.to_csv(fi_csv_path, index=False)
                mlflow.log_artifact(fi_csv_path)

            mlflow.sklearn.log_model(model, "model")

            print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
            print(f"  Precision: {result['metrics']['precision']:.4f}")
            print(f"  Recall: {result['metrics']['recall']:.4f}")
            print(f"  F1 Score: {result['metrics']['f1']:.4f}")
            print(f"  ROC AUC: {result['metrics']['roc_auc']:.4f}")
            print(f"  CV ROC AUC: {result['metrics']['cv_roc_auc_mean']:.4f} +/- {result['metrics']['cv_roc_auc_std']:.4f}")
            print(f"  Run ID: {run.info.run_id}")

    plot_all_roc_curves(results, y_test, output_dir)

    return results, run_ids


def compare_models(results):
    """Create a comparison table of all models."""
    print("\n" + "=" * 60)
    print("5. MODEL COMPARISON")
    print("=" * 60)

    comparison_data = []
    for model_name, result in results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1 Score': f"{metrics['f1']:.4f}",
            'ROC AUC': f"{metrics['roc_auc']:.4f}",
            'CV ROC AUC': f"{metrics['cv_roc_auc_mean']:.4f} +/- {metrics['cv_roc_auc_std']:.4f}"
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))

    best_model = max(results.items(), key=lambda x: x[1]['metrics']['roc_auc'])
    print(f"\n*** Best Model: {best_model[0]} (ROC AUC: {best_model[1]['metrics']['roc_auc']:.4f}) ***")

    return comparison_df, best_model[0]


def register_best_model(best_model_name, results, run_ids):
    """Register the best model in MLflow Model Registry."""
    print("\n" + "=" * 60)
    print("6. REGISTERING BEST MODEL")
    print("=" * 60)

    run_id = run_ids[best_model_name]
    model_uri = f"runs:/{run_id}/model"
    model_name = "Asthma_Prediction_Model"

    try:
        registered_model = mlflow.register_model(model_uri, model_name)

        print(f"[OK] Model registered: {model_name}")
        print(f"  Version: {registered_model.version}")
        print(f"  Run ID: {run_id}")

        # Try to update description and stage (may fail on some MLflow versions)
        try:
            client = MlflowClient()
            client.set_model_version_tag(
                name=model_name,
                version=registered_model.version,
                key="best_model_type",
                value=best_model_name
            )
            client.set_model_version_tag(
                name=model_name,
                version=registered_model.version,
                key="roc_auc",
                value=str(results[best_model_name]['metrics']['roc_auc'])
            )
            print(f"[OK] Model tags added")
        except Exception as e:
            print(f"[WARN] Could not add model tags: {e}")

        return registered_model
    except Exception as e:
        print(f"[WARN] Model registration issue: {e}")
        print(f"  Model was logged to run: {run_id}")
        return None


def print_feature_importance_summary(results, feature_names):
    """Print feature importance summary for the best tree-based model."""
    print("\n" + "=" * 60)
    print("7. FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    for model_name in ['XGBoost', 'Random Forest']:
        if model_name in results:
            model = results[model_name]['model']
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)

                print(f"\nTop 10 Most Important Features ({model_name}):")
                print("-" * 45)
                for _, row in importance_df.head(10).iterrows():
                    bar = "#" * int(row['Importance'] * 50)
                    print(f"  {row['Feature']:<30} {row['Importance']:.4f} {bar}")
                break


def main():
    """Main function to run the entire pipeline."""
    print("\n" + "=" * 60)
    print("   ASTHMA PREDICTION ML PIPELINE")
    print("=" * 60)

    df = load_and_explore_data(DATA_PATH)
    create_visualizations(df, OUTPUT_DIR)

    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_names = preprocess_data(df)

    results, run_ids = run_mlflow_experiment(
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled,
        feature_names, OUTPUT_DIR
    )

    comparison_df, best_model_name = compare_models(results)

    comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)
    print(f"\n[OK] Model comparison saved to: {os.path.join(OUTPUT_DIR, 'model_comparison.csv')}")

    register_best_model(best_model_name, results, run_ids)

    print_feature_importance_summary(results, feature_names)

    print("\n" + "=" * 60)
    print("   PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print(f"MLflow UI: Run 'mlflow ui' to view experiments")
    print(f"*** Best Model: {best_model_name} ***")

    return results, comparison_df


if __name__ == "__main__":
    results, comparison_df = main()
