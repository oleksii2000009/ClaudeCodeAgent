"""
Parkinson's Disease Telemonitoring Prediction
==============================================
Predicts total UPDRS scores from voice biomarkers using MLflow for experiment tracking.

Key considerations:
- Patient-aware train/test split to prevent data leakage
- Multiple model approaches for comparison
- Uncertainty quantification for medical predictions
- Full experiment tracking for reproducibility
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# =============================================================================
# Configuration
# =============================================================================
DATA_PATH = Path(__file__).parent / "Parkinsons-Telemonitoring-ucirvine.csv"
EXPERIMENT_NAME = "parkinsons_telemonitoring_prediction"
RANDOM_STATE = 42

# =============================================================================
# Data Loading and Exploration
# =============================================================================
def load_and_explore_data():
    """Load dataset and perform exploratory analysis."""
    print("=" * 60)
    print("PARKINSON'S DISEASE TELEMONITORING PREDICTION")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    print(f"\n[DATA] Dataset Shape: {df.shape[0]} recordings, {df.shape[1]} features")
    print(f"[DATA] Unique Patients: {df['subject'].nunique()}")
    print(f"[DATA] Recordings per Patient: {df.groupby('subject').size().mean():.1f} (avg)")

    # Identify feature groups
    voice_features = [
        'jitter', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp',
        'shimmer', 'shimmer_db', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda',
        'nhr', 'hnr', 'rpde', 'dfa', 'ppe'
    ]

    target = 'total_updrs'

    print(f"\n[TARGET] Target Variable: {target}")
    print(f"   Mean: {df[target].mean():.2f}")
    print(f"   Std:  {df[target].std():.2f}")
    print(f"   Range: [{df[target].min():.2f}, {df[target].max():.2f}]")

    return df, voice_features, target


def analyze_feature_correlations(df, voice_features, target):
    """Analyze correlations between voice features and UPDRS."""
    print("\n" + "=" * 60)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 60)

    correlations = df[voice_features + [target]].corr()[target].drop(target).sort_values(key=abs, ascending=False)

    print("\n[CORR] Top Voice Features Correlated with UPDRS Severity:")
    for feat, corr in correlations.head(10).items():
        direction = "(+)" if corr > 0 else "(-)"
        print(f"   {feat:20s}: {corr:+.4f} {direction}")

    return correlations


# =============================================================================
# Patient-Aware Data Splitting
# =============================================================================
def prepare_patient_split(df, voice_features, target, n_splits=5):
    """
    Create patient-aware train/test split to prevent data leakage.
    Same patient should NOT appear in both train and test sets.
    """
    X = df[voice_features].values
    y = df[target].values
    groups = df['subject'].values

    # Use GroupKFold for proper splitting
    group_kfold = GroupKFold(n_splits=n_splits)

    # Get the last fold for final test set
    splits = list(group_kfold.split(X, y, groups))
    train_idx, test_idx = splits[-1]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "=" * 60)
    print("PATIENT-AWARE DATA SPLIT")
    print("=" * 60)
    print(f"   Training: {len(X_train)} samples from {len(np.unique(groups_train))} patients")
    print(f"   Test:     {len(X_test)} samples from {len(np.unique(groups[test_idx]))} patients")
    print("   [OK] No patient overlap between train and test sets")

    return X_train_scaled, X_test_scaled, y_train, y_test, groups_train, scaler


# =============================================================================
# Model Training with MLflow Tracking
# =============================================================================
def train_and_evaluate_models(X_train, X_test, y_train, y_test, groups_train, voice_features):
    """Train multiple models and track with MLflow."""

    print("\n" + "=" * 60)
    print("MODEL TRAINING WITH MLFLOW TRACKING")
    print("=" * 60)

    # Set up MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Define models to try
    models = {
        "Ridge Regression": {
            "model": Ridge(alpha=1.0, random_state=RANDOM_STATE),
            "params": {"alpha": 1.0, "model_type": "linear"}
        },
        "ElasticNet": {
            "model": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=2000),
            "params": {"alpha": 0.1, "l1_ratio": 0.5, "model_type": "linear"}
        },
        "Random Forest": {
            "model": RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE),
            "params": {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 5, "model_type": "ensemble"}
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE),
            "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "model_type": "ensemble"}
        }
    }

    results = {}
    best_model = None
    best_rmse = float('inf')
    best_run_id = None

    for name, config in models.items():
        print(f"\n[TRAIN] Training {name}...")

        with mlflow.start_run(run_name=name) as run:
            model = config["model"]

            # Log parameters
            mlflow.log_params(config["params"])
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("n_features", len(voice_features))
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("patient_aware_split", True)

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Calculate metrics
            metrics = {
                "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                "train_mae": mean_absolute_error(y_train, y_pred_train),
                "test_mae": mean_absolute_error(y_test, y_pred_test),
                "train_r2": r2_score(y_train, y_pred_train),
                "test_r2": r2_score(y_test, y_pred_test),
            }

            # Calculate prediction intervals for uncertainty
            residuals = y_test - y_pred_test
            metrics["residual_std"] = np.std(residuals)
            metrics["residual_mean"] = np.mean(residuals)

            # Log all metrics
            mlflow.log_metrics(metrics)

            # Log model with signature
            signature = infer_signature(X_train[:5], y_pred_train[:5])
            mlflow.sklearn.log_model(model, "model", signature=signature)

            # Store results
            results[name] = {
                "model": model,
                "metrics": metrics,
                "predictions": y_pred_test,
                "residuals": residuals,
                "run_id": run.info.run_id
            }

            print(f"   Test RMSE: {metrics['test_rmse']:.3f}")
            print(f"   Test R2:   {metrics['test_r2']:.3f}")
            print(f"   Run ID:    {run.info.run_id}")

            # Track best model
            if metrics['test_rmse'] < best_rmse:
                best_rmse = metrics['test_rmse']
                best_model = name
                best_run_id = run.info.run_id

    print(f"\n[BEST] Best Model: {best_model} (RMSE: {best_rmse:.3f})")

    return results, best_model, best_run_id


# =============================================================================
# Feature Importance Analysis
# =============================================================================
def analyze_feature_importance(results, voice_features):
    """Analyze which voice features are most predictive."""

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    importance_data = {}

    # Get importance from Random Forest
    rf_model = results["Random Forest"]["model"]
    rf_importance = pd.Series(rf_model.feature_importances_, index=voice_features).sort_values(ascending=False)
    importance_data["Random Forest"] = rf_importance

    # Get importance from Gradient Boosting
    gb_model = results["Gradient Boosting"]["model"]
    gb_importance = pd.Series(gb_model.feature_importances_, index=voice_features).sort_values(ascending=False)
    importance_data["Gradient Boosting"] = gb_importance

    # Get coefficients from Ridge
    ridge_model = results["Ridge Regression"]["model"]
    ridge_importance = pd.Series(np.abs(ridge_model.coef_), index=voice_features).sort_values(ascending=False)
    importance_data["Ridge (|coef|)"] = ridge_importance

    print("\n[IMP] Top 5 Most Predictive Voice Features:")
    print("-" * 50)

    # Average importance across models
    avg_importance = (rf_importance / rf_importance.sum() +
                     gb_importance / gb_importance.sum() +
                     ridge_importance / ridge_importance.sum()) / 3
    avg_importance = avg_importance.sort_values(ascending=False)

    for i, (feat, imp) in enumerate(avg_importance.head(5).items(), 1):
        print(f"   {i}. {feat:20s} (avg normalized importance: {imp:.4f})")

    return importance_data, avg_importance


# =============================================================================
# Uncertainty and Error Analysis
# =============================================================================
def analyze_uncertainty(results, y_test, best_model):
    """Analyze model uncertainty and error distributions."""

    print("\n" + "=" * 60)
    print("UNCERTAINTY AND ERROR ANALYSIS")
    print("=" * 60)

    best_residuals = results[best_model]["residuals"]
    best_predictions = results[best_model]["predictions"]

    # Statistical tests
    _, normality_p = stats.shapiro(best_residuals[:min(500, len(best_residuals))])

    print(f"\n[ANALYSIS] {best_model} Error Analysis:")
    print(f"   Mean Absolute Error: {np.mean(np.abs(best_residuals)):.3f} UPDRS points")
    print(f"   Residual Std Dev:    {np.std(best_residuals):.3f} UPDRS points")
    print(f"   95% Prediction Interval: ±{1.96 * np.std(best_residuals):.3f} UPDRS points")

    # Error percentiles
    abs_errors = np.abs(best_residuals)
    print(f"\n   Error Percentiles:")
    print(f"   50th percentile: {np.percentile(abs_errors, 50):.3f}")
    print(f"   75th percentile: {np.percentile(abs_errors, 75):.3f}")
    print(f"   90th percentile: {np.percentile(abs_errors, 90):.3f}")
    print(f"   95th percentile: {np.percentile(abs_errors, 95):.3f}")

    # Normality of residuals
    print(f"\n   Residual Normality (Shapiro-Wilk p-value): {normality_p:.4f}")
    if normality_p > 0.05:
        print("   [OK] Residuals appear normally distributed (good for confidence intervals)")
    else:
        print("   [NOTE] Residuals may not be normally distributed (consider bootstrap intervals)")

    return best_residuals, best_predictions


# =============================================================================
# Visualization
# =============================================================================
def create_visualizations(df, results, y_test, voice_features, importance_data, best_model, correlations):
    """Create and save all visualizations."""

    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    output_dir = Path(__file__).parent

    # 1. Model Comparison Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Prediction vs Actual for all models
    ax = axes[0, 0]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    for i, (name, res) in enumerate(results.items()):
        ax.scatter(y_test, res["predictions"], alpha=0.5, label=name, c=colors[i], s=20)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect')
    ax.set_xlabel('Actual Total UPDRS', fontsize=12)
    ax.set_ylabel('Predicted Total UPDRS', fontsize=12)
    ax.set_title('Prediction Accuracy: All Models', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Residual Distribution for best model
    ax = axes[0, 1]
    best_residuals = results[best_model]["residuals"]
    ax.hist(best_residuals, bins=30, edgecolor='black', alpha=0.7, color='#3498db')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(x=np.mean(best_residuals), color='green', linestyle='-', linewidth=2, label=f'Mean: {np.mean(best_residuals):.2f}')
    ax.set_xlabel('Residual (Actual - Predicted)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Residual Distribution: {best_model}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Model Performance Comparison
    ax = axes[1, 0]
    model_names = list(results.keys())
    test_rmse = [results[m]["metrics"]["test_rmse"] for m in model_names]
    test_r2 = [results[m]["metrics"]["test_r2"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, test_rmse, width, label='Test RMSE', color='#e74c3c')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, test_r2, width, label='Test R2', color='#2ecc71')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('RMSE', color='#e74c3c', fontsize=12)
    ax2.set_ylabel('R2', color='#2ecc71', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=10)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # Feature Importance
    ax = axes[1, 1]
    rf_importance = importance_data["Random Forest"].head(10)
    colors_imp = plt.cm.viridis(np.linspace(0.2, 0.8, len(rf_importance)))
    bars = ax.barh(range(len(rf_importance)), rf_importance.values, color=colors_imp)
    ax.set_yticks(range(len(rf_importance)))
    ax.set_yticklabels(rf_importance.index, fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Top 10 Predictive Voice Features (Random Forest)', fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_dir / 'model_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved: model_analysis.png")

    # 2. Detailed Residual Analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    best_preds = results[best_model]["predictions"]

    # Residuals vs Predicted
    ax = axes[0, 0]
    ax.scatter(best_preds, best_residuals, alpha=0.5, c='#3498db', s=20)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.axhline(y=1.96*np.std(best_residuals), color='orange', linestyle=':', linewidth=2, label='95% CI')
    ax.axhline(y=-1.96*np.std(best_residuals), color='orange', linestyle=':', linewidth=2)
    ax.set_xlabel('Predicted Total UPDRS', fontsize=12)
    ax.set_ylabel('Residual', fontsize=12)
    ax.set_title('Residuals vs Predicted Values', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Q-Q Plot
    ax = axes[0, 1]
    stats.probplot(best_residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot of Residuals', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Absolute Error by UPDRS Range
    ax = axes[1, 0]
    abs_errors = np.abs(best_residuals)
    updrs_bins = pd.cut(y_test, bins=5)
    error_by_bin = pd.DataFrame({'error': abs_errors, 'bin': updrs_bins}).groupby('bin')['error'].mean()
    ax.bar(range(len(error_by_bin)), error_by_bin.values, color='#9b59b6', edgecolor='black')
    ax.set_xticks(range(len(error_by_bin)))
    ax.set_xticklabels([f'{int(b.left)}-{int(b.right)}' for b in error_by_bin.index], fontsize=10)
    ax.set_xlabel('UPDRS Score Range', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Prediction Error by Disease Severity', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Prediction Interval Coverage
    ax = axes[1, 1]
    std_res = np.std(best_residuals)
    coverage_90 = np.mean(np.abs(best_residuals) <= 1.645 * std_res) * 100
    coverage_95 = np.mean(np.abs(best_residuals) <= 1.96 * std_res) * 100
    coverage_99 = np.mean(np.abs(best_residuals) <= 2.576 * std_res) * 100

    coverages = [coverage_90, coverage_95, coverage_99]
    expected = [90, 95, 99]
    labels = ['90%', '95%', '99%']

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, expected, width, label='Expected', color='#95a5a6', edgecolor='black')
    ax.bar(x + width/2, coverages, width, label='Actual', color='#2ecc71', edgecolor='black')
    ax.set_xlabel('Confidence Level', fontsize=12)
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Prediction Interval Coverage', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(output_dir / 'residual_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved: residual_analysis.png")

    # 3. Feature Correlation Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = df[voice_features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Voice Feature Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved: feature_correlations.png")

    return ['model_analysis.png', 'residual_analysis.png', 'feature_correlations.png']


# =============================================================================
# Log Artifacts and Register Best Model
# =============================================================================
def log_artifacts_and_register_model(results, best_model, best_run_id, plots, voice_features, scaler):
    """Log visualizations as artifacts and register the best model."""

    print("\n" + "=" * 60)
    print("LOGGING ARTIFACTS AND REGISTERING MODEL")
    print("=" * 60)

    output_dir = Path(__file__).parent
    client = MlflowClient()

    # Log artifacts to the best run
    with mlflow.start_run(run_id=best_run_id):
        # Log plots
        for plot in plots:
            mlflow.log_artifact(str(output_dir / plot))
            print(f"   [OK] Logged artifact: {plot}")

        # Log feature names
        mlflow.log_text("\n".join(voice_features), "feature_names.txt")
        print("   [OK] Logged feature names")

        # Log model summary
        summary = f"""
Parkinson's Telemonitoring Model Summary
========================================
Best Model: {best_model}
Test RMSE: {results[best_model]['metrics']['test_rmse']:.4f}
Test R2: {results[best_model]['metrics']['test_r2']:.4f}
Test MAE: {results[best_model]['metrics']['test_mae']:.4f}

95% Prediction Interval: ±{1.96 * results[best_model]['metrics']['residual_std']:.3f} UPDRS points

Features Used: {len(voice_features)}
Patient-Aware Split: Yes (no data leakage)

Top Predictive Features:
"""
        if best_model in ["Random Forest", "Gradient Boosting"]:
            model = results[best_model]["model"]
            importance = pd.Series(model.feature_importances_, index=voice_features).sort_values(ascending=False)
            for feat, imp in importance.head(5).items():
                summary += f"  - {feat}: {imp:.4f}\n"

        mlflow.log_text(summary, "model_summary.txt")
        print("   [OK] Logged model summary")

    # Register the best model
    model_name = "parkinsons_updrs_predictor"
    model_uri = f"runs:/{best_run_id}/model"

    try:
        registered_model = mlflow.register_model(model_uri, model_name)
        print(f"\n[MODEL] Model registered: {model_name}")
        print(f"   Version: {registered_model.version}")

        # Add description
        client.update_model_version(
            name=model_name,
            version=registered_model.version,
            description=f"Parkinson's UPDRS predictor using {best_model}. RMSE: {results[best_model]['metrics']['test_rmse']:.3f}"
        )

        # Add tags
        client.set_model_version_tag(model_name, registered_model.version, "task", "regression")
        client.set_model_version_tag(model_name, registered_model.version, "domain", "medical")
        client.set_model_version_tag(model_name, registered_model.version, "algorithm", best_model)

        print("   [OK] Model metadata updated")

        return registered_model.version

    except Exception as e:
        print(f"   [NOTE] Model registration note: {e}")
        return None


# =============================================================================
# Main Execution
# =============================================================================
def main():
    """Main execution pipeline."""

    # 1. Load and explore data
    df, voice_features, target = load_and_explore_data()

    # 2. Analyze feature correlations
    correlations = analyze_feature_correlations(df, voice_features, target)

    # 3. Create patient-aware split
    X_train, X_test, y_train, y_test, groups_train, scaler = prepare_patient_split(
        df, voice_features, target
    )

    # 4. Train and evaluate models with MLflow tracking
    results, best_model, best_run_id = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, groups_train, voice_features
    )

    # 5. Analyze feature importance
    importance_data, avg_importance = analyze_feature_importance(results, voice_features)

    # 6. Analyze uncertainty
    best_residuals, best_predictions = analyze_uncertainty(results, y_test, best_model)

    # 7. Create visualizations
    plots = create_visualizations(
        df, results, y_test, voice_features, importance_data, best_model, correlations
    )

    # 8. Log artifacts and register model
    model_version = log_artifacts_and_register_model(
        results, best_model, best_run_id, plots, voice_features, scaler
    )

    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\n[WINNER] Best Model: {best_model}")
    print(f"[METRIC] Test RMSE: {results[best_model]['metrics']['test_rmse']:.3f} UPDRS points")
    print(f"[R2] Test R2: {results[best_model]['metrics']['test_r2']:.3f}")
    print(f"[CI] 95% CI: ±{1.96 * results[best_model]['metrics']['residual_std']:.3f} UPDRS points")
    print(f"\n[VIEW] View experiments: mlflow ui")
    print(f"[MODEL] Registered model: parkinsons_updrs_predictor (v{model_version})")

    return results, best_model


if __name__ == "__main__":
    results, best_model = main()
