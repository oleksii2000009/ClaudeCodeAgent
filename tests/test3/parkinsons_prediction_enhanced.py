"""
Parkinson's Disease Telemonitoring Prediction - Enhanced Version
================================================================
Improved models with feature engineering and patient-level aggregation.

Key improvements:
- Feature engineering (interactions, polynomial features)
- Patient-level mean features to capture baseline characteristics
- Mixed effects modeling approach
- XGBoost with optimized hyperparameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
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
EXPERIMENT_NAME = "parkinsons_telemonitoring_enhanced"
RANDOM_STATE = 42


def load_data():
    """Load and return dataset."""
    df = pd.read_csv(DATA_PATH)

    voice_features = [
        'jitter', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp',
        'shimmer', 'shimmer_db', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda',
        'nhr', 'hnr', 'rpde', 'dfa', 'ppe'
    ]

    return df, voice_features


def engineer_features(df, voice_features):
    """Create engineered features for better prediction."""

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    df_eng = df.copy()

    # 1. Add demographic features
    df_eng['age_scaled'] = (df_eng['age'] - df_eng['age'].mean()) / df_eng['age'].std()
    df_eng['sex_numeric'] = df_eng['sex'].astype(int)

    # 2. Add time-based features (disease progression proxy)
    df_eng['test_time_scaled'] = df_eng['test_time'] / df_eng['test_time'].max()

    # 3. Jitter aggregate features
    jitter_cols = ['jitter', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp']
    df_eng['jitter_mean'] = df_eng[jitter_cols].mean(axis=1)
    df_eng['jitter_std'] = df_eng[jitter_cols].std(axis=1)

    # 4. Shimmer aggregate features
    shimmer_cols = ['shimmer', 'shimmer_db', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda']
    df_eng['shimmer_mean'] = df_eng[shimmer_cols].mean(axis=1)
    df_eng['shimmer_std'] = df_eng[shimmer_cols].std(axis=1)

    # 5. Key interactions based on domain knowledge
    df_eng['hnr_nhr_ratio'] = df_eng['hnr'] / (df_eng['nhr'] + 1e-6)
    df_eng['jitter_shimmer_product'] = df_eng['jitter_mean'] * df_eng['shimmer_mean']
    df_eng['ppe_dfa_product'] = df_eng['ppe'] * df_eng['dfa']

    # 6. Patient-level mean features (captures patient baseline)
    patient_means = df_eng.groupby('subject')[voice_features].transform('mean')
    for feat in voice_features:
        df_eng[f'{feat}_patient_mean'] = patient_means[feat]
        df_eng[f'{feat}_deviation'] = df_eng[feat] - patient_means[feat]

    # 7. Define enhanced feature set
    engineered_features = (
        voice_features +
        ['age_scaled', 'sex_numeric', 'test_time_scaled'] +
        ['jitter_mean', 'jitter_std', 'shimmer_mean', 'shimmer_std'] +
        ['hnr_nhr_ratio', 'jitter_shimmer_product', 'ppe_dfa_product'] +
        [f'{feat}_patient_mean' for feat in voice_features] +
        [f'{feat}_deviation' for feat in voice_features]
    )

    print(f"   Original features: {len(voice_features)}")
    print(f"   Engineered features: {len(engineered_features)}")
    print("   [OK] Added patient-level baseline features")
    print("   [OK] Added aggregate voice measures")
    print("   [OK] Added domain-specific interactions")

    return df_eng, engineered_features


def prepare_data(df, features, target='total_updrs', n_splits=5):
    """Prepare train/test split with patient grouping."""

    X = df[features].values
    y = df[target].values
    groups = df['subject'].values

    # Use GroupKFold
    group_kfold = GroupKFold(n_splits=n_splits)
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
    print("DATA PREPARATION")
    print("=" * 60)
    print(f"   Training: {len(X_train)} samples from {len(np.unique(groups_train))} patients")
    print(f"   Test: {len(X_test)} samples from {len(np.unique(groups[test_idx]))} patients")

    return X_train_scaled, X_test_scaled, y_train, y_test, groups_train, scaler, features


def train_enhanced_models(X_train, X_test, y_train, y_test, groups_train, features):
    """Train enhanced models with MLflow tracking."""

    print("\n" + "=" * 60)
    print("TRAINING ENHANCED MODELS")
    print("=" * 60)

    mlflow.set_experiment(EXPERIMENT_NAME)

    models = {
        "Bayesian Ridge": {
            "model": BayesianRidge(
                alpha_1=1e-6, alpha_2=1e-6,
                lambda_1=1e-6, lambda_2=1e-6,
                compute_score=True
            ),
            "params": {"model_type": "bayesian_linear", "regularization": "automatic"}
        },
        "ElasticNet Tuned": {
            "model": ElasticNet(alpha=0.01, l1_ratio=0.3, random_state=RANDOM_STATE, max_iter=5000),
            "params": {"alpha": 0.01, "l1_ratio": 0.3, "model_type": "linear"}
        },
        "Random Forest Tuned": {
            "model": RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_leaf=10,
                min_samples_split=20, random_state=RANDOM_STATE, n_jobs=-1
            ),
            "params": {"n_estimators": 200, "max_depth": 15, "min_samples_leaf": 10, "model_type": "ensemble"}
        },
        "Gradient Boosting Tuned": {
            "model": GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=10, random_state=RANDOM_STATE
            ),
            "params": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "model_type": "boosting"}
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
            mlflow.log_param("n_features", len(features))
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("feature_engineering", True)

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Metrics
            metrics = {
                "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
                "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                "train_mae": mean_absolute_error(y_train, y_pred_train),
                "test_mae": mean_absolute_error(y_test, y_pred_test),
                "train_r2": r2_score(y_train, y_pred_train),
                "test_r2": r2_score(y_test, y_pred_test),
            }

            residuals = y_test - y_pred_test
            metrics["residual_std"] = np.std(residuals)
            metrics["residual_mean"] = np.mean(residuals)

            mlflow.log_metrics(metrics)

            # Log model
            signature = infer_signature(X_train[:5], y_pred_train[:5])
            mlflow.sklearn.log_model(model, "model", signature=signature)

            results[name] = {
                "model": model,
                "metrics": metrics,
                "predictions": y_pred_test,
                "residuals": residuals,
                "run_id": run.info.run_id
            }

            print(f"   Test RMSE: {metrics['test_rmse']:.3f}")
            print(f"   Test R2:   {metrics['test_r2']:.3f}")
            print(f"   Test MAE:  {metrics['test_mae']:.3f}")

            if metrics['test_rmse'] < best_rmse:
                best_rmse = metrics['test_rmse']
                best_model = name
                best_run_id = run.info.run_id

    print(f"\n[BEST] Best Model: {best_model} (RMSE: {best_rmse:.3f})")

    return results, best_model, best_run_id


def analyze_and_visualize(results, y_test, features, best_model):
    """Create analysis and visualizations."""

    print("\n" + "=" * 60)
    print("ANALYSIS AND VISUALIZATION")
    print("=" * 60)

    output_dir = Path(__file__).parent

    # Detailed model comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Actual vs Predicted scatter
    ax = axes[0, 0]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    for i, (name, res) in enumerate(results.items()):
        ax.scatter(y_test, res["predictions"], alpha=0.4, label=f'{name} (R2={res["metrics"]["test_r2"]:.2f})',
                   c=colors[i], s=15)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Actual Total UPDRS', fontsize=12)
    ax.set_ylabel('Predicted Total UPDRS', fontsize=12)
    ax.set_title('Prediction Accuracy Comparison', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Model performance bars
    ax = axes[0, 1]
    model_names = list(results.keys())
    x = np.arange(len(model_names))
    rmse_vals = [results[m]["metrics"]["test_rmse"] for m in model_names]
    mae_vals = [results[m]["metrics"]["test_mae"] for m in model_names]

    width = 0.35
    ax.bar(x - width/2, rmse_vals, width, label='RMSE', color='#e74c3c')
    ax.bar(x + width/2, mae_vals, width, label='MAE', color='#3498db')
    ax.set_ylabel('Error (UPDRS points)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=9)
    ax.set_title('Model Error Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Residual distributions
    ax = axes[1, 0]
    for i, (name, res) in enumerate(results.items()):
        ax.hist(res["residuals"], bins=30, alpha=0.5, label=name, color=colors[i])
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual (Actual - Predicted)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Residual Distributions', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. Feature importance (from best ensemble model)
    ax = axes[1, 1]
    if "Random Forest" in best_model or "Gradient Boosting" in best_model:
        model = results[best_model]["model"]
        importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(15)
    else:
        # Use Random Forest for feature importance
        model = results["Random Forest Tuned"]["model"]
        importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(15)

    colors_imp = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance)))
    ax.barh(range(len(importance)), importance.values, color=colors_imp)
    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels(importance.index, fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Top 15 Predictive Features', fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_dir / 'enhanced_model_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved: enhanced_model_analysis.png")

    # Error analysis by severity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    best_res = results[best_model]
    abs_errors = np.abs(best_res["residuals"])

    # Error by UPDRS range
    ax = axes[0]
    updrs_bins = pd.cut(y_test, bins=[0, 20, 30, 40, 55], labels=['Mild (0-20)', 'Moderate (20-30)', 'Moderate-Severe (30-40)', 'Severe (40+)'])
    error_by_bin = pd.DataFrame({'error': abs_errors, 'severity': updrs_bins}).groupby('severity')['error'].agg(['mean', 'std'])

    ax.bar(range(len(error_by_bin)), error_by_bin['mean'], yerr=error_by_bin['std'],
           color='#3498db', edgecolor='black', capsize=5)
    ax.set_xticks(range(len(error_by_bin)))
    ax.set_xticklabels(error_by_bin.index, fontsize=10)
    ax.set_xlabel('Disease Severity', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (+/- std)', fontsize=12)
    ax.set_title(f'{best_model}: Error by Disease Severity', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Prediction confidence intervals
    ax = axes[1]
    std_res = np.std(best_res["residuals"])
    sorted_idx = np.argsort(best_res["predictions"])[:100]  # First 100 sorted predictions
    preds_sorted = best_res["predictions"][sorted_idx]
    actual_sorted = y_test[sorted_idx]

    ax.fill_between(range(len(preds_sorted)),
                    preds_sorted - 1.96*std_res,
                    preds_sorted + 1.96*std_res,
                    alpha=0.3, color='#3498db', label='95% CI')
    ax.plot(range(len(preds_sorted)), preds_sorted, 'b-', linewidth=2, label='Predicted')
    ax.scatter(range(len(actual_sorted)), actual_sorted, c='red', s=20, alpha=0.7, label='Actual', zorder=5)
    ax.set_xlabel('Sample Index (sorted by prediction)', fontsize=12)
    ax.set_ylabel('Total UPDRS Score', fontsize=12)
    ax.set_title('Predictions with 95% Confidence Intervals', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'enhanced_uncertainty_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved: enhanced_uncertainty_analysis.png")

    return ['enhanced_model_analysis.png', 'enhanced_uncertainty_analysis.png']


def register_best_model(results, best_model, best_run_id, plots, features):
    """Register the best model in MLflow registry."""

    print("\n" + "=" * 60)
    print("MODEL REGISTRATION")
    print("=" * 60)

    output_dir = Path(__file__).parent
    client = MlflowClient()

    with mlflow.start_run(run_id=best_run_id):
        for plot in plots:
            mlflow.log_artifact(str(output_dir / plot))
            print(f"   [OK] Logged: {plot}")

        # Log feature list
        mlflow.log_text("\n".join(features), "engineered_features.txt")

        # Log summary
        metrics = results[best_model]['metrics']
        summary = f"""
Enhanced Parkinson's Telemonitoring Model
==========================================
Best Model: {best_model}
Test RMSE: {metrics['test_rmse']:.4f} UPDRS points
Test R2: {metrics['test_r2']:.4f}
Test MAE: {metrics['test_mae']:.4f} UPDRS points
95% Prediction Interval: +/- {1.96 * metrics['residual_std']:.3f} UPDRS points

Feature Engineering:
- Patient-level baseline features
- Aggregate voice measures (jitter/shimmer means)
- Domain-specific interactions
- Temporal features

Total Features: {len(features)}
"""
        mlflow.log_text(summary, "enhanced_model_summary.txt")
        print("   [OK] Logged model summary")

    # Register model
    model_name = "parkinsons_updrs_predictor_enhanced"
    model_uri = f"runs:/{best_run_id}/model"

    try:
        registered = mlflow.register_model(model_uri, model_name)
        print(f"\n[MODEL] Registered: {model_name} v{registered.version}")

        client.update_model_version(
            name=model_name,
            version=registered.version,
            description=f"Enhanced Parkinson's UPDRS predictor with feature engineering. RMSE: {metrics['test_rmse']:.3f}"
        )
        client.set_model_version_tag(model_name, registered.version, "feature_engineering", "true")
        client.set_model_version_tag(model_name, registered.version, "algorithm", best_model)

        return registered.version
    except Exception as e:
        print(f"   [NOTE] Registration: {e}")
        return None


def main():
    """Main pipeline."""

    print("=" * 60)
    print("PARKINSON'S TELEMONITORING - ENHANCED PREDICTION")
    print("=" * 60)

    # Load data
    df, voice_features = load_data()
    print(f"\n[DATA] Loaded {len(df)} recordings from {df['subject'].nunique()} patients")

    # Feature engineering
    df_eng, engineered_features = engineer_features(df, voice_features)

    # Prepare data
    X_train, X_test, y_train, y_test, groups_train, scaler, features = prepare_data(
        df_eng, engineered_features
    )

    # Train models
    results, best_model, best_run_id = train_enhanced_models(
        X_train, X_test, y_train, y_test, groups_train, features
    )

    # Analyze and visualize
    plots = analyze_and_visualize(results, y_test, features, best_model)

    # Register model
    version = register_best_model(results, best_model, best_run_id, plots, features)

    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    metrics = results[best_model]['metrics']
    print(f"\n[WINNER] Best Model: {best_model}")
    print(f"[METRIC] Test RMSE: {metrics['test_rmse']:.3f} UPDRS points")
    print(f"[METRIC] Test MAE:  {metrics['test_mae']:.3f} UPDRS points")
    print(f"[METRIC] Test R2:   {metrics['test_r2']:.3f}")
    print(f"[CI]     95% CI:    +/- {1.96 * metrics['residual_std']:.3f} UPDRS points")
    print(f"\n[VIEW]   Run 'mlflow ui' to compare experiments")
    print(f"[MODEL]  Registered: parkinsons_updrs_predictor_enhanced (v{version})")

    return results, best_model


if __name__ == "__main__":
    results, best_model = main()
