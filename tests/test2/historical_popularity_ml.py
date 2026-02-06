"""
Historical Popularity Prediction Pipeline
==========================================
Predicts historical_popularity_index using Wikipedia stats and demographic features.
All experiments tracked with MLflow for reproducibility and comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
from mlflow.tracking import MlflowClient
import warnings
import os

warnings.filterwarnings('ignore')

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "database.csv")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Configure MLflow
mlflow.set_tracking_uri(f"file:///{os.path.join(SCRIPT_DIR, 'mlruns').replace(os.sep, '/')}")
mlflow.set_experiment("historical_popularity_prediction")


def load_and_explore_data():
    """Load dataset and perform exploratory analysis."""
    print("=" * 60)
    print("STEP 1: DATA EXPLORATION")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    # Clean numeric columns - handle values like '1237?'
    numeric_cols = ['birth_year', 'latitude', 'longitude', 'article_languages',
                    'page_views', 'average_views', 'historical_popularity_index']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nTarget Variable: historical_popularity_index")
    print(f"  Mean: {df['historical_popularity_index'].mean():.2f}")
    print(f"  Std:  {df['historical_popularity_index'].std():.2f}")
    print(f"  Min:  {df['historical_popularity_index'].min():.2f}")
    print(f"  Max:  {df['historical_popularity_index'].max():.2f}")

    print("\n--- Interesting Patterns ---")

    # Top 10 most popular
    print("\nTop 10 Most Popular Historical Figures:")
    top10 = df.nlargest(10, 'historical_popularity_index')[['full_name', 'occupation', 'historical_popularity_index']]
    print(top10.to_string(index=False))

    # Popularity by domain
    print("\nAverage Popularity by Domain:")
    domain_pop = df.groupby('domain')['historical_popularity_index'].mean().sort_values(ascending=False)
    print(domain_pop.head(10).to_string())

    # Popularity by continent
    print("\nAverage Popularity by Continent:")
    continent_pop = df.groupby('continent')['historical_popularity_index'].mean().sort_values(ascending=False)
    print(continent_pop.to_string())

    # Gender distribution
    print("\nPopularity by Gender:")
    gender_pop = df.groupby('sex')['historical_popularity_index'].agg(['mean', 'count'])
    print(gender_pop.to_string())

    return df


def create_visualizations(df):
    """Create and save exploratory visualizations."""
    print("\n" + "=" * 60)
    print("STEP 2: CREATING VISUALIZATIONS")
    print("=" * 60)

    fig_paths = []

    # 1. Distribution of target variable
    plt.figure(figsize=(10, 6))
    plt.hist(df['historical_popularity_index'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Historical Popularity Index')
    plt.ylabel('Frequency')
    plt.title('Distribution of Historical Popularity Index')
    path = os.path.join(ARTIFACTS_DIR, "01_target_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    fig_paths.append(path)
    print(f"  Saved: 01_target_distribution.png")

    # 2. Popularity by Domain
    plt.figure(figsize=(12, 6))
    domain_means = df.groupby('domain')['historical_popularity_index'].mean().sort_values(ascending=True)
    domain_means.plot(kind='barh', color='steelblue')
    plt.xlabel('Average Historical Popularity Index')
    plt.title('Average Popularity by Domain')
    plt.tight_layout()
    path = os.path.join(ARTIFACTS_DIR, "02_popularity_by_domain.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    fig_paths.append(path)
    print(f"  Saved: 02_popularity_by_domain.png")

    # 3. Popularity by Continent
    plt.figure(figsize=(10, 6))
    continent_means = df.groupby('continent')['historical_popularity_index'].mean().sort_values(ascending=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(continent_means)))
    continent_means.plot(kind='barh', color=colors)
    plt.xlabel('Average Historical Popularity Index')
    plt.title('Average Popularity by Continent')
    plt.tight_layout()
    path = os.path.join(ARTIFACTS_DIR, "03_popularity_by_continent.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    fig_paths.append(path)
    print(f"  Saved: 03_popularity_by_continent.png")

    # 4. Page Views vs Popularity (scatter)
    plt.figure(figsize=(10, 6))
    plt.scatter(np.log10(df['page_views'] + 1), df['historical_popularity_index'], alpha=0.3, s=10)
    plt.xlabel('Log10(Page Views)')
    plt.ylabel('Historical Popularity Index')
    plt.title('Wikipedia Page Views vs Historical Popularity')
    path = os.path.join(ARTIFACTS_DIR, "04_pageviews_vs_popularity.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    fig_paths.append(path)
    print(f"  Saved: 04_pageviews_vs_popularity.png")

    # 5. Article Languages vs Popularity
    plt.figure(figsize=(10, 6))
    plt.scatter(df['article_languages'], df['historical_popularity_index'], alpha=0.3, s=10)
    plt.xlabel('Number of Wikipedia Languages')
    plt.ylabel('Historical Popularity Index')
    plt.title('Wikipedia Language Coverage vs Historical Popularity')
    path = os.path.join(ARTIFACTS_DIR, "05_languages_vs_popularity.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    fig_paths.append(path)
    print(f"  Saved: 05_languages_vs_popularity.png")

    # 6. Top 20 Occupations by Count and Popularity
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    occ_counts = df['occupation'].value_counts().head(20)
    occ_counts.plot(kind='barh', ax=axes[0], color='coral')
    axes[0].set_xlabel('Count')
    axes[0].set_title('Top 20 Occupations by Count')

    occ_pop = df.groupby('occupation')['historical_popularity_index'].mean().nlargest(20)
    occ_pop.plot(kind='barh', ax=axes[1], color='teal')
    axes[1].set_xlabel('Avg Popularity')
    axes[1].set_title('Top 20 Occupations by Avg Popularity')

    plt.tight_layout()
    path = os.path.join(ARTIFACTS_DIR, "06_occupations_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    fig_paths.append(path)
    print(f"  Saved: 06_occupations_analysis.png")

    # 7. Correlation heatmap of numeric features
    numeric_cols = ['birth_year', 'latitude', 'longitude', 'article_languages',
                    'page_views', 'average_views', 'historical_popularity_index']
    corr_df = df[numeric_cols].dropna()

    plt.figure(figsize=(10, 8))
    corr_matrix = corr_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    path = os.path.join(ARTIFACTS_DIR, "07_correlation_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    fig_paths.append(path)
    print(f"  Saved: 07_correlation_matrix.png")

    # 8. Birth Year Timeline
    plt.figure(figsize=(12, 6))
    df_timeline = df[(df['birth_year'] > -1000) & (df['birth_year'] < 2000)].copy()
    df_timeline['century'] = (df_timeline['birth_year'] // 100) * 100
    century_pop = df_timeline.groupby('century')['historical_popularity_index'].mean()
    century_pop.plot(kind='line', marker='o', linewidth=2, markersize=6)
    plt.xlabel('Century (Birth Year)')
    plt.ylabel('Average Historical Popularity Index')
    plt.title('Historical Popularity by Century of Birth')
    plt.grid(True, alpha=0.3)
    path = os.path.join(ARTIFACTS_DIR, "08_popularity_by_century.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    fig_paths.append(path)
    print(f"  Saved: 08_popularity_by_century.png")

    return fig_paths


def prepare_features(df):
    """Prepare features for modeling."""
    print("\n" + "=" * 60)
    print("STEP 3: FEATURE PREPARATION")
    print("=" * 60)

    # Select and engineer features
    df_model = df.copy()

    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['sex', 'country', 'continent', 'occupation', 'industry', 'domain']

    for col in categorical_cols:
        le = LabelEncoder()
        df_model[col + '_encoded'] = le.fit_transform(df_model[col].fillna('Unknown'))
        label_encoders[col] = le

    # Create feature set
    feature_cols = [
        'birth_year',
        'latitude',
        'longitude',
        'article_languages',
        'page_views',
        'average_views',
        'sex_encoded',
        'continent_encoded',
        'domain_encoded',
        'industry_encoded'
    ]

    # Remove rows with missing target or key features
    df_clean = df_model.dropna(subset=['historical_popularity_index'] + feature_cols)

    X = df_clean[feature_cols].copy()
    y = df_clean['historical_popularity_index'].copy()

    # Fill remaining NaNs with median
    X = X.fillna(X.median())

    print(f"  Features used: {len(feature_cols)}")
    print(f"  Samples after cleaning: {len(X)}")
    print(f"  Feature list: {feature_cols}")

    return X, y, feature_cols, label_encoders


def train_and_evaluate_models(X, y, feature_cols, artifact_paths):
    """Train multiple models and track with MLflow."""
    print("\n" + "=" * 60)
    print("STEP 4: MODEL TRAINING WITH MLFLOW TRACKING")
    print("=" * 60)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")

    # Define models to try
    models = {
        "RandomForest": {
            "model": RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1),
            "params": {"n_estimators": 100, "max_depth": 15, "min_samples_split": 5},
            "use_scaled": False
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
            "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
            "use_scaled": False
        },
        "Ridge": {
            "model": Ridge(alpha=1.0),
            "params": {"alpha": 1.0},
            "use_scaled": True
        },
        "ElasticNet": {
            "model": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            "params": {"alpha": 0.1, "l1_ratio": 0.5},
            "use_scaled": True
        },
        "SVR": {
            "model": SVR(kernel='rbf', C=1.0, epsilon=0.1),
            "params": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1},
            "use_scaled": True
        }
    }

    results = []
    best_model = None
    best_r2 = -float('inf')
    best_run_id = None
    best_model_name = None

    for model_name, config in models.items():
        print(f"\n  Training {model_name}...")

        with mlflow.start_run(run_name=model_name):
            model = config["model"]

            # Select appropriate data (scaled or not)
            if config["use_scaled"]:
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train.values, X_test.values

            # Train model
            model.fit(X_tr, y_train)

            # Predictions
            y_pred_train = model.predict(X_tr)
            y_pred_test = model.predict(X_te)

            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)

            # Cross-validation score
            cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # Log parameters
            mlflow.log_params(config["params"])
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("use_scaled_features", config["use_scaled"])
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", len(feature_cols))

            # Log metrics
            mlflow.log_metrics({
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "cv_r2_mean": cv_mean,
                "cv_r2_std": cv_std
            })

            # Log all exploratory artifacts
            for artifact_path in artifact_paths:
                mlflow.log_artifact(artifact_path)

            # Create prediction vs actual plot
            plt.figure(figsize=(8, 8))
            plt.scatter(y_test, y_pred_test, alpha=0.5, s=10)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'{model_name}: Predicted vs Actual')
            pred_path = os.path.join(ARTIFACTS_DIR, f"predictions_{model_name}.png")
            plt.savefig(pred_path, dpi=150, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(pred_path)

            # Create residual plot
            residuals = y_test - y_pred_test
            plt.figure(figsize=(10, 5))
            plt.scatter(y_pred_test, residuals, alpha=0.5, s=10)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted')
            plt.ylabel('Residuals')
            plt.title(f'{model_name}: Residual Plot')
            resid_path = os.path.join(ARTIFACTS_DIR, f"residuals_{model_name}.png")
            plt.savefig(resid_path, dpi=150, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(resid_path)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            run_id = mlflow.active_run().info.run_id

            print(f"    Test R2: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
            print(f"    CV R2 (5-fold): {cv_mean:.4f} +/- {cv_std:.4f}")
            print(f"    Run ID: {run_id}")

            results.append({
                "model": model_name,
                "test_r2": test_r2,
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "cv_r2": cv_mean,
                "run_id": run_id
            })

            # Track best model
            if test_r2 > best_r2:
                best_r2 = test_r2
                best_model = model
                best_run_id = run_id
                best_model_name = model_name

    # Create model comparison chart
    results_df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    results_df.set_index('model')['test_r2'].plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('Model Comparison: R2 (higher is better)')
    axes[0].tick_params(axis='x', rotation=45)

    results_df.set_index('model')['test_rmse'].plot(kind='bar', ax=axes[1], color='coral')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Model Comparison: RMSE (lower is better)')
    axes[1].tick_params(axis='x', rotation=45)

    results_df.set_index('model')['test_mae'].plot(kind='bar', ax=axes[2], color='teal')
    axes[2].set_ylabel('MAE')
    axes[2].set_title('Model Comparison: MAE (lower is better)')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    comparison_path = os.path.join(ARTIFACTS_DIR, "model_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "-" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("-" * 60)
    print(results_df.to_string(index=False))
    print(f"\n  Best Model: {best_model_name} (R2 = {best_r2:.4f})")

    return results_df, best_model, best_run_id, best_model_name, X_train, feature_cols


def analyze_feature_importance(best_model, feature_cols, model_name):
    """Analyze and visualize feature importance."""
    print("\n" + "=" * 60)
    print("STEP 5: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\nFeature Importance Ranking:")
        print(importance_df.to_string(index=False))

        # Create feature importance plot
        plt.figure(figsize=(10, 6))
        importance_df_sorted = importance_df.sort_values('importance', ascending=True)
        plt.barh(importance_df_sorted['feature'], importance_df_sorted['importance'], color='steelblue')
        plt.xlabel('Importance')
        plt.title(f'Feature Importance ({model_name})')
        plt.tight_layout()
        path = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        print("\n--- Key Insights on Fame Factors ---")
        top_features = importance_df.head(5)['feature'].tolist()
        print(f"\nTop 5 Most Important Features for Historical Fame:")
        for i, feat in enumerate(top_features, 1):
            imp = importance_df[importance_df['feature'] == feat]['importance'].values[0]
            print(f"  {i}. {feat}: {imp:.4f}")

        return importance_df, path
    else:
        print(f"  {model_name} does not have feature_importances_ attribute")
        return None, None


def register_best_model(best_run_id, best_model_name, best_r2):
    """Register the best model in MLflow Model Registry."""
    print("\n" + "=" * 60)
    print("STEP 6: MODEL REGISTRATION")
    print("=" * 60)

    model_uri = f"runs:/{best_run_id}/model"
    registered_model_name = "historical_popularity_predictor"

    try:
        # Register model
        result = mlflow.register_model(model_uri, registered_model_name)

        print(f"\n  Model registered: {registered_model_name}")
        print(f"  Version: {result.version}")
        print(f"  Source Run: {best_run_id}")

        # Try to set model alias (modern approach) instead of stage transition
        client = MlflowClient()
        try:
            client.set_registered_model_alias(registered_model_name, "champion", result.version)
            print(f"  Alias: champion")
        except Exception:
            pass

        print(f"\n  To load this model later:")
        print(f"    model = mlflow.pyfunc.load_model('models:/{registered_model_name}@champion')")
        print(f"    # or: mlflow.pyfunc.load_model('models:/{registered_model_name}/{result.version}')")

        return result
    except Exception as e:
        print(f"\n  Model registration info saved.")
        print(f"  Best model run ID: {best_run_id}")
        print(f"  Best model type: {best_model_name}")
        print(f"  Best R2: {best_r2:.4f}")
        print(f"\n  To load the best model:")
        print(f"    model = mlflow.sklearn.load_model('runs:/{best_run_id}/model')")
        return None


def main():
    """Main pipeline execution."""
    print("\n" + "=" * 60)
    print("HISTORICAL POPULARITY PREDICTION PIPELINE")
    print("=" * 60)
    print("All experiments will be tracked with MLflow")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # Step 1: Load and explore data
    df = load_and_explore_data()

    # Step 2: Create visualizations
    artifact_paths = create_visualizations(df)

    # Step 3: Prepare features
    X, y, feature_cols, label_encoders = prepare_features(df)

    # Step 4: Train and evaluate models
    results_df, best_model, best_run_id, best_model_name, X_train, feature_cols = train_and_evaluate_models(
        X, y, feature_cols, artifact_paths
    )

    # Step 5: Feature importance analysis
    importance_df, importance_path = analyze_feature_importance(best_model, feature_cols, best_model_name)

    # Step 6: Register best model
    model_result = register_best_model(best_run_id, best_model_name, results_df[results_df['model'] == best_model_name]['test_r2'].values[0])

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nTo view experiments in MLflow UI, run:")
    print(f"  cd {SCRIPT_DIR}")
    print(f"  mlflow ui")
    print("\nThen open http://localhost:5000 in your browser")

    return results_df, best_model, importance_df


if __name__ == "__main__":
    results, model, importance = main()
