"""
Model training stage: Trains an LGBM model with scikit-learn interface and logs comprehensive metrics to MLflow.
"""
import os
import sys
import yaml
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ML libraries
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold

# MLflow and DagsHub integration
import mlflow
import mlflow.sklearn
import dagshub
import joblib

# Custom Transformers
class TimeBasedFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        current_date = datetime.now()
        X['HireDateObj'] = pd.to_datetime(X['HireDate'])
        X['YearsSinceHire'] = (current_date - X['HireDateObj']).dt.days / 365
        X['TimeToPromotion'] = X['YearsSinceLastPromotion'] / np.maximum(X['YearsAtCompany'], 0.5)
        X['TimeToPromotion'] = X['TimeToPromotion'].clip(0, 1)
        X.drop('HireDateObj', axis=1, errors='ignore', inplace=True)
        return X

class CompensationFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.level_median_ = X.groupby('JobLevel')['MonthlyIncome'].median()
        self.dept_median_ = X.groupby('Department')['MonthlyIncome'].median()
        return self
    
    def transform(self, X):
        X = X.copy()
        X['SalaryRatioToLevel'] = X['MonthlyIncome'] / X['JobLevel'].map(self.level_median_)
        X['SalaryRatioDept'] = X['MonthlyIncome'] / X['Department'].map(self.dept_median_)
        return X

class TenureGroupTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        tenure_bins = [0, 1, 3, 5, 10, 100]
        X['TenureGroup'] = pd.cut(X['YearsAtCompany'], bins=tenure_bins, labels=False)
        return X

class SalaryRatioToTenure(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['SalaryRatioToTenure'] = X['MonthlyIncome'] / X.groupby('TenureGroup')['MonthlyIncome'].transform('median')
        X.drop('TenureGroup', axis=1, errors='ignore', inplace=True)
        return X

class CommuteDifficultyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['CommuteDifficulty'] = pd.cut(
            X['DistanceFromHome'], 
            bins=[0, 5, 15, 100], 
            labels=['Short', 'Medium', 'Long']
        )
        return X

class WorkLifeImbalanceTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['WorkLifeImbalance'] = ((X['WorkLifeBalance'] <= 2) & 
                                 (X['Overtime'] == True)).astype(int)
        return X

class SatisfactionCompositeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['OverallSatisfaction'] = (
            X['JobSatisfaction'] + 
            X['EnvironmentSatisfaction'] + 
            X['WorkLifeBalance']
        ) / 3
        X['EngagementScore'] = (X['OverallSatisfaction'] + X['PerformanceRating'] / 5 * 5) / 2
        return X

class CareerGrowthTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['StagnationRisk'] = ((X['YearsSinceLastPromotion'] >= 2) & 
                              (X['YearsAtCompany'] > 3)).astype(int)
        job_level_mapping = {
            'Entry Level': 1, 
            'Junior': 2, 
            'Mid-Level': 3, 
            'Senior': 4, 
            'Manager': 5, 
            'Director': 6, 
            'Executive': 7
        }
        X['CareerVelocity'] = X['JobLevel'].map(job_level_mapping) / np.maximum(X['YearsAtCompany'], 1)
        return X

class TrainingEngagementTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['TrainingEngagement'] = pd.cut(
            X['TrainingTimesLastYear'],
            bins=[-1, 20, 40, 100],
            labels=['Low', 'Medium', 'High']
        )
        return X

class EducationDemographicsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        education_map = {
            'High School': 1,
            'Diploma': 2,
            'Bachelor\'s Degree': 3,
            'Master\'s Degree': 4,
            'PhD': 5
        }
        X['EducationLevel'] = X['Education'].map(education_map)
        X['AgeGroup'] = pd.cut(
            X['Age'],
            bins=[20, 30, 40, 50, 100],
            labels=['20s', '30s', '40s', '50+']
        )
        return X

class IncomeToAgeRatioTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['IncomeToAgeRatio'] = X['MonthlyIncome'] / X['Age']
        return X

def create_preprocessing_pipeline(features_params):
    """
    Create a preprocessing pipeline based on the configuration in params.yaml.
    """
    # Configure preprocessing steps based on parameters
    preprocessing_steps = [
        ('time_based', TimeBasedFeatures()),
        ('tenure_group', TenureGroupTransformer()),
        ('compensation', CompensationFeatures()),
        ('salary_ratio_tenure', SalaryRatioToTenure()),
        ('commute', CommuteDifficultyTransformer()),
        ('work_life', WorkLifeImbalanceTransformer()),
        ('satisfaction', SatisfactionCompositeTransformer()),
        ('career_growth', CareerGrowthTransformer()),
        ('training_engagement', TrainingEngagementTransformer()),
        ('education_demographics', EducationDemographicsTransformer()),
        ('income_age_ratio', IncomeToAgeRatioTransformer()),
    ]
    
    # Define columns for different transformers
    numeric_cols = [
        'Age', 'DistanceFromHome', 'YearsAtCompany', 'YearsSinceLastPromotion',
        'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance',
        'PerformanceRating', 'MonthlyIncome', 'TrainingTimesLastYear',
        'YearsSinceHire', 'TimeToPromotion', 'SalaryRatioToLevel', 
        'SalaryRatioDept', 'SalaryRatioToTenure', 'OverallSatisfaction',
        'EngagementScore', 'CareerVelocity', 'IncomeToAgeRatio', 'EducationLevel'
    ]
    
    categorical_cols = [
        'Department', 'JobLevel', 'Gender', 'CommuteDifficulty', 
        'TrainingEngagement', 'AgeGroup'
    ]
    
    binary_cols = [
        'Overtime', 'WorkLifeImbalance', 'StagnationRisk'
    ]
    
    # Add derived binary features if created during transformation
    if features_params.get('add_performance_categories', False):
        binary_cols.extend(['HighPerformer', 'LowPerformer'])
    
    # Configure column transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler() if features_params.get('standardize_numeric', True) else 'passthrough')
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    column_transformer = ColumnTransformer([
        ('numeric', numeric_transformer, numeric_cols),
        ('categorical', categorical_transformer, categorical_cols),
        ('binary', 'passthrough', binary_cols)
    ], remainder='drop')
    
    # Add the column transformer to the pipeline
    preprocessing_steps.append(('preprocessor', column_transformer))
    
    return Pipeline(preprocessing_steps)

def main():
    """
    Main function to train the LGBM model with sklearn interface and track with MLflow.
    """
    # Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    # Extract model parameters
    model_params = params['model']
    features_params = params['features']
    
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("data/stages/features", exist_ok=True)
    
    print("Loading training data...")
    train_data = pd.read_csv("data/stages/preprocessed/train_data.csv")
    
    # Convert date columns to datetime
    if 'HireDate' in train_data.columns:
        train_data['HireDate'] = pd.to_datetime(train_data['HireDate'])
    if 'ExitDate' in train_data.columns and train_data['ExitDate'].notna().any():
        train_data['ExitDate'] = pd.to_datetime(train_data['ExitDate'])
    
    # Prepare target variable
    y_train = train_data['Attrition'].astype(int)
    X_train = train_data.drop('Attrition', axis=1)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training attrition rate: {y_train.mean():.2%}")
    
    # Configure DagsHub and MLflow integration
    if 'mlflow' in params and 'dagshub_repo_name' in params['mlflow']:
        repo_name = params['mlflow']['dagshub_repo_name']
        repo_owner = params['mlflow']['dagshub_repo_owner']
        
        print(f"Configuring DagsHub integration with repository: {repo_owner}/{repo_name}")
        
        # Initialize DagsHub tracking
        dagshub.init(repo_name=repo_name, repo_owner=repo_owner)
        
        # Set the MLflow tracking URI
        mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
    else:
        print("Using local MLflow tracking")
    
    # Set experiment name
    mlflow.set_experiment("hr-attrition-prediction")
    
    # Start MLflow run with a descriptive name
    with mlflow.start_run(run_name="lgbm_training") as run:
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run started: {run_id}")
        
        # Log dataset information
        mlflow.log_params({
            "dataset_size": len(train_data),
            "train_attrition_count": int(y_train.sum()),
            "train_non_attrition_count": int((~y_train.astype(bool)).sum()),
            "attrition_rate": float(y_train.mean())
        })
        
        # Create preprocessing pipeline
        print("Creating feature engineering pipeline...")
        preprocessing_pipeline = create_preprocessing_pipeline(features_params)
        
        # Create LGBM model with sklearn interface
        if model_params.get('class_weight') == 'balanced':
            class_weight = 'balanced'
        else:
            class_weight = None
        
        lgbm_model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            boosting_type=model_params.get('boosting_type', 'gbdt'),
            num_leaves=model_params.get('num_leaves', 31),
            learning_rate=model_params.get('learning_rate', 0.05),
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', -1),
            min_child_samples=model_params.get('min_child_samples', 20),
            reg_alpha=model_params.get('reg_alpha', 0.0),
            reg_lambda=model_params.get('reg_lambda', 0.0),
            feature_fraction=model_params.get('feature_fraction', 0.9),
            bagging_fraction=model_params.get('bagging_fraction', 0.8),
            bagging_freq=model_params.get('bagging_freq', 5),
            class_weight=class_weight,
            n_jobs=model_params.get('n_jobs', -1),
            random_state=42,
            verbose=-1
        )
        
        # Create full pipeline
        full_pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('model', lgbm_model)
        ])
        
        # Log preprocessing parameters
        mlflow.log_params({
            "preprocessing__numeric_imputer": "median",
            "preprocessing__categorical_imputer": "most_frequent",
            "preprocessing__scaling": "standard" if features_params.get('standardize_numeric', True) else "none",
            "preprocessing__encoding": features_params.get('categorical_encoding', 'one-hot')
        })
        
        # Log model parameters
        for key, value in lgbm_model.get_params().items():
            mlflow.log_param(f"model__{key}", value)
        
        # Perform cross-validation for metrics logging
        print("Performing cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            scores = cross_val_score(full_pipeline, X_train, y_train, cv=cv, scoring=metric)
            cv_metrics[f"cv_{metric}_mean"] = scores.mean()
            cv_metrics[f"cv_{metric}_std"] = scores.std()
        
        # Log all cross-validation metrics
        mlflow.log_metrics(cv_metrics)
        
        # Train the full model on all training data
        print("Training final model on full training dataset...")
        full_pipeline.fit(X_train, y_train)
        
        # Try to get feature importance
        try:
            # Get feature names from preprocessor
            feature_names = full_pipeline.named_steps['preprocessing'].named_steps['preprocessor'].get_feature_names_out()
            
            # Get feature importances from model
            importance = full_pipeline.named_steps['model'].feature_importances_
            
            # Create feature importance dataframe
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            # Save feature importance
            feature_importance.to_csv("data/stages/features/feature_importance.csv", index=False)
            
            # Log feature importance to MLflow
            mlflow.log_artifact("data/stages/features/feature_importance.csv")
            mlflow.log_text(feature_importance.to_markdown(), "feature_importance.md")
            mlflow.log_dict(feature_importance.to_dict(orient="records"), "feature_importance.json")
            
        except Exception as e:
            error_message = f"Warning: Could not extract feature importance: {e}"
            print(error_message)
            mlflow.log_text(error_message, "feature_importance_error.txt")
            
            # Create a dummy feature importance file to satisfy DVC
            pd.DataFrame({'Feature': ['NA'], 'Importance': [0]}).to_csv(
                "data/stages/features/feature_importance.csv", index=False
            )
        
        # Save performance metrics
        metrics = {
            **cv_metrics,
            "train_size": int(len(X_train)),
            "train_attrition_rate": float(y_train.mean()),
            "mlflow_run_id": run_id
        }
        
        with open("metrics/training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        # Log the metrics file as an artifact
        mlflow.log_artifact("metrics/training_metrics.json")
        
        # Save the model
        joblib.dump(full_pipeline, "models/attrition_model.joblib")
        
        # Log the model to MLflow with registration
        mlflow.sklearn.log_model(
            full_pipeline, 
            "model",
            registered_model_name="employee-attrition-model"
        )
        
        # Log the source files as artifacts
        try:
            mlflow.log_artifact(__file__)  # Log this training script
            mlflow.log_artifact("params.yaml")  # Log configuration
        except Exception as e:
            print(f"Warning: Could not log source files: {e}")
        
        print("Training stage complete!")
        print(f"Model saved to models/attrition_model.joblib")
        print(f"Training metrics saved to metrics/training_metrics.json")
        print(f"MLflow run: {run_id}")

if __name__ == "__main__":
    main()