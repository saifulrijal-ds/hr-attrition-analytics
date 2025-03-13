"""
Model evaluation stage: Evaluates the trained attrition model on the test dataset.
Generates comprehensive metrics and visualizations of model performance.
"""
import os
import sys
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# Custom Transformers - NEEDED FOR MODEL LOADING
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
    
def convert_to_serializable(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

def main():
    """
    Main function to evaluate the trained model.
    """
    # Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    # Create output directories
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    
    print("Loading test data...")
    test_data = pd.read_csv("data/stages/preprocessed/test_data.csv")
    
    # Convert date columns to datetime
    if 'HireDate' in test_data.columns:
        test_data['HireDate'] = pd.to_datetime(test_data['HireDate'])
    if 'ExitDate' in test_data.columns and test_data['ExitDate'].notna().any():
        test_data['ExitDate'] = pd.to_datetime(test_data['ExitDate'])
    
    # Prepare target variable
    y_test = test_data['Attrition'].astype(int)
    X_test = test_data.drop('Attrition', axis=1)
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test attrition rate: {y_test.mean():.2%}")
    
    # Load the trained model
    print("Loading trained model...")
    model = joblib.load("models/attrition_model.joblib")
    
    # Make predictions
    print("Making predictions on test data...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    average_precision = average_precision_score(y_test, y_pred_proba)
    
    # Print metrics summary
    print(f"Model evaluation metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC AUC: {auc:.4f}")
    print(f"  Average Precision: {average_precision:.4f}")
    
    # Save core metrics
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
        "average_precision": float(average_precision),
        "test_size": int(len(X_test)),
        "test_attrition_rate": float(y_test.mean())
    }
    
    # Generate confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Retained', 'Attrition'],
        yticklabels=['Retained', 'Attrition']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png", dpi=300)
    plt.close()
    
    # Generate ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("reports/figures/roc_curve.png", dpi=300)
    plt.close()
    
    # Generate Precision-Recall curve
    plt.figure(figsize=(8, 6))
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall_curve, precision_curve, label=f'PR Curve (AP = {average_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("reports/figures/precision_recall_curve.png", dpi=300)
    plt.close()
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("metrics/classification_report.csv")
    
    # Calculate predictions at different thresholds
    thresholds = np.linspace(0.1, 0.9, 9)
    threshold_metrics = []
    
    for threshold in thresholds:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        threshold_metrics.append({
            'threshold': threshold,
            'accuracy': accuracy_score(y_test, y_pred_threshold),
            'precision': precision_score(y_test, y_pred_threshold),
            'recall': recall_score(y_test, y_pred_threshold),
            'f1': f1_score(y_test, y_pred_threshold),
            'positives': y_pred_threshold.sum()
        })
    
    # Create threshold comparison DataFrame
    threshold_df = pd.DataFrame(threshold_metrics)
    threshold_df.to_csv("metrics/threshold_analysis.csv", index=False)
    
    # Plot threshold impact on precision and recall
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df['threshold'], threshold_df['precision'], 'b-', label='Precision')
    plt.plot(threshold_df['threshold'], threshold_df['recall'], 'r-', label='Recall')
    plt.plot(threshold_df['threshold'], threshold_df['f1'], 'g-', label='F1')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Impact of Classification Threshold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("reports/figures/threshold_impact.png", dpi=300)
    plt.close()
    
    # Extract employee segments with highest attrition risk
    test_with_proba = X_test.copy()
    test_with_proba['Attrition_Actual'] = y_test
    test_with_proba['Attrition_Predicted'] = y_pred
    test_with_proba['Attrition_Probability'] = y_pred_proba
    
    # Department risk analysis
    if 'Department' in test_with_proba.columns:
        dept_risk = test_with_proba.groupby('Department')['Attrition_Probability'].mean().reset_index()
        dept_risk = dept_risk.sort_values('Attrition_Probability', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Department', y='Attrition_Probability', data=dept_risk)
        plt.title('Average Attrition Risk by Department')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("reports/figures/department_risk.png", dpi=300)
        plt.close()
        
        dept_risk.to_csv("metrics/department_risk.csv", index=False)
    
    # Job Level risk analysis
    if 'JobLevel' in test_with_proba.columns:
        level_risk = test_with_proba.groupby('JobLevel')['Attrition_Probability'].mean().reset_index()
        level_risk = level_risk.sort_values('Attrition_Probability', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='JobLevel', y='Attrition_Probability', data=level_risk)
        plt.title('Average Attrition Risk by Job Level')
        plt.tight_layout()
        plt.savefig("reports/figures/joblevel_risk.png", dpi=300)
        plt.close()
        
        level_risk.to_csv("metrics/joblevel_risk.csv", index=False)
    
    # Age Group risk analysis
    if 'Age' in test_with_proba.columns:
        test_with_proba['AgeGroup'] = pd.cut(
            test_with_proba['Age'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['<25', '25-34', '35-44', '45-54', '55+']
        )
        age_risk = test_with_proba.groupby('AgeGroup', observed=True)['Attrition_Probability'].mean().reset_index()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='AgeGroup', y='Attrition_Probability', data=age_risk)
        plt.title('Average Attrition Risk by Age Group')
        plt.savefig("reports/figures/age_risk.png", dpi=300)
        plt.close()
        
        age_risk.to_csv("metrics/age_risk.csv", index=False)
    
    # Log model errors analysis
    errors = test_with_proba[y_test != y_pred].copy()
    false_positives = errors[errors['Attrition_Predicted'] == 1]
    false_negatives = errors[errors['Attrition_Predicted'] == 0]
    
    # Save error counts by segment
    error_analysis = {
        "total_samples": len(y_test),
        "correct_predictions": (y_test == y_pred).sum(),
        "incorrect_predictions": (y_test != y_pred).sum(),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "fp_rate": len(false_positives) / (y_test == 0).sum() if (y_test == 0).sum() > 0 else 0,
        "fn_rate": len(false_negatives) / (y_test == 1).sum() if (y_test == 1).sum() > 0 else 0
    }
    
    # Add error analysis to metrics
    metrics["error_analysis"] = error_analysis

    metrics = convert_to_serializable(metrics)
    
    # Save all performance metrics
    with open("metrics/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("Evaluation stage complete!")
    print(f"Metrics saved to metrics/evaluation_metrics.json")
    print(f"Visualizations saved to reports/figures/")

if __name__ == "__main__":
    main()