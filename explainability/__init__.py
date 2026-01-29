"""
Explainability module for XAI analysis using SHAP and LIME.

Usage:
    python explainability/explainers.py <experiment_name>
    
Example:
    python explainability/explainers.py xgboost_binary_smote
"""
from explainability.explainers import run_explainability

__all__ = ['run_explainability']
