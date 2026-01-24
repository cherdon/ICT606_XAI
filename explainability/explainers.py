"""
Shared SHAP and LIME explainability utilities for XGBoost models.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular
from typing import List, Optional, Dict, Any
from xgboost import XGBClassifier


def create_shap_explainer(model: XGBClassifier) -> shap.TreeExplainer:
    """
    Create a SHAP TreeExplainer for XGBoost model.
    
    Parameters
    ----------
    model : XGBClassifier
        Trained XGBoost model
        
    Returns
    -------
    shap.TreeExplainer
        SHAP explainer object
    """
    return shap.TreeExplainer(model)


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X: np.ndarray,
    feature_names: List[str]
) -> shap.Explanation:
    """
    Compute SHAP values for given data.
    
    Parameters
    ----------
    explainer : shap.TreeExplainer
        SHAP explainer object
    X : np.ndarray
        Feature matrix
    feature_names : List[str]
        List of feature names
        
    Returns
    -------
    shap.Explanation
        SHAP explanation object
    """
    shap_values = explainer(X)
    shap_values.feature_names = feature_names
    return shap_values


def plot_shap_summary(
    shap_values: shap.Explanation,
    X: np.ndarray,
    feature_names: List[str],
    output_path: str,
    title: str = "SHAP Summary Plot",
    class_names: Optional[List[str]] = None,
    is_multiclass: bool = False
):
    """
    Create and save SHAP summary plot.
    
    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP explanation object
    X : np.ndarray
        Feature matrix
    feature_names : List[str]
        List of feature names
    output_path : str
        Path to save the plot
    title : str
        Plot title
    class_names : List[str], optional
        Class names for multi-class
    is_multiclass : bool
        Whether this is a multi-class problem
    """
    plt.figure(figsize=(12, 8))
    
    if is_multiclass and class_names is not None:
        # For multi-class, create subplot for each class
        fig, axes = plt.subplots(1, len(class_names), figsize=(6*len(class_names), 8))
        
        for idx, class_name in enumerate(class_names):
            plt.sca(axes[idx])
            shap.summary_plot(
                shap_values[:, :, idx],
                X,
                feature_names=feature_names,
                show=False,
                title=f"Class: {class_name}"
            )
            axes[idx].set_title(f"Class: {class_name}", fontsize=12, fontweight='bold')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    else:
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            show=False
        )
        plt.title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to: {output_path}")


def plot_shap_bar(
    shap_values: shap.Explanation,
    output_path: str,
    title: str = "SHAP Feature Importance",
    class_names: Optional[List[str]] = None,
    is_multiclass: bool = False,
    max_display: int = 11
):
    """
    Create and save SHAP bar plot (mean absolute SHAP values).
    
    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP explanation object
    output_path : str
        Path to save the plot
    title : str
        Plot title
    class_names : List[str], optional
        Class names for multi-class
    is_multiclass : bool
        Whether this is a multi-class problem
    max_display : int
        Maximum number of features to display
    """
    if is_multiclass and class_names is not None:
        fig, axes = plt.subplots(1, len(class_names), figsize=(6*len(class_names), 8))
        
        for idx, class_name in enumerate(class_names):
            plt.sca(axes[idx])
            shap.plots.bar(shap_values[:, :, idx], max_display=max_display, show=False)
            axes[idx].set_title(f"Class: {class_name}", fontsize=12, fontweight='bold')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    else:
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_values, max_display=max_display, show=False)
        plt.title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SHAP bar plot saved to: {output_path}")


def plot_shap_waterfall(
    shap_values: shap.Explanation,
    instance_idx: int,
    output_path: str,
    title: str = "SHAP Waterfall Plot",
    class_idx: Optional[int] = None
):
    """
    Create and save SHAP waterfall plot for a single instance.
    
    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP explanation object
    instance_idx : int
        Index of the instance to explain
    output_path : str
        Path to save the plot
    title : str
        Plot title
    class_idx : int, optional
        Class index for multi-class problems
    """
    plt.figure(figsize=(12, 8))
    
    if class_idx is not None:
        shap.plots.waterfall(shap_values[instance_idx, :, class_idx], show=False)
    else:
        shap.plots.waterfall(shap_values[instance_idx], show=False)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SHAP waterfall plot saved to: {output_path}")


def create_lime_explainer(
    X_train: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    mode: str = 'classification'
) -> lime_tabular.LimeTabularExplainer:
    """
    Create a LIME explainer.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training data for LIME to sample from
    feature_names : List[str]
        List of feature names
    class_names : List[str]
        List of class names
    mode : str
        Either 'classification' or 'regression'
        
    Returns
    -------
    lime_tabular.LimeTabularExplainer
        LIME explainer object
    """
    return lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode=mode,
        discretize_continuous=True
    )


def explain_instance_lime(
    explainer: lime_tabular.LimeTabularExplainer,
    model: XGBClassifier,
    instance: np.ndarray,
    num_features: int = 10
) -> Any:
    """
    Generate LIME explanation for a single instance.
    
    Parameters
    ----------
    explainer : lime_tabular.LimeTabularExplainer
        LIME explainer object
    model : XGBClassifier
        Trained model with predict_proba method
    instance : np.ndarray
        Single instance to explain
    num_features : int
        Number of features to include in explanation
        
    Returns
    -------
    lime.explanation.Explanation
        LIME explanation object
    """
    return explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=num_features
    )


def plot_lime_explanation(
    explanation,
    output_path: str,
    title: str = "LIME Explanation"
):
    """
    Save LIME explanation as a figure.
    
    Parameters
    ----------
    explanation : lime.explanation.Explanation
        LIME explanation object
    output_path : str
        Path to save the plot
    title : str
        Plot title
    """
    fig = explanation.as_pyplot_figure()
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"LIME explanation plot saved to: {output_path}")


def plot_lime_multiple_instances(
    explainer: lime_tabular.LimeTabularExplainer,
    model: XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    instance_indices: List[int],
    class_names: List[str],
    output_path: str,
    title: str = "LIME Explanations",
    num_features: int = 10
):
    """
    Create LIME explanations for multiple instances and save as subplots.
    
    Parameters
    ----------
    explainer : lime_tabular.LimeTabularExplainer
        LIME explainer object
    model : XGBClassifier
        Trained model
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        True labels
    instance_indices : List[int]
        Indices of instances to explain
    class_names : List[str]
        Class names
    output_path : str
        Path to save the plot
    title : str
        Plot title
    num_features : int
        Number of features to show
    """
    n_instances = len(instance_indices)
    fig, axes = plt.subplots(1, n_instances, figsize=(7*n_instances, 6))
    
    if n_instances == 1:
        axes = [axes]
    
    for idx, (ax, inst_idx) in enumerate(zip(axes, instance_indices)):
        explanation = explain_instance_lime(
            explainer, model, X[inst_idx], num_features=num_features
        )
        
        # Get feature weights
        exp_list = explanation.as_list()
        features = [x[0] for x in exp_list]
        weights = [x[1] for x in exp_list]
        
        # Create horizontal bar plot
        colors = ['green' if w > 0 else 'red' for w in weights]
        y_pos = np.arange(len(features))
        
        ax.barh(y_pos, weights, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=9)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Get prediction
        pred = model.predict([X[inst_idx]])[0]
        true_label = y[inst_idx]
        
        if isinstance(pred, (int, np.integer)):
            pred_name = class_names[pred] if pred < len(class_names) else str(pred)
            true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
        else:
            pred_name = str(pred)
            true_name = str(true_label)
        
        ax.set_title(f"Instance {inst_idx}\nTrue: {true_name}, Pred: {pred_name}", fontsize=11)
        ax.set_xlabel("Feature Weight")
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"LIME multi-instance plot saved to: {output_path}")


def get_representative_instances(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    n_per_class: int = 1,
    include_misclassified: bool = True
) -> Dict[str, List[int]]:
    """
    Get representative instance indices for explanation.
    
    Parameters
    ----------
    y_test : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    n_per_class : int
        Number of instances per class to select
    include_misclassified : bool
        Whether to include misclassified instances
        
    Returns
    -------
    Dict[str, List[int]]
        Dictionary with 'correct' and 'misclassified' instance indices
    """
    correct_mask = y_test == y_pred
    incorrect_mask = ~correct_mask
    
    result = {'correct': [], 'misclassified': []}
    
    unique_classes = np.unique(y_test)
    
    # Get correctly classified instances per class
    for cls in unique_classes:
        class_correct = np.where((y_test == cls) & correct_mask)[0]
        if len(class_correct) > 0:
            selected = class_correct[:n_per_class].tolist()
            result['correct'].extend(selected)
    
    # Get misclassified instances
    if include_misclassified:
        misclassified_indices = np.where(incorrect_mask)[0]
        if len(misclassified_indices) > 0:
            n_misclassified = min(len(unique_classes), len(misclassified_indices))
            result['misclassified'] = misclassified_indices[:n_misclassified].tolist()
    
    return result
