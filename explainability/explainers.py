"""
Unified SHAP and LIME Explainability Runner
===========================================

Run explainability analysis on any experiment by providing the experiment name.

Usage:
    python explainability/explainers.py <experiment_name>
    
Example:
    python explainability/explainers.py xgboost_binning_binary_smote

This will:
1. Import and run the specified experiment
2. Generate SHAP explanations (summary, bar, waterfall plots)
3. Generate LIME explanations for representative instances
4. Save all outputs to: explainability_results/<experiment_name>/
"""
import sys
import os
import argparse
import importlib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular
from typing import List, Optional, Dict, Any
from xgboost import XGBClassifier


# ============================================================================
# SHAP UTILITIES
# ============================================================================

def create_shap_explainer(model: XGBClassifier) -> shap.TreeExplainer:
    """Create a SHAP TreeExplainer for XGBoost model."""
    return shap.TreeExplainer(model)


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X: np.ndarray,
    feature_names: List[str]
) -> shap.Explanation:
    """Compute SHAP values for given data."""
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
    """Create and save SHAP summary plot."""
    plt.figure(figsize=(12, 8))
    
    if is_multiclass and class_names is not None:
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
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_shap_bar(
    shap_values: shap.Explanation,
    output_path: str,
    title: str = "SHAP Feature Importance",
    class_names: Optional[List[str]] = None,
    is_multiclass: bool = False,
    max_display: int = 11
):
    """Create and save SHAP bar plot (mean absolute SHAP values)."""
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
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_shap_waterfall(
    shap_values: shap.Explanation,
    instance_idx: int,
    output_path: str,
    title: str = "SHAP Waterfall Plot",
    class_idx: Optional[int] = None
):
    """Create and save SHAP waterfall plot for a single instance."""
    plt.figure(figsize=(12, 8))
    
    if class_idx is not None:
        shap.plots.waterfall(shap_values[instance_idx, :, class_idx], show=False)
    else:
        shap.plots.waterfall(shap_values[instance_idx], show=False)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


# ============================================================================
# LIME UTILITIES
# ============================================================================

def create_lime_explainer(
    X_train: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    mode: str = 'classification'
) -> lime_tabular.LimeTabularExplainer:
    """Create a LIME explainer."""
    return lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode=mode,
        discretize_continuous=True
    )


def get_representative_instances(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    n_per_class: int = 1,
    include_misclassified: bool = True
) -> Dict[str, List[int]]:
    """Get representative instance indices for explanation."""
    correct_mask = y_test == y_pred
    incorrect_mask = ~correct_mask
    
    result = {'correct': [], 'misclassified': []}
    
    unique_classes = np.unique(y_test)
    
    for cls in unique_classes:
        class_correct = np.where((y_test == cls) & correct_mask)[0]
        if len(class_correct) > 0:
            selected = class_correct[:n_per_class].tolist()
            result['correct'].extend(selected)
    
    if include_misclassified:
        misclassified_indices = np.where(incorrect_mask)[0]
        if len(misclassified_indices) > 0:
            n_misclassified = min(len(unique_classes), len(misclassified_indices))
            result['misclassified'] = misclassified_indices[:n_misclassified].tolist()
    
    return result


def plot_lime_multiple_instances(
    explainer: lime_tabular.LimeTabularExplainer,
    model,
    X: np.ndarray,
    y: np.ndarray,
    instance_indices: List[int],
    class_names: List[str],
    output_path: str,
    title: str = "LIME Explanations",
    num_features: int = 10
):
    """Create LIME explanations for multiple instances and save as subplots."""
    n_instances = len(instance_indices)
    fig, axes = plt.subplots(1, n_instances, figsize=(7*n_instances, 6))
    
    if n_instances == 1:
        axes = [axes]
    
    for idx, (ax, inst_idx) in enumerate(zip(axes, instance_indices)):
        explanation = explainer.explain_instance(
            X[inst_idx],
            model.predict_proba,
            num_features=num_features
        )
        
        exp_list = explanation.as_list()
        features = [x[0] for x in exp_list]
        weights = [x[1] for x in exp_list]
        
        colors = ['green' if w > 0 else 'red' for w in weights]
        y_pos = np.arange(len(features))
        
        ax.barh(y_pos, weights, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=9)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
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
    print(f"  Saved: {os.path.basename(output_path)}")


# ============================================================================
# MAIN EXPLAINABILITY RUNNER
# ============================================================================

def run_shap_analysis(experiment_data: dict, output_dir: str):
    """Run SHAP analysis on the experiment results."""
    print("\n[SHAP Analysis]")
    
    feature_names = experiment_data['feature_names']
    results = experiment_data['results']
    is_binary = experiment_data.get('is_binary', True)
    class_names = results['class_names']
    
    # Create SHAP explainer
    explainer = create_shap_explainer(results['model'])
    shap_values = compute_shap_values(explainer, results['X_test'], feature_names)
    
    # Summary plot
    plot_shap_summary(
        shap_values,
        results['X_test'],
        feature_names,
        os.path.join(output_dir, "shap_summary.png"),
        title=f"SHAP Summary - {experiment_data['experiment_name']}",
        class_names=class_names if not is_binary else None,
        is_multiclass=not is_binary
    )
    
    # Bar plot
    plot_shap_bar(
        shap_values,
        os.path.join(output_dir, "shap_bar.png"),
        title=f"SHAP Feature Importance - {experiment_data['experiment_name']}",
        class_names=class_names if not is_binary else None,
        is_multiclass=not is_binary
    )
    
    # Waterfall plot for a correctly predicted positive class instance
    y_pred = results['model'].predict(results['X_test'])
    
    if is_binary:
        # Find a correctly predicted premium/positive instance
        positive_mask = (results['y_test'] == 1) & (y_pred == 1)
        positive_indices = np.where(positive_mask)[0]
        
        if len(positive_indices) > 0:
            plot_shap_waterfall(
                shap_values,
                positive_indices[0],
                os.path.join(output_dir, "shap_waterfall.png"),
                title=f"SHAP Waterfall - Positive Class Instance"
            )
    else:
        # Find a correctly predicted high quality instance
        if 'high' in class_names:
            high_class_idx = class_names.index('high')
        else:
            high_class_idx = len(class_names) - 1  # Last class
        
        high_mask = (results['y_test'] == high_class_idx) & (y_pred == high_class_idx)
        high_indices = np.where(high_mask)[0]
        
        if len(high_indices) > 0:
            plot_shap_waterfall(
                shap_values,
                high_indices[0],
                os.path.join(output_dir, "shap_waterfall.png"),
                title=f"SHAP Waterfall - {class_names[high_class_idx]} Class Instance",
                class_idx=high_class_idx
            )
    
    return shap_values


def run_lime_analysis(experiment_data: dict, output_dir: str):
    """Run LIME analysis on the experiment results."""
    print("\n[LIME Analysis]")
    
    feature_names = experiment_data['feature_names']
    results = experiment_data['results']
    class_names = results['class_names']
    
    # Create LIME explainer
    lime_explainer = create_lime_explainer(
        results['X_train'],
        feature_names,
        class_names
    )
    
    # Get representative instances
    y_pred = results['model'].predict(results['X_test'])
    instances = get_representative_instances(
        results['y_test'],
        y_pred,
        n_per_class=1,
        include_misclassified=True
    )
    
    # Correct predictions
    if instances['correct']:
        max_instances = min(3, len(instances['correct']))
        plot_lime_multiple_instances(
            lime_explainer,
            results['model'],
            results['X_test'],
            results['y_test'],
            instances['correct'][:max_instances],
            class_names,
            os.path.join(output_dir, "lime_correct.png"),
            title=f"LIME - Correct Predictions"
        )
    
    # Misclassified predictions
    if instances['misclassified']:
        max_instances = min(3, len(instances['misclassified']))
        plot_lime_multiple_instances(
            lime_explainer,
            results['model'],
            results['X_test'],
            results['y_test'],
            instances['misclassified'][:max_instances],
            class_names,
            os.path.join(output_dir, "lime_misclassified.png"),
            title=f"LIME - Misclassified Predictions"
        )


def run_explainability(experiment_name: str):
    """
    Run full explainability analysis on a given experiment.
    
    Parameters
    ----------
    experiment_name : str
        Name of the experiment file (without .py extension)
    """
    print("="*60)
    print("EXPLAINABILITY ANALYSIS: SHAP & LIME")
    print("="*60)
    print(f"\nExperiment: {experiment_name}")
    
    # Dynamically import the experiment module
    try:
        experiment_module = importlib.import_module(f"experiments.{experiment_name}")
    except ModuleNotFoundError:
        print(f"\nError: Experiment '{experiment_name}' not found in experiments/")
        print("Available experiments:")
        experiments_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments')
        for f in os.listdir(experiments_dir):
            if f.endswith('.py') and not f.startswith('_') and f != 'experiment_tracker.py':
                print(f"  - {f[:-3]}")
        return None
    
    # Run the experiment
    print("\n[Step 1] Running experiment...")
    experiment_data = experiment_module.run_experiment(verbose=False)
    
    print(f"  Features: {experiment_data['feature_names']}")
    print(f"  Classes: {experiment_data['results']['class_names']}")
    print(f"  Is Binary: {experiment_data.get('is_binary', 'Unknown')}")
    
    # Setup output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'explainability_results', experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run SHAP analysis
    print("\n[Step 2] Running SHAP analysis...")
    run_shap_analysis(experiment_data, output_dir)
    
    # Run LIME analysis
    print("\n[Step 3] Running LIME analysis...")
    run_lime_analysis(experiment_data, output_dir)
    
    print("\n" + "="*60)
    print("EXPLAINABILITY COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.png'):
            print(f"  - {f}")
    
    return experiment_data


def main():
    """Main entry point for command line usage."""
    parser = argparse.ArgumentParser(
        description='Run SHAP and LIME explainability analysis on an experiment.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python explainability/explainers.py xgboost_binning_binary_smote
    python explainability/explainers.py xgboost_binning_multiclass_nosmote
        """
    )
    parser.add_argument(
        'experiment_name',
        type=str,
        help='Name of the experiment (filename without .py extension)'
    )
    
    args = parser.parse_args()
    run_explainability(args.experiment_name)


if __name__ == "__main__":
    main()
