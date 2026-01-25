"""
SHAP and LIME Explainability Analysis for XGBoost Wine Quality Prediction.
WITH SMOTE applied.

This script imports the trained models from the experiment file and generates:
1. SHAP explanations (summary, bar, waterfall plots)
2. LIME explanations for representative instances

All outputs are saved to: explainability_results/{experiment_name}/shap/ and /lime/
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import shap
import matplotlib.pyplot as plt
from lime import lime_tabular

# Import the experiment's run function
from experiments.xgboost_binning_premium_smote import run_experiment

from explainability.explainers import (
    create_shap_explainer,
    compute_shap_values,
    plot_shap_summary,
    plot_shap_bar,
    plot_shap_waterfall,
    create_lime_explainer,
    plot_lime_multiple_instances,
    get_representative_instances
)


def run_shap_analysis(experiment_data: dict, shap_output_dir: str):
    """Run SHAP analysis for both models using the trained experiment models."""
    print("\n" + "="*60)
    print("SHAP ANALYSIS")
    print("="*60)
    
    os.makedirs(shap_output_dir, exist_ok=True)
    
    feature_names = experiment_data['feature_names']
    results_multi = experiment_data['multi']
    results_binary = experiment_data['binary']
    
    # =========================================================================
    # Multi-class SHAP Analysis
    # =========================================================================
    print("\n--- Multi-class Model SHAP Analysis ---")
    
    explainer_multi = create_shap_explainer(results_multi['model'])
    shap_values_multi = compute_shap_values(
        explainer_multi, 
        results_multi['X_test'], 
        feature_names
    )
    
    # Summary plot for each class
    plot_shap_summary(
        shap_values_multi,
        results_multi['X_test'],
        feature_names,
        os.path.join(shap_output_dir, "summary_multiclass.png"),
        title="SHAP Summary - Multi-class (with SMOTE)",
        class_names=results_multi['class_names'],
        is_multiclass=True
    )
    
    # Bar plot for each class
    plot_shap_bar(
        shap_values_multi,
        os.path.join(shap_output_dir, "bar_multiclass.png"),
        title="SHAP Feature Importance - Multi-class (with SMOTE)",
        class_names=results_multi['class_names'],
        is_multiclass=True
    )
    
    # Waterfall plot for high quality wine
    high_class_idx = list(results_multi['class_names']).index('high')
    y_pred_multi = results_multi['model'].predict(results_multi['X_test'])
    
    high_quality_mask = (results_multi['y_test'] == high_class_idx) & (y_pred_multi == high_class_idx)
    high_quality_indices = np.where(high_quality_mask)[0]
    
    if len(high_quality_indices) > 0:
        plot_shap_waterfall(
            shap_values_multi,
            high_quality_indices[0],
            os.path.join(shap_output_dir, "waterfall_high_quality.png"),
            title="SHAP Waterfall - High Quality Wine Instance (with SMOTE)",
            class_idx=high_class_idx
        )
    
    # =========================================================================
    # Binary SHAP Analysis
    # =========================================================================
    print("\n--- Binary Model SHAP Analysis ---")
    
    explainer_binary = create_shap_explainer(results_binary['model'])
    shap_values_binary = compute_shap_values(
        explainer_binary,
        results_binary['X_test'],
        feature_names
    )
    
    # Summary plot
    plot_shap_summary(
        shap_values_binary,
        results_binary['X_test'],
        feature_names,
        os.path.join(shap_output_dir, "summary_binary.png"),
        title="SHAP Summary - Binary is_premium (with SMOTE)"
    )
    
    # Bar plot
    plot_shap_bar(
        shap_values_binary,
        os.path.join(shap_output_dir, "bar_binary.png"),
        title="SHAP Feature Importance - Binary is_premium (with SMOTE)"
    )
    
    # Waterfall for a premium wine
    y_pred_binary = results_binary['model'].predict(results_binary['X_test'])
    premium_mask = (results_binary['y_test'] == 1) & (y_pred_binary == 1)
    premium_indices = np.where(premium_mask)[0]
    
    if len(premium_indices) > 0:
        plot_shap_waterfall(
            shap_values_binary,
            premium_indices[0],
            os.path.join(shap_output_dir, "waterfall_premium.png"),
            title="SHAP Waterfall - Premium Wine Instance (with SMOTE)"
        )
    
    return shap_values_multi, shap_values_binary


def run_lime_analysis(experiment_data: dict, lime_output_dir: str):
    """Run LIME analysis for both models using the trained experiment models."""
    print("\n" + "="*60)
    print("LIME ANALYSIS")
    print("="*60)
    
    os.makedirs(lime_output_dir, exist_ok=True)
    
    feature_names = experiment_data['feature_names']
    results_multi = experiment_data['multi']
    results_binary = experiment_data['binary']
    
    # =========================================================================
    # Multi-class LIME Analysis
    # =========================================================================
    print("\n--- Multi-class Model LIME Analysis ---")
    
    lime_explainer_multi = create_lime_explainer(
        results_multi['X_train'],
        feature_names,
        results_multi['class_names']
    )
    
    # Get representative instances
    y_pred_multi = results_multi['model'].predict(results_multi['X_test'])
    instances_multi = get_representative_instances(
        results_multi['y_test'],
        y_pred_multi,
        n_per_class=1,
        include_misclassified=True
    )
    
    # Plot correct predictions
    if instances_multi['correct']:
        plot_lime_multiple_instances(
            lime_explainer_multi,
            results_multi['model'],
            results_multi['X_test'],
            results_multi['y_test'],
            instances_multi['correct'][:3],
            results_multi['class_names'],
            os.path.join(lime_output_dir, "multiclass_correct.png"),
            title="LIME Explanations - Multi-class Correct Predictions (with SMOTE)"
        )
    
    # Plot misclassified if any
    if instances_multi['misclassified']:
        plot_lime_multiple_instances(
            lime_explainer_multi,
            results_multi['model'],
            results_multi['X_test'],
            results_multi['y_test'],
            instances_multi['misclassified'][:3],
            results_multi['class_names'],
            os.path.join(lime_output_dir, "multiclass_misclassified.png"),
            title="LIME Explanations - Multi-class Misclassified (with SMOTE)"
        )
    
    # =========================================================================
    # Binary LIME Analysis
    # =========================================================================
    print("\n--- Binary Model LIME Analysis ---")
    
    lime_explainer_binary = create_lime_explainer(
        results_binary['X_train'],
        feature_names,
        results_binary['class_names']
    )
    
    # Get representative instances
    y_pred_binary = results_binary['model'].predict(results_binary['X_test'])
    instances_binary = get_representative_instances(
        results_binary['y_test'],
        y_pred_binary,
        n_per_class=1,
        include_misclassified=True
    )
    
    # Plot correct predictions
    if instances_binary['correct']:
        plot_lime_multiple_instances(
            lime_explainer_binary,
            results_binary['model'],
            results_binary['X_test'],
            results_binary['y_test'],
            instances_binary['correct'][:2],
            results_binary['class_names'],
            os.path.join(lime_output_dir, "binary_correct.png"),
            title="LIME Explanations - Binary Correct Predictions (with SMOTE)"
        )
    
    # Plot misclassified if any
    if instances_binary['misclassified']:
        plot_lime_multiple_instances(
            lime_explainer_binary,
            results_binary['model'],
            results_binary['X_test'],
            results_binary['y_test'],
            instances_binary['misclassified'][:2],
            results_binary['class_names'],
            os.path.join(lime_output_dir, "binary_misclassified.png"),
            title="LIME Explanations - Binary Misclassified (with SMOTE)"
        )


def main():
    """Main explainability pipeline."""
    print("="*60)
    print("XAI ANALYSIS: SHAP & LIME")
    print("Experiment: XGBoost with SMOTE")
    print("="*60)
    
    # Run the experiment to get trained models and data
    print("\n[Step 1] Running experiment to get trained models...")
    experiment_data = run_experiment(verbose=False)
    print(f"Experiment: {experiment_data['experiment_name']}")
    print(f"Feature names: {experiment_data['feature_names']}")
    print(f"Multi-class classes: {experiment_data['multi']['class_names']}")
    print(f"Binary classes: {experiment_data['binary']['class_names']}")
    
    # Setup output directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = os.path.join(script_dir, '..', 'explainability_results', experiment_data['experiment_name'])
    shap_output_dir = os.path.join(base_output_dir, 'shap')
    lime_output_dir = os.path.join(base_output_dir, 'lime')
    
    # Run SHAP analysis
    print("\n[Step 2] Running SHAP analysis...")
    run_shap_analysis(experiment_data, shap_output_dir)
    
    # Run LIME analysis
    print("\n[Step 3] Running LIME analysis...")
    run_lime_analysis(experiment_data, lime_output_dir)
    
    print("\n" + "="*60)
    print("XAI ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nSHAP results saved to: {shap_output_dir}")
    print(f"LIME results saved to: {lime_output_dir}")
    
    # List generated files
    print("\nGenerated SHAP files:")
    if os.path.exists(shap_output_dir):
        for f in sorted(os.listdir(shap_output_dir)):
            print(f"  - shap/{f}")
    
    print("\nGenerated LIME files:")
    if os.path.exists(lime_output_dir):
        for f in sorted(os.listdir(lime_output_dir)):
            print(f"  - lime/{f}")


if __name__ == "__main__":
    main()
