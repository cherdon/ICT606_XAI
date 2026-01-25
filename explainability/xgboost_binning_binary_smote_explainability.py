"""
SHAP and LIME Explainability Analysis for XGBoost Binary Classification (is_premium).
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

# Import the experiment's run function
from experiments.xgboost_binning_binary_smote import run_experiment

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
    """Run SHAP analysis for binary model."""
    print("\n" + "="*60)
    print("SHAP ANALYSIS")
    print("="*60)
    
    os.makedirs(shap_output_dir, exist_ok=True)
    
    feature_names = experiment_data['feature_names']
    results = experiment_data['results']
    
    print("\n--- Binary Model SHAP Analysis ---")
    
    explainer = create_shap_explainer(results['model'])
    shap_values = compute_shap_values(explainer, results['X_test'], feature_names)
    
    # Summary plot
    plot_shap_summary(
        shap_values,
        results['X_test'],
        feature_names,
        os.path.join(shap_output_dir, "summary.png"),
        title="SHAP Summary - Binary is_premium (with SMOTE)"
    )
    
    # Bar plot
    plot_shap_bar(
        shap_values,
        os.path.join(shap_output_dir, "bar.png"),
        title="SHAP Feature Importance - Binary is_premium (with SMOTE)"
    )
    
    # Waterfall for a premium wine
    y_pred = results['model'].predict(results['X_test'])
    premium_mask = (results['y_test'] == 1) & (y_pred == 1)
    premium_indices = np.where(premium_mask)[0]
    
    if len(premium_indices) > 0:
        plot_shap_waterfall(
            shap_values,
            premium_indices[0],
            os.path.join(shap_output_dir, "waterfall_premium.png"),
            title="SHAP Waterfall - Premium Wine Instance (with SMOTE)"
        )
    
    return shap_values


def run_lime_analysis(experiment_data: dict, lime_output_dir: str):
    """Run LIME analysis for binary model."""
    print("\n" + "="*60)
    print("LIME ANALYSIS")
    print("="*60)
    
    os.makedirs(lime_output_dir, exist_ok=True)
    
    feature_names = experiment_data['feature_names']
    results = experiment_data['results']
    
    print("\n--- Binary Model LIME Analysis ---")
    
    lime_explainer = create_lime_explainer(
        results['X_train'],
        feature_names,
        results['class_names']
    )
    
    # Get representative instances
    y_pred = results['model'].predict(results['X_test'])
    instances = get_representative_instances(
        results['y_test'],
        y_pred,
        n_per_class=1,
        include_misclassified=True
    )
    
    # Plot correct predictions
    if instances['correct']:
        plot_lime_multiple_instances(
            lime_explainer,
            results['model'],
            results['X_test'],
            results['y_test'],
            instances['correct'][:2],
            results['class_names'],
            os.path.join(lime_output_dir, "correct.png"),
            title="LIME Explanations - Correct Predictions (with SMOTE)"
        )
    
    # Plot misclassified if any
    if instances['misclassified']:
        plot_lime_multiple_instances(
            lime_explainer,
            results['model'],
            results['X_test'],
            results['y_test'],
            instances['misclassified'][:2],
            results['class_names'],
            os.path.join(lime_output_dir, "misclassified.png"),
            title="LIME Explanations - Misclassified (with SMOTE)"
        )


def main():
    """Main explainability pipeline."""
    print("="*60)
    print("XAI ANALYSIS: SHAP & LIME")
    print("Experiment: Binary Classification with SMOTE")
    print("="*60)
    
    # Run the experiment to get trained models and data
    print("\n[Step 1] Running experiment to get trained models...")
    experiment_data = run_experiment(verbose=False)
    print(f"Experiment: {experiment_data['experiment_name']}")
    print(f"Feature names: {experiment_data['feature_names']}")
    print(f"Classes: {experiment_data['results']['class_names']}")
    
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


if __name__ == "__main__":
    main()
