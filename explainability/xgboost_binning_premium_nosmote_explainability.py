"""
SHAP and LIME Explainability Analysis for XGBoost Wine Quality Prediction.
WITHOUT SMOTE - uses original imbalanced data.

This script:
1. Trains the same XGBoost models as the experiment
2. Generates SHAP explanations (summary, bar, waterfall plots)
3. Generates LIME explanations for representative instances
4. Saves all visualizations to experiment_results/

Model Performance (without SMOTE):
- Multi-class: Accuracy=0.6029, F1=0.6008, AUC-ROC=0.7466
- Binary: Accuracy=0.8787, F1=0.4407, AUC-ROC=0.8635
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from preprocessing import (
    load_wine_data,
    bin_column,
    create_binary_column,
    handle_missing_values,
    remove_duplicates
)

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

# Experiment name
EXPERIMENT_NAME = "xgboost_binning_premium_nosmote"


def prepare_data():
    """Prepare data WITHOUT SMOTE."""
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'winequality-red.csv')
    
    # Load data
    df = load_wine_data(data_path, wine_type='red')
    
    # Prepare multi-class dataset
    bin_logic = {
        'low': (None, 6),
        'medium': (6, 7),
        'high': (7, None)
    }
    df_multiclass = bin_column(
        df.copy(), 
        column='quality', 
        bin_logic=bin_logic,
        new_column_name='quality_category',
        drop_original=True
    )
    
    # Prepare binary dataset
    df_binary = create_binary_column(
        df.copy(),
        column='quality',
        threshold=7,
        new_column_name='is_premium',
        drop_original=True
    )
    
    # Clean data
    df_multiclass = handle_missing_values(df_multiclass, strategy='drop')
    df_multiclass = remove_duplicates(df_multiclass)
    df_binary = handle_missing_values(df_binary, strategy='drop')
    df_binary = remove_duplicates(df_binary)
    
    # Prepare features
    feature_cols = [col for col in df_multiclass.columns if col != 'quality_category']
    
    # Multi-class
    X_multi = df_multiclass[feature_cols].values
    y_multi_raw = df_multiclass['quality_category'].values
    label_encoder = LabelEncoder()
    y_multi = label_encoder.fit_transform(y_multi_raw)
    
    # Binary
    X_binary = df_binary[feature_cols].values
    y_binary = df_binary['is_premium'].values
    
    # NO SMOTE - use original data directly
    
    # Train-test split
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y_multi,
        test_size=TEST_SIZE, stratify=y_multi, random_state=RANDOM_STATE
    )
    
    X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(
        X_binary, y_binary,
        test_size=TEST_SIZE, stratify=y_binary, random_state=RANDOM_STATE
    )
    
    return {
        'multi': {
            'X_train': X_train_multi,
            'X_test': X_test_multi,
            'y_train': y_train_multi,
            'y_test': y_test_multi,
            'label_encoder': label_encoder,
            'class_names': list(label_encoder.classes_)
        },
        'binary': {
            'X_train': X_train_binary,
            'X_test': X_test_binary,
            'y_train': y_train_binary,
            'y_test': y_test_binary,
            'class_names': ['Not Premium', 'Premium']
        },
        'feature_names': feature_cols
    }


def train_models(data: dict) -> dict:
    """Train XGBoost models."""
    RANDOM_STATE = 42
    
    # Multi-class model
    model_multi = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric='mlogloss'
    )
    model_multi.fit(data['multi']['X_train'], data['multi']['y_train'])
    
    # Binary model
    model_binary = XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )
    model_binary.fit(data['binary']['X_train'], data['binary']['y_train'])
    
    return {
        'multi': model_multi,
        'binary': model_binary
    }


def run_shap_analysis(models: dict, data: dict, output_dir: str):
    """Run SHAP analysis for both models."""
    print("\n" + "="*60)
    print("SHAP ANALYSIS")
    print("="*60)
    
    feature_names = data['feature_names']
    
    # =========================================================================
    # Multi-class SHAP Analysis
    # =========================================================================
    print("\n--- Multi-class Model SHAP Analysis ---")
    
    explainer_multi = create_shap_explainer(models['multi'])
    shap_values_multi = compute_shap_values(
        explainer_multi, 
        data['multi']['X_test'], 
        feature_names
    )
    
    # Summary plot for each class
    plot_shap_summary(
        shap_values_multi,
        data['multi']['X_test'],
        feature_names,
        os.path.join(output_dir, f"{EXPERIMENT_NAME}_shap_summary_multiclass.png"),
        title="SHAP Summary - Multi-class (without SMOTE)",
        class_names=data['multi']['class_names'],
        is_multiclass=True
    )
    
    # Bar plot for each class
    plot_shap_bar(
        shap_values_multi,
        os.path.join(output_dir, f"{EXPERIMENT_NAME}_shap_bar_multiclass.png"),
        title="SHAP Feature Importance - Multi-class (without SMOTE)",
        class_names=data['multi']['class_names'],
        is_multiclass=True
    )
    
    # Waterfall plot for high quality wine (class index for 'high')
    high_class_idx = list(data['multi']['class_names']).index('high')
    y_pred_multi = models['multi'].predict(data['multi']['X_test'])
    
    # Find a correctly predicted high quality instance
    high_quality_mask = (data['multi']['y_test'] == high_class_idx) & (y_pred_multi == high_class_idx)
    high_quality_indices = np.where(high_quality_mask)[0]
    
    if len(high_quality_indices) > 0:
        plot_shap_waterfall(
            shap_values_multi,
            high_quality_indices[0],
            os.path.join(output_dir, f"{EXPERIMENT_NAME}_shap_waterfall_high_quality.png"),
            title="SHAP Waterfall - High Quality Wine Instance (without SMOTE)",
            class_idx=high_class_idx
        )
    
    # =========================================================================
    # Binary SHAP Analysis
    # =========================================================================
    print("\n--- Binary Model SHAP Analysis ---")
    
    explainer_binary = create_shap_explainer(models['binary'])
    shap_values_binary = compute_shap_values(
        explainer_binary,
        data['binary']['X_test'],
        feature_names
    )
    
    # Summary plot
    plot_shap_summary(
        shap_values_binary,
        data['binary']['X_test'],
        feature_names,
        os.path.join(output_dir, f"{EXPERIMENT_NAME}_shap_summary_binary.png"),
        title="SHAP Summary - Binary is_premium (without SMOTE)"
    )
    
    # Bar plot
    plot_shap_bar(
        shap_values_binary,
        os.path.join(output_dir, f"{EXPERIMENT_NAME}_shap_bar_binary.png"),
        title="SHAP Feature Importance - Binary is_premium (without SMOTE)"
    )
    
    # Waterfall for a premium wine
    y_pred_binary = models['binary'].predict(data['binary']['X_test'])
    premium_mask = (data['binary']['y_test'] == 1) & (y_pred_binary == 1)
    premium_indices = np.where(premium_mask)[0]
    
    if len(premium_indices) > 0:
        plot_shap_waterfall(
            shap_values_binary,
            premium_indices[0],
            os.path.join(output_dir, f"{EXPERIMENT_NAME}_shap_waterfall_premium.png"),
            title="SHAP Waterfall - Premium Wine Instance (without SMOTE)"
        )
    
    return shap_values_multi, shap_values_binary


def run_lime_analysis(models: dict, data: dict, output_dir: str):
    """Run LIME analysis for both models."""
    print("\n" + "="*60)
    print("LIME ANALYSIS")
    print("="*60)
    
    feature_names = data['feature_names']
    
    # =========================================================================
    # Multi-class LIME Analysis
    # =========================================================================
    print("\n--- Multi-class Model LIME Analysis ---")
    
    lime_explainer_multi = create_lime_explainer(
        data['multi']['X_train'],
        feature_names,
        data['multi']['class_names']
    )
    
    # Get representative instances
    y_pred_multi = models['multi'].predict(data['multi']['X_test'])
    instances_multi = get_representative_instances(
        data['multi']['y_test'],
        y_pred_multi,
        n_per_class=1,
        include_misclassified=True
    )
    
    # Plot correct predictions
    if instances_multi['correct']:
        plot_lime_multiple_instances(
            lime_explainer_multi,
            models['multi'],
            data['multi']['X_test'],
            data['multi']['y_test'],
            instances_multi['correct'][:3],  # Limit to 3 instances
            data['multi']['class_names'],
            os.path.join(output_dir, f"{EXPERIMENT_NAME}_lime_multiclass_correct.png"),
            title="LIME Explanations - Multi-class Correct Predictions (without SMOTE)"
        )
    
    # Plot misclassified if any
    if instances_multi['misclassified']:
        plot_lime_multiple_instances(
            lime_explainer_multi,
            models['multi'],
            data['multi']['X_test'],
            data['multi']['y_test'],
            instances_multi['misclassified'][:3],
            data['multi']['class_names'],
            os.path.join(output_dir, f"{EXPERIMENT_NAME}_lime_multiclass_misclassified.png"),
            title="LIME Explanations - Multi-class Misclassified (without SMOTE)"
        )
    
    # =========================================================================
    # Binary LIME Analysis
    # =========================================================================
    print("\n--- Binary Model LIME Analysis ---")
    
    lime_explainer_binary = create_lime_explainer(
        data['binary']['X_train'],
        feature_names,
        data['binary']['class_names']
    )
    
    # Get representative instances
    y_pred_binary = models['binary'].predict(data['binary']['X_test'])
    instances_binary = get_representative_instances(
        data['binary']['y_test'],
        y_pred_binary,
        n_per_class=1,
        include_misclassified=True
    )
    
    # Plot correct predictions (one premium, one not premium)
    if instances_binary['correct']:
        plot_lime_multiple_instances(
            lime_explainer_binary,
            models['binary'],
            data['binary']['X_test'],
            data['binary']['y_test'],
            instances_binary['correct'][:2],
            data['binary']['class_names'],
            os.path.join(output_dir, f"{EXPERIMENT_NAME}_lime_binary_correct.png"),
            title="LIME Explanations - Binary Correct Predictions (without SMOTE)"
        )
    
    # Plot misclassified if any
    if instances_binary['misclassified']:
        plot_lime_multiple_instances(
            lime_explainer_binary,
            models['binary'],
            data['binary']['X_test'],
            data['binary']['y_test'],
            instances_binary['misclassified'][:2],
            data['binary']['class_names'],
            os.path.join(output_dir, f"{EXPERIMENT_NAME}_lime_binary_misclassified.png"),
            title="LIME Explanations - Binary Misclassified (without SMOTE)"
        )


def main():
    """Main explainability pipeline."""
    print("="*60)
    print("XAI ANALYSIS: SHAP & LIME")
    print("Experiment: XGBoost WITHOUT SMOTE")
    print("="*60)
    
    # Setup output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'explainability_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    print("\n[Step 1] Preparing data (NO SMOTE)...")
    data = prepare_data()
    print(f"Feature names: {data['feature_names']}")
    print(f"Multi-class classes: {data['multi']['class_names']}")
    print(f"Binary classes: {data['binary']['class_names']}")
    
    # Train models
    print("\n[Step 2] Training models...")
    models = train_models(data)
    print("Models trained successfully.")
    
    # Run SHAP analysis
    print("\n[Step 3] Running SHAP analysis...")
    run_shap_analysis(models, data, output_dir)
    
    # Run LIME analysis
    print("\n[Step 4] Running LIME analysis...")
    run_lime_analysis(models, data, output_dir)
    
    print("\n" + "="*60)
    print("XAI ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nAll visualizations saved to: {output_dir}")
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        if EXPERIMENT_NAME in f and ('shap' in f.lower() or 'lime' in f.lower()):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
