"""
XGBoost experiment with binning strategies for wine quality prediction.
WITHOUT SMOTE - uses original imbalanced data.

================================================================================
RESULTS SUMMARY (without SMOTE)
================================================================================

Multi-class Classification (low/medium/high):
  - Accuracy:      0.6029
  - F1 (weighted): 0.6008
  - F1 (macro):    0.5754
  - AUC-ROC:       0.7466

Binary Classification (is_premium):
  - Accuracy:      0.8787
  - F1-Score:      0.4407
  - AUC-ROC:       0.8635

================================================================================

This experiment:
1. Loads red wine data
2. Creates two datasets: multi-class (low/medium/high) and binary (is_premium)
3. Does NOT apply SMOTE (uses original imbalanced data)
4. Trains XGBoost models for both classification tasks
5. Evaluates using Accuracy, F1-Score, and AUC-ROC
6. Saves confusion matrix visualizations

Goal: Determine XAI for why a wine is good quality (baseline without SMOTE)
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from preprocessing import (
    load_wine_data,
    bin_column,
    create_binary_column,
    check_class_distribution,
    handle_missing_values,
    remove_duplicates
)

# Experiment name (used for saving results)
EXPERIMENT_NAME = os.path.splitext(os.path.basename(__file__))[0]


def prepare_multiclass_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataset for multi-class classification.
    Bins quality into: low (<6), medium (6), high (>=7)
    """
    bin_logic = {
        'low': (None, 6),      # quality < 6
        'medium': (6, 7),      # quality >= 6 and < 7 (i.e., quality == 6)
        'high': (7, None)      # quality >= 7
    }
    
    df_binned = bin_column(
        df, 
        column='quality', 
        bin_logic=bin_logic,
        new_column_name='quality_category',
        drop_original=True
    )
    
    return df_binned


def prepare_binary_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataset for binary classification.
    Creates is_premium: 1 if quality >= 7, else 0
    """
    df_binary = create_binary_column(
        df,
        column='quality',
        threshold=7,
        new_column_name='is_premium',
        drop_original=True
    )
    
    return df_binary


def clean_for_xgboost(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean data for XGBoost training.
    """
    df = handle_missing_values(df, strategy='drop')
    df = remove_duplicates(df)
    
    if verbose:
        print(f"Data shape after cleaning: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
    
    return df


def print_class_distribution(y: np.ndarray, dataset_name: str = "Dataset", verbose: bool = True):
    """
    Print class distribution for a dataset.
    """
    if not verbose:
        return
        
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    percentages = counts / total * 100
    
    print(f"\n{dataset_name} class distribution:")
    for cls, count, pct in zip(unique, counts, percentages):
        print(f"  Class {cls}: {count} samples ({pct:.2f}%)")


def train_and_evaluate_multiclass(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    random_state: int = 42,
    verbose: bool = True
) -> dict:
    """
    Train and evaluate XGBoost for multi-class classification.
    """
    if verbose:
        print("\n" + "="*60)
        print("MULTI-CLASS CLASSIFICATION (low/medium/high)")
        print("="*60)
    
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    try:
        auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    except ValueError as e:
        if verbose:
            print(f"Warning: Could not compute AUC-ROC: {e}")
        auc_roc = None
    
    cm = confusion_matrix(y_test, y_pred)
    
    if verbose:
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"F1-Score (weighted): {f1_weighted:.4f}")
        print(f"F1-Score (macro): {f1_macro:.4f}")
        if auc_roc is not None:
            print(f"AUC-ROC (weighted OvR): {auc_roc:.4f}")
        
        class_names = label_encoder.classes_
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        print("Confusion Matrix:")
        print(pd.DataFrame(cm, index=class_names, columns=class_names))
    
    return {
        'model': model,
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'auc_roc': auc_roc,
        'label_encoder': label_encoder,
        'y_test': y_test,
        'y_pred': y_pred,
        'confusion_matrix': cm
    }


def train_and_evaluate_binary(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    verbose: bool = True
) -> dict:
    """
    Train and evaluate XGBoost for binary classification (is_premium).
    """
    if verbose:
        print("\n" + "="*60)
        print("BINARY CLASSIFICATION (is_premium)")
        print("="*60)
    
    model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    cm = confusion_matrix(y_test, y_pred)
    
    if verbose:
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Premium', 'Premium']))
        
        print("Confusion Matrix:")
        print(pd.DataFrame(cm, 
                           index=['Not Premium', 'Premium'], 
                           columns=['Not Premium', 'Premium']))
    
    return {
        'model': model,
        'accuracy': accuracy,
        'f1': f1,
        'auc_roc': auc_roc,
        'y_test': y_test,
        'y_pred': y_pred,
        'confusion_matrix': cm
    }


def plot_confusion_matrices(
    results_multi: dict,
    results_binary: dict,
    output_dir: str,
    experiment_name: str
):
    """
    Plot and save confusion matrices for both models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    class_names_multi = results_multi['label_encoder'].classes_
    cm_multi = results_multi['confusion_matrix']
    
    sns.heatmap(
        cm_multi, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names_multi,
        yticklabels=class_names_multi,
        ax=axes[0]
    )
    axes[0].set_title('Multi-class Classification\n(low/medium/high)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=10)
    axes[0].set_ylabel('Actual', fontsize=10)
    
    metrics_text_multi = (
        f"Accuracy: {results_multi['accuracy']:.4f}\n"
        f"F1 (weighted): {results_multi['f1_weighted']:.4f}\n"
        f"AUC-ROC: {results_multi['auc_roc']:.4f}"
    )
    axes[0].text(
        1.02, 0.5, metrics_text_multi,
        transform=axes[0].transAxes,
        fontsize=9,
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    )
    
    class_names_binary = ['Not Premium', 'Premium']
    cm_binary = results_binary['confusion_matrix']
    
    sns.heatmap(
        cm_binary, 
        annot=True, 
        fmt='d', 
        cmap='Greens',
        xticklabels=class_names_binary,
        yticklabels=class_names_binary,
        ax=axes[1]
    )
    axes[1].set_title('Binary Classification\n(is_premium)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=10)
    axes[1].set_ylabel('Actual', fontsize=10)
    
    metrics_text_binary = (
        f"Accuracy: {results_binary['accuracy']:.4f}\n"
        f"F1-Score: {results_binary['f1']:.4f}\n"
        f"AUC-ROC: {results_binary['auc_roc']:.4f}"
    )
    axes[1].text(
        1.02, 0.5, metrics_text_binary,
        transform=axes[1].transAxes,
        fontsize=9,
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
    )
    
    fig.suptitle(f'Confusion Matrices - {experiment_name}', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{experiment_name}_confusion_matrices.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nConfusion matrices saved to: {output_path}")
    
    return output_path


def run_experiment(verbose: bool = True) -> dict:
    """
    Run the full experiment and return all data needed for explainability.
    
    This function is designed to be called by explainability scripts.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'experiment_name': Name of the experiment
        - 'feature_names': List of feature names
        - 'multi': Dict with multiclass model, data, and results
        - 'binary': Dict with binary model, data, and results
    """
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'winequality-red.csv')
    
    if verbose:
        print("="*60)
        print("WINE QUALITY PREDICTION WITH XGBoost (WITHOUT SMOTE)")
        print("="*60)
        print("\n[Step 1] Loading red wine data...")
    
    df = load_wine_data(data_path, wine_type='red')
    
    if verbose:
        print(f"\nOriginal data shape: {df.shape}")
        print(f"\nQuality distribution in original data:")
        print(df['quality'].value_counts().sort_index())
        print("\n[Step 2] Creating dataset copies...")
    
    df_multiclass = df.copy()
    df_binary = df.copy()
    
    if verbose:
        print("\n[Step 3] Preparing multi-class dataset...")
    df_multiclass = prepare_multiclass_dataset(df_multiclass)
    if verbose:
        print("\nMulti-class distribution:")
        print(check_class_distribution(df_multiclass, 'quality_category'))
        print("\n[Step 4] Preparing binary dataset...")
    
    df_binary = prepare_binary_dataset(df_binary)
    if verbose:
        print("\nBinary distribution:")
        print(check_class_distribution(df_binary, 'is_premium'))
        print("\n[Step 5] Cleaning data for XGBoost...")
        print("\nMulti-class dataset:")
    
    df_multiclass = clean_for_xgboost(df_multiclass, verbose=verbose)
    if verbose:
        print("\nBinary dataset:")
    df_binary = clean_for_xgboost(df_binary, verbose=verbose)
    
    # Prepare features and targets
    feature_cols = [col for col in df_multiclass.columns if col != 'quality_category']
    X_multi = df_multiclass[feature_cols].values
    y_multi_raw = df_multiclass['quality_category'].values
    
    label_encoder = LabelEncoder()
    y_multi = label_encoder.fit_transform(y_multi_raw)
    if verbose:
        print(f"\nLabel encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    X_binary = df_binary[feature_cols].values
    y_binary = df_binary['is_premium'].values
    
    if verbose:
        print("\n[Step 6] Class distribution (NO SMOTE applied)...")
    print_class_distribution(y_multi, "Multi-class", verbose=verbose)
    print_class_distribution(y_binary, "Binary", verbose=verbose)
    
    if verbose:
        print("\n[Step 7] Performing stratified train-test split...")
    
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y_multi,
        test_size=TEST_SIZE,
        stratify=y_multi,
        random_state=RANDOM_STATE
    )
    if verbose:
        print(f"\nMulti-class - Train: {len(y_train_multi)}, Test: {len(y_test_multi)}")
    
    X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(
        X_binary, y_binary,
        test_size=TEST_SIZE,
        stratify=y_binary,
        random_state=RANDOM_STATE
    )
    if verbose:
        print(f"Binary - Train: {len(y_train_binary)}, Test: {len(y_test_binary)}")
        print("\n[Step 8 & 9] Training XGBoost and evaluating...")
    
    results_multi = train_and_evaluate_multiclass(
        X_train_multi, X_test_multi,
        y_train_multi, y_test_multi,
        label_encoder,
        random_state=RANDOM_STATE,
        verbose=verbose
    )
    
    results_binary = train_and_evaluate_binary(
        X_train_binary, X_test_binary,
        y_train_binary, y_test_binary,
        random_state=RANDOM_STATE,
        verbose=verbose
    )
    
    # Add data to results for explainability
    results_multi['X_train'] = X_train_multi
    results_multi['X_test'] = X_test_multi
    results_multi['y_train'] = y_train_multi
    results_multi['feature_names'] = feature_cols
    results_multi['class_names'] = list(label_encoder.classes_)
    
    results_binary['X_train'] = X_train_binary
    results_binary['X_test'] = X_test_binary
    results_binary['y_train'] = y_train_binary
    results_binary['feature_names'] = feature_cols
    results_binary['class_names'] = ['Not Premium', 'Premium']
    
    if verbose:
        print("\n" + "="*60)
        print("SUMMARY OF RESULTS (WITHOUT SMOTE)")
        print("="*60)
        
        print("\nMulti-class Classification (low/medium/high):")
        print(f"  Accuracy:     {results_multi['accuracy']:.4f}")
        print(f"  F1 (weighted): {results_multi['f1_weighted']:.4f}")
        print(f"  F1 (macro):    {results_multi['f1_macro']:.4f}")
        if results_multi['auc_roc'] is not None:
            print(f"  AUC-ROC:      {results_multi['auc_roc']:.4f}")
        
        print("\nBinary Classification (is_premium):")
        print(f"  Accuracy: {results_binary['accuracy']:.4f}")
        print(f"  F1-Score: {results_binary['f1']:.4f}")
        print(f"  AUC-ROC:  {results_binary['auc_roc']:.4f}")
    
    return {
        'experiment_name': EXPERIMENT_NAME,
        'feature_names': feature_cols,
        'multi': results_multi,
        'binary': results_binary
    }


def main():
    """Main experiment pipeline with visualization."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'experiment_results')
    
    # Run experiment
    experiment_data = run_experiment(verbose=True)
    
    # Save confusion matrices
    print("\n[Step 10] Saving confusion matrix visualizations...")
    plot_confusion_matrices(
        experiment_data['multi'], 
        experiment_data['binary'], 
        output_dir, 
        experiment_data['experiment_name']
    )
    
    return experiment_data['multi'], experiment_data['binary']


if __name__ == "__main__":
    results_multi, results_binary = main()
