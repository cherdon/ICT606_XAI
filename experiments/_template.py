"""
EXPERIMENT TEMPLATE
===================

Copy this file and rename it to create a new experiment.
File name becomes the experiment name (e.g., xgboost_binary_smote.py -> "xgboost_binary_smote")

Instructions:
1. Copy this file: cp _template.py my_new_experiment.py
2. Update the docstring with your experiment description and results
3. Implement the required functions: prepare_data(), train_and_evaluate()
4. Run: python experiments/my_new_experiment.py

The experiment will automatically:
- Save confusion matrix to experiment_results/
- Update all_experiments.csv with metrics

================================================================================
RESULTS SUMMARY
================================================================================

[TO BE UPDATED after running]

================================================================================
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
from experiments.experiment_tracker import save_experiment_results

# Experiment name is automatically derived from filename
EXPERIMENT_NAME = os.path.splitext(os.path.basename(__file__))[0]

# ============================================================================
# CONFIGURATION - Modify these for your experiment
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
USE_SMOTE = False  # Set to True to use SMOTE for class balancing
MIN_CLASS_PERCENTAGE = 0.15  # Threshold for SMOTE (if enabled)
IS_BINARY = True  # Set to False for multi-class classification


# ============================================================================
# DATA PREPARATION - Implement your data preparation logic
# ============================================================================
def prepare_data(df: pd.DataFrame, verbose: bool = True) -> tuple:
    """
    Prepare the data for the experiment.
    
    Returns
    -------
    tuple: (X, y, feature_names, class_names, label_encoder or None)
    """
    # Example for binary classification:
    if IS_BINARY:
        df = create_binary_column(
            df,
            column='quality',
            threshold=7,
            new_column_name='target',
            drop_original=True
        )
        target_col = 'target'
        class_names = ['Not Premium', 'Premium']
        label_encoder = None
    else:
        # Example for multi-class classification:
        bin_logic = {
            'low': (None, 6),
            'medium': (6, 7),
            'high': (7, None)
        }
        df = bin_column(
            df,
            column='quality',
            bin_logic=bin_logic,
            new_column_name='target',
            drop_original=True
        )
        target_col = 'target'
        label_encoder = LabelEncoder()
        class_names = None  # Will be set after encoding
    
    # Clean data
    df = handle_missing_values(df, strategy='drop')
    df = remove_duplicates(df)
    
    if verbose:
        print(f"Data shape after cleaning: {df.shape}")
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].values
    
    if label_encoder:
        y = label_encoder.fit_transform(df[target_col].values)
        class_names = list(label_encoder.classes_)
    else:
        y = df[target_col].values
    
    return X, y, feature_cols, class_names, label_encoder


# ============================================================================
# MODEL TRAINING - Implement your model training logic
# ============================================================================
def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    label_encoder=None,
    verbose: bool = True
) -> dict:
    """
    Train and evaluate the model.
    
    Returns
    -------
    dict: Results including model, metrics, predictions, confusion matrix
    """
    if IS_BINARY:
        model = XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric='logloss'
        )
    else:
        model = XGBClassifier(
            objective='multi:softprob',
            num_class=len(class_names),
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric='mlogloss'
        )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    if IS_BINARY:
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1])
        f1_weighted = None
        f1_macro = None
    else:
        f1 = None
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        try:
            auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc_roc = None
    
    cm = confusion_matrix(y_test, y_pred)
    
    if verbose:
        print(f"\nAccuracy: {accuracy:.4f}")
        if IS_BINARY:
            print(f"F1-Score: {f1:.4f}")
        else:
            print(f"F1 (weighted): {f1_weighted:.4f}")
            print(f"F1 (macro): {f1_macro:.4f}")
        if auc_roc:
            print(f"AUC-ROC: {auc_roc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        print("Confusion Matrix:")
        print(pd.DataFrame(cm, index=class_names, columns=class_names))
    
    return {
        'model': model,
        'accuracy': accuracy,
        'f1': f1,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'auc_roc': auc_roc,
        'y_test': y_test,
        'y_pred': y_pred,
        'confusion_matrix': cm,
        'class_names': class_names,
        'label_encoder': label_encoder
    }


# ============================================================================
# PLOTTING - Confusion matrix visualization
# ============================================================================
def plot_confusion_matrix(results: dict, output_dir: str, experiment_name: str):
    """Plot and save confusion matrix."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    
    class_names = results['class_names']
    cm = results['confusion_matrix']
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues' if not IS_BINARY else 'Greens',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f'Confusion Matrix - {experiment_name}', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=10)
    plt.ylabel('Actual', fontsize=10)
    
    # Add metrics
    if IS_BINARY:
        metrics_text = (
            f"Accuracy: {results['accuracy']:.4f}\n"
            f"F1-Score: {results['f1']:.4f}\n"
            f"AUC-ROC: {results['auc_roc']:.4f}"
        )
    else:
        metrics_text = (
            f"Accuracy: {results['accuracy']:.4f}\n"
            f"F1 (weighted): {results['f1_weighted']:.4f}\n"
            f"AUC-ROC: {results['auc_roc']:.4f}"
        )
    
    plt.text(
        1.35, 0.5, metrics_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    )
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{experiment_name}_confusion_matrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nConfusion matrix saved to: {output_path}")


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================
def run_experiment(verbose: bool = True) -> dict:
    """
    Run the full experiment pipeline.
    
    This function is called by the explainability runner.
    
    Returns
    -------
    dict: Contains experiment_name, feature_names, results
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'winequality-red.csv')
    
    if verbose:
        print("="*60)
        print(f"EXPERIMENT: {EXPERIMENT_NAME}")
        print("="*60)
        print("\n[Step 1] Loading data...")
    
    df = load_wine_data(data_path, wine_type='red')
    
    if verbose:
        print(f"Original data shape: {df.shape}")
        print("\n[Step 2] Preparing data...")
    
    X, y, feature_names, class_names, label_encoder = prepare_data(df.copy(), verbose=verbose)
    
    # Apply SMOTE if enabled
    if USE_SMOTE:
        if verbose:
            print("\n[Step 3] Applying SMOTE...")
        from imblearn.over_sampling import SMOTE
        unique, counts = np.unique(y, return_counts=True)
        min_pct = counts.min() / len(y)
        
        if min_pct < MIN_CLASS_PERCENTAGE:
            smote = SMOTE(random_state=RANDOM_STATE)
            X, y = smote.fit_resample(X, y)
            if verbose:
                print(f"SMOTE applied. New shape: {X.shape}")
        else:
            if verbose:
                print("SMOTE not needed - classes balanced enough.")
    
    if verbose:
        step = 4 if USE_SMOTE else 3
        print(f"\n[Step {step}] Splitting data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    
    if verbose:
        print(f"Train: {len(y_train)}, Test: {len(y_test)}")
        step = 5 if USE_SMOTE else 4
        print(f"\n[Step {step}] Training and evaluating...")
    
    results = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        class_names, label_encoder, verbose=verbose
    )
    
    # Add data to results for explainability
    results['X_train'] = X_train
    results['X_test'] = X_test
    results['y_train'] = y_train
    results['feature_names'] = feature_names
    
    return {
        'experiment_name': EXPERIMENT_NAME,
        'feature_names': feature_names,
        'is_binary': IS_BINARY,
        'results': results
    }


def main():
    """Main entry point when running the experiment directly."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'experiment_results')
    
    # Run experiment
    experiment_data = run_experiment(verbose=True)
    results = experiment_data['results']
    
    # Save confusion matrix
    print("\n[Final] Saving confusion matrix...")
    plot_confusion_matrix(results, output_dir, EXPERIMENT_NAME)
    
    # Save to CSV tracker
    print("\n[Final] Updating experiments tracker...")
    metrics = {'accuracy': results['accuracy'], 'auc_roc': results['auc_roc']}
    if IS_BINARY:
        metrics['f1'] = results['f1']
    else:
        metrics['f1_weighted'] = results['f1_weighted']
        metrics['f1_macro'] = results['f1_macro']
    
    save_experiment_results(
        experiment_name=EXPERIMENT_NAME,
        results=metrics,
        output_dir=output_dir
    )
    
    return experiment_data


if __name__ == "__main__":
    experiment_data = main()
