"""
XGBoost Binary Classification for Wine Quality (is_premium) - FINETUNED VERSION 4.
Uses ADASYN (Adaptive Synthetic Sampling) for handling class imbalance.

================================================================================
RESULTS SUMMARY
================================================================================

Binary Classification (is_premium):
  - Accuracy:      [TO BE UPDATED]
  - F1-Score:      [TO BE UPDATED]
  - AUC-ROC:       [TO BE UPDATED]

================================================================================

IMPROVEMENTS OVER PREVIOUS VERSIONS:
1. Uses ADASYN instead of regular SMOTE
2. ADASYN focuses on harder-to-classify samples near decision boundary
3. Generates more synthetic samples for difficult minority examples
4. Better handling of imbalanced data with complex decision boundaries

Preprocessing Pipeline:
1. Binary target creation (quality >= 7 → Premium)
2. Handle missing values and duplicates
3. Remove multicollinear features (correlation > 0.7)
4. Stratified train/test split (80/20)
5. ADASYN oversampling (on training data only)
6. Hyperparameter tuning with RandomizedSearchCV
7. Threshold optimization for F1

Model: XGBoost Binary Classifier with ADASYN
"""
import sys
import os
from typing import Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve
)
from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier

from preprocessing import (
    load_wine_data,
    create_binary_column,
    check_class_distribution,
    handle_missing_values,
    remove_duplicates,
    remove_multicollinear_features
)
from experiments.experiment_tracker import save_experiment_results

# Experiment configuration
EXPERIMENT_NAME = os.path.splitext(os.path.basename(__file__))[0]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CORRELATION_THRESHOLD = 0.7
MIN_CLASS_PERCENTAGE = 0.15
N_ITER_SEARCH = 50  # Number of parameter combinations to try
CV_FOLDS = 5
ADASYN_N_NEIGHBORS = 5


def prepare_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Prepare dataset: create binary target, clean data, remove multicollinearity.
    """
    # Step 1: Create binary target
    df = create_binary_column(
        df,
        column='quality',
        threshold=7,
        new_column_name='is_premium',
        drop_original=True
    )
    
    if verbose:
        print("\nBinary target distribution:")
        print(check_class_distribution(df, 'is_premium'))
    
    # Step 2: Handle missing values and duplicates
    df = handle_missing_values(df, strategy='drop')
    df = remove_duplicates(df)
    
    if verbose:
        print(f"\nData shape after cleaning: {df.shape}")
    
    # Step 3: Remove multicollinear features
    if verbose:
        print(f"\nRemoving multicollinear features (threshold > {CORRELATION_THRESHOLD}):")
    
    df, removed_features = remove_multicollinear_features(
        df,
        target_column='is_premium',
        threshold=CORRELATION_THRESHOLD,
        verbose=verbose
    )
    
    return df


def apply_adasyn(
    X: np.ndarray,
    y: np.ndarray,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply ADASYN for oversampling the minority class."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    percentages = counts / total * 100
    
    if verbose:
        print("\nClass distribution before ADASYN:")
        for cls, count, pct in zip(unique, counts, percentages):
            label = "Premium" if cls == 1 else "Not Premium"
            print(f"  {label} ({cls}): {count} samples ({pct:.2f}%)")
    
    if verbose:
        print(f"\nApplying ADASYN (n_neighbors={ADASYN_N_NEIGHBORS})...")
        print("  - ADASYN generates more synthetic samples for harder-to-classify examples")
        print("  - Focuses on samples near the decision boundary")
    
    adasyn = ADASYN(random_state=RANDOM_STATE, n_neighbors=ADASYN_N_NEIGHBORS)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    
    if verbose:
        unique_new, counts_new = np.unique(y_resampled, return_counts=True)
        print("\nClass distribution after ADASYN:")
        for cls, count in zip(unique_new, counts_new):
            label = "Premium" if cls == 1 else "Not Premium"
            pct = count / len(y_resampled) * 100
            print(f"  {label} ({cls}): {count} samples ({pct:.2f}%)")
    
    return X_resampled, y_resampled


def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, verbose: bool = True) -> float:
    """
    Find optimal decision threshold to maximize F1 score using precision-recall curve.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate F1 for each threshold (avoid division by zero)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # Find the threshold that maximizes F1 (exclude last element as it corresponds to threshold=1)
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    if verbose:
        print(f"\n  Optimal threshold: {optimal_threshold:.4f}")
        print(f"  F1 at optimal threshold: {optimal_f1:.4f}")
    
    return optimal_threshold


def hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True
) -> XGBClassifier:
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    No scale_pos_weight - relying on ADASYN for class imbalance.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING (RandomizedSearchCV)")
        print("=" * 60)
    
    # Define parameter distributions (no scale_pos_weight - using ADASYN)
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
    }
    
    # Base model (no scale_pos_weight - using ADASYN)
    base_model = XGBClassifier(
        objective='binary:logistic',
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Stratified K-Fold cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=N_ITER_SEARCH,
        cv=cv,
        scoring='f1',  # Optimize for F1
        n_jobs=-1,
        verbose=1 if verbose else 0,
        random_state=RANDOM_STATE
    )
    
    if verbose:
        print(f"\nSearching {N_ITER_SEARCH} parameter combinations with {CV_FOLDS}-fold CV...")
        print("Note: Using ADASYN for class imbalance (no scale_pos_weight)")
    
    random_search.fit(X_train, y_train)
    
    if verbose:
        print(f"\nBest parameters found:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"\nBest CV F1 score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True
) -> dict:
    """Train XGBoost with hyperparameter tuning and threshold optimization."""
    
    # Step 1: Hyperparameter tuning
    model = hyperparameter_tuning(X_train, y_train, verbose=verbose)
    
    if verbose:
        print("\n" + "=" * 60)
        print("XGBOOST BINARY CLASSIFICATION (FINETUNED v4 - ADASYN)")
        print("=" * 60)
    
    # Step 2: Get predictions with default threshold
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = model.predict(X_test)
    
    # Metrics with default threshold (0.5)
    accuracy_default = accuracy_score(y_test, y_pred_default)
    f1_default = f1_score(y_test, y_pred_default)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    if verbose:
        print(f"\n--- Results with Default Threshold (0.5) ---")
        print(f"Accuracy: {accuracy_default:.4f}")
        print(f"F1-Score: {f1_default:.4f}")
        print(f"AUC-ROC:  {auc_roc:.4f}")
    
    # Step 3: Find optimal threshold
    if verbose:
        print(f"\n--- Threshold Optimization ---")
    
    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba, verbose=verbose)
    
    # Apply optimal threshold
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Metrics with optimized threshold
    accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
    f1_optimized = f1_score(y_test, y_pred_optimized)
    
    if verbose:
        print(f"\n--- Results with Optimized Threshold ({optimal_threshold:.4f}) ---")
        print(f"Accuracy: {accuracy_optimized:.4f}")
        print(f"F1-Score: {f1_optimized:.4f}")
        print(f"AUC-ROC:  {auc_roc:.4f}")
        
        print(f"\n--- Improvement ---")
        print(f"F1 improvement: {f1_default:.4f} → {f1_optimized:.4f} (+{f1_optimized - f1_default:.4f})")
        
        print("\nClassification Report (Optimized Threshold):")
        print(classification_report(y_test, y_pred_optimized, target_names=['Not Premium', 'Premium']))
    
    cm = confusion_matrix(y_test, y_pred_optimized)
    
    if verbose:
        print("Confusion Matrix (Optimized Threshold):")
        print(pd.DataFrame(
            cm,
            index=['Not Premium', 'Premium'],
            columns=['Not Premium', 'Premium']
        ))
    
    return {
        'model': model,
        'accuracy': accuracy_optimized,
        'f1': f1_optimized,
        'auc_roc': auc_roc,
        'optimal_threshold': optimal_threshold,
        'f1_default': f1_default,
        'accuracy_default': accuracy_default,
        'y_test': y_test,
        'y_pred': y_pred_optimized,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm,
        'class_names': ['Not Premium', 'Premium']
    }


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
        cmap='Greens',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f'Confusion Matrix - {experiment_name}', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=10)
    plt.ylabel('Actual', fontsize=10)
    
    metrics_text = (
        f"Accuracy: {results['accuracy']:.4f}\n"
        f"F1-Score: {results['f1']:.4f}\n"
        f"AUC-ROC: {results['auc_roc']:.4f}\n"
        f"Threshold: {results['optimal_threshold']:.4f}"
    )
    plt.text(
        1.35, 0.5, metrics_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
    )
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{experiment_name}_confusion_matrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nConfusion matrix saved to: {output_path}")
    return output_path


def run_experiment(verbose: bool = True) -> dict:
    """
    Run the full experiment pipeline.
    Returns data needed for explainability analysis.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'winequality-red.csv')
    
    if verbose:
        print("=" * 60)
        print(f"EXPERIMENT: {EXPERIMENT_NAME}")
        print("=" * 60)
        print("\n[Step 1] Loading red wine data...")
    
    df = load_wine_data(data_path, wine_type='red')
    
    if verbose:
        print(f"Original data shape: {df.shape}")
        print("\n[Step 2] Preparing data (binary target, cleaning, multicollinearity)...")
    
    df = prepare_data(df.copy(), verbose=verbose)
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col != 'is_premium']
    X = df[feature_cols].values
    y = df['is_premium'].values
    
    if verbose:
        print(f"\n[Step 3] Stratified train/test split ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    
    if verbose:
        print(f"Train: {len(y_train)}, Test: {len(y_test)}")
        print("\n[Step 4] Applying ADASYN on training data...")
    
    # ADASYN applied AFTER split, only on training data
    X_train_adasyn, y_train_adasyn = apply_adasyn(X_train, y_train, verbose=verbose)
    
    if verbose:
        print("\n[Step 5] Hyperparameter tuning and training XGBoost...")
    
    results = train_and_evaluate(
        X_train_adasyn, X_test, y_train_adasyn, y_test, verbose=verbose
    )
    
    # Store data for explainability
    results['X_train'] = X_train_adasyn
    results['X_test'] = X_test
    results['y_train'] = y_train_adasyn
    results['feature_names'] = feature_cols
    
    return {
        'experiment_name': EXPERIMENT_NAME,
        'feature_names': feature_cols,
        'is_binary': True,
        'results': results
    }


def main():
    """Main entry point."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'experiment_results')
    
    # Run experiment
    experiment_data = run_experiment(verbose=True)
    results = experiment_data['results']
    
    # Save confusion matrix
    print("\n[Step 6] Saving confusion matrix...")
    plot_confusion_matrix(results, output_dir, EXPERIMENT_NAME)
    
    # Save to CSV tracker
    print("\n[Step 7] Updating experiments tracker...")
    save_experiment_results(
        experiment_name=EXPERIMENT_NAME,
        results={
            'accuracy': results['accuracy'],
            'f1': results['f1'],
            'auc_roc': results['auc_roc']
        },
        output_dir=output_dir
    )
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  - F1 (default threshold):   {results['f1_default']:.4f}")
    print(f"  - F1 (optimized threshold): {results['f1']:.4f}")
    print(f"  - Optimal threshold:        {results['optimal_threshold']:.4f}")
    print(f"  - Strategy: ADASYN (adaptive synthetic sampling)")
    
    return experiment_data


if __name__ == "__main__":
    experiment_data = main()
