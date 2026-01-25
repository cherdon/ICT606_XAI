"""
SVM Binary Classification for Wine Quality (Premium vs Non-Premium).
Uses enhanced preprocessing pipeline with SMOTE for class balancing.

================================================================================
RESULTS SUMMARY
================================================================================

Binary Classification (is_premium):
  - Accuracy:      0.8309
  - F1-Score:      0.5208
  - AUC-ROC:       0.8657

Note: Results unchanged from before (correlation threshold 0.7 removed no features)
================================================================================

Preprocessing Pipeline:
1. Binary target creation (quality >= 7 → Premium)
2. Handle missing values and duplicates
3. Remove multicollinear features (correlation > 0.8)
4. Stratified train/test split (80/20)
5. Yeo-Johnson transformation (fit on train)
6. RobustScaler (fit on train)
7. SMOTE oversampling (on training data only)

Model: Support Vector Machine (SVM) with RBF kernel
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
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE

from preprocessing import (
    load_wine_data,
    create_binary_column,
    check_class_distribution,
    handle_missing_values,
    remove_duplicates,
    remove_multicollinear_features,
    apply_yeo_johnson_transform,
    apply_robust_scaling
)
from experiments.experiment_tracker import save_experiment_results

# Experiment configuration
EXPERIMENT_NAME = os.path.splitext(os.path.basename(__file__))[0]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CORRELATION_THRESHOLD = 0.7
MIN_CLASS_PERCENTAGE = 0.15


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


def apply_smote_if_needed(
    X: np.ndarray,
    y: np.ndarray,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Apply SMOTE if minority class is below threshold."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    percentages = counts / total * 100
    
    if verbose:
        print("\nClass distribution before SMOTE:")
        for cls, count, pct in zip(unique, counts, percentages):
            label = "Premium" if cls == 1 else "Not Premium"
            print(f"  {label} ({cls}): {count} samples ({pct:.2f}%)")
    
    min_class_pct = percentages.min() / 100
    
    if min_class_pct < MIN_CLASS_PERCENTAGE:
        if verbose:
            print(f"\nMinority class ({min_class_pct*100:.2f}%) below {MIN_CLASS_PERCENTAGE*100:.0f}% threshold.")
            print("Applying SMOTE...")
        
        smote = SMOTE(random_state=RANDOM_STATE)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        if verbose:
            unique_new, counts_new = np.unique(y_resampled, return_counts=True)
            print("\nClass distribution after SMOTE:")
            for cls, count in zip(unique_new, counts_new):
                label = "Premium" if cls == 1 else "Not Premium"
                pct = count / len(y_resampled) * 100
                print(f"  {label} ({cls}): {count} samples ({pct:.2f}%)")
        
        return X_resampled, y_resampled, True
    else:
        if verbose:
            print(f"\nAll classes above {MIN_CLASS_PERCENTAGE*100:.0f}% threshold. SMOTE not needed.")
        return X, y, False


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True
) -> dict:
    """Train SVM and evaluate performance."""
    if verbose:
        print("\n" + "=" * 60)
        print("SVM BINARY CLASSIFICATION")
        print("=" * 60)
    
    # SVM with RBF kernel
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,  # Required for predict_proba (AUC-ROC)
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    
    if verbose:
        print("\nTraining SVM (RBF kernel)...")
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    cm = confusion_matrix(y_test, y_pred)
    
    if verbose:
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC:  {auc_roc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Premium', 'Premium']))
        
        print("Confusion Matrix:")
        print(pd.DataFrame(
            cm,
            index=['Not Premium', 'Premium'],
            columns=['Not Premium', 'Premium']
        ))
    
    return {
        'model': model,
        'accuracy': accuracy,
        'f1': f1,
        'auc_roc': auc_roc,
        'y_test': y_test,
        'y_pred': y_pred,
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
        cmap='Purples',  # Purple for SVM
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f'Confusion Matrix - {experiment_name}', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=10)
    plt.ylabel('Actual', fontsize=10)
    
    metrics_text = (
        f"Accuracy: {results['accuracy']:.4f}\n"
        f"F1-Score: {results['f1']:.4f}\n"
        f"AUC-ROC: {results['auc_roc']:.4f}"
    )
    plt.text(
        1.35, 0.5, metrics_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5)
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
        print("\n[Step 4] Applying Yeo-Johnson transformation...")
    
    X_train, X_test, yj_transformer = apply_yeo_johnson_transform(
        X_train, X_test, feature_cols, verbose=verbose
    )
    
    if verbose:
        print("\n[Step 5] Applying RobustScaler...")
    
    X_train, X_test, scaler = apply_robust_scaling(
        X_train, X_test, verbose=verbose
    )
    
    if verbose:
        print("\n[Step 6] Applying SMOTE on training data...")
    
    X_train_smote, y_train_smote, smote_applied = apply_smote_if_needed(
        X_train, y_train, verbose=verbose
    )
    
    if verbose:
        print("\n[Step 7] Training and evaluating SVM...")
    
    results = train_and_evaluate(
        X_train_smote, X_test, y_train_smote, y_test, verbose=verbose
    )
    
    # Store data for explainability
    results['X_train'] = X_train_smote
    results['X_test'] = X_test
    results['y_train'] = y_train_smote
    results['feature_names'] = feature_cols
    results['yj_transformer'] = yj_transformer
    results['scaler'] = scaler
    
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
    print("\n[Step 8] Saving confusion matrix...")
    plot_confusion_matrix(results, output_dir, EXPERIMENT_NAME)
    
    # Save to CSV tracker
    print("\n[Step 9] Updating experiments tracker...")
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
    
    return experiment_data


if __name__ == "__main__":
    experiment_data = main()
