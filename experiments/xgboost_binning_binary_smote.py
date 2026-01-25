"""
XGBoost Binary Classification for Wine Quality (is_premium).
Uses SMOTE for handling class imbalance.

================================================================================
RESULTS SUMMARY
================================================================================

Binary Classification (is_premium):
  - Accuracy:      0.9149
  - F1-Score:      0.9187
  - AUC-ROC:       0.9753

================================================================================

This experiment:
1. Loads red wine data
2. Creates binary target: is_premium (1 if quality >= 7, else 0)
3. Applies SMOTE if minority class is underrepresented (<15%)
4. Trains XGBoost binary classifier
5. Evaluates using Accuracy, F1-Score, and AUC-ROC
6. Saves confusion matrix visualization and results to CSV

Goal: Determine XAI for why a wine is premium quality
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
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from preprocessing import (
    load_wine_data,
    create_binary_column,
    check_class_distribution,
    handle_missing_values,
    remove_duplicates
)
from experiments.experiment_tracker import save_experiment_results

# Experiment name (used for saving results)
EXPERIMENT_NAME = os.path.splitext(os.path.basename(__file__))[0]


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
    """Clean data for XGBoost training."""
    df = handle_missing_values(df, strategy='drop')
    df = remove_duplicates(df)
    
    if verbose:
        print(f"Data shape after cleaning: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
    
    return df


def apply_smote_if_needed(
    X: np.ndarray, 
    y: np.ndarray, 
    min_percentage: float = 0.15,
    random_state: int = 42,
    verbose: bool = True
) -> tuple:
    """Apply SMOTE if minority class is below threshold."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    percentages = counts / total * 100
    
    if verbose:
        print("\nClass distribution before SMOTE:")
        for cls, count, pct in zip(unique, counts, percentages):
            print(f"  Class {cls}: {count} samples ({pct:.2f}%)")
    
    min_class_pct = percentages.min() / 100
    
    if min_class_pct < min_percentage:
        if verbose:
            print(f"\nMinority class ({min_class_pct*100:.2f}%) is below {min_percentage*100:.0f}% threshold.")
            print("Applying SMOTE...")
        
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        if verbose:
            unique_new, counts_new = np.unique(y_resampled, return_counts=True)
            print("\nClass distribution after SMOTE:")
            for cls, count in zip(unique_new, counts_new):
                pct = count / len(y_resampled) * 100
                print(f"  Class {cls}: {count} samples ({pct:.2f}%)")
        
        return X_resampled, y_resampled, True
    else:
        if verbose:
            print(f"\nAll classes are above {min_percentage*100:.0f}% threshold. SMOTE not needed.")
        return X, y, False


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    verbose: bool = True
) -> dict:
    """Train and evaluate XGBoost for binary classification."""
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


def plot_confusion_matrix(results: dict, output_dir: str, experiment_name: str):
    """Plot and save confusion matrix."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    
    class_names = ['Not Premium', 'Premium']
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
        f"AUC-ROC: {results['auc_roc']:.4f}"
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
    Run the full experiment and return all data needed for explainability.
    """
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    MIN_CLASS_PERCENTAGE = 0.15
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'winequality-red.csv')
    
    if verbose:
        print("="*60)
        print("WINE QUALITY - BINARY CLASSIFICATION WITH SMOTE")
        print("="*60)
        print("\n[Step 1] Loading red wine data...")
    
    df = load_wine_data(data_path, wine_type='red')
    
    if verbose:
        print(f"\nOriginal data shape: {df.shape}")
        print("\n[Step 2] Preparing binary dataset...")
    
    df_binary = prepare_binary_dataset(df.copy())
    if verbose:
        print("\nBinary distribution:")
        print(check_class_distribution(df_binary, 'is_premium'))
        print("\n[Step 3] Cleaning data for XGBoost...")
    
    df_binary = clean_for_xgboost(df_binary, verbose=verbose)
    
    # Prepare features and targets
    feature_cols = [col for col in df_binary.columns if col != 'is_premium']
    X = df_binary[feature_cols].values
    y = df_binary['is_premium'].values
    
    if verbose:
        print("\n[Step 4] Applying SMOTE if needed...")
    
    X_resampled, y_resampled, _ = apply_smote_if_needed(
        X, y, min_percentage=MIN_CLASS_PERCENTAGE,
        random_state=RANDOM_STATE, verbose=verbose
    )
    
    if verbose:
        print("\n[Step 5] Performing stratified train-test split...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled,
        test_size=TEST_SIZE,
        stratify=y_resampled,
        random_state=RANDOM_STATE
    )
    if verbose:
        print(f"Train: {len(y_train)}, Test: {len(y_test)}")
        print("\n[Step 6] Training XGBoost and evaluating...")
    
    results = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        random_state=RANDOM_STATE, verbose=verbose
    )
    
    # Add data to results for explainability
    results['X_train'] = X_train
    results['X_test'] = X_test
    results['y_train'] = y_train
    results['feature_names'] = feature_cols
    results['class_names'] = ['Not Premium', 'Premium']
    
    if verbose:
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"\nAccuracy: {results['accuracy']:.4f}")
        print(f"F1-Score: {results['f1']:.4f}")
        print(f"AUC-ROC:  {results['auc_roc']:.4f}")
    
    return {
        'experiment_name': EXPERIMENT_NAME,
        'feature_names': feature_cols,
        'results': results
    }


def main():
    """Main experiment pipeline with visualization and CSV tracking."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'experiment_results')
    
    # Run experiment
    experiment_data = run_experiment(verbose=True)
    results = experiment_data['results']
    
    # Save confusion matrix
    print("\n[Step 7] Saving confusion matrix visualization...")
    plot_confusion_matrix(results, output_dir, EXPERIMENT_NAME)
    
    # Save results to CSV tracker
    print("\n[Step 8] Saving results to experiments tracker...")
    save_experiment_results(
        experiment_name=EXPERIMENT_NAME,
        results={
            'accuracy': results['accuracy'],
            'f1': results['f1'],
            'auc_roc': results['auc_roc']
        },
        output_dir=output_dir
    )
    
    return experiment_data


if __name__ == "__main__":
    experiment_data = main()
