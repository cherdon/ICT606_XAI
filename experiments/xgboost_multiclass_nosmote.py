"""
XGBoost Multi-class Classification for Wine Quality (low/medium/high).
WITHOUT SMOTE - uses original imbalanced data.

================================================================================
RESULTS SUMMARY
================================================================================

Multi-class Classification (low/medium/high):
  - Accuracy:      0.6029
  - F1 (weighted): 0.6008
  - F1 (macro):    0.5754
  - AUC-ROC:       0.7466

================================================================================

Preprocessing Pipeline:
1. Multi-class target creation (low: <6, medium: 6, high: >=7)
2. Handle missing values and duplicates
3. Remove multicollinear features (correlation > 0.7)
4. Stratified train/test split (80/20)
5. NO SMOTE (baseline with imbalanced data)

Model: XGBoost Multi-class Classifier
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


def prepare_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Prepare dataset: create multi-class target, clean data, remove multicollinearity.
    """
    # Step 1: Create multi-class target
    bin_logic = {
        'low': (None, 6),      # quality < 6
        'medium': (6, 7),      # quality >= 6 and < 7 (i.e., quality == 6)
        'high': (7, None)      # quality >= 7
    }
    
    df = bin_column(
        df,
        column='quality',
        bin_logic=bin_logic,
        new_column_name='quality_category',
        drop_original=True
    )
    
    if verbose:
        print("\nMulti-class target distribution:")
        print(check_class_distribution(df, 'quality_category'))
    
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
        target_column='quality_category',
        threshold=CORRELATION_THRESHOLD,
        verbose=verbose
    )
    
    return df


def print_class_distribution(y: np.ndarray, label_encoder: LabelEncoder, verbose: bool = True):
    """Print class distribution."""
    if not verbose:
        return
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    print("\nClass distribution (NO SMOTE):")
    for cls, count in zip(unique, counts):
        label = label_encoder.inverse_transform([cls])[0]
        pct = count / total * 100
        print(f"  {label} ({cls}): {count} samples ({pct:.2f}%)")


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    verbose: bool = True
) -> dict:
    """Train XGBoost and evaluate performance."""
    if verbose:
        print("\n" + "=" * 60)
        print("XGBOOST MULTI-CLASS CLASSIFICATION (NO SMOTE)")
        print("=" * 60)
    
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric='mlogloss'
    )
    
    if verbose:
        print("\nTraining XGBoost...")
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Metrics
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
    class_names = list(label_encoder.classes_)
    
    if verbose:
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"F1-Score (weighted): {f1_weighted:.4f}")
        print(f"F1-Score (macro): {f1_macro:.4f}")
        if auc_roc is not None:
            print(f"AUC-ROC (weighted OvR): {auc_roc:.4f}")
        
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
        'y_test': y_test,
        'y_pred': y_pred,
        'confusion_matrix': cm,
        'class_names': class_names,
        'label_encoder': label_encoder
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
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f'Confusion Matrix - {experiment_name}', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=10)
    plt.ylabel('Actual', fontsize=10)
    
    auc_text = f"{results['auc_roc']:.4f}" if results['auc_roc'] else "N/A"
    metrics_text = (
        f"Accuracy: {results['accuracy']:.4f}\n"
        f"F1 (weighted): {results['f1_weighted']:.4f}\n"
        f"AUC-ROC: {auc_text}"
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
        print("\n[Step 2] Preparing data (multi-class target, cleaning, multicollinearity)...")
    
    df = prepare_data(df.copy(), verbose=verbose)
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col != 'quality_category']
    X = df[feature_cols].values
    y_raw = df['quality_category'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    if verbose:
        print(f"\nLabel encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
        print(f"\n[Step 3] Stratified train/test split ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    
    if verbose:
        print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    print_class_distribution(y_train, label_encoder, verbose=verbose)
    
    if verbose:
        print("\n[Step 4] Training and evaluating XGBoost...")
    
    results = train_and_evaluate(
        X_train, X_test, y_train, y_test, label_encoder, verbose=verbose
    )
    
    # Store data for explainability
    results['X_train'] = X_train
    results['X_test'] = X_test
    results['y_train'] = y_train
    results['feature_names'] = feature_cols
    
    return {
        'experiment_name': EXPERIMENT_NAME,
        'feature_names': feature_cols,
        'is_binary': False,
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
    print("\n[Step 5] Saving confusion matrix...")
    plot_confusion_matrix(results, output_dir, EXPERIMENT_NAME)
    
    # Save to CSV tracker
    print("\n[Step 6] Updating experiments tracker...")
    save_experiment_results(
        experiment_name=EXPERIMENT_NAME,
        results={
            'accuracy': results['accuracy'],
            'f1_weighted': results['f1_weighted'],
            'f1_macro': results['f1_macro'],
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
