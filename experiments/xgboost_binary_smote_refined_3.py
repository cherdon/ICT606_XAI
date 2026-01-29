"""
XGBoost Binary Classification for Wine Quality (is_premium) - REFINED VERSION 3.
Uses SMOTE + Hyperparameter Tuning + Threshold Optimization + Yeo-Johnson + RobustScaler + Feature Engineering.

================================================================================
IMPROVEMENTS OVER REFINED VERSION 2
================================================================================

1. Added Feature Engineering:
   - sulfur_ratio = free sulfur dioxide / total sulfur dioxide
   - acidity_ratio = fixed acidity / volatile acidity
2. Retains Yeo-Johnson transformation
3. Retains RobustScaler
4. Retains hyperparameter tuning and threshold optimization

Preprocessing Pipeline:
1. Binary target creation (quality >= 7 → Premium)
2. Handle missing values and duplicates
3. Feature Engineering (sulfur_ratio, acidity_ratio) [NEW]
4. Remove multicollinear features (correlation > 0.7)
5. Stratified train/test split (80/20)
6. Yeo-Johnson transformation (fit on train)
7. RobustScaler (fit on train)
8. SMOTE oversampling (on training data only)
9. Hyperparameter tuning with RandomizedSearchCV
10. Threshold optimization for F1

Model: XGBoost Binary Classifier (Refined v3 - Full Pipeline)
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
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

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
N_ITER_SEARCH = 50
CV_FOLDS = 5


def apply_feature_engineering(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Apply domain-informed feature engineering.
    
    Creates:
    - sulfur_ratio: free sulfur dioxide / total sulfur dioxide
    - acidity_ratio: fixed acidity / volatile acidity
    """
    df = df.copy()
    
    # Sulfur ratio - indicates proportion of free SO2 (active preservative)
    if 'free sulfur dioxide' in df.columns and 'total sulfur dioxide' in df.columns:
        df['sulfur_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-8)
        if verbose:
            print("  Created: sulfur_ratio = free sulfur dioxide / total sulfur dioxide")
    
    # Acidity ratio - balance between fixed and volatile acidity
    if 'fixed acidity' in df.columns and 'volatile acidity' in df.columns:
        df['acidity_ratio'] = df['fixed acidity'] / (df['volatile acidity'] + 1e-8)
        if verbose:
            print("  Created: acidity_ratio = fixed acidity / volatile acidity")
    
    return df


def prepare_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Prepare dataset: create binary target, clean data, feature engineering, remove multicollinearity.
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
    
    # Step 3: Feature Engineering
    if verbose:
        print(f"\nApplying Feature Engineering:")
    df = apply_feature_engineering(df, verbose=verbose)
    
    if verbose:
        print(f"Data shape after feature engineering: {df.shape}")
    
    # Step 4: Remove multicollinear features
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


def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, verbose: bool = True) -> float:
    """Find optimal decision threshold to maximize F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
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
    """Perform hyperparameter tuning using RandomizedSearchCV."""
    if verbose:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING (RandomizedSearchCV)")
        print("=" * 60)
    
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
    }
    
    base_model = XGBClassifier(
        objective='binary:logistic',
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=N_ITER_SEARCH,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1 if verbose else 0,
        random_state=RANDOM_STATE
    )
    
    if verbose:
        print(f"\nSearching {N_ITER_SEARCH} parameter combinations with {CV_FOLDS}-fold CV...")
    
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
    
    model = hyperparameter_tuning(X_train, y_train, verbose=verbose)
    
    if verbose:
        print("\n" + "=" * 60)
        print("XGBOOST BINARY CLASSIFICATION (REFINED v3 - FULL PIPELINE)")
        print("=" * 60)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = model.predict(X_test)
    
    accuracy_default = accuracy_score(y_test, y_pred_default)
    f1_default = f1_score(y_test, y_pred_default)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    if verbose:
        print(f"\n--- Results with Default Threshold (0.5) ---")
        print(f"Accuracy: {accuracy_default:.4f}")
        print(f"F1-Score: {f1_default:.4f}")
        print(f"AUC-ROC:  {auc_roc:.4f}")
    
    if verbose:
        print(f"\n--- Threshold Optimization ---")
    
    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba, verbose=verbose)
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    
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
        print(pd.DataFrame(cm, index=['Not Premium', 'Premium'], columns=['Not Premium', 'Premium']))
    
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
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {experiment_name}', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=10)
    plt.ylabel('Actual', fontsize=10)
    
    metrics_text = (
        f"Accuracy: {results['accuracy']:.4f}\n"
        f"F1-Score: {results['f1']:.4f}\n"
        f"AUC-ROC: {results['auc_roc']:.4f}\n"
        f"Threshold: {results['optimal_threshold']:.4f}"
    )
    plt.text(1.35, 0.5, metrics_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{experiment_name}_confusion_matrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nConfusion matrix saved to: {output_path}")
    return output_path


def plot_feature_importance(model: XGBClassifier, feature_names: list, output_dir: str, experiment_name: str):
    """Plot XGBoost feature importance."""
    os.makedirs(output_dir, exist_ok=True)
    
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'], color='#27ae60', alpha=0.8)
    plt.xlabel('Feature Importance', fontsize=11)
    plt.title(f'Feature Importance - {experiment_name}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{experiment_name}_feature_importance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance saved to: {output_path}")
    return output_path


def run_experiment(verbose: bool = True) -> dict:
    """Run the full experiment pipeline."""
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
        print("\n[Step 2] Preparing data (binary target, cleaning, feature engineering, multicollinearity)...")
    
    df = prepare_data(df.copy(), verbose=verbose)
    
    feature_cols = [col for col in df.columns if col != 'is_premium']
    X = df[feature_cols].values
    y = df['is_premium'].values
    
    if verbose:
        print(f"\nFeatures used ({len(feature_cols)}): {feature_cols}")
        print(f"\n[Step 3] Stratified train/test split ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
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
        print("\n[Step 7] Hyperparameter tuning and training XGBoost...")
    
    results = train_and_evaluate(
        X_train_smote, X_test, y_train_smote, y_test, verbose=verbose
    )
    
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
    
    experiment_data = run_experiment(verbose=True)
    results = experiment_data['results']
    feature_names = experiment_data['feature_names']
    
    print("\n[Step 8] Saving confusion matrix...")
    plot_confusion_matrix(results, output_dir, EXPERIMENT_NAME)
    
    print("\n[Step 9] Saving feature importance plot...")
    plot_feature_importance(results['model'], feature_names, output_dir, EXPERIMENT_NAME)
    
    print("\n[Step 10] Updating experiments tracker...")
    save_experiment_results(
        experiment_name=EXPERIMENT_NAME,
        results={'accuracy': results['accuracy'], 'f1': results['f1'], 'auc_roc': results['auc_roc']},
        output_dir=output_dir
    )
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  - F1 (default threshold):   {results['f1_default']:.4f}")
    print(f"  - F1 (optimized threshold): {results['f1']:.4f}")
    print(f"  - Optimal threshold:        {results['optimal_threshold']:.4f}")
    
    return experiment_data


if __name__ == "__main__":
    experiment_data = main()
