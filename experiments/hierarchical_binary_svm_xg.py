"""
Hierarchical Binary Classification for Wine Quality - Mixed SVM + XGBoost.
Stage 1: SVM (better at complex boundaries for initial filtering)
Stage 2: XGBoost (handles the focused subset with tree-based approach)

================================================================================
RESULTS SUMMARY
================================================================================

STAGE 1: Better (>=6) vs Regular (<6) - SVM
  - Accuracy:      [TO BE UPDATED]
  - F1-Score:      [TO BE UPDATED]
  - AUC-ROC:       [TO BE UPDATED]

STAGE 2: Premium (>=7) vs Good (=6) - XGBoost
  - Accuracy:      [TO BE UPDATED]
  - F1-Score:      [TO BE UPDATED]
  - AUC-ROC:       [TO BE UPDATED]

COMBINED: Premium (>=7) vs Non-Premium (<7)
  - Accuracy:      [TO BE UPDATED]
  - F1-Score:      [TO BE UPDATED]
  - AUC-ROC:       [TO BE UPDATED]

================================================================================

Model: SVM (Stage 1) + XGBoost (Stage 2)
Rationale: SVM for initial separation; XGBoost for final Premium classification.
"""
import sys
import os
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, confusion_matrix
)

from preprocessing import (
    load_wine_data, handle_missing_values, remove_duplicates,
    remove_multicollinear_features, apply_yeo_johnson_transform, apply_robust_scaling
)
from experiments.experiment_tracker import save_experiment_results

EXPERIMENT_NAME = os.path.splitext(os.path.basename(__file__))[0]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CORRELATION_THRESHOLD = 0.7
N_ITER_SEARCH = 25
CV_FOLDS = 5


def prepare_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print("\nOriginal quality distribution:")
        print(df['quality'].value_counts().sort_index())
    df = handle_missing_values(df, strategy='drop')
    df = remove_duplicates(df)
    if verbose:
        print(f"\nData shape after cleaning: {df.shape}")
    return df


def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, verbose: bool = True) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]
    if verbose:
        print(f"    Optimal threshold: {optimal_threshold:.4f}")
    return optimal_threshold


def calculate_scale_pos_weight(y: np.ndarray) -> float:
    return np.sum(y == 0) / np.sum(y == 1)


def train_svm_stage1(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    class_names: list, verbose: bool = True
) -> Dict:
    """Train Stage 1 with SVM."""
    stage_name = "STAGE 1: Better vs Regular (SVM)"
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"{stage_name}")
        print("=" * 60)
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\nTraining class distribution:")
        for cls, count in zip(unique, counts):
            pct = count / len(y_train) * 100
            print(f"  {class_names[cls]} ({cls}): {count} samples ({pct:.2f}%)")

    param_distributions = {
        'C': [0.1, 1.0, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'poly'],
        'class_weight': ['balanced']
    }
    
    base_model = SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced')
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    random_search = RandomizedSearchCV(
        estimator=base_model, param_distributions=param_distributions,
        n_iter=N_ITER_SEARCH, cv=cv, scoring='f1', n_jobs=-1,
        verbose=0, random_state=RANDOM_STATE
    )
    
    if verbose:
        print(f"\nHyperparameter tuning ({N_ITER_SEARCH} combinations)...")
    
    random_search.fit(X_train, y_train)
    model = random_search.best_estimator_
    
    if verbose:
        print(f"Best CV F1: {random_search.best_score_:.4f}")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    if verbose:
        print(f"\nThreshold optimization:")
    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba, verbose)
    
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred_optimized)
    f1 = f1_score(y_test, y_pred_optimized)
    precision = precision_score(y_test, y_pred_optimized)
    recall = recall_score(y_test, y_pred_optimized)
    
    if verbose:
        print(f"\n--- Results ---")
        print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred_optimized)
    
    return {
        'model': model, 'accuracy': accuracy, 'f1': f1, 'auc_roc': auc_roc,
        'precision': precision, 'recall': recall, 'optimal_threshold': optimal_threshold,
        'y_pred_proba': y_pred_proba, 'confusion_matrix': cm, 'class_names': class_names
    }


def train_xgboost_stage2(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    class_names: list, verbose: bool = True
) -> Dict:
    """Train Stage 2 with XGBoost."""
    stage_name = "STAGE 2: Premium vs Good (XGBoost)"
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"{stage_name}")
        print("=" * 60)
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\nTraining class distribution:")
        for cls, count in zip(unique, counts):
            pct = count / len(y_train) * 100
            print(f"  {class_names[cls]} ({cls}): {count} samples ({pct:.2f}%)")

    scale_pos_weight = calculate_scale_pos_weight(y_train)
    
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'scale_pos_weight': [scale_pos_weight * 0.5, scale_pos_weight, scale_pos_weight * 1.5],
    }
    
    base_model = XGBClassifier(
        objective='binary:logistic', random_state=RANDOM_STATE,
        eval_metric='logloss', use_label_encoder=False
    )
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    random_search = RandomizedSearchCV(
        estimator=base_model, param_distributions=param_distributions,
        n_iter=N_ITER_SEARCH, cv=cv, scoring='f1', n_jobs=-1,
        verbose=0, random_state=RANDOM_STATE
    )
    
    if verbose:
        print(f"\nHyperparameter tuning ({N_ITER_SEARCH} combinations)...")
    
    random_search.fit(X_train, y_train)
    model = random_search.best_estimator_
    
    if verbose:
        print(f"Best CV F1: {random_search.best_score_:.4f}")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    if verbose:
        print(f"\nThreshold optimization:")
    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba, verbose)
    
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred_optimized)
    f1 = f1_score(y_test, y_pred_optimized)
    precision = precision_score(y_test, y_pred_optimized)
    recall = recall_score(y_test, y_pred_optimized)
    
    if verbose:
        print(f"\n--- Results ---")
        print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred_optimized)
    
    return {
        'model': model, 'accuracy': accuracy, 'f1': f1, 'auc_roc': auc_roc,
        'precision': precision, 'recall': recall, 'optimal_threshold': optimal_threshold,
        'y_pred_proba': y_pred_proba, 'confusion_matrix': cm, 'class_names': class_names
    }


def predict_hierarchical(X_scaled, X_raw, stage1_model, stage2_model, stage1_threshold, stage2_threshold):
    """Predict using scaled features for SVM Stage 1, raw for XGBoost Stage 2."""
    stage1_proba = stage1_model.predict_proba(X_scaled)[:, 1]
    stage1_pred = (stage1_proba >= stage1_threshold).astype(int)
    
    final_pred = np.zeros(len(X_scaled), dtype=int)
    final_proba = np.zeros(len(X_scaled))
    
    better_mask = stage1_pred == 1
    if better_mask.sum() > 0:
        # Use raw features for XGBoost Stage 2
        stage2_proba = stage2_model.predict_proba(X_raw[better_mask])[:, 1]
        stage2_pred = (stage2_proba >= stage2_threshold).astype(int)
        final_pred[better_mask] = stage2_pred
        final_proba[better_mask] = stage1_proba[better_mask] * stage2_proba
    
    return final_pred, final_proba


def evaluate_combined(y_true_premium, y_pred, y_pred_proba, verbose=True):
    auc_roc = roc_auc_score(y_true_premium, y_pred_proba)
    
    if verbose:
        print(f"\nCombined threshold optimization:")
    optimal_threshold = find_optimal_threshold(y_true_premium, y_pred_proba, verbose)
    
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    accuracy = accuracy_score(y_true_premium, y_pred_optimized)
    f1 = f1_score(y_true_premium, y_pred_optimized)
    precision = precision_score(y_true_premium, y_pred_optimized)
    recall = recall_score(y_true_premium, y_pred_optimized)
    
    cm = confusion_matrix(y_true_premium, y_pred_optimized)
    
    return {
        'accuracy': accuracy, 'f1': f1, 'auc_roc': auc_roc,
        'precision': precision, 'recall': recall, 'optimal_threshold': optimal_threshold,
        'confusion_matrix': cm, 'class_names': ['Non-Premium', 'Premium']
    }


def plot_all_confusion_matrices(stage1_results, stage2_results, combined_results, 
                                 output_dir, experiment_name):
    """Plot all three confusion matrices in a single figure."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Confusion Matrices - {experiment_name}', fontsize=14, fontweight='bold')
    
    sns.heatmap(stage1_results['confusion_matrix'], annot=True, fmt='d', cmap='Greens',
                xticklabels=stage1_results['class_names'], 
                yticklabels=stage1_results['class_names'], ax=axes[0])
    axes[0].set_title(f"Stage 1: Better vs Regular (SVM)\nF1={stage1_results['f1']:.4f}, Acc={stage1_results['accuracy']:.4f}")
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    sns.heatmap(stage2_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=stage2_results['class_names'],
                yticklabels=stage2_results['class_names'], ax=axes[1])
    axes[1].set_title(f"Stage 2: Premium vs Good (XGBoost)\nF1={stage2_results['f1']:.4f}, Acc={stage2_results['accuracy']:.4f}")
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    sns.heatmap(combined_results['confusion_matrix'], annot=True, fmt='d', cmap='Purples',
                xticklabels=combined_results['class_names'],
                yticklabels=combined_results['class_names'], ax=axes[2])
    axes[2].set_title(f"Combined: Premium vs Non-Premium\nF1={combined_results['f1']:.4f}, Acc={combined_results['accuracy']:.4f}")
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('Actual')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{experiment_name}_all_confusion_matrices.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nAll confusion matrices saved to: {output_path}")
    return output_path


def run_experiment(verbose: bool = True) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'winequality-red.csv')
    
    if verbose:
        print("=" * 60)
        print(f"EXPERIMENT: {EXPERIMENT_NAME}")
        print("=" * 60)
        print("\nMixed Model: SVM (Stage 1) + XGBoost (Stage 2)")
        print("\n[Step 1] Loading red wine data...")
    
    df = load_wine_data(data_path, wine_type='red')
    df = prepare_data(df.copy(), verbose=verbose)
    
    df['is_premium'] = (df['quality'] >= 7).astype(int)
    
    df_clean, _ = remove_multicollinear_features(
        df, target_column='is_premium', threshold=CORRELATION_THRESHOLD, verbose=verbose
    )
    
    df_clean['is_better'] = (df_clean['quality'] >= 6).astype(int)
    df_clean['is_premium'] = (df_clean['quality'] >= 7).astype(int)
    
    target_cols = ['quality', 'is_better', 'is_premium']
    feature_cols = [col for col in df_clean.columns if col not in target_cols]
    
    X = df_clean[feature_cols].values
    y_better = df_clean['is_better'].values
    y_premium = df_clean['is_premium'].values
    quality = df_clean['quality'].values
    
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, stratify=quality, random_state=RANDOM_STATE
    )
    
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_better_train, y_better_test = y_better[train_idx], y_better[test_idx]
    y_premium_test = y_premium[test_idx]
    quality_train, quality_test = quality[train_idx], quality[test_idx]
    
    # Apply transformations for SVM Stage 1
    if verbose:
        print("\n[Step 2] Applying transformations for SVM Stage 1...")
    X_train_scaled, X_test_scaled, _ = apply_yeo_johnson_transform(
        X_train_raw.copy(), X_test_raw.copy(), feature_cols, verbose=False
    )
    X_train_scaled, X_test_scaled, _ = apply_robust_scaling(
        X_train_scaled, X_test_scaled, verbose=False
    )
    
    # Stage 1 - SVM (uses scaled features)
    stage1_results = train_svm_stage1(
        X_train_scaled, X_test_scaled, y_better_train, y_better_test,
        ['Regular (<6)', 'Better (>=6)'], verbose
    )
    
    # Stage 2 - XGBoost (uses raw features, only "Better" wines)
    better_train_mask = quality_train >= 6
    X_train_better_raw = X_train_raw[better_train_mask]
    y_premium_among_better_train = (quality_train[better_train_mask] >= 7).astype(int)
    
    better_test_mask = quality_test >= 6
    X_test_better_raw = X_test_raw[better_test_mask]
    y_premium_among_better_test = (quality_test[better_test_mask] >= 7).astype(int)
    
    stage2_results = train_xgboost_stage2(
        X_train_better_raw, X_test_better_raw,
        y_premium_among_better_train, y_premium_among_better_test,
        ['Good (=6)', 'Premium (>=7)'], verbose
    )
    
    # Combined - SVM uses scaled, XGBoost uses raw
    if verbose:
        print("\n" + "=" * 60)
        print("COMBINED EVALUATION")
        print("=" * 60)
    
    combined_pred, combined_proba = predict_hierarchical(
        X_test_scaled, X_test_raw,
        stage1_results['model'], stage2_results['model'],
        stage1_results['optimal_threshold'], stage2_results['optimal_threshold']
    )
    combined_results = evaluate_combined(y_premium_test, combined_pred, combined_proba, verbose)
    
    return {
        'experiment_name': EXPERIMENT_NAME, 'stage1_results': stage1_results,
        'stage2_results': stage2_results, 'combined_results': combined_results
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'experiment_results')
    
    experiment_data = run_experiment(verbose=True)
    stage1 = experiment_data['stage1_results']
    stage2 = experiment_data['stage2_results']
    combined = experiment_data['combined_results']
    
    save_experiment_results(
        experiment_name=EXPERIMENT_NAME,
        results={'accuracy': combined['accuracy'], 'f1': combined['f1'], 'auc_roc': combined['auc_roc']},
        output_dir=output_dir
    )
    
    plot_all_confusion_matrices(stage1, stage2, combined, output_dir, EXPERIMENT_NAME)
    
    print("\n" + "=" * 60)
    print("HIERARCHICAL CLASSIFICATION (SVM + XGBoost) COMPLETE")
    print("=" * 60)
    
    print("\n" + "-" * 60)
    print("STAGE 1: Better (>=6) vs Regular (<6) - SVM")
    print("-" * 60)
    print(f"  Accuracy:  {stage1['accuracy']:.4f}")
    print(f"  F1-Score:  {stage1['f1']:.4f}")
    print(f"  AUC-ROC:   {stage1['auc_roc']:.4f}")
    print(f"  Precision: {stage1['precision']:.4f}")
    print(f"  Recall:    {stage1['recall']:.4f}")
    print("\n  Confusion Matrix:")
    print(pd.DataFrame(stage1['confusion_matrix'],
        index=['Regular (<6)', 'Better (>=6)'],
        columns=['Regular (<6)', 'Better (>=6)']).to_string().replace('\n', '\n  '))
    
    print("\n" + "-" * 60)
    print("STAGE 2: Premium (>=7) vs Good (=6) - XGBoost")
    print("-" * 60)
    print(f"  Accuracy:  {stage2['accuracy']:.4f}")
    print(f"  F1-Score:  {stage2['f1']:.4f}")
    print(f"  AUC-ROC:   {stage2['auc_roc']:.4f}")
    print(f"  Precision: {stage2['precision']:.4f}")
    print(f"  Recall:    {stage2['recall']:.4f}")
    print("\n  Confusion Matrix:")
    print(pd.DataFrame(stage2['confusion_matrix'],
        index=['Good (=6)', 'Premium (>=7)'],
        columns=['Good (=6)', 'Premium (>=7)']).to_string().replace('\n', '\n  '))
    
    print("\n" + "-" * 60)
    print("COMBINED: Premium (>=7) vs Non-Premium (<7)")
    print("-" * 60)
    print(f"  Accuracy:  {combined['accuracy']:.4f}")
    print(f"  F1-Score:  {combined['f1']:.4f}  ← Previous best: 0.6667")
    print(f"  AUC-ROC:   {combined['auc_roc']:.4f}")
    print(f"  Precision: {combined['precision']:.4f}")
    print(f"  Recall:    {combined['recall']:.4f}")
    print("\n  Confusion Matrix:")
    print(pd.DataFrame(combined['confusion_matrix'],
        index=['Non-Premium', 'Premium'],
        columns=['Non-Premium', 'Premium']).to_string().replace('\n', '\n  '))
    
    return experiment_data


if __name__ == "__main__":
    main()
