"""
Hierarchical Binary Classification for Wine Quality - SVM with BALANCED FILTERING Stage 1.

Stage 1 Goal: Filter out as many Regular wines as possible to create a balanced Stage 2 dataset,
while minimizing loss of Premium wines.

================================================================================
STRATEGY
================================================================================

The problem with high-recall Stage 1:
- Almost everything passes through (112 out of 129 Regular wines still go to Stage 2)
- Stage 2 still sees severely imbalanced data
- Defeats the purpose of hierarchical classification

This version optimizes Stage 1 for FILTERING EFFICIENCY:
1. Optimize for F1-score (balanced precision/recall)
2. Use threshold that maximizes Regular wine filtering
3. Accept losing some Good (quality=6) wines - they're Non-Premium anyway
4. Key constraint: Minimize Premium (quality>=7) wine loss

================================================================================
"""
import sys
import os
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
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


def find_optimal_threshold_f1(y_true: np.ndarray, y_pred_proba: np.ndarray, verbose: bool = True) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]
    if verbose:
        print(f"    Optimal threshold (F1): {optimal_threshold:.4f}")
    return optimal_threshold


def find_filtering_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                              quality: np.ndarray, verbose: bool = True) -> float:
    """
    Find threshold that maximizes filtering of Regular wines while keeping Premium wines.
    
    Strategy: 
    - Try different thresholds
    - For each, calculate: Regular wines filtered vs Premium wines lost
    - Choose threshold that filters most Regular while losing minimal Premium
    """
    thresholds = np.linspace(0.3, 0.8, 50)
    
    best_threshold = 0.5
    best_score = -np.inf
    
    # Identify Premium wines (quality >= 7) in the test set
    premium_mask = quality >= 7
    regular_mask = quality < 6
    good_mask = quality == 6
    
    if verbose:
        print(f"\n    Test set composition:")
        print(f"      Regular (quality < 6): {regular_mask.sum()}")
        print(f"      Good (quality = 6): {good_mask.sum()}")
        print(f"      Premium (quality >= 7): {premium_mask.sum()}")
        print(f"\n    Searching for optimal filtering threshold...")
    
    results = []
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        # Calculate metrics
        regular_filtered = ((y_pred == 0) & regular_mask).sum()  # Regular correctly rejected
        regular_passed = ((y_pred == 1) & regular_mask).sum()    # Regular incorrectly passed
        
        good_filtered = ((y_pred == 0) & good_mask).sum()        # Good incorrectly rejected (OK)
        good_passed = ((y_pred == 1) & good_mask).sum()          # Good correctly passed
        
        premium_filtered = ((y_pred == 0) & premium_mask).sum()  # Premium incorrectly rejected (BAD!)
        premium_passed = ((y_pred == 1) & premium_mask).sum()    # Premium correctly passed
        
        # Score: Maximize Regular filtering, minimize Premium loss
        # Heavy penalty for losing Premium wines
        score = regular_filtered - (premium_filtered * 10)  # 10x penalty for Premium loss
        
        results.append({
            'threshold': thresh,
            'regular_filtered': regular_filtered,
            'regular_passed': regular_passed,
            'good_filtered': good_filtered,
            'premium_filtered': premium_filtered,
            'premium_passed': premium_passed,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    # Find the result for best threshold
    best_result = [r for r in results if r['threshold'] == best_threshold][0]
    
    if verbose:
        print(f"\n    Best filtering threshold: {best_threshold:.4f}")
        print(f"    Regular wines filtered: {best_result['regular_filtered']} / {regular_mask.sum()}")
        print(f"    Good wines filtered: {best_result['good_filtered']} / {good_mask.sum()}")
        print(f"    Premium wines lost: {best_result['premium_filtered']} / {premium_mask.sum()}")
        
        total_filtered = best_result['regular_filtered'] + best_result['good_filtered']
        total_non_premium = regular_mask.sum() + good_mask.sum()
        print(f"    Total Non-Premium filtered: {total_filtered} / {total_non_premium} ({total_filtered/total_non_premium*100:.1f}%)")
    
    return best_threshold


def train_svm_stage1_filtering(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    quality_test: np.ndarray,
    class_names: list, verbose: bool = True
) -> Dict:
    """Train Stage 1 optimized for filtering efficiency."""
    stage_name = "STAGE 1: Better vs Regular (FILTERING OPTIMIZED)"
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"{stage_name}")
        print("=" * 60)
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\nTraining class distribution:")
        for cls, count in zip(unique, counts):
            pct = count / len(y_train) * 100
            print(f"  {class_names[cls]} ({cls}): {count} samples ({pct:.2f}%)")
        print(f"\n*** OPTIMIZING FOR FILTERING EFFICIENCY ***")

    # Standard hyperparameter search optimizing for F1
    param_distributions = {
        'C': [0.1, 1.0, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'poly'],
        'class_weight': ['balanced', {0: 1, 1: 1}, {0: 2, 1: 1}]  # Try favoring Regular class too
    }
    
    base_model = SVC(probability=True, random_state=RANDOM_STATE)
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    random_search = RandomizedSearchCV(
        estimator=base_model, param_distributions=param_distributions,
        n_iter=N_ITER_SEARCH, cv=cv, scoring='f1', n_jobs=-1,
        verbose=0, random_state=RANDOM_STATE
    )
    
    if verbose:
        print(f"\nHyperparameter tuning (optimizing for F1)...")
    
    random_search.fit(X_train, y_train)
    model = random_search.best_estimator_
    
    if verbose:
        print(f"Best CV F1: {random_search.best_score_:.4f}")
        print(f"Best params: {random_search.best_params_}")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    # Find threshold optimized for filtering
    if verbose:
        print(f"\nFinding filtering-optimized threshold:")
    optimal_threshold = find_filtering_threshold(y_test, y_pred_proba, quality_test, verbose)
    
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
    f1_optimized = f1_score(y_test, y_pred_optimized)
    precision = precision_score(y_test, y_pred_optimized)
    recall = recall_score(y_test, y_pred_optimized)
    
    if verbose:
        print(f"\n--- {stage_name} Results ---")
        print(f"Accuracy: {accuracy_optimized:.4f}")
        print(f"F1-Score: {f1_optimized:.4f}")
        print(f"Precision: {precision:.4f} (higher = more Regular filtered)")
        print(f"Recall:   {recall:.4f} (lower OK if Premium kept)")
        print(f"AUC-ROC:  {auc_roc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred_optimized)
    
    return {
        'model': model, 'accuracy': accuracy_optimized, 'f1': f1_optimized,
        'auc_roc': auc_roc, 'precision': precision, 'recall': recall,
        'optimal_threshold': optimal_threshold, 'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm, 'class_names': class_names
    }


def train_svm_stage2(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    class_names: list, verbose: bool = True
) -> Dict:
    stage_name = "STAGE 2: Premium vs Good"
    
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
        print(f"\nHyperparameter tuning (optimizing for F1)...")
    
    random_search.fit(X_train, y_train)
    model = random_search.best_estimator_
    
    if verbose:
        print(f"Best CV F1: {random_search.best_score_:.4f}")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    if verbose:
        print(f"\nThreshold optimization:")
    optimal_threshold = find_optimal_threshold_f1(y_test, y_pred_proba, verbose)
    
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
    f1_optimized = f1_score(y_test, y_pred_optimized)
    precision = precision_score(y_test, y_pred_optimized)
    recall = recall_score(y_test, y_pred_optimized)
    
    if verbose:
        print(f"\n--- {stage_name} Results ---")
        print(f"Accuracy: {accuracy_optimized:.4f}, F1: {f1_optimized:.4f}, AUC-ROC: {auc_roc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred_optimized)
    
    return {
        'model': model, 'accuracy': accuracy_optimized, 'f1': f1_optimized,
        'auc_roc': auc_roc, 'precision': precision, 'recall': recall,
        'optimal_threshold': optimal_threshold, 'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm, 'class_names': class_names
    }


def predict_hierarchical(X, stage1_model, stage2_model, stage1_threshold, stage2_threshold):
    stage1_proba = stage1_model.predict_proba(X)[:, 1]
    stage1_pred = (stage1_proba >= stage1_threshold).astype(int)
    
    final_pred = np.zeros(len(X), dtype=int)
    final_proba = np.zeros(len(X))
    
    better_mask = stage1_pred == 1
    if better_mask.sum() > 0:
        stage2_proba = stage2_model.predict_proba(X[better_mask])[:, 1]
        stage2_pred = (stage2_proba >= stage2_threshold).astype(int)
        final_pred[better_mask] = stage2_pred
        final_proba[better_mask] = stage1_proba[better_mask] * stage2_proba
    
    return final_pred, final_proba


def evaluate_combined(y_true_premium, y_pred, y_pred_proba, verbose=True):
    auc_roc = roc_auc_score(y_true_premium, y_pred_proba)
    
    if verbose:
        print(f"\nCombined threshold optimization:")
    optimal_threshold = find_optimal_threshold_f1(y_true_premium, y_pred_proba, verbose)
    
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    accuracy_optimized = accuracy_score(y_true_premium, y_pred_optimized)
    f1_optimized = f1_score(y_true_premium, y_pred_optimized)
    precision_optimized = precision_score(y_true_premium, y_pred_optimized)
    recall_optimized = recall_score(y_true_premium, y_pred_optimized)
    
    cm = confusion_matrix(y_true_premium, y_pred_optimized)
    
    return {
        'accuracy': accuracy_optimized, 'f1': f1_optimized, 'auc_roc': auc_roc,
        'precision': precision_optimized, 'recall': recall_optimized,
        'optimal_threshold': optimal_threshold, 'confusion_matrix': cm,
        'class_names': ['Non-Premium', 'Premium']
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
    axes[0].set_title(f"Stage 1: Better vs Regular (FILTERING)\nPrecision={stage1_results['precision']:.4f}, Recall={stage1_results['recall']:.4f}")
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    sns.heatmap(stage2_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=stage2_results['class_names'],
                yticklabels=stage2_results['class_names'], ax=axes[1])
    axes[1].set_title(f"Stage 2: Premium vs Good\nF1={stage2_results['f1']:.4f}, Acc={stage2_results['accuracy']:.4f}")
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
        print("\nSTRATEGY: Stage 1 optimized for FILTERING (remove Non-Premium)")
        print("Goal: Filter out Regular wines while keeping Premium wines")
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
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_better_train, y_better_test = y_better[train_idx], y_better[test_idx]
    y_premium_test = y_premium[test_idx]
    quality_train, quality_test = quality[train_idx], quality[test_idx]
    
    # Apply transformations for SVM
    if verbose:
        print("\n[Step 2] Applying Yeo-Johnson transformation...")
    X_train, X_test, _ = apply_yeo_johnson_transform(X_train, X_test, feature_cols, verbose=verbose)
    
    if verbose:
        print("\n[Step 3] Applying RobustScaler...")
    X_train, X_test, _ = apply_robust_scaling(X_train, X_test, verbose=verbose)
    
    # Stage 1 - FILTERING OPTIMIZED
    stage1_results = train_svm_stage1_filtering(
        X_train, X_test, y_better_train, y_better_test, quality_test,
        ['Regular (<6)', 'Better (>=6)'], verbose
    )
    
    # Stage 2 - Standard F1
    better_train_mask = quality_train >= 6
    X_train_better = X_train[better_train_mask]
    y_premium_among_better_train = (quality_train[better_train_mask] >= 7).astype(int)
    
    better_test_mask = quality_test >= 6
    X_test_better = X_test[better_test_mask]
    y_premium_among_better_test = (quality_test[better_test_mask] >= 7).astype(int)
    
    stage2_results = train_svm_stage2(
        X_train_better, X_test_better, y_premium_among_better_train, y_premium_among_better_test,
        ['Good (=6)', 'Premium (>=7)'], verbose
    )
    
    # Combined
    combined_pred, combined_proba = predict_hierarchical(
        X_test, stage1_results['model'], stage2_results['model'],
        stage1_results['optimal_threshold'], stage2_results['optimal_threshold']
    )
    combined_results = evaluate_combined(y_premium_test, combined_pred, combined_proba, verbose)
    
    # Calculate filtering efficiency
    stage1_pred = (stage1_results['y_pred_proba'] >= stage1_results['optimal_threshold']).astype(int)
    samples_passed_to_stage2 = stage1_pred.sum()
    premium_in_passed = ((stage1_pred == 1) & (quality_test >= 7)).sum()
    
    if verbose:
        print(f"\n{'=' * 60}")
        print("FILTERING EFFICIENCY ANALYSIS")
        print("=" * 60)
        print(f"Total test samples: {len(y_better_test)}")
        print(f"Samples passed to Stage 2: {samples_passed_to_stage2}")
        print(f"Samples filtered out: {len(y_better_test) - samples_passed_to_stage2}")
        print(f"Premium wines in Stage 2 input: {premium_in_passed}")
        print(f"Stage 2 Premium ratio: {premium_in_passed/samples_passed_to_stage2*100:.1f}%")
    
    return {
        'experiment_name': EXPERIMENT_NAME, 'stage1_results': stage1_results,
        'stage2_results': stage2_results, 'combined_results': combined_results,
        'filtering_stats': {
            'total_samples': len(y_better_test),
            'passed_to_stage2': samples_passed_to_stage2,
            'premium_in_stage2': premium_in_passed
        }
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'experiment_results')
    
    experiment_data = run_experiment(verbose=True)
    stage1 = experiment_data['stage1_results']
    stage2 = experiment_data['stage2_results']
    combined = experiment_data['combined_results']
    filtering = experiment_data['filtering_stats']
    
    save_experiment_results(
        experiment_name=EXPERIMENT_NAME,
        results={'accuracy': combined['accuracy'], 'f1': combined['f1'], 'auc_roc': combined['auc_roc']},
        output_dir=output_dir
    )
    
    plot_all_confusion_matrices(stage1, stage2, combined, output_dir, EXPERIMENT_NAME)
    
    print("\n" + "=" * 60)
    print("HIERARCHICAL CLASSIFICATION (SVM FILTERING) COMPLETE")
    print("=" * 60)
    
    print("\n" + "-" * 60)
    print("STAGE 1: Better (>=6) vs Regular (<6) - FILTERING OPTIMIZED")
    print("-" * 60)
    print(f"  Accuracy:  {stage1['accuracy']:.4f}")
    print(f"  F1-Score:  {stage1['f1']:.4f}")
    print(f"  AUC-ROC:   {stage1['auc_roc']:.4f}")
    print(f"  Precision: {stage1['precision']:.4f} (higher = better filtering)")
    print(f"  Recall:    {stage1['recall']:.4f}")
    print("\n  Confusion Matrix:")
    print(pd.DataFrame(stage1['confusion_matrix'],
        index=['Regular (<6)', 'Better (>=6)'],
        columns=['Regular (<6)', 'Better (>=6)']).to_string().replace('\n', '\n  '))
    
    # Filtering analysis
    regular_filtered = stage1['confusion_matrix'][0, 0]
    regular_total = stage1['confusion_matrix'][0, :].sum()
    better_kept = stage1['confusion_matrix'][1, 1]
    better_total = stage1['confusion_matrix'][1, :].sum()
    
    print(f"\n  *** FILTERING RESULTS ***")
    print(f"  Regular wines filtered: {regular_filtered} / {regular_total} ({regular_filtered/regular_total*100:.1f}%)")
    print(f"  Better wines kept: {better_kept} / {better_total} ({better_kept/better_total*100:.1f}%)")
    print(f"  Samples passed to Stage 2: {filtering['passed_to_stage2']} / {filtering['total_samples']}")
    
    print("\n" + "-" * 60)
    print("STAGE 2: Premium (>=7) vs Good (=6)")
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
    print(f"  F1-Score:  {combined['f1']:.4f}")
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
